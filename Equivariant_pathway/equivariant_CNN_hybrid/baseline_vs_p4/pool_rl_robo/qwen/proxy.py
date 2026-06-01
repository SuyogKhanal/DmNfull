"""OpenAI-compatible router that fans calls to two local vLLM servers.

Why this exists
---------------
The upstream P4 pipeline reads ``llm.model`` and ``llm.vlm_model`` from
config and includes the appropriate one in every chat-completions
request body. With two vLLM backends running side-by-side on a single
SLURM job (one Qwen3-32B on GPU 1 port 8000, one Qwen3-VL-32B on GPU 0
port 8001), we just need to route each request to the right port based
on the ``model`` field of the body.

Everything else passes through unchanged: headers, streaming bytes,
status codes. The pipeline OpenAI client (with
``OPENAI_BASE_URL=http://127.0.0.1:8002/v1``) talks to us, we talk to
the right backend.

Endpoints proxied:
  POST /v1/chat/completions   — routed by body["model"]
  POST /v1/completions        — routed by body["model"]   (legacy)
  GET  /v1/models             — forwarded to BOTH backends, results merged
  GET  /healthz               — local readiness probe used by the sbatch

Run via:
  python -m pool_x_selector.qwen.proxy \\
      --port 8002 --llm_port 8000 --vlm_port 8001 \\
      --vlm_model qwen3-vl-32b
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Set

# Qwen3 is a reasoning model: it emits <think>...</think> blocks inline in
# the response text. The upstream aggregator does a strict json.loads on
# the model output and the per-episode parser scans for a FINAL_REC block;
# a leading <think> prefix makes json.loads fail (→ empty prescription →
# 0 demos prescribed). Strip the think blocks from response text before
# returning to the pipeline. GPT on the OpenAI cluster never emitted these,
# which is why this only bites on the Qwen path.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_think(text: str) -> str:
    if not isinstance(text, str) or "<think>" not in text:
        return text
    cleaned = _THINK_RE.sub("", text)
    # Truncated/unclosed <think> (no matching </think>): drop the dangling
    # opener and everything after it — there is no usable answer there.
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.lstrip()


def _strip_think_recursive(obj: Any) -> Any:
    # Walk the responses-API envelope and strip think blocks from every
    # string leaf. Only text/content fields ever carry <think>, so this is
    # safe; it covers output_text, output[].content[].text, and variants.
    if isinstance(obj, str):
        return _strip_think(obj)
    if isinstance(obj, list):
        return [_strip_think_recursive(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _strip_think_recursive(v) for k, v in obj.items()}
    return obj

# Late imports — only required when the proxy is actually run, so the
# rest of the suite can import this package without pulling in fastapi.
try:
    import httpx
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as e:  # pragma: no cover
    sys.stderr.write(
        f"[qwen-proxy] missing dependency: {e}.\n"
        "Install with: pip install fastapi uvicorn httpx\n"
    )
    raise


def make_app(
    llm_port: int,
    vlm_port: int,
    vlm_model_names: Set[str],
    host: str = "127.0.0.1",
) -> "FastAPI":
    app = FastAPI(title="qwen-vllm-router")
    timeout = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=30.0)
    client = httpx.AsyncClient(timeout=timeout)
    llm_base = f"http://{host}:{llm_port}"
    vlm_base = f"http://{host}:{vlm_port}"

    @app.on_event("shutdown")
    async def _shutdown():
        await client.aclose()

    def _route_base_for_model(model: str) -> str:
        return vlm_base if model in vlm_model_names else llm_base

    async def _forward(base: str, path: str, request: Request) -> Response:
        body = await request.body()
        # Strip hop-by-hop / connection headers; pass everything else.
        fwd_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in {"host", "content-length", "connection"}
        }
        url = f"{base}{path}"
        # Stream-friendly: open a stream to the backend and pipe its
        # response back so chat-completions streaming chunks reach the
        # client without buffering.
        upstream = await client.send(
            client.build_request(
                request.method, url, headers=fwd_headers, content=body,
            ),
            stream=True,
        )
        # Drop hop-by-hop response headers.
        resp_headers = {
            k: v for k, v in upstream.headers.items()
            if k.lower() not in {"content-length", "transfer-encoding", "connection"}
        }

        async def _iter():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            _iter(), status_code=upstream.status_code, headers=resp_headers,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}
        model = str(payload.get("model", "") or "")
        target = _route_base_for_model(model)
        # Rebuild request with the buffered body since we already read it.
        async def _stream_once():
            yield body

        async def _send():
            req = client.build_request(
                "POST", f"{target}/v1/chat/completions",
                headers={
                    k: v for k, v in request.headers.items()
                    if k.lower() not in {"host", "content-length", "connection"}
                },
                content=body,
            )
            return await client.send(req, stream=True)

        upstream = await _send()
        resp_headers = {
            k: v for k, v in upstream.headers.items()
            if k.lower() not in {"content-length", "transfer-encoding", "connection"}
        }
        # Inject a one-line audit so the sbatch log shows what was routed.
        print(
            f"[qwen-proxy] POST /v1/chat/completions  model={model!r}  "
            f"-> port={target.rsplit(':', 1)[-1]}  status={upstream.status_code}",
            flush=True,
        )

        async def _iter():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            _iter(), status_code=upstream.status_code, headers=resp_headers,
        )

    @app.post("/v1/completions")
    async def completions(request: Request):
        # Legacy completions endpoint; route the same way.
        return await chat_completions(request)

    def _patch_responses_payload(payload: dict) -> bytes:
        # (1) vLLM 0.17.x ships with openai SDK 2.24, whose
        # ResponseInputImageParam marks `detail` as Required (auto/low/
        # high). The upstream pipeline runs on openai SDK 2.30 and omits
        # `detail`. OpenAI's hosted API tolerates the omission; vLLM
        # pydantic-rejects with 400. Inject `detail: "auto"` on every
        # input_image item that lacks one before forwarding.
        inp = payload.get("input")
        if isinstance(inp, list):
            for item in inp:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for c in content:
                        if (
                            isinstance(c, dict)
                            and c.get("type") == "input_image"
                            and "detail" not in c
                        ):
                            c["detail"] = "auto"
        # (2) The pipeline requests max_output_tokens=16384. Qwen3's hard
        # context ceiling is 40960 (max_position_embeddings), so a big
        # aggregator input + 16384 output blows past it → 400 "input+output
        # > context length". Cap output so input always has ~32k of room.
        # 8192 is ample for a <think> block plus the structured JSON/
        # FINAL_REC answer these calls produce. Paired with correction_n=20
        # (fewer failures → smaller aggregator prompt) this fits even an
        # all-fail round (~28k input + 8192 < 40960).
        cap = int(os.environ.get("PROXY_MAX_OUTPUT_TOKENS", "8192"))
        cur = payload.get("max_output_tokens")
        if not isinstance(cur, int) or cur > cap:
            payload["max_output_tokens"] = cap
        return json.dumps(payload).encode()

    @app.post("/v1/responses")
    async def responses_create(request: Request):
        # The upstream pipeline (vlm_analyser, reasoning, aggregator,
        # knowledge_fetcher) uses client.responses.create, which posts
        # to /v1/responses. Route by body["model"] same as chat.
        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}
        model = str(payload.get("model", "") or "")
        target = _route_base_for_model(model)
        # Rewrite the body to satisfy vLLM's stricter schema validation.
        forward_body = _patch_responses_payload(payload) if payload else body
        upstream = await client.send(
            client.build_request(
                "POST", f"{target}/v1/responses",
                headers={
                    k: v for k, v in request.headers.items()
                    if k.lower() not in {"host", "content-length", "connection"}
                },
                content=forward_body,
            ),
            stream=True,
        )
        resp_headers = {
            k: v for k, v in upstream.headers.items()
            if k.lower() not in {"content-length", "transfer-encoding", "connection"}
        }
        print(
            f"[qwen-proxy] POST /v1/responses  model={model!r}  "
            f"-> port={target.rsplit(':', 1)[-1]}  status={upstream.status_code}",
            flush=True,
        )

        ctype = upstream.headers.get("content-type", "")
        # Non-streaming JSON response: buffer it, strip <think> blocks from
        # the text fields, and return clean JSON. The pipeline calls
        # responses.create without stream=True, so this is the hot path.
        if "text/event-stream" not in ctype:
            raw = await upstream.aread()
            await upstream.aclose()
            try:
                data = json.loads(raw)
                data = _strip_think_recursive(data)
                out = json.dumps(data).encode()
            except Exception:
                out = raw  # not JSON / parse failed — pass through untouched
            return Response(
                content=out, status_code=upstream.status_code,
                headers=resp_headers, media_type="application/json",
            )

        # Streaming (SSE) fallback — pass through unmodified. The pipeline
        # doesn't stream these calls, so think-stripping isn't applied here.
        async def _iter():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            _iter(), status_code=upstream.status_code, headers=resp_headers,
        )

    @app.get("/v1/responses/{response_id}")
    async def responses_get(response_id: str, request: Request):
        # Read-side of the responses API. No body to inspect for model
        # routing, so try LLM backend first, fall back to VLM. Used by
        # SDK polling paths; rare in our workload.
        for base in (llm_base, vlm_base):
            try:
                r = await client.get(
                    f"{base}/v1/responses/{response_id}",
                    headers={
                        k: v for k, v in request.headers.items()
                        if k.lower() not in {"host", "connection"}
                    },
                    timeout=30.0,
                )
                if r.status_code != 404:
                    return Response(
                        content=r.content, status_code=r.status_code,
                        headers={
                            k: v for k, v in r.headers.items()
                            if k.lower() not in {"content-length", "transfer-encoding", "connection"}
                        },
                    )
            except Exception as exc:
                print(f"[qwen-proxy] GET /v1/responses/{response_id} {base} failed: {exc!r}", flush=True)
        return JSONResponse({"detail": "Not Found"}, status_code=404)

    @app.get("/v1/models")
    async def models():
        # Forward to both backends; merge data lists.
        out_data = []
        for base in (llm_base, vlm_base):
            try:
                r = await client.get(f"{base}/v1/models", timeout=10.0)
                if r.status_code == 200:
                    out_data.extend((r.json() or {}).get("data", []) or [])
            except Exception as exc:
                print(f"[qwen-proxy] /v1/models {base} failed: {exc!r}", flush=True)
        return JSONResponse({"object": "list", "data": out_data})

    @app.get("/healthz")
    async def healthz():
        out = {}
        for name, base in (("llm", llm_base), ("vlm", vlm_base)):
            try:
                r = await client.get(f"{base}/v1/models", timeout=5.0)
                out[name] = {"port": base.rsplit(":", 1)[-1], "ok": r.status_code == 200}
            except Exception as exc:
                out[name] = {"port": base.rsplit(":", 1)[-1], "ok": False, "error": repr(exc)}
        status = 200 if all(v.get("ok") for v in out.values()) else 503
        return JSONResponse(out, status_code=status)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002,
                    help="Port this proxy listens on.")
    ap.add_argument("--llm_port", type=int, default=8000,
                    help="Port of the LLM (text) vLLM server.")
    ap.add_argument("--vlm_port", type=int, default=8001,
                    help="Port of the VLM (vision) vLLM server.")
    ap.add_argument(
        "--vlm_model", action="append", default=None,
        help="Served model name(s) to route to the VLM port (default "
             "comma-separated VLM_MODEL_NAME env var, fallback "
             "'qwen3-vl-32b'). May be passed multiple times.",
    )
    ap.add_argument("--host", type=str, default="127.0.0.1")
    args = ap.parse_args()

    if args.vlm_model:
        vlm_models = set(args.vlm_model)
    else:
        env = os.environ.get("VLM_MODEL_NAME") or "qwen3-vl-32b"
        vlm_models = {m.strip() for m in env.split(",") if m.strip()}

    print(
        f"[qwen-proxy] starting  listen={args.host}:{args.port}  "
        f"llm_port={args.llm_port}  vlm_port={args.vlm_port}  "
        f"vlm_models={sorted(vlm_models)}",
        flush=True,
    )

    import uvicorn
    app = make_app(
        llm_port=args.llm_port, vlm_port=args.vlm_port,
        vlm_model_names=vlm_models, host=args.host,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
