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
  python -m baseline_vs_p4_sequential_batch.qwen.proxy \\
      --port 8002 --llm_port 8000 --vlm_port 8001 \\
      --vlm_model qwen3-vl-32b
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Set

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
