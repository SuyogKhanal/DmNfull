"""VLM frame analyser — the ONLY module that uses the response cache.

Cache policy (enforced here, stripped from reasoning.py):
  - VLM outputs are expensive (vision API calls per frame) and deterministic
    given the same frame + prompt, so they ARE cached keyed on
    (rollout_id, episode_id).  Including rollout_id ensures that two
    ablation suites run against different rollouts never share cached VLM
    results, while all profiles within the SAME suite (same shared rollout)
    DO share VLM results — saving API cost without polluting ablations.
  - Reasoning, prescription, summary, aggregator, and TKF outputs are NOT
    cached — see reasoning.py and aggregator.py.
"""
import base64
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _oai_client():
    from openai import OpenAI
    return OpenAI()


def _encode_image(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[VLM] Could not encode image {path}: {e}")
        return None


def _frame_prompt(role: str, step_idx: int, episode: Dict) -> str:
    dyn = episode.get("dynamic_config", {})
    return (
        f"You are analysing frame '{role}' (step {step_idx}) of a failed maze navigation episode.\n"
        f"Episode config: start={dyn.get('start_pos','?')}, goal={dyn.get('goal_pos','?')}, "
        f"fires={dyn.get('fire_positions','?')}.\n\n"
        "Provide a structured analysis (~150 words):\n"
        "1. Agent location and adjacent cells (note any walls/fires/goal nearby).\n"
        "2. Goal location and Manhattan distance.\n"
        "3. Fire hazards relative to the agent.\n"
        "4. First corrective demonstration move (exact action + brief justification)."
    )


def _analyse_frame(
    client,
    model: str,
    image_path: str,
    role: str,
    step_idx: int,
    episode: Dict,
    max_tokens: int,
) -> str:
    b64 = _encode_image(image_path)
    if b64 is None:
        return f"[VLM: could not load frame {role}]"
    prompt = _frame_prompt(role, step_idx, episode)
    from pipeline._oai_retry import call_with_retry
    try:
        r = call_with_retry(
            client.responses.create,
            label="vlm",
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text",  "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                    ],
                }
            ],
            max_output_tokens=max_tokens,
        )
        return r.output_text or ""
    except Exception as e:
        traceback.print_exc()
        return f"[VLM ERROR on {role}: {e}]"


def analyse_failure(
    episode: Dict,
    llm_cfg: Dict,
    cache=None,
    rollout_id: str = "default",
) -> Tuple[str, List[Dict]]:
    """Run VLM analysis on the 3 key frames.  Results ARE cached per (rollout_id, episode_id).

    The rollout_id parameter must be passed by the caller and should identify
    the specific rollout directory (e.g. the rollout folder name or its absolute
    path).  This ensures:
      - All ablation profiles that share the same rollout reuse the same VLM
        outputs (saves API cost — the rollout frames are identical).
      - Two different ablation suites with different rollouts never share
        cached VLM results, preventing stale cache pollution.
    """
    episode_id = episode["episode_id"]
    model      = str(llm_cfg.get("vlm_model", llm_cfg.get("model", "gpt-5-nano-2025-08-07")))
    max_tokens = int(llm_cfg.get("max_output_tokens", 4096))

    # --- Cache hit path (VLM only) ---
    # Scope is now (rollout_id, episode_id) so different rollouts never collide.
    if cache is not None:
        scope  = f"rollout_{rollout_id}_episode_{int(episode_id)}"
        cached = cache.load(scope, "vlm_report")
        if cached is not None:
            print(f"  [VLM][ep {episode_id}] cache HIT — skipping API calls")
            return cached.get("report", ""), cached.get("per_frame", [])
    else:
        scope = f"rollout_{rollout_id}_episode_{int(episode_id)}"

    frame_paths = episode.get("frame_paths", {})
    key_frames  = episode.get("key_frames", [])

    client_obj = _oai_client()
    per_frame: List[Dict] = []
    report_sections: List[str] = []

    for kf in key_frames:
        role     = kf.get("role", "unknown")
        step_idx = kf.get("step_idx", 0)
        img_path = frame_paths.get(role)

        if not img_path or not Path(img_path).exists():
            txt = f"[VLM: frame '{role}' not found at {img_path}]"
        else:
            txt = _analyse_frame(client_obj, model, img_path, role, step_idx, episode, max_tokens)

        per_frame.append({"role": role, "step_idx": step_idx, "text": txt})
        report_sections.append(f"--- {role} frame step {step_idx} ---\n{txt}")

    report = (
        f"VISION REPORT\nMaze={episode.get('maze_name','?')} "
        f"Steps={episode.get('total_steps','?')} "
        f"Reward={episode.get('total_reward',0.0):.2f} "
        f"Success={episode.get('success',False)}\n"
        "---\n" + "\n\n".join(report_sections)
    )

    # --- Cache miss path: save for next run ---
    if cache is not None:
        cache.save(
            scope,
            "vlm_report",
            {"report": report, "per_frame": per_frame},
        )

    return report, per_frame


def save_vlm_report(text: str, episode_dir: Path) -> Path:
    episode_dir.mkdir(parents=True, exist_ok=True)
    out = episode_dir / "vlm_report.txt"
    out.write_text(text, encoding="utf-8")
    return out
