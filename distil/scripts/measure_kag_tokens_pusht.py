"""Measure the Push-T KAG block's exact token contribution — PAIRED PROMPT DIFF.

Same methodology as distil/scripts/measure_kag_tokens.py (Door/Wipe), adapted to
the fork-backed Push-T suite, whose LLM is a LOCAL vLLM serving Qwen3-32B. There
the "serving tokenizer" is not behind an API — it is the HF tokenizer that vLLM
itself loads from the model directory. So we render the REAL analysis /
prescription / cross-episode / aggregator prompts WITH and WITHOUT the KAG block
and diff the token counts with that exact tokenizer. No estimate, no tiktoken
stand-in, no API call.

    kag_tokens_per_round = kag_calls * tokens(kag_block)

where kag_calls is counted per round from the run's own telemetry
(``kag_in_prompt`` in d5_events.jsonl).

Run with the interpreter that has `transformers` (the orchestrator env does not):

    /home/s226137394/.conda/envs/vllm_embed/bin/python \
        distil/scripts/measure_kag_tokens_pusht.py

Writes distil/results/_compute/kag_tokens_pusht.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DMN = Path("/weka/s226137394/DmNfull")
FORK = Path("/weka/s226137394/diff-dagger")
TOKENIZER_DIR = Path("/weka/s226137394/models/Qwen3-32B")   # the text model vLLM serves
KAG_TXT = (DMN / "Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4"
                 "/pool_rl_robo/p4/kag/PushT-v1.kag.txt")
OUT = DMN / "distil/results/_compute/kag_tokens_pusht.json"

sys.path.insert(0, str(FORK))

from diffdagger.main_analysis.kag_loader import (          # noqa: E402
    load_kag_document, format_kag_for_prompt,
)
from diffdagger.main_pipeline.stage2_prescriptive import (  # noqa: E402
    build_analysis_prompt, build_prescription_prompt, _kag_block,
)

# A representative Stage-A (VLM) report and episode meta. The KAG delta is an
# insertion-verbatim diff, so it does not depend on this body; a realistic body
# just keeps any tokenizer boundary effect inside the measurement.
VISION = (
    "--- Frame 1: start @ timestep 0 ---\n"
    "The TCP is at [-0.318, 0.271, 0.011] m and the T-block centre at "
    "[-0.218, -0.015, 0.021] m; the TCP sits ~0.10 m in -x and ~0.29 m in +y of "
    "the block. --- Frame 2: high-loss @ t*=63 --- The stick has contacted the "
    "T's long bar off-centre and is rotating it away from the goal pose instead "
    "of translating it. --- Frame 3: end --- The T is short of the goal region "
    "and mis-rotated by ~0.6 rad.")
META = {"episode_id": 109781, "seed": 109781, "success": False,
        "total_steps": 200, "peak_loss": 1.2152, "mean_loss": 0.4413,
        "t_star": 63}


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    def n(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    if not KAG_TXT.is_file():
        print(f"ERROR: rendered KAG missing: {KAG_TXT}\n"
              f"  (it is written by p4/kag.py::kag_text_path at run time; run the "
              f"arm once, or call load_kag_text('PushT-v1'))", file=sys.stderr)
        return 2

    kag_raw = load_kag_document(str(KAG_TXT))
    kag = format_kag_for_prompt(kag_raw)        # exactly what pipeline injects

    pairs = {}
    # 1. per-episode ANALYSIS prompt
    a_with = build_analysis_prompt(META, VISION, kag, "", "PushT")
    a_none = build_analysis_prompt(META, VISION, "", "", "PushT")
    pairs["analysis"] = n(a_with) - n(a_none)
    # 2. per-episode PRESCRIPTION prompt
    p_with = build_prescription_prompt(META, VISION, kag, "", "PushT")
    p_none = build_prescription_prompt(META, VISION, "", "", "PushT")
    pairs["prescription"] = n(p_with) - n(p_none)

    # The verbatim text that _kag_block() splices into EVERY KAG-carrying prompt
    # (analysis, prescription, cross-episode, stage-3 aggregator). This is the
    # exact per-call KAG cost.
    block_tokens = n(_kag_block(kag))

    result = {
        "task": "PushT",
        "modality": "image",
        "tokenizer": str(TOKENIZER_DIR),
        "method": "exact tokenization, with the SERVING tokenizer (the HF "
                  "tokenizer the vLLM server loads for the served text model), of "
                  "the verbatim block _kag_block(format_kag_for_prompt(KAG)) that "
                  "the pipeline splices into every KAG-carrying prompt; "
                  "cross-checked by a paired prompt-token diff (same prompt "
                  "rendered with and without the KAG block)",
        "kag_text_path": str(KAG_TXT),
        "kag_chars": len(kag),
        "kag_tokens": block_tokens,               # <- the per-CALL number to use
        "kag_tokens_text_only": n(kag),           # KAG doc without the block header
        "kag_tokens_paired_diff": pairs,          # cross-check per prompt type
        "note": "kag_tokens is the per-CALL cost; per-round cost = kag_calls * "
                "kag_tokens, where kag_calls is counted from the run's own "
                "d5_events.jsonl (kag_in_prompt==true). The VLM prompts carry no "
                "KAG block. The paired diffs differ slightly per prompt type "
                "because each builder also emits a short KAG-conditional "
                "instruction line; the spliced block itself is kag_tokens.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
