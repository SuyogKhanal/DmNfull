# PROVENANCE — Table 8 rebuilt with Diff-DAgger as the baseline

## Reused, not re-run: the DISEIL (`full`) BUDGET=5 runs

| setting | run dir | rounds | final SR | ran on |
|---|---|---:|---:|---|
| Door/state | `distil/results/_compute/Door/state/full/seed1` | 6 | 0.88 | `a100-m-02` |
| Door/image | `distil/results/_compute/Door/image/full/seed1` | 6 | 0.73 | `a100-m-03` |
| Wipe/image | `distil/results/_compute/Wipe/image/full/seed1` | 6 | 0.12 | `h100-m-12` |

## Run: the Diff-DAgger baseline arm (seed 1)

`diffdagger` is a valid arm: `distil/config.py:160` `BASELINE_ARMS` → `distil/baselines.py:109`
→ `distil/diffdagger.py::run_diffdagger`.

### SLURM history — every attempt, including the failures

| job ID | name | budget | node | outcome |
|---|---|---|---|---|
| 110474-6 | d5dd_* | 5 | a100-m-01 | **FAILED @5 s** — launcher default `CONDA_ENV=distil` does not exist here; `conda activate` failed silently (`\|\| true`), bare `python` had no torch. |
| 110477 | d5dd_Door_state | 5 | a100-m-01 | COMPLETED (2 rounds) — **superseded**, see hardware confound |
| 110478 | d5dd_Door_image | 5 | a100-m-01 | **FAILED @37 m** — `KeyError: 'image'` in the FINAL `evaluate_policy`; all rounds had completed. |
| 110479 | d5dd_Wipe_image | 5 | a100-m-01 | CANCELLED (same latent crash) |
| 110486-92 | d5dd*/d5dd20_* | 5, 20 | a100-m-01 | COMPLETED — **superseded** (confounded: 4-5 concurrent jobs on one node) |
| **110502** | m_dd_Ds5 | 5 | **a100-m-02** | **COMPLETED (2 rounds)** ✅ hardware-matched |
| **110503** | m_dd_Ds20 | 20 | **a100-m-02** | **COMPLETED (5 rounds)** ✅ |
| **110504** | m_dd_Di5 | 5 | **a100-m-03** | **COMPLETED (2 rounds)** ✅ |
| **110505** | m_dd_Di20 | 20 | **a100-m-03** | **COMPLETED (5 rounds)** ✅ |
| **110506** | m_dd_Wi5 | 5 | **h100-m-12** | **COMPLETED (2 rounds)** ✅ |
| 110507 → 110518 | m_dd_Wi20 | 20 | h100 (any) | **PENDING** — H100 queue ~9 h out. Wipe/image P5 baseline falls back to the matched BUDGET=5 run (n=2 rounds). See below. |

### The hardware confound (found, then fixed by re-running)

The first batch put **all six** Diff-DAgger jobs on `a100-m-01`, 4-5 concurrent, while the reused
DISEIL runs had executed on `a100-m-02`, `a100-m-03` and — for Wipe/image — **an H100**
(`h100-m-12`). Round-0 **training** is provably identical work across arms (same demos, windows,
8000 steps, architecture — all printed), so any difference is pure environment:

```
round-0 train_s (identical work)     DISEIL   confounded DD   MATCHED DD
  Door/state                          250 s      426 s (1.70x)   229 s (0.92x)
  Door/image                          709 s      592 s (0.83x)   581 s (0.82x)
  Wipe/image                          479 s      883 s (1.84x)   476 s (0.99x)   <- H100 vs A100
```
The confound made DISEIL look **cheaper than the baseline** on Wipe/image (Overhead ×0.893). After
re-running on matched nodes that ratio is **×1.329**, and all six ratios are above 1. The
superseded runs are preserved at `distil/results/_compute_confounded/`.

### Exact sbatch commands (the matched re-run)

```bash
cd /weka/s226137394/DmNfull
PY=/home/s226137394/.conda/envs/diffdagger/bin/python

# Each Diff-DAgger job pinned to the SAME node its DISEIL counterpart used.
#   Door/state -> a100-m-02 | Door/image -> a100-m-03 | Wipe/image -> h100-m-12 (gpu-large)
sbatch --job-name=m_dd_Ds5 --partition=gpu --nodelist=a100-m-02 \
  --export=ALL,TASK=Door,MODALITY=state,ABLATION=diffdagger,SEED=1,BUDGET=5,\
OUTPUT_DIR=$PWD/distil/results/_compute/Door/state/diffdagger/seed1,\
CONDA_ENV=diffdagger,PYTHON_BIN=$PY \
  distil/scripts/run_distil.sbatch
# ... and likewise for BUDGET=20 (dir: diffdagger_b20), Door/image on a100-m-03,
#     and Wipe/image on h100-m-12 with --partition=gpu-large.
```
**Two env vars must be added to the documented launcher call:** `CONDA_ENV=diffdagger` (the sbatch
default `distil` does not exist on this cluster) and an explicit `PYTHON_BIN` (because `~/.bashrc`
prepends another env's `bin` to `PATH`, so bare `python` can resolve to the wrong env).

### Run directories used by the table

| setting | arm | budget | run dir | rounds | final SR |
|---|---|---:|---|---:|---:|
| Door/state | diffdagger | 5 | `distil/results/_compute/Door/state/diffdagger/seed1` | 2 | 0.5 |
| Door/state | diffdagger | 20 | `distil/results/_compute/Door/state/diffdagger_b20/seed1` | 5 | 0.93 |
| Door/image | diffdagger | 5 | `distil/results/_compute/Door/image/diffdagger/seed1` | 2 | 0.38 |
| Door/image | diffdagger | 20 | `distil/results/_compute/Door/image/diffdagger_b20/seed1` | 5 | 0.48 |
| Wipe/image | diffdagger | 5 | `distil/results/_compute/Wipe/image/diffdagger/seed1` | 2 | 0.07 |
| Wipe/image | diffdagger | 20 | *(not scheduled — see 110518)* | — | — |

## Bugs fixed to make this table possible

**1. `distil/diffdagger.py:299` — image modality never reached the final eval.** The in-loop eval
at `:240-245` passes `image_size`; the FINAL `evaluate_policy` did not, so every image-modality
Diff-DAgger run raised `KeyError: 'image'` *after all its rounds had completed*. (The commit
"Thread image modality through the baseline loop (all 6 arms)" missed this call site.) One line:

```python
     final = evaluate_policy(
         policy, eval_env, num_episodes=eval_eps, ...,
+        image_size=image_size,
     )
```
**2. `run.log` is APPEND-mode → re-submitted runs concatenate attempts.** Because failed/cancelled
jobs wrote into the same `OUTPUT_DIR` the re-runs then reused, several `run.log`s contain **two**
`===== DISTIL |` banners, and a naive parse invents a phantom round 0 from the aborted attempt (it
has `[train]` and `[calibrate]` but never reaches `[eval]`), shifting every later round.
`parse_d5_stages._last_run()` truncates each log to the final banner. Verified: every arm now
parses to exactly its expected round count.

## Scripts

| script | role |
|---|---|
| `distil/scripts/parse_d5_stages.py` | **REUSED.** `parse_full()` unchanged. Extended additively with **`parse_diffdagger()`** (the existing `parse_safe()` cannot read this arm — its regex hardcodes `===== safe Round (\d+)` while `run_diffdagger` prints `===== Round N \| dataset=N trajs =====`) and **`_last_run()`**. |
| `distil/scripts/build_table8_diffdagger.py` | NEW — computes the P1/P5 blocks; writes `table8.csv`. |
| `distil/scripts/write_table8_md.py` | NEW — emits `table8.md` **from the CSV**, so no number is hand-transcribed. |
| `distil/scripts/write_d5_vs_diffdagger_sheet.py` | NEW — appends the `D5_vs_DiffDAgger` sheet via openpyxl. |
| `distil/scripts/measure_kag_tokens.py` | **REUSED, not re-run** — its `build/kag_tokens.json` supplies the measured KAG deltas (Door 798/801, Wipe 598/598). |

## Timing sources — every number traces to a logged event

- **DISEIL round wall-clock**: `result.json` `history[i]["sec"]` (authoritative).
- **Diff-DAgger round wall-clock**: derived from `run.log` timestamps (`[train]` → last `[dagger ep]`),
  because `baselines.py::_wrap_history` writes **no** `sec` key for this arm.
- **Stage spans**: `[HH:MM:SS]` prefixes on `[train]`, `[calibrate]`, `[eval]`, `[screen]`,
  `[distil-llm] ... failures analysed`, `[collect a0]`, `[dagger ep*]`.
- **Cross-check**: for DISEIL, the log-derived spans sum to the independently-recorded
  `result.json` `sec` within **±1 s per round**. No unattributed cost bucket exists.
- **Tokens**: `result.json` `history[i]["tokens"]` (`by_stage.{vlm,analysis,decision}`, `reasoning`,
  `kag_calls`) ← the OpenRouter `usage` object (`distil/p4/llm.py:142-186`).
- **KAG**: measured paired with/without-KAG prompt-token diff (`build/kag_tokens.json`).

**Nothing is estimated, interpolated, or fabricated. But precision is not accuracy** — see the
noise floor and hardware caveats in `table8.md`.

## Outputs

- `table8.md` — the finished table (P1 + P5), measurement-validity section, caveats, interpretation
- `table8.csv` — machine-readable (incl. a `baseline_source` column)
- `table8_rows.json`, `raw_stage_spans.json` — per-round stage spans for both arms
- Workbook sheet **`D5_vs_DiffDAgger`** appended to `DISTIL_ablation_results.xlsx` (24 → 25 sheets;
  all 24 originals verified byte-identical in content against a pristine backup; `D5_Compute` is
  **not** modified)
