"""
MOM: Minutes-of-Meeting audio → speaker-attributed JSON.

Pipeline
--------
1. Decode any audio (WAV / MP4 / M4A / MP3 / ...) → 16 kHz mono WAV via ffmpeg.
2. Diarize the full recording on GPU with pyannote/speaker-diarization-3.1
   (full-file pass is required so the same speaker keeps a stable label across
   the whole meeting — chunked diarization would re-cluster per chunk).
3. Split the WAV into chunks and transcribe them IN PARALLEL with the Groq
   Whisper API (chunking is also needed because Groq caps each request at
   25 MB / 1 h).
4. Align each transcript segment to the most-overlapping diarization turn,
   relabel speakers as speaker0 / speaker1 / speaker2 (in order of first
   appearance), and write a structured JSON.

Run
---
    export GROQ_API_KEY=...
    export HF_TOKEN=...           # accept pyannote/speaker-diarization-3.1 on HF first
    python mom_transcribe.py <audio> [--out meeting.json] [--num-speakers 3]

Deps
----
    pip install groq pyannote.audio==3.1.* torch torchaudio soundfile tqdm python-dotenv
    (ffmpeg must be on PATH)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"
CHUNK_SECONDS = 600          # 10-min chunks → well under Groq's 25 MB / 1 h cap
PARALLEL_WORKERS = 4         # concurrent Groq calls
TARGET_SR = 16_000           # pyannote + Whisper both prefer 16 kHz mono


# ---------------------------------------------------------------------------
# audio prep
# ---------------------------------------------------------------------------
def to_wav_16k_mono(src: Path, work_dir: Path) -> Path:
    """Decode anything ffmpeg understands → 16 kHz mono WAV."""
    out = work_dir / f"{src.stem}_16k.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ac", "1", "-ar", str(TARGET_SR),
        "-loglevel", "error",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def split_chunks(wav: Path, work_dir: Path, chunk_seconds: int) -> list[tuple[Path, float]]:
    """Write fixed-length WAV chunks. Returns [(chunk_path, absolute_start_sec)]."""
    data, sr = sf.read(str(wav))
    total_sec = len(data) / sr
    n = max(1, math.ceil(total_sec / chunk_seconds))
    out = []
    for i in range(n):
        start = i * chunk_seconds
        end = min(start + chunk_seconds, total_sec)
        s_idx, e_idx = int(start * sr), int(end * sr)
        cp = work_dir / f"chunk_{i:04d}.wav"
        sf.write(str(cp), data[s_idx:e_idx], sr)
        out.append((cp, float(start)))
    return out


# ---------------------------------------------------------------------------
# diarization (GPU, full file)
# ---------------------------------------------------------------------------
def _patch_torchaudio_for_pyannote() -> None:
    """Shim torchaudio ≥2.11 (which dropped AudioMetaData / info / list_audio_backends)
    so pyannote.audio 3.3.x can still import. Backed by soundfile."""
    import torchaudio
    import soundfile as _sf
    from dataclasses import dataclass

    if not hasattr(torchaudio, "AudioMetaData"):
        @dataclass
        class AudioMetaData:
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int = 0
            encoding: str = "PCM_S"
        torchaudio.AudioMetaData = AudioMetaData

    if not hasattr(torchaudio, "info"):
        def _info(path, backend=None):  # noqa: ARG001
            i = _sf.info(str(path))
            return torchaudio.AudioMetaData(
                sample_rate=int(i.samplerate),
                num_frames=int(i.frames),
                num_channels=int(i.channels),
            )
        torchaudio.info = _info

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]


def _patch_hf_hub_for_pyannote() -> None:
    """huggingface_hub ≥ 1.0 removed `use_auth_token`; pyannote 3.3.x still passes it.
    Wrap hf_hub_download / snapshot_download to translate use_auth_token → token,
    BEFORE any pyannote module does `from huggingface_hub import hf_hub_download`."""
    import functools
    import huggingface_hub

    def _wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            if "use_auth_token" in kwargs:
                tok = kwargs.pop("use_auth_token")
                kwargs.setdefault("token", tok)
            return fn(*args, **kwargs)
        inner.__wrapped_for_pyannote__ = True
        return inner

    for name in ("hf_hub_download", "snapshot_download"):
        fn = getattr(huggingface_hub, name, None)
        if fn is not None and not getattr(fn, "__wrapped_for_pyannote__", False):
            setattr(huggingface_hub, name, _wrap(fn))


def diarize_full(wav: Path, hf_token: str, num_speakers: int) -> list[dict[str, Any]]:
    _patch_torchaudio_for_pyannote()
    _patch_hf_hub_for_pyannote()
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        dev_name = torch.cuda.get_device_name(0)
    else:
        dev_name = "CPU"
    print(f"  diarizing on {dev_name} (full-file pass)")

    with ProgressHook() as hook:
        diar = pipeline(str(wav), num_speakers=num_speakers, hook=hook)

    turns: list[dict[str, Any]] = []
    for segment, _, speaker in diar.itertracks(yield_label=True):
        turns.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker_raw": speaker,
        })
    turns.sort(key=lambda t: t["start"])
    return turns


def normalize_speakers(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map pyannote's SPEAKER_xx → speaker0/1/2 in first-appearance order."""
    mapping: dict[str, str] = {}
    for t in turns:
        raw = t["speaker_raw"]
        if raw not in mapping:
            mapping[raw] = f"speaker{len(mapping)}"
        t["speaker"] = mapping[raw]
    return turns


# ---------------------------------------------------------------------------
# transcription (Groq Whisper, parallel chunks)
# ---------------------------------------------------------------------------
def transcribe_one(client: Groq, chunk_path: Path, offset: float, model: str) -> list[dict[str, Any]]:
    with open(chunk_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            file=(chunk_path.name, f.read()),
            model=model,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    raw = getattr(resp, "segments", None) or []
    out = []
    for s in raw:
        s = dict(s) if not isinstance(s, dict) else s
        out.append({
            "start": float(s.get("start", 0.0)) + offset,
            "end":   float(s.get("end",   0.0)) + offset,
            "text":  str(s.get("text", "")).strip(),
        })
    return out


def transcribe_parallel(
    client: Groq,
    chunks: list[tuple[Path, float]],
    model: str,
    workers: int,
) -> list[dict[str, Any]]:
    all_segs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(transcribe_one, client, cp, off, model): cp for cp, off in chunks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Whisper (Groq)"):
            all_segs.extend(fut.result())
    all_segs.sort(key=lambda x: x["start"])
    return all_segs


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------
def attach_speakers(segments: list[dict[str, Any]], turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each transcript segment, pick the diarization turn with max time-overlap."""
    out = []
    for seg in tqdm(segments, desc="Aligning segments → speakers"):
        best_spk, best_ov = None, 0.0
        for t in turns:
            if t["end"] <= seg["start"]:
                continue
            if t["start"] >= seg["end"]:
                break
            ov = min(seg["end"], t["end"]) - max(seg["start"], t["start"])
            if ov > best_ov:
                best_ov, best_spk = ov, t["speaker"]
        out.append({
            "start": round(seg["start"], 2),
            "end":   round(seg["end"],   2),
            "speaker": best_spk or "unknown",
            "text": seg["text"],
        })
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Meeting audio → speaker-attributed JSON")
    ap.add_argument("audio", type=str, help="Path to audio file (WAV / MP4 / M4A / MP3 / ...)")
    ap.add_argument("--out", type=str, default=None, help="Output JSON path (default: <audio>.mom.json)")
    ap.add_argument("--num-speakers", type=int, default=3, help="Expected speaker count (default 3)")
    ap.add_argument("--chunk-seconds", type=int, default=CHUNK_SECONDS)
    ap.add_argument("--workers", type=int, default=PARALLEL_WORKERS)
    ap.add_argument("--model", type=str, default=GROQ_WHISPER_MODEL)
    args = ap.parse_args()

    load_dotenv()
    groq_key = os.environ.get("GROQ_API_KEY")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not groq_key:
        sys.exit("GROQ_API_KEY missing — set it in .env or export it")
    if not hf_token:
        sys.exit("HF_TOKEN missing — pyannote/speaker-diarization-3.1 needs a HuggingFace token "
                 "(accept the model gating page first)")

    src = Path(args.audio).expanduser().resolve()
    if not src.exists():
        sys.exit(f"Audio not found: {src}")
    out_path = Path(args.out).expanduser().resolve() if args.out else src.with_suffix(".mom.json")

    with tempfile.TemporaryDirectory(prefix="mom_") as tmp:
        work = Path(tmp)

        print("[1/4] Decoding → 16 kHz mono WAV")
        wav = to_wav_16k_mono(src, work)

        print("[2/4] Speaker diarization")
        turns = normalize_speakers(diarize_full(wav, hf_token, args.num_speakers))

        print(f"[3/4] Chunking audio ({args.chunk_seconds}s) and transcribing in parallel "
              f"({args.workers} workers)")
        chunks = split_chunks(wav, work, args.chunk_seconds)
        client = Groq(api_key=groq_key)
        segments = transcribe_parallel(client, chunks, args.model, args.workers)

        print("[4/4] Aligning transcript with speaker turns")
        merged = attach_speakers(segments, turns)

    speaker_set = sorted({m["speaker"] for m in merged if m["speaker"] != "unknown"})
    out_doc = {
        "audio_file": str(src),
        "duration_sec": (turns[-1]["end"] if turns else (merged[-1]["end"] if merged else 0.0)),
        "model": args.model,
        "num_speakers_expected": args.num_speakers,
        "speakers": speaker_set,
        "diarization_turns": [
            {"start": round(t["start"], 2), "end": round(t["end"], 2), "speaker": t["speaker"]}
            for t in turns
        ],
        "segments": merged,
    }
    out_path.write_text(json.dumps(out_doc, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}  ({len(merged)} segments · {len(turns)} turns · {len(speaker_set)} speakers)")


if __name__ == "__main__":
    main()
