#!/usr/bin/env python3
"""
Transcribe meeting recordings with speaker diarization.

Usage:
    python transcribe.py meeting.mp4  ./output/
    python transcribe.py meeting.mp3  ./output/  --model medium
    python transcribe.py meeting.m4a  ./output/  --force   # re-run all steps

Artifacts written to output_dir/
    audio.wav               — extracted 16 kHz mono audio
    transcript_raw.txt      — timestamped Whisper segments, no speaker labels
    diarization.rttm        — speaker turns in RTTM format
    transcript_diarized.txt — full transcript with speaker labels

Existing artifacts are reused automatically (skip completed steps).
Pass --force to ignore cached artifacts and re-run everything.

Requires HF_TOKEN env var (or --hf-token) for pyannote speaker diarization.
Get a free token at https://huggingface.co/settings/tokens and accept the
model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
"""

import argparse
import contextlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

from dotenv import load_dotenv
load_dotenv()


class Segment(NamedTuple):
    start: float
    end: float
    text: str


MLX_MODELS = {
    "tiny":     "mlx-community/whisper-tiny-mlx",
    "base":     "mlx-community/whisper-base-mlx",
    "small":    "mlx-community/whisper-small-mlx",
    "medium":   "mlx-community/whisper-medium-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def detect_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_time_precise(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def parse_time_precise(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


@contextlib.contextmanager
def timed_step(label: str):
    """Print step start; on exit print elapsed time and optional artifact path."""
    print(f"\n▶ {label}...")
    t0 = time.perf_counter()
    ctx: dict = {}
    yield ctx
    elapsed = time.perf_counter() - t0
    artifact = f"  →  {ctx['artifact']}" if ctx.get("artifact") else ""
    print(f"✓ {label}  {elapsed:.1f}s{artifact}")


def skip_step(label: str, artifact: Path) -> None:
    print(f"\n– {label}  skipped  →  {artifact}")


# ── pipeline steps ────────────────────────────────────────────────────────────

def extract_audio(input_path: Path, output_dir: Path) -> Path:
    """Decode any audio/video input to a canonical 16 kHz mono WAV artifact."""
    out = output_dir / "audio.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ac", "1", "-ar", "16000",
            "-loglevel", "error",
            str(out),
        ],
        check=True,
    )
    return out


def load_audio_tensor(audio_path: str, sample_rate: int = 16000):
    """Load audio as a torch tensor for pyannote via the ffmpeg binary."""
    import numpy as np
    import torch

    proc = subprocess.run(
        [
            "ffmpeg", "-i", audio_path,
            "-ac", "1", "-ar", str(sample_rate),
            "-f", "f32le", "-loglevel", "error",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    waveform = torch.from_numpy(
        np.frombuffer(proc.stdout, dtype=np.float32).copy()
    ).unsqueeze(0)  # (1, samples)
    return waveform, sample_rate


def transcribe_audio(audio_path: str, model_size: str, language: str | None) -> list[Segment]:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=MLX_MODELS[model_size],
        language=language,
    )
    print(f"  Language: {result['language']}")
    return [Segment(s["start"], s["end"], s["text"]) for s in result["segments"]]


def diarize(audio_path: str, hf_token: str, device: str):
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    if device == "mps":
        pipeline = pipeline.to(torch.device("mps"))

    waveform, sample_rate = load_audio_tensor(audio_path)
    return pipeline({"waveform": waveform, "sample_rate": sample_rate})


def assign_speakers(segments: list[Segment], diarization) -> list[dict]:
    # Newer pyannote wraps the Annotation in a DiarizeOutput dataclass
    annotation = getattr(diarization, "speaker_diarization", diarization)
    results = []
    for seg in segments:
        midpoint = (seg.start + seg.end) / 2
        speaker = "UNKNOWN"
        for turn, _, label in annotation.itertracks(yield_label=True):
            if turn.start <= midpoint <= turn.end:
                speaker = label
                break
        results.append({
            "start": seg.start,
            "end": seg.end,
            "speaker": speaker,
            "text": seg.text.strip(),
        })
    return results


# ── artifact readers / writers ────────────────────────────────────────────────

def write_raw_transcript(segments: list[Segment], output_dir: Path) -> Path:
    out = output_dir / "transcript_raw.txt"
    lines = [
        f"[{format_time_precise(s.start)}-{format_time_precise(s.end)}] {s.text.strip()}"
        for s in segments
        if s.text.strip()
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


_RAW_PATTERN = re.compile(
    r'^\[(\d+:\d+:\d+\.\d+)-(\d+:\d+:\d+\.\d+)\] (.*)$'
)

def load_raw_transcript(path: Path) -> list[Segment]:
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _RAW_PATTERN.match(line)
        if m:
            segments.append(Segment(
                parse_time_precise(m.group(1)),
                parse_time_precise(m.group(2)),
                m.group(3),
            ))
    return segments


def write_rttm(diarization, path: Path) -> None:
    annotation = getattr(diarization, "speaker_diarization", diarization)
    with open(path, "w") as f:
        for turn, _, label in annotation.itertracks(yield_label=True):
            duration = turn.end - turn.start
            f.write(
                f"SPEAKER audio 1 {turn.start:.3f} {duration:.3f}"
                f" <NA> <NA> {label} <NA> <NA>\n"
            )


def load_rttm(path: Path):
    from pyannote.core import Annotation
    from pyannote.core import Segment as PySegment

    annotation = Annotation()
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if not parts or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7]
        annotation[PySegment(start, start + duration)] = speaker
    return annotation


def write_diarized_transcript(results: list[dict], output_dir: Path) -> Path:
    out = output_dir / "transcript_diarized.txt"
    out.write_text(format_transcript(results), encoding="utf-8")
    return out


def format_transcript(results: list[dict]) -> str:
    lines = []
    current_speaker = None
    for r in results:
        if not r["text"]:
            continue
        if r["speaker"] != current_speaker:
            current_speaker = r["speaker"]
            lines.append(f"\n[{format_time(r['start'])}] {r['speaker']}:")
        lines.append(f"  {r['text']}")
    return "\n".join(lines).strip()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe meeting recordings with speaker diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio_file", help="Audio or video file to transcribe")
    parser.add_argument(
        "output_dir",
        help="Directory where all artifacts are written (created if absent)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: large-v3; use small/medium for speed)",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (default: $HF_TOKEN env var)",
    )
    parser.add_argument(
        "--language",
        choices=["en", "es"],
        default=None,
        help="Force a single language for the whole recording (default: auto-detect).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default=None,
        help="Compute device (default: auto-detect; mps on Apple Silicon, cpu otherwise)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all steps even if artifacts already exist in output_dir.",
    )
    args = parser.parse_args()

    input_path = Path(args.audio_file)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not args.hf_token:
        print("Error: HuggingFace token required for speaker diarization.", file=sys.stderr)
        print("  Set HF_TOKEN env var or pass --hf-token.", file=sys.stderr)
        print("  Get a token at https://huggingface.co/settings/tokens", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or detect_device()
    print(f"Device:     {device}")
    print(f"Output dir: {output_dir}/")

    # ── 1. Extract audio ──────────────────────────────────────────────────────
    audio_wav = output_dir / "audio.wav"
    if not args.force and audio_wav.exists():
        skip_step("Extracting audio", audio_wav)
        audio_path = audio_wav
    else:
        with timed_step("Extracting audio") as ctx:
            audio_path = extract_audio(input_path, output_dir)
            ctx["artifact"] = audio_path

    # ── 2. Transcribe ─────────────────────────────────────────────────────────
    raw_txt = output_dir / "transcript_raw.txt"
    if not args.force and raw_txt.exists():
        skip_step("Transcribing", raw_txt)
        segments = load_raw_transcript(raw_txt)
    else:
        lang_label = args.language or "auto-detect"
        with timed_step(f"Transcribing ({args.model}, language: {lang_label})") as ctx:
            segments = transcribe_audio(str(audio_path), args.model, args.language)
            raw_path = write_raw_transcript(segments, output_dir)
            ctx["artifact"] = raw_path

    # ── 3. Diarize ────────────────────────────────────────────────────────────
    rttm_path = output_dir / "diarization.rttm"
    if not args.force and rttm_path.exists():
        skip_step("Speaker diarization", rttm_path)
        diarization = load_rttm(rttm_path)
    else:
        with timed_step("Running speaker diarization") as ctx:
            diarization = diarize(str(audio_path), args.hf_token, device)
            write_rttm(diarization, rttm_path)
            ctx["artifact"] = rttm_path

    # ── 4. Merge and write ────────────────────────────────────────────────────
    with timed_step("Merging speakers and writing transcript") as ctx:
        results = assign_speakers(segments, diarization)
        diarized_path = write_diarized_transcript(results, output_dir)
        ctx["artifact"] = diarized_path

    # ── Preview ───────────────────────────────────────────────────────────────
    transcript = diarized_path.read_text(encoding="utf-8")
    preview_lines = transcript.split("\n")[:15]
    print("\n--- Preview ---")
    print("\n".join(preview_lines))
    if len(transcript.split("\n")) > 15:
        print("...")


if __name__ == "__main__":
    main()
