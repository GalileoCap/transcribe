#!/usr/bin/env python3
"""
Transcribe meeting recordings with speaker diarization.

Usage:
    python transcribe.py meeting.mp4
    python transcribe.py meeting.mp3 --model medium
    python transcribe.py meeting.m4a --output transcript.txt

Requires HF_TOKEN env var (or --hf-token) for pyannote speaker diarization.
Get a free token at https://huggingface.co/settings/tokens and accept the
model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


def detect_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_audio_tensor(audio_path: str, sample_rate: int = 16000):
    """Decode audio via the ffmpeg binary to avoid torchcodec shared-library issues."""
    import subprocess
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


def diarize(audio_path: str, hf_token: str, device: str):
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    if device in ("mps", "cuda"):
        pipeline = pipeline.to(torch.device(device))

    waveform, sample_rate = load_audio_tensor(audio_path)
    return pipeline({"waveform": waveform, "sample_rate": sample_rate})


def _whisper_device(device: str) -> tuple[str, str]:
    # CTranslate2 (faster-whisper's backend) only supports cpu and cuda — not mps.
    ct2_device = "cuda" if device == "cuda" else "cpu"
    compute_type = "float16" if ct2_device == "cuda" else "int8"
    return ct2_device, compute_type


def transcribe_audio(audio_path: str, model_size: str, device: str, language: str | None):
    from faster_whisper import WhisperModel

    ct2_device, compute_type = _whisper_device(device)
    model = WhisperModel(model_size, device=ct2_device, compute_type=compute_type)
    segments, info = model.transcribe(audio_path, beam_size=5, language=language)
    segments = list(segments)

    print(f"  Language: {info.language} (confidence {info.language_probability:.0%})")
    return segments


def transcribe_audio_chunked(
    audio_path: str,
    model_size: str,
    device: str,
    candidates: list[str] | None,
    chunk_duration: int = 60,
):
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    ct2_device, compute_type = _whisper_device(device)
    model = WhisperModel(model_size, device=ct2_device, compute_type=compute_type)

    audio = decode_audio(audio_path)
    sample_rate = 16000  # faster-whisper always resamples to 16 kHz
    chunk_samples = chunk_duration * sample_rate
    n_chunks = max(1, (len(audio) + chunk_samples - 1) // chunk_samples)

    all_segments = []
    prev_lang = (candidates or ["en"])[0]

    for i in range(n_chunks):
        chunk = audio[i * chunk_samples : (i + 1) * chunk_samples]
        offset = i * chunk_duration

        lang, prob = model.detect_language(chunk)
        if candidates and lang not in candidates:
            lang = prev_lang  # keep previous language rather than jumping to an unexpected one
        prev_lang = lang

        print(f"  [{format_time(offset)}] Language: {lang} ({prob:.0%})")

        segments, _ = model.transcribe(chunk, language=lang, beam_size=5)
        for seg in segments:
            all_segments.append(seg._replace(
                start=seg.start + offset,
                end=seg.end + offset,
            ))

    return all_segments


def assign_speakers(segments, diarization) -> list[dict]:
    # Newer pyannote wraps the Annotation in a DiarizeOutput namedtuple
    annotation = getattr(diarization, "diarization", diarization)
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


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe meeting recordings with speaker diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio_file", help="Audio or video file to transcribe")
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
    lang_group = parser.add_mutually_exclusive_group()
    lang_group.add_argument(
        "--language",
        choices=["en", "es"],
        default=None,
        help="Force a single language for the whole recording.",
    )
    lang_group.add_argument(
        "--chunk-languages",
        action="store_true",
        help="Detect language per chunk (for meetings that switch languages mid-way).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        metavar="LANG",
        default=None,
        help="Candidate languages for --chunk-languages, e.g. --languages en es. "
             "Detects from all Whisper languages if omitted.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Compute device (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file (default: <input>.txt)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    if not args.hf_token:
        print("Error: HuggingFace token required for speaker diarization.", file=sys.stderr)
        print("  Set HF_TOKEN env var or pass --hf-token.", file=sys.stderr)
        print("  Get a token at https://huggingface.co/settings/tokens", file=sys.stderr)
        sys.exit(1)

    device = args.device or detect_device()
    ct2_device = "cuda" if device == "cuda" else "cpu"
    print(f"Device: {device} (whisper: {ct2_device}, diarization: {device})")

    t0 = time.perf_counter()
    if args.chunk_languages:
        candidates = args.languages
        label = f"candidates: {', '.join(candidates)}" if candidates else "any language"
        print(f"Transcribing with per-chunk language detection ({label})...")
        segments = transcribe_audio_chunked(str(audio_path), args.model, device, candidates)
    else:
        lang_label = args.language or "auto-detect"
        print(f"Transcribing with Whisper ({args.model}, language: {lang_label})...")
        segments = transcribe_audio(str(audio_path), args.model, device, args.language)
    t1 = time.perf_counter()
    print(f"Transcription done in {t1 - t0:.1f}s")

    print("Running speaker diarization...")
    diarization = diarize(str(audio_path), args.hf_token, device)

    print("Merging speakers and transcript...")
    results = assign_speakers(segments, diarization)

    transcript = format_transcript(results)

    output_path = Path(args.output) if args.output else audio_path.with_suffix(".txt")
    output_path.write_text(transcript, encoding="utf-8")
    print(f"\nSaved: {output_path}")

    preview = transcript.split("\n")[:15]
    print("\n--- Preview ---")
    print("\n".join(preview))
    if len(transcript.split("\n")) > 15:
        print("...")


if __name__ == "__main__":
    main()
