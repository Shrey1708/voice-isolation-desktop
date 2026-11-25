"""
Offline Target Speaker Extraction (TSE) verification test.

Tests VoxLock using the enrolled data/voiceprint.wav on a real noisy
multi-speaker recording (or a pre-recorded WAV), writes the isolated audio
to data/test_isolated.wav, and reports measured CPU Real-Time Factor (RTF).

Usage:
    python -m tests.test_offline_tse              # record 10s noisy mix & isolate
    python -m tests.test_offline_tse --reuse      # reuse existing data/test_mix.wav
    python -m tests.test_offline_tse --input a.wav
    python -m tests.test_offline_tse --smoke      # quick synthetic CPU RTF check
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import voxlock

from src.devices import select_physical_mic
from src.enrollment import countdown, load_config, record_mono_16k
from src.resampler import RATE_AUDIO, RATE_MODEL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MIX = PROJECT_ROOT / "data" / "test_mix.wav"
DEFAULT_OUT = PROJECT_ROOT / "data" / "test_isolated.wav"
CHUNK_SAMPLES_16K = 640  # 40 ms at 16 kHz (5 x 128-sample VoxLock hop)
RTF_TARGET = 0.65


def run_streaming_tse(
    vl: voxlock.VoxLock,
    voiceprint: np.ndarray,
    mix_16k: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run VoxLock in 40 ms streaming chunks and return (isolated_audio, rtf)."""
    stream = vl.stream(voiceprint)
    output_chunks = []

    t0 = time.perf_counter()
    for start in range(0, len(mix_16k), CHUNK_SAMPLES_16K):
        chunk = mix_16k[start : start + CHUNK_SAMPLES_16K]
        out = stream.push(chunk)
        if len(out):
            output_chunks.append(out)
    tail = stream.flush()
    if len(tail):
        output_chunks.append(tail)
    elapsed = time.perf_counter() - t0

    isolated = np.concatenate(output_chunks)[: len(mix_16k)]
    audio_duration = len(mix_16k) / RATE_MODEL
    rtf = elapsed / audio_duration if audio_duration > 0 else 0.0
    return isolated, rtf


def record_test_mix(duration_s: float, capture_rate: int) -> np.ndarray:
    """Record a noisy / multi-speaker test clip from the physical mic."""
    mic = select_physical_mic()
    if mic is None:
        print("ERROR: No physical microphone found!")
        sys.exit(1)

    print(f"Using mic: [{mic['index']}] {mic['name']} ({mic['host_api']})")
    print(f"\nPlay another person's voice (YouTube/TV/speaker) + room noise nearby,")
    print(f"and speak normally over it for {duration_s:.0f} seconds.")
    input("Press Enter when ready to record test mix...")
    countdown(3)

    print("  ● RECORDING TEST MIX — speak while background voice plays...", flush=True)
    mix_16k = record_mono_16k(mic["index"], duration_s, capture_rate=capture_rate)
    print("  ■ Done.\n")

    DEFAULT_MIX.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(DEFAULT_MIX), mix_16k, RATE_MODEL, subtype="FLOAT")
    print(f"Saved raw mix to: {DEFAULT_MIX}")
    return mix_16k


def run_smoke_check(threads: int) -> None:
    """Verify VoxLock streaming pipeline and CPU RTF using synthetic signals."""
    t = np.arange(5 * RATE_MODEL, dtype=np.float32) / RATE_MODEL
    voiceprint = (0.1 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    interferer = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    noise = (0.02 * np.random.default_rng(0).standard_normal(len(t))).astype(np.float32)
    mix = voiceprint + interferer + noise

    vl = voxlock.VoxLock(threads=threads)
    isolated, rtf = run_streaming_tse(vl, voiceprint, mix)

    assert len(isolated) == len(mix), f"Length mismatch: {len(isolated)} != {len(mix)}"
    assert np.isfinite(isolated).all(), "Non-finite samples in isolated output!"
    print(f"Smoke test passed: {len(mix) / RATE_MODEL:.1f}s processed | RTF={rtf:.3f} (target < {RTF_TARGET})")
    assert rtf < RTF_TARGET, f"RTF {rtf:.3f} exceeds target {RTF_TARGET}"


def main() -> None:
    config = load_config()
    threads = config.get("onnx_threads", 2)
    voiceprint_path = PROJECT_ROOT / config.get("voiceprint_path", "data/voiceprint.wav")

    parser = argparse.ArgumentParser(description="Offline VoxLock TSE verification")
    parser.add_argument("--input", type=Path, default=None, help="Existing noisy mix WAV to isolate")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing data/test_mix.wav")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording length if recording a mix")
    parser.add_argument("--smoke", action="store_true", help="Run synthetic CPU RTF check without mic")
    args = parser.parse_args()

    if args.smoke:
        run_smoke_check(threads)
        return

    if not voiceprint_path.exists():
        print(f"ERROR: Voiceprint not found at {voiceprint_path}")
        print("Run `python -m src.enrollment` first.")
        sys.exit(1)

    print("=== Offline TSE Verification (Phase 2) ===")
    voiceprint = voxlock.load_audio(str(voiceprint_path))
    print(f"Loaded voiceprint: {voiceprint_path} ({len(voiceprint) / RATE_MODEL:.2f}s)")

    if args.input is not None:
        mix_16k = voxlock.load_audio(str(args.input))
        print(f"Loaded input mix:  {args.input} ({len(mix_16k) / RATE_MODEL:.2f}s)")
    elif args.reuse and DEFAULT_MIX.exists():
        mix_16k = voxlock.load_audio(str(DEFAULT_MIX))
        print(f"Reusing test mix:  {DEFAULT_MIX} ({len(mix_16k) / RATE_MODEL:.2f}s)")
    else:
        mix_16k = record_test_mix(args.duration, config.get("sample_rate", RATE_AUDIO))

    print(f"Loading VoxLock (onnx_threads={threads})...")
    vl = voxlock.VoxLock(threads=threads)

    print("Running streaming TSE in 40 ms chunks...")
    isolated, rtf = run_streaming_tse(vl, voiceprint, mix_16k)

    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(DEFAULT_OUT), isolated, RATE_MODEL, subtype="FLOAT")

    status = "PASS" if rtf < RTF_TARGET else "WARN (exceeds target)"
    print(f"\nResults:")
    print(f"  Audio duration: {len(mix_16k) / RATE_MODEL:.2f}s")
    print(f"  Measured RTF:   {rtf:.3f}  [{status}: target < {RTF_TARGET}]")
    print(f"  Isolated WAV:   {DEFAULT_OUT}")
    if DEFAULT_MIX.exists() and args.input is None:
        print(f"  Compare against {DEFAULT_MIX} to verify competing speech & noise suppression.")


if __name__ == "__main__":
    main()
