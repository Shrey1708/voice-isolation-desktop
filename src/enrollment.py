"""
Voiceprint enrollment recorder.

Records a clean 5-10 second speech sample from the physical microphone,
resamples it to 16 kHz mono float32, validates audio levels, and saves it
to data/voiceprint.wav for Target Speaker Extraction.

Usage:
    python -m src.enrollment
    python -m src.enrollment --duration 7
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import soxr

from .devices import select_physical_mic
from .resampler import RATE_AUDIO, RATE_MODEL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.json"


def load_config() -> dict:
    """Load config.json from the project root if present."""
    try:
        return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def record_mono_16k(
    device_index: int,
    duration_s: float,
    capture_rate: int = RATE_AUDIO,
) -> np.ndarray:
    """Record mono float32 audio from `device_index` and resample to 16 kHz."""
    total_samples = int(duration_s * capture_rate)
    recorded = sd.rec(
        total_samples,
        samplerate=capture_rate,
        channels=1,
        dtype="float32",
        device=device_index,
    )
    sd.wait()

    audio_48k = recorded[:, 0]
    if capture_rate == RATE_MODEL:
        return audio_48k.astype(np.float32)
    return soxr.resample(audio_48k, capture_rate, RATE_MODEL, quality="HQ").astype(np.float32)


def check_audio_quality(audio_16k: np.ndarray) -> tuple[float, float, list[str]]:
    """Compute peak/RMS levels and return any quality warnings."""
    peak = float(np.max(np.abs(audio_16k))) if len(audio_16k) else 0.0
    rms = float(np.sqrt(np.mean(audio_16k ** 2))) if len(audio_16k) else 0.0
    duration_s = len(audio_16k) / RATE_MODEL

    warnings = []
    if duration_s < 3.0:
        warnings.append(f"Clip is {duration_s:.1f}s; VoxLock requires at least 3.0s.")
    if peak >= 0.99:
        warnings.append(f"Audio is clipping (peak={peak:.3f}). Lower mic gain or move back.")
    if rms < 0.008 or peak < 0.05:
        warnings.append(f"Audio is too quiet (rms={rms:.4f}, peak={peak:.3f}). Speak closer or raise mic gain.")

    return peak, rms, warnings


def countdown(seconds: int = 3) -> None:
    """Print a short recording countdown."""
    for n in range(seconds, 0, -1):
        print(f"  Recording in {n}...", flush=True)
        time.sleep(1)


def main() -> None:
    config = load_config()
    default_out = PROJECT_ROOT / config.get("voiceprint_path", "data/voiceprint.wav")

    parser = argparse.ArgumentParser(description="Record voiceprint for VoxLock TSE")
    parser.add_argument("--duration", type=float, default=7.0, help="Recording length in seconds (5-10s recommended)")
    parser.add_argument("--output", type=Path, default=default_out, help="Output WAV path")
    parser.add_argument("--device", type=int, default=None, help="Optional mic device index override")
    args = parser.parse_args()

    if args.duration < 3.0:
        print("ERROR: Duration must be at least 3.0 seconds.")
        sys.exit(1)

    if args.device is not None:
        mic_index = args.device
        mic_name = sd.query_devices(mic_index)["name"]
    else:
        mic = select_physical_mic(config.get("input_device"))
        if mic is None:
            print("ERROR: No physical microphone found!")
            sys.exit(1)
        mic_index = mic["index"]
        mic_name = f"{mic['name']} ({mic['host_api']})"

    print("=== Voice Enrollment (Phase 2) ===")
    print(f"Mic:      [{mic_index}] {mic_name}")
    print(f"Output:   {args.output}")
    print(f"Duration: {args.duration:.1f}s (16 kHz mono float32)\n")
    print("Speak continuously in your normal voice in a quiet room (no background speech).")

    input("Press Enter when ready to start...")
    countdown(3)

    print("  ● RECORDING NOW — speak naturally...", flush=True)
    audio_16k = record_mono_16k(mic_index, args.duration, capture_rate=config.get("sample_rate", RATE_AUDIO))
    print("  ■ Done.\n")

    peak, rms, warnings = check_audio_quality(audio_16k)
    print(f"Audio stats: duration={len(audio_16k) / RATE_MODEL:.2f}s | peak={peak:.3f} | rms={rms:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), audio_16k, RATE_MODEL, subtype="FLOAT")
    print(f"Saved voiceprint to: {args.output}")

    if warnings:
        print("\nQUALITY WARNINGS:")
        for w in warnings:
            print(f"  ! {w}")
        print("Consider re-running `python -m src.enrollment` for a cleaner voiceprint.")
    else:
        print("\n✓ Voiceprint looks healthy!")


if __name__ == "__main__":
    main()
