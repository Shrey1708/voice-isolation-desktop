"""
Test: Stateful streaming resampler produces continuous output.

Verifies that splitting a signal into chunks, resampling each chunk
through the streaming resampler, and concatenating the results matches
a single-pass resample of the entire signal — no boundary clicks or
discontinuities.
"""

import numpy as np
import soxr

from src.resampler import StreamingResampler, RATE_AUDIO, RATE_MODEL


def generate_test_tone(freq_hz: float, duration_s: float, rate: int) -> np.ndarray:
    """Generate a pure sine tone at the given frequency and sample rate."""
    t = np.arange(int(rate * duration_s), dtype=np.float32) / rate
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def test_downsample_continuity():
    """
    Chunk-by-chunk streaming downsample should closely match
    a single-pass stateless downsample of the same signal.
    """
    tone = generate_test_tone(440.0, duration_s=2.0, rate=RATE_AUDIO)
    chunk_size = 1920  # 40ms at 48kHz

    # Streaming: process in chunks
    resampler = StreamingResampler()
    streamed_chunks = []
    for i in range(0, len(tone), chunk_size):
        chunk = tone[i : i + chunk_size]
        streamed_chunks.append(resampler.downsample(chunk))
    streamed = np.concatenate(streamed_chunks)

    # Reference: single-pass resample
    reference = soxr.resample(tone, RATE_AUDIO, RATE_MODEL, quality="HQ")

    # Allow a small length difference due to streaming delay
    min_len = min(len(streamed), len(reference))
    # Skip the first few samples (resampler delay) and compare
    skip = 50
    diff = np.abs(streamed[skip:min_len] - reference[skip:min_len])
    max_diff = np.max(diff)

    print(f"Downsample continuity test:")
    print(f"  Streamed length: {len(streamed)}, Reference length: {len(reference)}")
    print(f"  Max difference (after skip={skip}): {max_diff:.8f}")
    assert max_diff < 0.01, f"Discontinuity detected! Max diff = {max_diff}"
    print("  ✓ PASSED\n")


def test_roundtrip_continuity():
    """
    48k -> 16k -> 48k roundtrip in streaming chunks should produce
    output close to the original signal (within resampling tolerance).
    """
    tone = generate_test_tone(300.0, duration_s=2.0, rate=RATE_AUDIO)
    chunk_size = 1920  # 40ms at 48kHz

    resampler = StreamingResampler()
    output_chunks = []

    for i in range(0, len(tone), chunk_size):
        chunk = tone[i : i + chunk_size]
        down = resampler.downsample(chunk)
        up = resampler.upsample(down)
        output_chunks.append(up)

    output = np.concatenate(output_chunks)

    # Compare middle portion (avoid edge effects)
    start = 2000
    end = min(len(tone), len(output)) - 2000
    diff = np.abs(tone[start:end] - output[start:end])
    max_diff = np.max(diff)
    rms_diff = np.sqrt(np.mean(diff**2))

    print(f"Roundtrip continuity test:")
    print(f"  Input length: {len(tone)}, Output length: {len(output)}")
    print(f"  Max difference: {max_diff:.6f}, RMS difference: {rms_diff:.6f}")
    assert max_diff < 0.05, f"Roundtrip distortion too high! Max diff = {max_diff}"
    print("  ✓ PASSED\n")


def test_no_boundary_clicks():
    """
    Check that chunk boundaries don't introduce discontinuities
    by measuring the maximum sample-to-sample jump in the output.
    """
    tone = generate_test_tone(1000.0, duration_s=1.0, rate=RATE_AUDIO)
    chunk_size = 1920

    resampler = StreamingResampler()
    down_chunks = []
    for i in range(0, len(tone), chunk_size):
        chunk = tone[i : i + chunk_size]
        down_chunks.append(resampler.downsample(chunk))

    output = np.concatenate(down_chunks)

    # Max expected jump for a 1kHz tone at 16kHz:
    # max derivative ≈ 2π * 1000 / 16000 ≈ 0.39
    # Allow some headroom
    jumps = np.abs(np.diff(output))
    max_jump = np.max(jumps)
    expected_max = 2 * np.pi * 1000 / RATE_MODEL * 1.5  # 50% headroom

    print(f"Boundary click test:")
    print(f"  Max sample-to-sample jump: {max_jump:.6f}")
    print(f"  Expected max (with headroom): {expected_max:.6f}")
    assert max_jump < expected_max, f"Boundary click detected! Jump = {max_jump}"
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    test_downsample_continuity()
    test_roundtrip_continuity()
    test_no_boundary_clicks()
    print("All resampler tests passed! ✓")
