"""
Stateful streaming resampler using soxr.

Wraps soxr.ResampleStream for glitch-free 48kHz <-> 16kHz conversion.
Maintains internal state across chunks to avoid boundary clicks/pops.
"""

import numpy as np
import soxr

RATE_AUDIO = 48_000   # PortAudio / virtual cable sample rate
RATE_MODEL = 16_000   # TSE model sample rate


class StreamingResampler:
    """
    Bidirectional stateful resampler for the audio pipeline.

    Provides two operations:
        downsample(chunk_48k) -> chunk_16k   (for model input)
        upsample(chunk_16k)   -> chunk_48k   (for cable output)

    Both maintain internal state so consecutive chunks produce
    continuous output without boundary artifacts.
    """

    def __init__(self, in_rate=RATE_AUDIO, out_rate=RATE_MODEL):
        self._downsampler = soxr.ResampleStream(
            in_rate=in_rate,
            out_rate=out_rate,
            num_channels=1,
            dtype="float32",
            quality="HQ",
        )
        self._upsampler = soxr.ResampleStream(
            in_rate=out_rate,
            out_rate=in_rate,
            num_channels=1,
            dtype="float32",
            quality="HQ",
        )

    def downsample(self, chunk: np.ndarray) -> np.ndarray:
        """48kHz mono float32 -> 16kHz mono float32."""
        return self._downsampler.resample_chunk(chunk)

    def upsample(self, chunk: np.ndarray) -> np.ndarray:
        """16kHz mono float32 -> 48kHz mono float32."""
        return self._upsampler.resample_chunk(chunk)

    def reset(self):
        """Clear internal state for both directions."""
        self._downsampler.clear()
        self._upsampler.clear()
