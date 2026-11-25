"""
Streaming VoxLock Target Speaker Extraction (TSE) processor.

Wraps voxlock.VoxLock conditioned on a pre-recorded voiceprint so only
the enrolled speaker's voice passes through in real time. Includes an
optional smooth noise gate to suppress residual hiss between words.
"""

from pathlib import Path

import numpy as np
import voxlock

from .processor_base import ProcessorBase
from .resampler import RATE_MODEL


def _find_cached_model_dir(precision: str) -> Path | None:
    """Return the local Hugging Face cache directory if weights already exist."""
    try:
        from huggingface_hub import hf_hub_download
        from voxlock.core import PRECISIONS, REPO, REVISION

        for name in PRECISIONS[precision]:
            path = hf_hub_download(
                REPO,
                f"onnx/v2/{name}.onnx",
                revision=REVISION,
                local_files_only=True,
            )
        return Path(path).parent
    except Exception:
        return None


class SmoothNoiseGate:
    """
    Lightweight RMS-based noise gate with hangover and per-sample gain ramping
    to avoid clicks or clipped word endings.
    """

    def __init__(self, threshold_db: float = -45.0, hangover_ms: float = 200.0, floor_gain: float = 0.05):
        self.threshold_db = threshold_db
        self.hangover_samples = int(RATE_MODEL * hangover_ms / 1000)
        self.floor_gain = floor_gain
        self._hold_left = 0
        self._gain = 1.0

    def reset(self) -> None:
        self._hold_left = 0
        self._gain = 1.0

    def apply(self, audio_16k: np.ndarray) -> np.ndarray:
        if len(audio_16k) == 0:
            return audio_16k

        rms = float(np.sqrt(np.mean(audio_16k ** 2) + 1e-12))
        level_db = 20.0 * np.log10(rms)

        if level_db >= self.threshold_db:
            self._hold_left = self.hangover_samples
            target_gain = 1.0
        elif self._hold_left > 0:
            self._hold_left = max(0, self._hold_left - len(audio_16k))
            target_gain = 1.0
        else:
            target_gain = self.floor_gain

        if target_gain == self._gain:
            return audio_16k if self._gain == 1.0 else (audio_16k * self._gain).astype(np.float32)

        ramp = np.linspace(self._gain, target_gain, len(audio_16k), dtype=np.float32)
        self._gain = target_gain
        return (audio_16k * ramp).astype(np.float32)


class VoxLockProcessor(ProcessorBase):
    """
    Real-time streaming Target Speaker Extraction backed by VoxLock ONNX.

    Args:
        voiceprint_path: Path to the enrolled 16 kHz mono WAV clip.
        threads:         ONNX Runtime intra-op thread count (default 2).
        precision:       Model quantization ('int8', 'int8-full', 'fp16', 'fp32').
        noise_gate:      Enable optional post-TSE smooth noise gate.
        gate_threshold_db: RMS threshold in dBFS when noise_gate is enabled.
    """

    def __init__(
        self,
        voiceprint_path: str | Path,
        threads: int = 2,
        precision: str = "int8",
        noise_gate: bool = False,
        gate_threshold_db: float = -45.0,
    ):
        vp_path = Path(voiceprint_path)
        if not vp_path.exists():
            raise FileNotFoundError(f"Voiceprint not found: {vp_path}")

        self._voiceprint = voxlock.load_audio(str(vp_path))
        model_dir = _find_cached_model_dir(precision)
        self._vl = voxlock.VoxLock(model_dir=model_dir, precision=precision, threads=threads)
        self._stream = self._vl.stream(self._voiceprint)

        self.noise_gate = noise_gate
        self._gate = SmoothNoiseGate(threshold_db=gate_threshold_db)

    def process(self, chunk_16k: np.ndarray) -> np.ndarray:
        """Isolate target speaker in a 16 kHz mono float32 chunk."""
        isolated = self._stream.push(chunk_16k)
        if self.noise_gate and len(isolated) > 0:
            isolated = self._gate.apply(isolated)
        return isolated

    def reset(self) -> None:
        """Reset VoxLock LSTM and overlap-add state while keeping the enrolled voice."""
        self._stream = self._vl.stream(self._voiceprint)
        self._gate.reset()
