"""
Rolling telemetry tracker for latency, RTF, and XRUNs.

Collects per-frame timing and reports rolling averages every N seconds.
All methods are designed to be called from the worker thread (non-blocking).
"""

import time
from collections import deque


class Telemetry:
    """
    Tracks processing performance metrics.

    Args:
        window_seconds: How many seconds of history to keep for rolling averages.
        report_interval: Seconds between printed status reports.
    """

    def __init__(self, window_seconds: float = 10.0, report_interval: float = 5.0):
        self._window = window_seconds
        self._report_interval = report_interval

        # (timestamp, rtf) pairs for rolling average
        self._rtf_history: deque[tuple[float, float]] = deque()

        self.underruns = 0
        self.overruns = 0
        self.frames_processed = 0
        self._last_report = time.monotonic()
        self._start_time = time.monotonic()

    def record_frame(self, processing_time: float, frame_duration: float):
        """
        Record one processed frame's timing.

        Args:
            processing_time: Wall-clock seconds spent processing this frame.
            frame_duration: Duration of the audio frame in seconds.
        """
        now = time.monotonic()
        rtf = processing_time / frame_duration if frame_duration > 0 else 0.0

        self._rtf_history.append((now, rtf))
        self.frames_processed += 1

        # Trim old entries outside the rolling window
        cutoff = now - self._window
        while self._rtf_history and self._rtf_history[0][0] < cutoff:
            self._rtf_history.popleft()

    def record_underrun(self):
        self.underruns += 1

    def record_overrun(self):
        self.overruns += 1

    @property
    def rolling_rtf(self) -> float:
        """Rolling average Real-Time Factor over the window."""
        if not self._rtf_history:
            return 0.0
        return sum(rtf for _, rtf in self._rtf_history) / len(self._rtf_history)

    def should_report(self) -> bool:
        """Check if it's time to print a status line."""
        return (time.monotonic() - self._last_report) >= self._report_interval

    def report(self, in_depth: int = 0, out_depth: int = 0) -> str:
        """
        Generate a status report string and reset the report timer.

        Args:
            in_depth: Current input queue depth.
            out_depth: Current output queue depth.
        """
        self._last_report = time.monotonic()
        elapsed = self._last_report - self._start_time

        return (
            f"[{elapsed:6.0f}s] "
            f"RTF={self.rolling_rtf:.3f}  "
            f"Frames={self.frames_processed}  "
            f"Q(in={in_depth}, out={out_depth})  "
            f"Underruns={self.underruns}  "
            f"Overruns={self.overruns}"
        )
