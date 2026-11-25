"""
Bounded thread-safe audio queue.

Thin wrapper around queue.Queue sized for real-time audio buffering.
Drops oldest frames on overflow rather than blocking the audio callback.
"""

import queue
import numpy as np


class AudioQueue:
    """
    Thread-safe bounded queue for passing audio chunks between
    the sounddevice callbacks and the worker thread.

    Args:
        maxsize: Maximum number of chunks before overflow.
                 Default 64 ≈ 64 * 40ms = 2.5 seconds of buffer.
    """

    def __init__(self, maxsize: int = 64):
        self._queue = queue.Queue(maxsize=maxsize)
        self.overruns = 0

    def put(self, chunk: np.ndarray):
        """
        Enqueue a chunk. If full, discard the oldest chunk and increment
        the overrun counter — never block the audio callback.
        """
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._queue.get_nowait()   # drop oldest
            except queue.Empty:
                pass
            self._queue.put_nowait(chunk)
            self.overruns += 1

    def get(self, timeout: float = 0.1) -> np.ndarray | None:
        """
        Dequeue a chunk, blocking up to `timeout` seconds.
        Returns None if the queue is empty after the timeout.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_nowait(self) -> np.ndarray | None:
        """Non-blocking dequeue. Returns None if empty. Safe for audio callbacks."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def depth(self) -> int:
        """Current number of queued chunks."""
        return self._queue.qsize()

    def clear(self):
        """Drain the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
