"""
Full-duplex audio I/O manager using sounddevice.

Opens separate input (physical mic) and output (virtual cable) streams,
routes audio through queues, and runs a worker thread that applies
resampling + processing between them.

Design rules (from workflow guardrails):
  - Audio callbacks NEVER run ML, resampling, disk I/O, or print().
  - Callbacks only copy float32 PCM into/out of bounded queues.
  - All heavy work happens in the worker thread.
"""

import threading
import time

import numpy as np
import sounddevice as sd

from .resampler import StreamingResampler, RATE_AUDIO
from .ring_buffer import AudioQueue
from .processor_base import ProcessorBase, PassthroughProcessor
from .telemetry import Telemetry


class AudioEngine:
    """
    Manages the full audio pipeline:
        Physical mic -> input queue -> worker thread -> output queue -> virtual cable.

    Args:
        input_device:  PortAudio device index or name for the physical mic.
        output_device: PortAudio device index or name for the virtual cable output.
        processor:     Audio processor (PassthroughProcessor or TSE).
        chunk_ms:      Frame size in milliseconds (default 40ms).
        sample_rate:   Stream sample rate (default 48000).
    """

    def __init__(
        self,
        input_device,
        output_device,
        processor: ProcessorBase | None = None,
        chunk_ms: int = 40,
        sample_rate: int = RATE_AUDIO,
    ):
        self.input_device = input_device
        self.output_device = output_device
        self.processor = processor or PassthroughProcessor()
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)

        self._in_queue = AudioQueue(maxsize=64)
        self._out_queue = AudioQueue(maxsize=64)
        self._resampler = StreamingResampler()
        self._telemetry = Telemetry()

        self._running = False
        self._bypass = False
        self._reset_requested = threading.Event()
        self._out_buffer = np.array([], dtype=np.float32)  # accumulator for exact chunking
        self._worker_thread: threading.Thread | None = None
        self._input_stream: sd.InputStream | None = None
        self._output_stream: sd.OutputStream | None = None

    # ── Properties ──────────────────────────────────────────────

    @property
    def bypass(self) -> bool:
        return self._bypass

    @bypass.setter
    def bypass(self, value: bool):
        if value != self._bypass:
            self._bypass = value
            self._reset_requested.set()

    def reset(self):
        """Request a thread-safe reset of resampler, processor, and buffer state."""
        self._reset_requested.set()

    @property
    def telemetry(self) -> Telemetry:
        return self._telemetry

    # ── Audio Callbacks (minimal — no processing!) ──────────────

    def _input_callback(self, indata, frames, time_info, status):
        """Copy captured audio into the input queue. Never process here."""
        # indata shape: (frames, channels) — take channel 0 as mono
        self._in_queue.put(indata[:, 0].copy())

    def _output_callback(self, outdata, frames, time_info, status):
        """Pull processed audio from the output queue. Output silence on underrun."""
        chunk = self._out_queue.get_nowait()
        if chunk is not None:
            outdata[:, 0] = chunk
            outdata[:, 1] = chunk
        else:
            outdata[:] = 0.0
            self._telemetry.record_underrun()

    # ── Worker Thread ───────────────────────────────────────────

    def _worker_loop(self):
        """
        Runs in a dedicated thread. Pulls from input queue, resamples,
        processes, resamples back, and pushes to output queue.

        Uses an accumulator buffer so the output queue always receives
        exact chunk_samples-sized arrays, regardless of resampler output
        length variations.
        """
        frame_duration = self.chunk_samples / self.sample_rate

        while self._running:
            if self._reset_requested.is_set():
                self._reset_requested.clear()
                self._resampler.reset()
                self.processor.reset()
                self._out_buffer = np.array([], dtype=np.float32)

            chunk_48k = self._in_queue.get(timeout=0.05)
            if chunk_48k is None:
                continue

            t0 = time.perf_counter()

            if self._bypass:
                processed_48k = chunk_48k
            else:
                chunk_16k = self._resampler.downsample(chunk_48k)
                processed_16k = self.processor.process(chunk_16k) if len(chunk_16k) > 0 else chunk_16k
                processed_48k = (
                    self._resampler.upsample(processed_16k)
                    if len(processed_16k) > 0
                    else np.zeros(0, dtype=np.float32)
                )

            elapsed = time.perf_counter() - t0
            self._telemetry.record_frame(elapsed, frame_duration)

            # Accumulate and push exact-sized chunks
            if len(processed_48k) > 0:
                self._out_buffer = np.concatenate([self._out_buffer, processed_48k])
                while len(self._out_buffer) >= self.chunk_samples:
                    exact = self._out_buffer[: self.chunk_samples]
                    self._out_buffer = self._out_buffer[self.chunk_samples :]
                    self._out_queue.put(exact)

            # Periodic status report
            if self._telemetry.should_report():
                self._telemetry.overruns = self._in_queue.overruns + self._out_queue.overruns
                report = self._telemetry.report(
                    in_depth=self._in_queue.depth,
                    out_depth=self._out_queue.depth,
                )
                print(report, flush=True)

    # ── Lifecycle ───────────────────────────────────────────────

    def start(self):
        """Open audio streams and start the worker thread."""
        if self._running:
            return

        self._running = True

        self._input_stream = sd.InputStream(
            device=self.input_device,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_samples,
            callback=self._input_callback,
        )

        self._output_stream = sd.OutputStream(
            device=self.output_device,
            samplerate=self.sample_rate,
            channels=2,
            dtype="float32",
            blocksize=self.chunk_samples,
            callback=self._output_callback,
        )

        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="audio-worker", daemon=True
        )

        self._input_stream.start()
        self._output_stream.start()
        self._worker_thread.start()

    def stop(self):
        """Stop streams and worker thread gracefully."""
        self._running = False

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

        if self._input_stream:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None

        if self._output_stream:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None

        # Final telemetry
        self._telemetry.overruns = self._in_queue.overruns + self._out_queue.overruns
        print(self._telemetry.report(
            in_depth=self._in_queue.depth,
            out_depth=self._out_queue.depth,
        ))
