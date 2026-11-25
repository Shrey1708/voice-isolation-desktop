"""
Abstract audio processor interface.

Defines the contract for audio processing stages in the pipeline.
Processors receive 16kHz mono float32 chunks and return processed chunks.
"""

import numpy as np
from abc import ABC, abstractmethod


class ProcessorBase(ABC):
    """Base class for audio processors in the pipeline."""

    @abstractmethod
    def process(self, chunk_16k: np.ndarray) -> np.ndarray:
        """
        Process a 16kHz mono float32 audio chunk.

        Args:
            chunk_16k: Input audio at 16kHz, mono, float32.

        Returns:
            Processed audio at 16kHz, mono, float32.
        """

    def reset(self):
        """Reset processor state. Override if the processor is stateful."""


class PassthroughProcessor(ProcessorBase):
    """No-op processor that returns audio unchanged. Used for baseline testing."""

    def process(self, chunk_16k: np.ndarray) -> np.ndarray:
        return chunk_16k
