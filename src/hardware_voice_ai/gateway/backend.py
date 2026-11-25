"""The speech-to-speech call the gateway makes for one turn.

An adapter receives the utterance and yields spoken audio. Text inside the
model stays inside the adapter. The device session never receives it.
"""

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class SpokenChunk:
    """A piece of the spoken reply, float32 PCM."""

    samples: np.ndarray
    sample_rate_hz: int

    def __post_init__(self) -> None:
        samples = np.array(self.samples, dtype=np.float32, copy=True)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        if samples.ndim != 2 or samples.shape[0] < 1 or samples.shape[1] < 1:
            raise ValueError("spoken audio must have shape (samples, channels)")
        if not np.isfinite(samples).all():
            raise ValueError("spoken audio must be finite")
        if self.sample_rate_hz <= 0:
            raise ValueError("spoken sample rate must be positive")
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)


class SpeechBackend(Protocol):
    """One utterance in, spoken audio out. No separate STT, LLM, or TTS call."""

    def speak(
        self,
        utterance: np.ndarray,
        sample_rate_hz: int,
        channels: int,
        cancel: threading.Event,
    ) -> Iterator[SpokenChunk]: ...
