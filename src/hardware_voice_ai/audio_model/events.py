"""Events inside the one Audio Model loop.

The spoken reply is reply audio. That is the only audio the model returns
for the speaker. Control messages are errors, end of turn, and cancel.
"""

import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np


class TurnOutcome(StrEnum):
    SPOKEN = "spoken"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    ERROR = "error"
    LOW_CONFIDENCE = "low_confidence"
    REFUSED = "refused"


class Clock(Protocol):
    def monotonic(self) -> float: ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()


class ManualClock:
    """A clock a test can move forward without sleeping."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


@dataclass(frozen=True, slots=True)
class Deadline:
    clock: Clock
    started: float
    first_audio_s: float
    turn_s: float

    def elapsed(self) -> float:
        return self.clock.monotonic() - self.started

    def turn_expired(self) -> bool:
        return self.elapsed() >= self.turn_s

    def first_audio_expired(self, got_audio: bool) -> bool:
        if got_audio:
            return False
        return self.elapsed() >= self.first_audio_s

    def expired(self, got_audio: bool) -> bool:
        return self.turn_expired() or self.first_audio_expired(got_audio)


class TurnCancel:
    """Cooperative cancel for the turn that is in the audio model."""

    def __init__(self) -> None:
        self._flag = threading.Event()

    def cancel(self) -> None:
        self._flag.set()

    def is_cancelled(self) -> bool:
        return self._flag.is_set()


@dataclass(frozen=True, slots=True)
class ReplyChunk:
    """A piece of the spoken reply, float32 PCM shaped ``(samples, channels)``."""

    samples: np.ndarray
    sample_rate_hz: int

    def __post_init__(self) -> None:
        samples = np.array(self.samples, dtype=np.float32, copy=True)
        if samples.ndim != 2 or samples.shape[0] < 1 or samples.shape[1] < 1:
            raise ValueError("reply audio must have shape (samples, channels)")
        if self.sample_rate_hz <= 0:
            raise ValueError("reply sample rate must be positive")
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)


@dataclass(frozen=True, slots=True)
class Failure:
    outcome: TurnOutcome
    message: str


@dataclass(frozen=True, slots=True)
class End:
    """The model finished this turn. Any spoken reply already streamed."""
