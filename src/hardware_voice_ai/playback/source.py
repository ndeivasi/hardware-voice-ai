"""Spoken-reply audio for the speaker.

The Audio Model is not implemented here. ``ReplyAudio`` is the stream the
session orchestrator plays. ``StubReplyAudio`` is the in-memory stand-in
so playback can be tested before a local or remote model exists. Chunks
can be pushed after playback has started.
"""

import threading
from typing import Protocol

import numpy as np

from hardware_voice_ai.playback.pcm import as_pcm


class ReplyAudio(Protocol):
    """PCM stream of one spoken reply.

    ``read`` returns the next chunk, an empty array when nothing is ready
    yet, or ``None`` when the reply is finished. Chunks are float32 in
    nominal range ``[-1, 1]``, shape ``(samples, channels)``.

    ``read`` must return promptly or wait in a way ``interrupt`` can wake.
    Playback calls ``interrupt`` on stop so a blocked read does not hold
    the speaker.
    """

    sample_rate_hz: int
    channels: int

    def read(self, max_samples: int) -> np.ndarray | None: ...

    def interrupt(self) -> None: ...


class MemoryReplyAudio:
    """A finished reply held entirely in memory.

    Used for local cues and for tests that already have the whole buffer.
    This is not an Audio Model.
    """

    def __init__(
        self,
        samples: np.ndarray,
        *,
        sample_rate_hz: int,
        channels: int,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self._data = as_pcm(samples, channels)
        self._pos = 0
        self._lock = threading.Lock()

    def read(self, max_samples: int) -> np.ndarray | None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        with self._lock:
            if self._pos >= len(self._data):
                return None
            end = min(self._pos + max_samples, len(self._data))
            chunk = self._data[self._pos : end].copy()
            self._pos = end
            return chunk

    def interrupt(self) -> None:
        return None


class StubReplyAudio:
    """Push-based stand-in for Audio Model reply audio.

    Playback may start after the first ``push``. Later chunks are read as
    they arrive. ``finish`` marks the end of the reply. Nothing is written
    to disk and nothing is sent over the network.

    This is not a local model, a remote model, or a cloud gateway.
    """

    def __init__(self, sample_rate_hz: int, channels: int) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if channels < 1:
            raise ValueError("channels must be at least 1")
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self._pending = np.zeros((0, channels), dtype=np.float32)
        self._finished = False
        self._cv = threading.Condition()

    def push(self, samples: np.ndarray) -> None:
        data = as_pcm(samples, self.channels)
        with self._cv:
            if self._finished:
                raise RuntimeError("reply audio is already finished")
            if len(self._pending) == 0:
                self._pending = data
            else:
                self._pending = np.concatenate((self._pending, data))
            self._cv.notify_all()

    def finish(self) -> None:
        with self._cv:
            self._finished = True
            self._cv.notify_all()

    def read_would_block(self) -> bool:
        """True when the next ``read`` would wait for a chunk that is not here."""
        with self._cv:
            return len(self._pending) == 0 and not self._finished

    def read(self, max_samples: int) -> np.ndarray | None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        with self._cv:
            if len(self._pending) == 0 and not self._finished:
                self._cv.wait(timeout=0.02)
            if len(self._pending) == 0:
                if self._finished:
                    return None
                return np.zeros((0, self.channels), dtype=np.float32)
            count = min(max_samples, len(self._pending))
            chunk = self._pending[:count].copy()
            self._pending = self._pending[count:].copy()
            return chunk

    def interrupt(self) -> None:
        with self._cv:
            self._cv.notify_all()
