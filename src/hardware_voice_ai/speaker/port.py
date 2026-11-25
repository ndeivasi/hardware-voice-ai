"""Where the Audio Model's spoken reply is recorded for the voice loop.

Device playback, volume, and immediate silence live in ``playback``. The
runtime plays these chunks with ``play_reply`` and local cues with
``play_cue``. ``MemorySpeaker`` here records the reply in memory and does
not open an audio device, so the voice loop can be tested without a speaker.

``stop`` ignores later reply writes until the next ``start_reply``. Cue
playback is separate from the spoken reply. Speaker methods must not call
back into the session.
"""

import threading
from typing import Protocol

import numpy as np


class SpeakerPort(Protocol):
    def start_reply(self, sample_rate_hz: int, channels: int) -> None: ...

    def write_reply(self, samples: np.ndarray) -> None: ...

    def end_reply(self) -> None: ...

    def stop(self) -> None: ...

    def play_cue(self, cue: str, samples: np.ndarray, sample_rate_hz: int) -> None: ...


class MemorySpeaker:
    """In-memory stand-in. It keeps the current reply and any local cues."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reply_chunks: list[np.ndarray] = []
        self.reply_rate: int | None = None
        self.reply_channels: int | None = None
        self.cues: list[tuple[str, np.ndarray, int]] = []
        self.stops = 0
        self.endings = 0
        self._accepting = False

    def start_reply(self, sample_rate_hz: int, channels: int) -> None:
        with self._lock:
            self.reply_rate = sample_rate_hz
            self.reply_channels = channels
            self.reply_chunks = []
            self._accepting = True

    def write_reply(self, samples: np.ndarray) -> None:
        with self._lock:
            if not self._accepting:
                return
            copied = np.array(samples, dtype=np.float32, copy=True)
            self.reply_chunks.append(copied)

    def end_reply(self) -> None:
        with self._lock:
            self._accepting = False
            self.endings += 1

    def stop(self) -> None:
        """Record an immediate stop. Already-handed chunks stay for inspection.

        A device implementation discards its playback buffer here. This
        stand-in rejects any further reply writes until the next reply.
        """
        with self._lock:
            self._accepting = False
            self.stops += 1

    def play_cue(self, cue: str, samples: np.ndarray, sample_rate_hz: int) -> None:
        with self._lock:
            copied = np.array(samples, dtype=np.float32, copy=True)
            self.cues.append((cue, copied, sample_rate_hz))

    def reply_audio(self) -> np.ndarray:
        with self._lock:
            channels = self.reply_channels or 1
            if not self.reply_chunks:
                return np.zeros((0, channels), dtype=np.float32)
            return np.concatenate(self.reply_chunks, axis=0)
