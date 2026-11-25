"""Swappable speaker output.

Hardware is not chosen yet. The live backend is PortAudio through
``sounddevice``, aimed at a Linux-class single-board computer with a
speaker. ``MemorySpeaker`` and ``NullSpeaker`` keep tests off that
hardware. PCM is float32 in ``[-1, 1]``, the same layout as the microphone.
"""

import threading
from typing import Protocol

import numpy as np

from hardware_voice_ai.audio.frame import AudioFormat
from hardware_voice_ai.errors import AudioBackendError, ConfigurationError
from hardware_voice_ai.playback.constants import DEFAULT_QUEUE_MS, STOP_LATENCY_BUDGET_MS
from hardware_voice_ai.playback.pcm import as_pcm


class Speaker(Protocol):
    """One local playback device, or a stand-in.

    ``write`` queues float32 PCM shaped ``(samples, channels)``. The
    implementation keeps at most ``queue_ms`` of audio that ``drop_queued``
    can still discard. Audio older than that has left the queue and is not
    pulled back. ``flush`` plays out the queue. That is how a finished reply
    reaches the speaker and how a stop keeps the leftover under the latency
    budget.

    Implementations must not persist samples.
    """

    format: AudioFormat
    queue_ms: int

    def write(self, samples: np.ndarray) -> None: ...

    def drop_queued(self) -> int: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...

    @property
    def queued_samples(self) -> int: ...

    @property
    def played_samples(self) -> int: ...

    @property
    def dropped_samples(self) -> int: ...


def _check_queue_ms(queue_ms: int) -> int:
    if not isinstance(queue_ms, int) or isinstance(queue_ms, bool):
        raise ConfigurationError("playback queue must be a whole number of milliseconds.")
    if not 1 <= queue_ms <= STOP_LATENCY_BUDGET_MS:
        raise ConfigurationError(
            "playback queue must be from 1 to "
            f"{STOP_LATENCY_BUDGET_MS} ms so a stop can quiet the speaker "
            "inside the latency budget."
        )
    return queue_ms


def _check_level(samples: np.ndarray) -> None:
    if samples.size == 0:
        return
    peak = float(np.max(np.abs(samples)))
    if peak > 1.0 + 1e-5:
        raise ValueError("refusing speaker samples outside [-1, 1]")


class NullSpeaker:
    """Drops PCM immediately after counting it.

    The default when a caller did not ask for a speaker. It keeps the queue
    accounting used to bound stop latency and does not retain samples.
    """

    def __init__(self, fmt: AudioFormat, queue_ms: int = DEFAULT_QUEUE_MS) -> None:
        self.format = fmt
        self.queue_ms = _check_queue_ms(queue_ms)
        self._limit = max(1, fmt.sample_rate_hz * self.queue_ms // 1000)
        self._queued = 0
        self._played = 0
        self._dropped = 0
        self._closed = False
        self._lock = threading.Lock()

    def write(self, samples: np.ndarray) -> None:
        data = as_pcm(samples, self.format.channels)
        _check_level(data)
        with self._lock:
            self._require_open()
            self._queued += int(data.shape[0])
            overflow = self._queued - self._limit
            if overflow > 0:
                self._played += overflow
                self._queued -= overflow

    def drop_queued(self) -> int:
        with self._lock:
            dropped = self._queued
            self._dropped += dropped
            self._queued = 0
            return dropped

    def flush(self) -> None:
        with self._lock:
            self._played += self._queued
            self._queued = 0

    def close(self) -> None:
        with self._lock:
            self._queued = 0
            self._closed = True

    @property
    def queued_samples(self) -> int:
        with self._lock:
            return self._queued

    @property
    def played_samples(self) -> int:
        with self._lock:
            return self._played

    @property
    def dropped_samples(self) -> int:
        with self._lock:
            return self._dropped

    def retains_samples(self) -> bool:
        return False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("speaker is closed")


class MemorySpeaker:
    """In-memory speaker for tests and injected audio.

    Samples stay in RAM so a test can see what would have been heard.
    The live device uses ``SoundDeviceSpeaker``, which does not keep a copy.
    The last ``queue_ms`` of audio stays queued until more audio arrives,
    ``flush`` plays it, or ``drop_queued`` discards it.
    """

    def __init__(self, fmt: AudioFormat, queue_ms: int = DEFAULT_QUEUE_MS) -> None:
        self.format = fmt
        self.queue_ms = _check_queue_ms(queue_ms)
        self._limit = max(1, fmt.sample_rate_hz * self.queue_ms // 1000)
        self._heard = np.zeros((0, fmt.channels), dtype=np.float32)
        self._queued = np.zeros((0, fmt.channels), dtype=np.float32)
        self._dropped = 0
        self._closed = False
        self._lock = threading.Lock()

    def write(self, samples: np.ndarray) -> None:
        data = as_pcm(samples, self.format.channels)
        _check_level(data)
        with self._lock:
            self._require_open()
            if len(self._queued) == 0:
                self._queued = data
            else:
                self._queued = np.concatenate((self._queued, data))
            overflow = len(self._queued) - self._limit
            if overflow > 0:
                head = self._queued[:overflow]
                self._queued = self._queued[overflow:].copy()
                self._heard = (
                    head if len(self._heard) == 0 else np.concatenate((self._heard, head))
                )

    def drop_queued(self) -> int:
        with self._lock:
            dropped = len(self._queued)
            self._dropped += dropped
            self._queued = np.zeros((0, self.format.channels), dtype=np.float32)
            return dropped

    def flush(self) -> None:
        with self._lock:
            if len(self._queued) == 0:
                return
            self._heard = (
                self._queued
                if len(self._heard) == 0
                else np.concatenate((self._heard, self._queued))
            )
            self._queued = np.zeros((0, self.format.channels), dtype=np.float32)

    def close(self) -> None:
        with self._lock:
            self._queued = np.zeros((0, self.format.channels), dtype=np.float32)
            self._closed = True

    @property
    def queued_samples(self) -> int:
        with self._lock:
            return len(self._queued)

    @property
    def played_samples(self) -> int:
        with self._lock:
            return len(self._heard)

    @property
    def dropped_samples(self) -> int:
        with self._lock:
            return self._dropped

    def heard(self) -> np.ndarray:
        """Copy of audio that has left the hardware queue."""
        with self._lock:
            return self._heard.copy()

    def accepted(self) -> np.ndarray:
        """Copy of heard audio plus audio still sitting in the queue."""
        with self._lock:
            if len(self._queued) == 0:
                return self._heard.copy()
            if len(self._heard) == 0:
                return self._queued.copy()
            return np.concatenate((self._heard, self._queued))

    def retains_samples(self) -> bool:
        return True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("speaker is closed")


class SoundDeviceSpeaker:
    """Live playback through ``sounddevice`` (PortAudio).

    The import is deferred so tests run without the audio extra and without
    a speaker. A callback pulls one frame at a time from a short queue.
    ``drop_queued`` clears that queue. The frame already inside the callback
    can still finish; at the default 20 ms frame that is inside the stop
    budget. Samples are not copied into a file or a retained history.
    """

    def __init__(
        self,
        fmt: AudioFormat,
        device: str | int | None = None,
        queue_ms: int = DEFAULT_QUEUE_MS,
    ) -> None:
        sd = _import_sounddevice()
        self.format = fmt
        self.queue_ms = _check_queue_ms(queue_ms)
        if fmt.frame_ms > STOP_LATENCY_BUDGET_MS:
            raise ConfigurationError(
                "speaker frame is longer than the stop-latency budget"
            )
        self._limit = max(fmt.frame_samples, fmt.sample_rate_hz * self.queue_ms // 1000)
        self._pending = np.zeros((0, fmt.channels), dtype=np.float32)
        self._played = 0
        self._dropped = 0
        self._closed = False
        self._flush_epoch = 0
        self._cv = threading.Condition()
        self._stream = sd.OutputStream(
            samplerate=fmt.sample_rate_hz,
            channels=fmt.channels,
            dtype="float32",
            blocksize=fmt.frame_samples,
            device=device,
            latency="low",
            callback=self._on_audio,
        )
        try:
            self._stream.start()
        except Exception as exc:
            self._stream.close()
            raise AudioBackendError(f"Could not open the speaker: {exc}") from exc

    def write(self, samples: np.ndarray) -> None:
        data = np.ascontiguousarray(as_pcm(samples, self.format.channels))
        _check_level(data)
        offset = 0
        with self._cv:
            while offset < len(data):
                if self._closed:
                    raise RuntimeError("speaker is closed")
                room = self._limit - len(self._pending)
                if room <= 0:
                    self._cv.wait(timeout=0.02)
                    continue
                take = min(room, len(data) - offset)
                piece = data[offset : offset + take]
                self._append_pending(piece)
                offset += take

    def write_available(self, samples: np.ndarray) -> int:
        """Queue what fits right now. Return how many samples were queued.

        Unlike ``write``, this does not wait for the callback to make room.
        The playback thread uses that so it can tell the caller it is about
        to block, and the caller can drain the callback.
        """
        data = np.ascontiguousarray(as_pcm(samples, self.format.channels))
        _check_level(data)
        with self._cv:
            if self._closed:
                raise RuntimeError("speaker is closed")
            room = self._limit - len(self._pending)
            take = min(room, len(data))
            if take <= 0:
                return 0
            self._append_pending(data[:take])
            return int(take)

    def drop_queued(self) -> int:
        with self._cv:
            dropped = len(self._pending)
            self._dropped += dropped
            self._pending = np.zeros((0, self.format.channels), dtype=np.float32)
            self._flush_epoch += 1
            self._cv.notify_all()
            return dropped

    def flush(self) -> None:
        """Block until the callback has played the queue, or stop/close aborts it.

        ``drop_queued`` and ``close`` wake this wait. Playback stays in
        progress until ``flush`` returns, so mute and stop can still discard
        the tail. A no-op return would mark the sound complete while audio
        is sitting in ``_pending``.
        """
        with self._cv:
            epoch = self._flush_epoch
            while len(self._pending) > 0 and not self._closed and self._flush_epoch == epoch:
                self._cv.wait(timeout=0.05)

    def flush_blocks(self) -> bool:
        """True when ``flush`` will wait for the callback to drain the queue."""
        with self._cv:
            return len(self._pending) > 0 and not self._closed

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._pending = np.zeros((0, self.format.channels), dtype=np.float32)
            self._flush_epoch += 1
            self._cv.notify_all()
        self._stream.stop()
        self._stream.close()

    @property
    def queued_samples(self) -> int:
        with self._cv:
            return len(self._pending)

    @property
    def played_samples(self) -> int:
        with self._cv:
            return self._played

    @property
    def dropped_samples(self) -> int:
        with self._cv:
            return self._dropped

    def retains_samples(self) -> bool:
        return False

    def _append_pending(self, piece: np.ndarray) -> None:
        if len(piece) == 0:
            return
        self._pending = (
            piece if len(self._pending) == 0 else np.concatenate((self._pending, piece))
        )

    def _on_audio(self, outdata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        del time_info, status
        with self._cv:
            take = min(frames, len(self._pending))
            if take:
                outdata[:take] = self._pending[:take]
                self._pending = self._pending[take:].copy()
                self._played += take
            if take < frames:
                outdata[take:] = 0
            self._cv.notify_all()


def build_speaker(
    fmt: AudioFormat,
    backend: str,
    *,
    device: str | int | None = None,
    queue_ms: int = DEFAULT_QUEUE_MS,
) -> Speaker:
    """Open the speaker backend named by the device config.

    ``pcm`` is the in-memory speaker used by tests. ``sounddevice`` is the
    live PortAudio output. Both speak the same PCM format.
    """
    if backend == "pcm":
        return MemorySpeaker(fmt, queue_ms=queue_ms)
    if backend == "sounddevice":
        return SoundDeviceSpeaker(fmt, device=device, queue_ms=queue_ms)
    raise ConfigurationError(
        f"Unknown audio backend {backend!r}. Use 'sounddevice' or 'pcm'."
    )


def _import_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise AudioBackendError(
            "The sounddevice audio backend is not installed. "
            "Install the 'audio' extra to open a speaker."
        ) from exc
    return sd
