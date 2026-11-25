"""In-process handoff from capture to the session orchestrator.

The orchestrator routes a finished utterance to the Audio Model (``local`` or
``remote``). Capture does not run that model and does not open a network
connection. Always-on microphone audio is never placed on this queue; only
the post-wake utterance, including the pre-roll that was handed over, is.
"""

import queue
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from hardware_voice_ai.session.audio_model import AudioModel
from hardware_voice_ai.wake.events import WakeEvent


@dataclass(frozen=True, slots=True)
class CapturedUtterance:
    """One utterance for the Audio Model, kept in memory only.

    ``samples`` is float32 PCM shaped ``(samples, channels)`` in ``[-1, 1]``.
    The leading ``pre_roll_sample_count`` rows are the wake snapshot, so audio
    from before the wake word ended is still present. ``wake_word_end_offset``
    is the row where that wake word ends (exclusive). Rows from there through
    the rest of the pre-roll, then the frames from ``on_frame``, are the
    continuation.
    """

    wake: WakeEvent
    audio_model: AudioModel
    samples: np.ndarray
    sample_rate_hz: int
    pre_roll_sample_count: int
    wake_word_end_offset: int
    endpoint: str

    def __post_init__(self) -> None:
        samples = np.array(self.samples, dtype=np.float32, copy=True)
        if samples.ndim != 2:
            raise ValueError("utterance samples must have shape (samples, channels)")
        object.__setattr__(self, "samples", samples)

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])


class UtteranceSink(Protocol):
    """Where a finished utterance goes.

    The session orchestrator, or a queue it drains, implements this. The call
    runs on the microphone thread, under the orchestrator's lock, because
    ``close_capture`` calls ``stop`` directly. Implementations should enqueue
    and return. They should not run the Audio Model and should not open a
    network connection.
    """

    def accept_utterance(self, utterance: CapturedUtterance) -> None: ...


class CaptureLifecycle(Protocol):
    """The slice of ``SessionOrchestrator`` capture calls on end-of-utterance.

    ``close_capture`` stops the tap with reason ``end_of_utterance`` and leaves
    the session in ``capturing``. Cancel and hardware mute are not sent through
    this method; the orchestrator calls ``stop`` for those itself.
    """

    def close_capture(self) -> None: ...


class QueueUtteranceSink:
    """In-process queue from the capture thread to the session orchestrator.

    ``accept_utterance`` only enqueues the in-memory utterance. It does not
    write samples to disk and it does not start an Audio Model.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[CapturedUtterance] = queue.Queue()

    def accept_utterance(self, utterance: CapturedUtterance) -> None:
        self._queue.put(utterance)

    def take(self) -> CapturedUtterance | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def drain(self) -> list[CapturedUtterance]:
        items: list[CapturedUtterance] = []
        while True:
            item = self.take()
            if item is None:
                return items
            items.append(item)
