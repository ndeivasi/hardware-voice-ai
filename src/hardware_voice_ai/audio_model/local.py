"""On-device Audio Model.

The model takes utterance audio and yields the spoken reply. Nothing here
opens a socket. ``reference-audio-v1`` is a deterministic stand-in so the
loop can run without a model download. The production local model is an
open question; pass another ``LocalEngine`` to swap it in. An unknown id
is unavailable, and the wake loop still runs.
"""

from collections.abc import Iterator
from typing import Protocol

import numpy as np

from hardware_voice_ai.audio_model.defaults import REFERENCE_LOCAL_MODEL_ID
from hardware_voice_ai.audio_model.events import (
    Deadline,
    End,
    Failure,
    ReplyChunk,
    TurnCancel,
    TurnOutcome,
)
from hardware_voice_ai.session.audio_model import AudioModel

_RATE_HZ = 16_000
_FRAME = 320
_LOW_RMS = 0.01
_REPLY_S = 0.4


class LocalEngine(Protocol):
    def run(
        self,
        samples: np.ndarray,
        sample_rate_hz: int,
        cancel: TurnCancel,
        deadline: Deadline,
    ) -> Iterator[object]: ...


class ReferenceAudioEngine:
    """In-process stand-in. Reply audio is a tone shaped by the utterance level.

    It is not a conversational model and it does not leave the device. The
    same utterance always produces the same reply. Near-silence is reported
    as low confidence instead of a reply.
    """

    model_id = REFERENCE_LOCAL_MODEL_ID

    def run(
        self,
        samples: np.ndarray,
        sample_rate_hz: int,
        cancel: TurnCancel,
        deadline: Deadline,
    ) -> Iterator[object]:
        del sample_rate_hz
        if cancel.is_cancelled():
            yield Failure(TurnOutcome.CANCELLED, "audio model cancelled")
            return
        if deadline.expired(False):
            yield Failure(TurnOutcome.TIMEOUT, "audio model timed out")
            return
        level = _rms(samples)
        if level < _LOW_RMS:
            yield Failure(TurnOutcome.LOW_CONFIDENCE, "audio was hard to hear")
            return
        frequency = 400.0 + min(level, 1.0) * 400.0
        reply = _tone(frequency, _REPLY_S, _RATE_HZ)
        got_audio = False
        for chunk in _chunks(reply, _FRAME):
            if cancel.is_cancelled():
                yield Failure(TurnOutcome.CANCELLED, "audio model cancelled")
                return
            if deadline.expired(got_audio):
                yield Failure(TurnOutcome.TIMEOUT, "audio model timed out")
                return
            got_audio = True
            yield ReplyChunk(chunk, _RATE_HZ)
        yield End()


class LocalAudioModel:
    """The device's Audio Model when the setting is ``local``."""

    setting = AudioModel.LOCAL

    def __init__(self, engine: LocalEngine | None) -> None:
        self._engine = engine

    def run(
        self,
        utterance,
        cancel: TurnCancel,
        deadline: Deadline,
    ) -> Iterator[object]:
        if self._engine is None:
            yield Failure(TurnOutcome.UNAVAILABLE, "local audio model is unavailable")
            return
        samples = np.array(utterance.samples, dtype=np.float32, copy=True)
        yield from self._engine.run(samples, int(utterance.sample_rate_hz), cancel, deadline)


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    per_channel = np.sqrt(np.mean(np.square(samples, dtype=np.float64), axis=0))
    return float(np.max(per_channel))


def _tone(frequency_hz: float, duration_s: float, rate: int) -> np.ndarray:
    count = int(duration_s * rate)
    time = np.arange(count, dtype=np.float32) / rate
    wave = (0.2 * np.sin(2 * np.pi * frequency_hz * time)).astype(np.float32)
    fade = min(int(0.005 * rate), count // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        wave[:fade] *= ramp
        wave[-fade:] *= ramp[::-1]
    return wave.reshape(-1, 1)


def _chunks(samples: np.ndarray, frame: int):
    for start in range(0, int(samples.shape[0]), frame):
        yield samples[start : start + frame]
