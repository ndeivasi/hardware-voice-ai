"""Remote Audio Model.

One session per turn, opened only after a captured utterance is submitted.
The device sends that utterance and may send a cancel. The gateway sends
spoken-reply audio, errors, and end of turn. The spoken reply is audio.
This class does not name a vendor and does not import a socket; the session
opener does that, and only when a gateway URL is configured.
"""

from collections.abc import Callable, Iterator

import numpy as np

from hardware_voice_ai.audio_model.events import (
    Deadline,
    End,
    Failure,
    ReplyChunk,
    TurnCancel,
    TurnOutcome,
)
from hardware_voice_ai.audio_model.framing import iter_chunks
from hardware_voice_ai.audio_model.gateway import (
    GatewayAudio,
    GatewayEnd,
    GatewayError,
    redact,
)
from hardware_voice_ai.audio_model.session import GatewaySession
from hardware_voice_ai.errors import ModelUnavailable
from hardware_voice_ai.session.audio_model import AudioModel

_CHUNK = 320
_FAILURE_FROM_CODE = {
    "unavailable": TurnOutcome.UNAVAILABLE,
    "timeout": TurnOutcome.TIMEOUT,
    "low_confidence": TurnOutcome.LOW_CONFIDENCE,
    "cancelled": TurnOutcome.CANCELLED,
}


SessionOpener = Callable[[], GatewaySession]


class RemoteAudioModel:
    """The device's Audio Model when the setting is ``remote``."""

    setting = AudioModel.REMOTE

    def __init__(self, opener: SessionOpener, *, poll_s: float = 0.05, token: str = "") -> None:
        self._opener = opener
        self._poll_s = poll_s
        self._token = token

    def run(self, utterance, cancel: TurnCancel, deadline: Deadline) -> Iterator[object]:
        if cancel.is_cancelled():
            yield Failure(TurnOutcome.CANCELLED, "audio model cancelled")
            return
        try:
            session = self._opener()
        except (ModelUnavailable, OSError, TimeoutError, ConnectionError):
            yield Failure(TurnOutcome.UNAVAILABLE, "remote audio model is unavailable")
            return
        try:
            yield from self._exchange(session, utterance, cancel, deadline)
        finally:
            try:
                session.close()
            except Exception:
                pass

    def _exchange(
        self,
        session: GatewaySession,
        utterance,
        cancel: TurnCancel,
        deadline: Deadline,
    ) -> Iterator[object]:
        samples = np.array(utterance.samples, dtype=np.float32, copy=True)
        try:
            session.start_turn(
                sample_rate_hz=int(utterance.sample_rate_hz),
                channels=int(samples.shape[1]),
            )
            for chunk in iter_chunks(samples, _CHUNK):
                stopped = self._stop_if_needed(session, cancel, deadline, got_audio=False)
                if stopped is not None:
                    yield stopped
                    return
                session.send_audio(chunk)
            session.finish_audio()
            got_audio = False
            while True:
                stopped = self._stop_if_needed(session, cancel, deadline, got_audio)
                if stopped is not None:
                    yield stopped
                    return
                remaining = max(0.0, deadline.turn_s - deadline.elapsed())
                event = session.receive(min(self._poll_s, remaining) if remaining else 0.0)
                if event is None:
                    continue
                if isinstance(event, GatewayAudio):
                    got_audio = True
                    yield ReplyChunk(event.samples, event.sample_rate_hz)
                    continue
                if isinstance(event, GatewayEnd):
                    yield End()
                    return
                if isinstance(event, GatewayError):
                    yield self._failure(event)
                    return
                yield Failure(TurnOutcome.ERROR, "audio model failed")
                return
        except (ModelUnavailable, OSError, TimeoutError, ConnectionError):
            yield Failure(TurnOutcome.UNAVAILABLE, "remote audio model is unavailable")
        except Exception:
            yield Failure(TurnOutcome.ERROR, "audio model failed")

    def _stop_if_needed(
        self,
        session: GatewaySession,
        cancel: TurnCancel,
        deadline: Deadline,
        got_audio: bool,
    ) -> Failure | None:
        if cancel.is_cancelled():
            self._cancel(session)
            return Failure(TurnOutcome.CANCELLED, "audio model cancelled")
        if deadline.expired(got_audio):
            self._cancel(session)
            return Failure(TurnOutcome.TIMEOUT, "audio model timed out")
        return None

    def _cancel(self, session: GatewaySession) -> None:
        try:
            session.send_cancel()
        except Exception:
            pass

    def _failure(self, event: GatewayError) -> Failure:
        outcome = _FAILURE_FROM_CODE.get(event.code, TurnOutcome.ERROR)
        message = redact(event.message, self._token).strip()
        if not message or len(message) > 200:
            message = "audio model failed"
        if outcome is TurnOutcome.UNAVAILABLE:
            message = "remote audio model is unavailable"
        elif outcome is TurnOutcome.TIMEOUT:
            message = "audio model timed out"
        elif outcome is TurnOutcome.LOW_CONFIDENCE:
            message = "audio was hard to hear"
        elif outcome is TurnOutcome.CANCELLED:
            message = "audio model cancelled"
        return Failure(outcome, message)
