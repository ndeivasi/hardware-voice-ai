"""Handoff from an accepted wake into utterance capture.

Issue 2 replaces ``PendingCapturePort``. This module defines the contract:
start on wake with the pre-roll, receive later microphone frames, and stop
on end-of-utterance, cancel, or hardware mute. Nothing here decides that the
user has finished speaking, and nothing here forwards audio to a model.

Issue 2 reports end-of-utterance through ``SessionOrchestrator.close_capture``.
Cancel and hardware mute are session-driven and return the device to idle.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from hardware_voice_ai.audio.frame import AudioFrame, AudioSnapshot
from hardware_voice_ai.session.audio_model import AudioModel
from hardware_voice_ai.wake.events import WakeEvent


class CaptureStopReason(StrEnum):
    """Why capture should release the microphone utterance.

    ``end_of_utterance`` is part of the contract for issue 2. This package
    does not emit it.
    """

    END_OF_UTTERANCE = "end_of_utterance"
    CANCEL = "cancel"
    MUTE = "mute"


@dataclass(frozen=True, slots=True)
class CaptureRequest:
    """What issue 2 needs in order to open an utterance.

    ``pre_roll`` includes microphone audio up to and including the wake
    decision frame, so it still holds samples from before the wake word
    ended. ``audio_model`` is the device setting for the turn (``local`` or
    ``remote``). The wake path records it and does not start a model.
    """

    wake: WakeEvent
    pre_roll: AudioSnapshot
    audio_model: AudioModel


class CapturePort(Protocol):
    """Issue 2's microphone-capture boundary.

    ``start`` is the handoff. Frames after the decision frame are delivered
    to ``on_frame`` until ``stop``. Implementations must not upload always-on
    audio; only this post-wake utterance, including the pre-roll, is in scope
    for a later turn. ``on_frame`` must not call back into the wake runtime.

    When issue 2 decides the utterance has ended, it calls
    ``SessionOrchestrator.close_capture``. That stops the tap with
    ``end_of_utterance`` and leaves the session in ``capturing``.
    """

    def start(self, request: CaptureRequest) -> None: ...

    def on_frame(self, frame: AudioFrame) -> None: ...

    def stop(self, reason: CaptureStopReason) -> None: ...


@dataclass
class PendingCapturePort:
    """Records the handoff and leaves the utterance unimplemented.

    ``on_frame`` discards samples. End-of-utterance, silence trimming, and
    passing the utterance to the audio model belong to later issues.
    """

    request: CaptureRequest | None = None
    active: bool = False
    stop_reasons: list[CaptureStopReason] = field(default_factory=list)

    def start(self, request: CaptureRequest) -> None:
        self.request = request
        self.active = True

    def on_frame(self, frame: AudioFrame) -> None:
        del frame

    def stop(self, reason: CaptureStopReason) -> None:
        self.stop_reasons.append(reason)
        self.active = False
