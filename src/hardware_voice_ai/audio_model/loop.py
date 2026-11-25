"""The one voice loop for Audio Model ``local`` and ``remote``.

A captured utterance goes in. The selected model produces the spoken reply,
which is written to the speaker port as it arrives. Failures play a local
cue. A finished turn moves through ``speaking`` and back to ``idle``. Cancel
silences the speaker and does not play a cue.
"""

from dataclasses import dataclass

from hardware_voice_ai.audio_model.cues import LocalCue, render_cue
from hardware_voice_ai.audio_model.events import (
    Clock,
    Deadline,
    End,
    Failure,
    ReplyChunk,
    SystemClock,
    TurnCancel,
    TurnOutcome,
)
from hardware_voice_ai.session.audio_model import AudioModel
from hardware_voice_ai.session.state import DeviceState
from hardware_voice_ai.speaker.port import SpeakerPort


@dataclass(frozen=True, slots=True)
class TurnResult:
    outcome: TurnOutcome
    message: str = ""
    cue: str | None = None
    reply_sample_count: int = 0


class VoiceLoop:
    """Run one turn of the selected Audio Model and hand audio to the speaker."""

    def __init__(
        self,
        model,
        speaker: SpeakerPort,
        *,
        first_audio_timeout_s: float,
        turn_timeout_s: float,
        clock: Clock | None = None,
    ) -> None:
        self.model = model
        self.speaker = speaker
        self.first_audio_timeout_s = first_audio_timeout_s
        self.turn_timeout_s = turn_timeout_s
        self.clock = clock if clock is not None else SystemClock()

    @property
    def setting(self) -> AudioModel:
        return self.model.setting

    def play_cue(self, cue: LocalCue) -> None:
        self.speaker.play_cue(cue.value, render_cue(cue), 16_000)

    def run(self, utterance, cancel: TurnCancel, on_state) -> TurnResult:
        if cancel.is_cancelled():
            return TurnResult(TurnOutcome.CANCELLED, message="audio model cancelled")

        deadline = Deadline(
            self.clock,
            self.clock.monotonic(),
            self.first_audio_timeout_s,
            self.turn_timeout_s,
        )
        speaking = False
        reply_count = 0
        outcome = TurnOutcome.ERROR
        message = "audio model failed"
        generator = self.model.run(utterance, cancel, deadline)
        try:
            for event in generator:
                if cancel.is_cancelled():
                    outcome = TurnOutcome.CANCELLED
                    message = "audio model cancelled"
                    break
                if isinstance(event, ReplyChunk):
                    if not speaking:
                        on_state(DeviceState.SPEAKING)
                        self.speaker.start_reply(
                            event.sample_rate_hz, int(event.samples.shape[1])
                        )
                        speaking = True
                    self.speaker.write_reply(event.samples)
                    reply_count += int(event.samples.shape[0])
                    continue
                if isinstance(event, Failure):
                    outcome = event.outcome
                    message = event.message
                    break
                if isinstance(event, End):
                    if reply_count > 0:
                        outcome = TurnOutcome.SPOKEN
                        message = ""
                    else:
                        outcome = TurnOutcome.ERROR
                        message = "audio model returned no spoken reply"
                    break
            else:
                if reply_count > 0:
                    outcome = TurnOutcome.SPOKEN
                    message = ""
                else:
                    outcome = TurnOutcome.ERROR
                    message = "audio model returned no spoken reply"
        except Exception:
            outcome = TurnOutcome.ERROR
            message = "audio model failed"
        finally:
            # Close even on break, so a remote session's finally runs before
            # the device returns to idle and can accept another wake.
            try:
                generator.close()
            except Exception:
                outcome = TurnOutcome.ERROR
                message = "audio model failed"

        if cancel.is_cancelled():
            outcome = TurnOutcome.CANCELLED
            message = "audio model cancelled"

        if outcome is TurnOutcome.SPOKEN and speaking:
            self._call(self.speaker.end_reply)
        elif outcome is not TurnOutcome.SPOKEN:
            self._call(self.speaker.stop)

        cue = _cue_for(outcome, self.setting)
        if cue is not None:
            if not speaking:
                on_state(DeviceState.SPEAKING)
                speaking = True
            try:
                self.play_cue(cue)
            except Exception:
                cue = None

        if not speaking and outcome is not TurnOutcome.CANCELLED:
            on_state(DeviceState.SPEAKING)
        # Idle is the session's step, under its lock, after this call returns.
        # Doing it here would leave the old turn id set while the state is
        # already idle, and a new wake could be forced back to idle.
        return TurnResult(
            outcome,
            message=message,
            cue=None if cue is None else cue.value,
            reply_sample_count=reply_count,
        )

    def _call(self, method) -> None:
        try:
            method()
        except Exception:
            return


def _cue_for(outcome: TurnOutcome, setting: AudioModel) -> LocalCue | None:
    if outcome in (TurnOutcome.SPOKEN, TurnOutcome.CANCELLED, TurnOutcome.REFUSED):
        return None
    if setting is AudioModel.REMOTE and outcome in (
        TurnOutcome.UNAVAILABLE,
        TurnOutcome.TIMEOUT,
    ):
        return LocalCue.OFFLINE
    return LocalCue.ERROR
