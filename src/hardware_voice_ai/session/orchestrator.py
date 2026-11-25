"""Session owner for an accepted wake, one Audio Model turn, and playback.

The wake path moves ``idle → woken → capturing`` and hands the pre-roll to
capture. After capture closes, the Audio Model runs off the microphone
thread. ``play_reply`` plays spoken-reply audio on the device speaker.
``play_cue`` plays a local wake, offline, or error cue. Neither call opens
a cloud session.

A new accepted wake during speaking cancels the model turn, cuts playback,
and starts a new capture. Hardware mute cuts playback and returns the
device to idle. Releasing mute does not resume audio.
"""

import threading

from hardware_voice_ai.audio.frame import AudioFormat, AudioSnapshot
from hardware_voice_ai.audio_model.cues import LocalCue
from hardware_voice_ai.audio_model.events import TurnCancel, TurnOutcome
from hardware_voice_ai.audio_model.loop import TurnResult, VoiceLoop
from hardware_voice_ai.audio_model.utterance import utterance_problem
from hardware_voice_ai.capture.port import CapturePort, CaptureRequest, CaptureStopReason
from hardware_voice_ai.errors import ConfigurationError
from hardware_voice_ai.playback.cues import Cue
from hardware_voice_ai.playback.player import (
    PlaybackResult,
    PlaybackStatus,
    PlaybackStopReason,
    SpeakerPlayback,
    muted_result,
)
from hardware_voice_ai.playback.source import ReplyAudio
from hardware_voice_ai.playback.speaker import NullSpeaker
from hardware_voice_ai.session.audio_model import AudioModel
from hardware_voice_ai.session.state import DeviceState, IllegalTransitionError, assert_transition
from hardware_voice_ai.wake.events import WakeEvent


class SessionOrchestrator:
    """On-device session state for wake, capture, the Audio Model, and playback.

    Audio Model is stored for the turn and copied onto the capture request.
    ``local`` and ``remote`` use the same speaker path. Reply audio is
    whatever the caller streams in. When a voice loop is attached, an
    accepted utterance is run through that one model.
    """

    def __init__(
        self,
        capture: CapturePort,
        audio_model: AudioModel = AudioModel.REMOTE,
        playback: SpeakerPlayback | None = None,
        *,
        voice: VoiceLoop | None = None,
    ) -> None:
        if isinstance(audio_model, str):
            audio_model = AudioModel(audio_model)
        if voice is not None and voice.setting is not audio_model:
            raise ConfigurationError(
                "Voice loop does not match the Audio Model setting."
            )
        self.capture = capture
        self.audio_model = audio_model
        self.voice = voice
        self.playback = (
            playback
            if playback is not None
            else SpeakerPlayback(NullSpeaker(AudioFormat(16_000, 1, 320)))
        )
        self.state = DeviceState.IDLE
        self.trace = [DeviceState.IDLE]
        self._muted = False
        self._forward_frames = False
        self._lock = threading.RLock()
        self._reply_token = 0
        self.last_reply_result: PlaybackResult | None = None
        self._active_turn: TurnCancel | None = None
        self._turn_id = 0
        self._run_lock = threading.Lock()

    @property
    def is_muted(self) -> bool:
        return self._muted

    @property
    def model_is_running(self) -> bool:
        """True while an Audio Model turn has been started and not finished."""
        with self._lock:
            return self._active_turn is not None

    @property
    def capture_is_active(self) -> bool:
        """True while microphone frames should be tapped to capture."""
        with self._lock:
            return self._forward_frames

    def on_wake_accepted(self, event: WakeEvent, pre_roll: AudioSnapshot) -> bool:
        """Notify the session of a wake the engine already accepted.

        Hardware mute refuses the wake. A wake during capture is ignored.
        A wake during speaking stops playback, cancels the model turn, then
        accepts the new turn. An accepted wake plays the local wake cue.
        The runtime keeps that cue out of the capture tap and ignores a
        further wake while it plays.
        """
        if not self._cut_playback_for_wake():
            return False
        with self._lock:
            if self._muted or self.state is not DeviceState.IDLE:
                return False
            accepted = self._begin_capture(event, pre_roll)
        if accepted:
            self.play_cue(Cue.WAKE)
        return accepted

    def close_capture(self) -> None:
        """Stop the capture tap because the utterance ended.

        The session stays in ``capturing`` so the caller can run the Audio
        Model or pass reply audio to ``play_reply``. This does not infer
        end-of-utterance, start the model, or start playback.
        """
        with self._lock:
            if not self._forward_frames:
                return
            self._forward_frames = False
            self.capture.stop(CaptureStopReason.END_OF_UTTERANCE)

    def play_reply(self, source: ReplyAudio) -> PlaybackResult:
        """Play spoken-reply audio on the local speaker.

        Call this after capture has closed, while the session is still
        ``capturing``, or while it is already ``speaking``. The session
        moves to ``speaking`` and, when no Audio Model turn owns the reply,
        back to ``idle`` when the reply completes. A later ``play_reply``
        replaces the one in progress. Hardware mute refuses the call and
        plays nothing.
        """
        with self._lock:
            if self._muted:
                return muted_result()
            if self.state is DeviceState.CAPTURING and self._forward_frames:
                raise IllegalTransitionError(
                    "Reply playback starts after capture has closed."
                )
            if self.state is DeviceState.CAPTURING:
                self._transition(DeviceState.SPEAKING)
            elif self.state is not DeviceState.SPEAKING:
                raise IllegalTransitionError(
                    f"Cannot play a reply from {self.state.value}."
                )
            self._reply_token += 1
            token = self._reply_token
            playback = self.playback
        try:
            result = playback.play_reply(
                source,
                on_finished=lambda done: self._on_reply_finished(token, done),
            )
        except Exception:
            with self._lock:
                if (
                    self._active_turn is None
                    and token == self._reply_token
                    and self.state is DeviceState.SPEAKING
                ):
                    self._transition(DeviceState.IDLE)
            raise
        with self._lock:
            if result.reason is PlaybackStopReason.MUTE:
                if token == self._reply_token and self.state is DeviceState.SPEAKING:
                    self._transition(DeviceState.IDLE)
                return muted_result()
        return result

    def play_cue(self, cue: Cue) -> PlaybackResult:
        """Play a local wake, offline, or error cue.

        Cues are rendered in-process. The Audio Model mode does not choose
        a cloud path for them. A cue during ``speaking`` stops the reply
        and returns the session to ``idle`` before the cue starts.
        """
        with self._lock:
            if self._muted:
                return muted_result()
            leave_speaking = self.state is DeviceState.SPEAKING
            if leave_speaking:
                self._reply_token += 1
                self._drop_model_locked()
            playback = self.playback
        if leave_speaking:
            self._stop_voice_speaker()
        stopped: PlaybackResult | None = None
        if leave_speaking or playback.is_playing:
            stopped = playback.stop(PlaybackStopReason.REPLACED)
        with self._lock:
            if self._muted:
                return muted_result()
            if leave_speaking:
                if stopped is not None:
                    self.last_reply_result = stopped
                if self.state is DeviceState.SPEAKING:
                    self._transition(DeviceState.IDLE)
        return self.playback.play_cue(cue)

    def set_volume(self, volume: float) -> None:
        """Set speaker gain in ``[0, 1]``. This does not change mute."""
        self.playback.set_volume(volume)

    def stop_playback(self) -> PlaybackResult:
        """Stop the current sound. A spoken reply returns the session to idle."""
        with self._lock:
            leave_speaking = self.state is DeviceState.SPEAKING
            if leave_speaking:
                self._reply_token += 1
                self._drop_model_locked()
            playback = self.playback
        result = playback.stop(PlaybackStopReason.STOP)
        self._stop_voice_speaker()
        with self._lock:
            if leave_speaking:
                self.last_reply_result = result
                if self.state is DeviceState.SPEAKING:
                    self._transition(DeviceState.IDLE)
        return result

    def playback_status(self) -> PlaybackResult:
        """Status of the sound playing now, if any.

        ``last_reply_result`` keeps the spoken reply's own completion or
        stop result after a later cue replaces it.
        """
        return self.playback.status()

    def arm_for_model(self, utterance) -> TurnResult | tuple[int, TurnCancel]:
        """Enter ``speaking`` on the caller thread before the model thread runs.

        The microphone loop calls this as soon as capture has queued an
        utterance, before it reads another frame. A wake that arrives in
        that gap is barge-in rather than a refused wake. The Audio Model
        itself still runs on another thread.
        """
        with self._lock:
            refused = self._refusal(utterance)
            if refused is not None:
                if refused.outcome is TurnOutcome.ERROR:
                    self._transition(DeviceState.IDLE)
                return refused
            self._turn_id += 1
            self._active_turn = TurnCancel()
            self._transition(DeviceState.SPEAKING)
            return self._turn_id, self._active_turn

    def owns_turn(self, turn_id: int) -> bool:
        with self._lock:
            return self._turn_id == turn_id and self._active_turn is not None

    def run_armed_turn(self, utterance, turn_id: int, cancel: TurnCancel) -> TurnResult:
        """Run a turn that ``arm_for_model`` already moved to ``speaking``."""
        with self._run_lock:
            with self._lock:
                if self._turn_id != turn_id or cancel.is_cancelled():
                    return TurnResult(
                        TurnOutcome.CANCELLED, message="audio model cancelled"
                    )
            try:
                return self._run_voice(utterance, turn_id, cancel)
            except Exception:
                return TurnResult(TurnOutcome.ERROR, message="audio model failed")
            finally:
                self._finish_turn(turn_id)

    def submit_utterance(self, utterance) -> TurnResult:
        """Run the selected Audio Model on one captured utterance.

        Call this after ``close_capture``, and not from the microphone
        callback. The model receives this utterance's samples. Always-on
        audio is not included unless it was already part of that utterance.
        ``local`` does not open a gateway session. ``remote`` opens one
        session for this turn only.
        """
        if self.voice is None:
            raise ConfigurationError("No Audio Model is attached to this session.")
        with self._run_lock:
            prepared = self._prepare_turn(utterance)
            if isinstance(prepared, TurnResult):
                if prepared.cue is not None:
                    try:
                        self.voice.play_cue(LocalCue(prepared.cue))
                    except Exception:
                        pass
                return prepared
            turn_id, cancel = prepared
            try:
                return self._run_voice(utterance, turn_id, cancel)
            except Exception:
                return TurnResult(TurnOutcome.ERROR, message="audio model failed")
            finally:
                self._finish_turn(turn_id)

    def cancel(self) -> None:
        """Return to idle. Capture and playback both stop. No cue is played."""
        with self._lock:
            if (
                self.state is DeviceState.IDLE
                and not self.playback.is_playing
                and self._active_turn is None
            ):
                return
            self._forward_frames = False
            to_idle = self.state is not DeviceState.IDLE
            was_speaking = self.state is DeviceState.SPEAKING
            if to_idle:
                self.capture.stop(CaptureStopReason.CANCEL)
                self._reply_token += 1
            self._drop_model_locked()
            playback = self.playback
        result = playback.stop(PlaybackStopReason.STOP)
        self._stop_voice_speaker()
        with self._lock:
            if was_speaking:
                self.last_reply_result = result
            if to_idle and self.state is not DeviceState.IDLE:
                self._transition(DeviceState.IDLE)

    def apply_mute(self, muted: bool) -> None:
        """Hardware mute is authoritative and returns the device to idle.

        Mute stops playback immediately. Releasing mute does not resume it.
        While muted, reply and cue playback write no samples.
        """
        with self._lock:
            self._muted = muted
            self.playback.set_suppressed(muted)
            if not muted:
                return
            self._forward_frames = False
            to_idle = self.state is not DeviceState.IDLE
            was_speaking = self.state is DeviceState.SPEAKING
            if to_idle:
                self.capture.stop(CaptureStopReason.MUTE)
                self._reply_token += 1
            self._drop_model_locked()
            playback = self.playback
        result = playback.stop(PlaybackStopReason.MUTE)
        self._stop_voice_speaker()
        with self._lock:
            if was_speaking:
                self.last_reply_result = result
            if to_idle and self.state is not DeviceState.IDLE:
                self._transition(DeviceState.IDLE)

    def _finish_turn(self, turn_id: int) -> None:
        """Clear this turn. Stay in ``speaking`` while playback is still flushing.

        A wake can already have cancelled this turn and opened a new capture.
        That capture must stay. ``_active_turn`` is cleared once, under this
        lock. If this turn is still the one in ``speaking`` and the speaker
        has not finished ``flush``, the session stays in ``speaking`` so a
        wake can stop that tail. ``_on_reply_finished`` goes idle after the
        flush completes. If playback is already quiet, this method goes idle,
        still once. VoiceLoop does not move the session to idle.
        """
        with self._lock:
            if self._turn_id != turn_id:
                return
            newer_capture = self._forward_frames or self.state is DeviceState.WOKEN
            self._active_turn = None
            if newer_capture:
                return
            if self.state is DeviceState.SPEAKING and self.playback.is_playing:
                return
            if self.state is not DeviceState.IDLE:
                self._transition(DeviceState.IDLE)

    def _run_voice(self, utterance, turn_id: int, cancel: TurnCancel) -> TurnResult:
        if self.voice is None:
            raise ConfigurationError("No Audio Model is attached to this session.")
        return self.voice.run(
            utterance,
            cancel,
            lambda new, turn_id=turn_id: self._on_model_state(new, turn_id),
        )

    def _prepare_turn(self, utterance) -> TurnResult | tuple[int, TurnCancel]:
        with self._lock:
            refused = self._refusal(utterance)
            if refused is not None:
                if refused.outcome is TurnOutcome.ERROR:
                    self._transition(DeviceState.IDLE)
                return refused
            self._turn_id += 1
            self._active_turn = TurnCancel()
            return self._turn_id, self._active_turn

    def _refusal(self, utterance) -> TurnResult | None:
        if self._active_turn is not None:
            return TurnResult(
                TurnOutcome.REFUSED, message="a turn is already running"
            )
        if self._muted or self.state is not DeviceState.CAPTURING:
            return TurnResult(
                TurnOutcome.REFUSED,
                message="the device is not holding a captured utterance",
            )
        if self._forward_frames:
            return TurnResult(TurnOutcome.REFUSED, message="capture is still open")
        try:
            problem = utterance_problem(utterance)
        except Exception:
            problem = "utterance audio is missing"
        if problem is None and utterance.audio_model is not self.audio_model:
            problem = "utterance does not match the Audio Model setting"
        if problem is not None:
            return TurnResult(
                TurnOutcome.ERROR,
                message=problem,
                cue=LocalCue.ERROR.value,
            )
        return None

    def _drop_model_locked(self) -> None:
        """Cancel the model turn. The caller holds the session lock.

        The finishing turn then sees a newer id and does not move the session.
        """
        turn = self._active_turn
        if turn is None:
            return
        turn.cancel()
        self._active_turn = None
        self._turn_id += 1

    def _on_model_state(self, new: DeviceState, turn_id: int) -> None:
        with self._lock:
            if self._turn_id != turn_id or self.state is new:
                return
            self._transition(new)

    def _stop_voice_speaker(self) -> None:
        if self.voice is None:
            return
        try:
            self.voice.speaker.stop()
        except Exception:
            return

    def _cut_playback_for_wake(self) -> bool:
        with self._lock:
            if self._muted:
                return False
            if self.state is not DeviceState.SPEAKING:
                return self.state is DeviceState.IDLE
            self._reply_token += 1
            self._drop_model_locked()
            playback = self.playback
        result = playback.stop(PlaybackStopReason.BARGE_IN)
        self._stop_voice_speaker()
        with self._lock:
            self.last_reply_result = result
            if self.state is DeviceState.SPEAKING:
                self._transition(DeviceState.IDLE)
            return not self._muted and self.state is DeviceState.IDLE

    def _begin_capture(self, event: WakeEvent, pre_roll: AudioSnapshot) -> bool:
        request = CaptureRequest(
            wake=event,
            pre_roll=pre_roll,
            audio_model=self.audio_model,
        )
        self._transition(DeviceState.WOKEN)
        try:
            self.capture.start(request)
        except Exception:
            self._transition(DeviceState.IDLE)
            raise
        self._forward_frames = True
        self._transition(DeviceState.CAPTURING)
        return True

    def _on_reply_finished(self, token: int, result: PlaybackResult) -> None:
        with self._lock:
            if token != self._reply_token:
                return
            self.last_reply_result = result
            # ``_finish_turn`` clears the model token and leaves ``speaking``
            # in place while ``flush`` is still running. This callback is the
            # idle step for that tail, and only after the token is clear. A
            # callback that arrives first must not idle the turn; ``_finish_turn``
            # does that when playback is already quiet.
            if self._active_turn is not None:
                return
            if (
                result.status is PlaybackStatus.COMPLETED
                and self.state is DeviceState.SPEAKING
            ):
                self._transition(DeviceState.IDLE)

    def _transition(self, new: DeviceState) -> None:
        assert_transition(self.state, new)
        self.state = new
        self.trace.append(new)
