"""Session owner for an accepted wake, one Audio Model turn, and playback.

The wake path moves ``idle → woken → capturing`` and hands the pre-roll to
capture. After capture closes, the Audio Model runs off the microphone
thread. ``play_reply`` plays spoken-reply audio on the device speaker.
``play_cue`` plays a local cue. Neither call opens a cloud session.

An accepted wake plays the wake cue and no second cue. Capture end plays
one ``thinking`` earcon. A completed spoken reply opens a 5-second
follow-up listen while the session is ``idle``. A new wake word is not
required in that window. Hardware mute, stop, and cancel do not play
``follow_up_end``.

A new accepted wake during speaking cancels the model turn, cuts playback,
and starts a new capture. Hardware mute cuts playback and returns the
device to idle. Releasing mute does not resume audio.
"""

import threading

from hardware_voice_ai.audio.frame import AudioFormat, AudioFrame, AudioSnapshot
from hardware_voice_ai.audio_model.cues import LocalCue
from hardware_voice_ai.audio_model.events import TurnCancel, TurnOutcome
from hardware_voice_ai.audio_model.loop import TurnResult, VoiceLoop
from hardware_voice_ai.audio_model.utterance import utterance_problem
from hardware_voice_ai.capture.endpoint import EndpointConfig, channel_rms
from hardware_voice_ai.capture.port import CapturePort, CaptureRequest, CaptureStopReason
from hardware_voice_ai.errors import ConfigurationError
from hardware_voice_ai.playback.constants import FOLLOW_UP_WINDOW_MS
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

# Same floor as capture's endpoint. Not a wake-word score, and not the
# noise-relative multiplier: a follow-up already as loud as the pre-roll
# must still count as speech.
_FOLLOW_UP_SPEECH_RMS = EndpointConfig().min_speech_rms


class SessionOrchestrator:
    """On-device session state for wake, capture, the Audio Model, and playback.

    Audio Model is stored for the turn and copied onto the capture request.
    ``local`` and ``remote`` use the same speaker path. Reply audio is
    whatever the caller streams in. When a voice loop is attached, an
    accepted utterance is run through that one model. Local cues, including
    ``thinking``, ``listening``, and ``follow_up_end``, stay in this process.
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
        self._thinking_due = False
        self._follow_up_open = False
        self._follow_up_remaining = 0
        self._pending_follow_up_token: int | None = None
        self._follow_up_consumed_token: int | None = None
        self._last_wake: WakeEvent | None = None

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

    @property
    def follow_up_is_active(self) -> bool:
        """True during the 5-second listen after a completed spoken reply.

        The session stays ``idle``. This is not a new ``DeviceState``.
        """
        with self._lock:
            return self._follow_up_open

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
            # A real wake ends the follow-up. The wake cue is the only cue.
            self._follow_up_open = False
            self._follow_up_remaining = 0
            self._thinking_due = False
            accepted = self._begin_capture(event, pre_roll)
        if accepted:
            self.play_cue(Cue.WAKE)
        return accepted

    def close_capture(self) -> None:
        """Stop the capture tap because the utterance ended.

        The session stays in ``capturing`` so the caller can run the Audio
        Model or pass reply audio to ``play_reply``. This does not infer
        end-of-utterance or start the model. It does play one ``thinking``
        earcon. That cue is not a session state.
        """
        with self._lock:
            if not self._forward_frames:
                return
            self._forward_frames = False
            self.capture.stop(CaptureStopReason.END_OF_UTTERANCE)
            self._thinking_due = not self._muted and self.state is DeviceState.CAPTURING
        self._play_thinking_if_due()

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
            # Reply audio is ready. Do not start thinking, and cut it if
            # it has already claimed the speaker.
            self._thinking_due = False
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
        """Play a local cue.

        Cues are rendered in-process. The Audio Model mode does not choose
        a cloud path for them. A cue during ``speaking`` stops the reply
        and returns the session to ``idle`` before the cue starts. The
        follow-up earcons use ``_play_idle_cue`` instead, so they do not
        take that path.
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
        """Stop the current sound. A spoken reply returns the session to idle.

        Stopping during the follow-up window ends it with no
        ``follow_up_end`` cue.
        """
        self._close_follow_up()
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
        self._close_follow_up()
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
            self._follow_up_open = False
            self._follow_up_remaining = 0
            self._thinking_due = False
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
            # A local cue is not the spoken reply. The reply's own flush is
            # the tail that must keep the session in speaking. ``play_reply``
            # does not set the cue flag.
            reply_still_playing = (
                self.state is DeviceState.SPEAKING
                and self.playback.is_playing
                and not self.playback.is_cue_playing
            )
            if reply_still_playing:
                return
            if self.state is not DeviceState.IDLE:
                self._transition(DeviceState.IDLE)
        self._open_follow_up_after_reply()

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
        self._last_wake = event
        self._forward_frames = True
        self._transition(DeviceState.CAPTURING)
        return True

    def _on_reply_finished(self, token: int, result: PlaybackResult) -> None:
        should_follow = False
        with self._lock:
            if token != self._reply_token:
                return
            self.last_reply_result = result
            if result.status is PlaybackStatus.COMPLETED:
                self._pending_follow_up_token = token
            # ``_finish_turn`` clears the model token and leaves ``speaking``
            # in place while ``flush`` is still running. This callback is the
            # idle step for that tail, and only after the token is clear. A
            # callback that arrives first must not idle the turn; ``_finish_turn``
            # does that when playback is already quiet.
            if self._active_turn is not None:
                return
            if result.status is not PlaybackStatus.COMPLETED:
                return
            if self.state is DeviceState.SPEAKING:
                self._transition(DeviceState.IDLE)
            should_follow = self.state is DeviceState.IDLE
        if should_follow:
            self._open_follow_up_after_reply()

    def follow_up_hears_speech(self, frame: AudioFrame) -> bool:
        """True when this microphone frame is speech during the follow-up window.

        The threshold is the capture endpoint's ``min_speech_rms`` on the
        loudest channel. A wake word is not required. The small slack covers
        a constant at that level stored as float32.
        """
        if not self.follow_up_is_active:
            return False
        return channel_rms(frame.samples) + 1e-5 >= _FOLLOW_UP_SPEECH_RMS

    def accept_follow_up_speech(self, pre_roll: AudioSnapshot) -> bool:
        """Open a capture for speech in the follow-up window.

        No wake cue and no ``follow_up_end``. The request reuses the previous
        wake's word and model id so ``CaptureRequest`` stays valid. The
        sample indexes point at this speech, not at a new wake decision.
        ``pre_roll`` must already include the speech frame.
        """
        with self._lock:
            if (
                not self._follow_up_open
                or self._muted
                or self.state is not DeviceState.IDLE
                or self._last_wake is None
            ):
                return False
            self._follow_up_open = False
            self._follow_up_remaining = 0
            event = self._follow_up_event(pre_roll)
            playback = self.playback
        if playback.is_playing:
            playback.stop(PlaybackStopReason.STOP)
        with self._lock:
            if self._muted or self.state is not DeviceState.IDLE:
                return False
            return self._begin_capture(event, pre_roll)

    def advance_follow_up(self, sample_count: int) -> None:
        """Count microphone samples toward the 5-second window.

        Expiry plays ``follow_up_end`` once and leaves the session ``idle``.
        A wake word is required again after that.
        """
        play_end = False
        with self._lock:
            if not self._follow_up_open or self._muted or sample_count <= 0:
                return
            self._follow_up_remaining -= sample_count
            if self._follow_up_remaining > 0:
                return
            self._follow_up_open = False
            self._follow_up_remaining = 0
            play_end = self.state is DeviceState.IDLE
        if play_end:
            self._play_idle_cue(Cue.FOLLOW_UP_END, replace=True)

    def _play_thinking_if_due(self) -> None:
        """Play one thinking earcon unless a reply already owns the turn."""
        with self._lock:
            if not self._thinking_due:
                return
            if self._muted or self.state is not DeviceState.CAPTURING:
                self._thinking_due = False
                return
            self._thinking_due = False
            # Hold the session lock so ``play_reply`` cannot move to
            # speaking until this cue has the speaker. ``play_reply`` then
            # cuts it. The cue is not a session state.
            self.playback.play_cue(Cue.THINKING)

    def _open_follow_up_after_reply(self) -> None:
        """Open the follow-up listen once after this reply completed."""
        with self._lock:
            token = self._pending_follow_up_token
            if token is None or token != self._reply_token:
                return
            if token == self._follow_up_consumed_token:
                return
            if self._muted or self.state is not DeviceState.IDLE or self._follow_up_open:
                return
            self._follow_up_consumed_token = token
            self._pending_follow_up_token = None
            self._follow_up_open = True
            rate = self.playback.speaker.format.sample_rate_hz
            self._follow_up_remaining = rate * FOLLOW_UP_WINDOW_MS // 1000
        self._play_idle_cue(Cue.LISTENING)

    def _play_idle_cue(self, cue: Cue, *, replace: bool = False) -> None:
        """Play ``cue`` only while idle.

        ``replace`` cuts whatever is already playing. The listening cue does
        not replace a wake that won the speaker. ``follow_up_end`` does
        replace the listening cue if that cue is still going.
        """
        with self._lock:
            if self._muted or self.state is not DeviceState.IDLE:
                return
            if cue is Cue.LISTENING and not self._follow_up_open:
                return
            if self.playback.is_playing and not replace:
                return
            self.playback.play_cue(cue)

    def _close_follow_up(self) -> None:
        """End the window with no ``follow_up_end`` cue."""
        with self._lock:
            self._follow_up_open = False
            self._follow_up_remaining = 0

    def _follow_up_event(self, pre_roll: AudioSnapshot) -> WakeEvent:
        previous = self._last_wake
        assert previous is not None
        end = pre_roll.end_sample_index
        frame = previous.frame_samples if previous.frame_samples > 0 else 1
        word_end = end - frame
        if word_end <= pre_roll.start_sample_index:
            word_end = min(end, pre_roll.start_sample_index + 1)
        channels = int(pre_roll.samples.shape[1]) if pre_roll.samples.ndim == 2 else 1
        return WakeEvent(
            wake_word=previous.wake_word,
            model_id=previous.model_id,
            confidence=previous.confidence,
            decision_latency_ms=0,
            winning_channel=0,
            channels=channels if channels > 0 else previous.channels,
            sample_rate_hz=pre_roll.sample_rate_hz,
            frame_samples=frame,
            decision_frame_index=end // frame,
            word_end_frame_index=word_end // frame,
            word_end_sample_index=word_end,
            decision_sample_index=end,
        )

    def _transition(self, new: DeviceState) -> None:
        assert_transition(self.state, new)
        self.state = new
        self.trace.append(new)
