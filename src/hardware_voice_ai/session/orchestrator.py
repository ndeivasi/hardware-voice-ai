"""Session owner for an accepted wake and for local speaker playback.

The wake path moves ``idle → woken → capturing`` and hands the pre-roll to
capture. After capture closes, ``play_reply`` moves ``capturing → speaking``
and plays spoken-reply audio on the device speaker. It does not run an
Audio Model and it does not open a cloud session.

Wake, offline, and error cues are rendered locally in both Audio Model
modes. A new accepted wake during speaking cuts playback and starts a new
turn. Hardware mute cuts playback and returns the device to idle.
"""

import threading

from hardware_voice_ai.audio.frame import AudioFormat, AudioSnapshot
from hardware_voice_ai.capture.port import CapturePort, CaptureRequest, CaptureStopReason
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
    """On-device session state for wake, capture, and speaker playback.

    Audio Model is stored for the turn and copied onto the capture request.
    ``local`` and ``remote`` use the same speaker path. Reply audio is
    whatever the caller streams in; this class does not produce it.
    """

    def __init__(
        self,
        capture: CapturePort,
        audio_model: AudioModel = AudioModel.REMOTE,
        playback: SpeakerPlayback | None = None,
    ) -> None:
        self.capture = capture
        self.audio_model = audio_model
        self.playback = (
            playback
            if playback is not None
            else SpeakerPlayback(NullSpeaker(AudioFormat(16_000, 1, 320)))
        )
        self.state = DeviceState.IDLE
        self._muted = False
        self._forward_frames = False
        self._lock = threading.RLock()
        self._reply_token = 0
        self.last_reply_result: PlaybackResult | None = None

    @property
    def is_muted(self) -> bool:
        return self._muted

    @property
    def capture_is_active(self) -> bool:
        """True while microphone frames should be tapped to capture."""
        with self._lock:
            return self._forward_frames

    def on_wake_accepted(self, event: WakeEvent, pre_roll: AudioSnapshot) -> bool:
        """Notify the session of a wake the engine already accepted.

        Hardware mute refuses the wake. A wake during capture is ignored.
        A wake during speaking stops playback, then accepts the new turn.
        An accepted wake plays the local wake cue.
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

        The session stays in ``capturing`` so the caller can pass that
        utterance's reply audio to ``play_reply``. This does not infer
        end-of-utterance and it does not start playback.
        """
        with self._lock:
            if not self._forward_frames:
                return
            self._forward_frames = False
            self.capture.stop(CaptureStopReason.END_OF_UTTERANCE)

    def play_reply(self, source: ReplyAudio) -> PlaybackResult:
        """Play spoken-reply audio on the local speaker.

        Call this after capture has closed, while the session is still
        ``capturing``. The session moves to ``speaking`` and back to
        ``idle`` when the reply completes. A later ``play_reply`` replaces
        the one in progress. Hardware mute refuses the call and plays nothing.
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
                if token == self._reply_token and self.state is DeviceState.SPEAKING:
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
            playback = self.playback
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
            playback = self.playback
        result = playback.stop(PlaybackStopReason.STOP)
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

    def cancel(self) -> None:
        """Return to idle. Capture and playback both stop."""
        with self._lock:
            if self.state is DeviceState.IDLE and not self.playback.is_playing:
                return
            self._forward_frames = False
            to_idle = self.state is not DeviceState.IDLE
            was_speaking = self.state is DeviceState.SPEAKING
            if to_idle:
                self.capture.stop(CaptureStopReason.CANCEL)
                self._reply_token += 1
            playback = self.playback
        result = playback.stop(PlaybackStopReason.STOP)
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
            playback = self.playback
        result = playback.stop(PlaybackStopReason.MUTE)
        with self._lock:
            if was_speaking:
                self.last_reply_result = result
            if to_idle and self.state is not DeviceState.IDLE:
                self._transition(DeviceState.IDLE)

    def _cut_playback_for_wake(self) -> bool:
        with self._lock:
            if self._muted:
                return False
            if self.state is not DeviceState.SPEAKING:
                return self.state is DeviceState.IDLE
            self._reply_token += 1
            playback = self.playback
        result = playback.stop(PlaybackStopReason.BARGE_IN)
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
            if (
                result.status is PlaybackStatus.COMPLETED
                and self.state is DeviceState.SPEAKING
            ):
                self._transition(DeviceState.IDLE)

    def _transition(self, new: DeviceState) -> None:
        assert_transition(self.state, new)
        self.state = new
