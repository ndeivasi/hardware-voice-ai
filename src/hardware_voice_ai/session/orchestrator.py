"""Session owner for an accepted wake.

The wake path moves ``idle → woken → capturing`` and hands the pre-roll to
capture. It does not run an audio model and it does not open a cloud session.
"""

import threading

from hardware_voice_ai.audio.frame import AudioSnapshot
from hardware_voice_ai.capture.port import CapturePort, CaptureRequest, CaptureStopReason
from hardware_voice_ai.session.audio_model import AudioModel
from hardware_voice_ai.session.state import DeviceState, assert_transition
from hardware_voice_ai.wake.events import WakeEvent


class SessionOrchestrator:
    """On-device session state for the wake handoff.

    Audio Model is stored for the turn and copied onto the capture request.
    ``local`` and ``remote`` take the same wake path in this package; later
    issues are what actually run one audio model on the captured utterance.
    """

    def __init__(
        self,
        capture: CapturePort,
        audio_model: AudioModel = AudioModel.REMOTE,
    ) -> None:
        self.capture = capture
        self.audio_model = audio_model
        self.state = DeviceState.IDLE
        self._muted = False
        self._forward_frames = False
        self._lock = threading.RLock()

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

        Hardware mute and any state other than idle refuse the wake. A wake
        during capture or playback is ignored here; barge-in is later work.
        """
        with self._lock:
            if self._muted or self.state is not DeviceState.IDLE:
                return False
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

    def close_capture(self) -> None:
        """Stop the capture tap because the utterance ended.

        Issue 2 calls this when it decides the user has stopped speaking.
        The session stays in ``capturing`` so later work can pass that
        utterance to the one audio model. This wake path never calls it
        and does not infer end-of-utterance from silence.
        """
        with self._lock:
            if not self._forward_frames:
                return
            self._forward_frames = False
            self.capture.stop(CaptureStopReason.END_OF_UTTERANCE)

    def cancel(self) -> None:
        """Return to idle. Capture is told to stop; no audio is retained here."""
        with self._lock:
            if self.state is DeviceState.IDLE:
                return
            self._forward_frames = False
            self.capture.stop(CaptureStopReason.CANCEL)
            self._transition(DeviceState.IDLE)

    def apply_mute(self, muted: bool) -> None:
        """Hardware mute is authoritative and returns the device to idle."""
        with self._lock:
            self._muted = muted
            if not muted:
                return
            self._forward_frames = False
            if self.state is DeviceState.IDLE:
                return
            self.capture.stop(CaptureStopReason.MUTE)
            self._transition(DeviceState.IDLE)

    def _transition(self, new: DeviceState) -> None:
        assert_transition(self.state, new)
        self.state = new
