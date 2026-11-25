"""Wire the microphone, pre-roll, wake engine, and session orchestrator.

This is the always-on loop. It is the only reader of the microphone. Audio
stays in the pre-roll and the wake model. After an accepted wake, the
orchestrator receives the decision and a snapshot; later frames are tapped
to capture. The loop does not open a socket.
"""

import threading

from hardware_voice_ai.audio.ring_buffer import PreRollBuffer
from hardware_voice_ai.audio.source import AudioSource
from hardware_voice_ai.capture.handoff import QueueUtteranceSink
from hardware_voice_ai.capture.port import CapturePort
from hardware_voice_ai.capture.utterance import UtteranceCapture
from hardware_voice_ai.config import DeviceConfig
from hardware_voice_ai.errors import ConfigurationError
from hardware_voice_ai.log.detections import DetectionLog
from hardware_voice_ai.mute.control import InMemoryMuteControl, MuteControl
from hardware_voice_ai.playback.player import SpeakerPlayback
from hardware_voice_ai.playback.speaker import NullSpeaker
from hardware_voice_ai.session.orchestrator import SessionOrchestrator
from hardware_voice_ai.session.state import DeviceState
from hardware_voice_ai.wake.engine import WakeEngine
from hardware_voice_ai.wake.factory import build_wake_model
from hardware_voice_ai.wake.model import WakeModel


class WakeRuntime:
    """Always-on on-device wake loop.

    The default capture is ``UtteranceCapture``. This constructor binds it to
    the session orchestrator, so callers do not call ``bind_session``. A
    custom ``CapturePort`` can still be passed for tests. ``utterances`` is
    the queue for the default capture; it stays ``None`` when the caller
    supplies a different port.
    """

    def __init__(
        self,
        config: DeviceConfig,
        source: AudioSource,
        *,
        model: WakeModel | None = None,
        capture: CapturePort | None = None,
        mute: MuteControl | None = None,
        detection_log: DetectionLog | None = None,
        playback: SpeakerPlayback | None = None,
    ) -> None:
        if (
            source.format.sample_rate_hz != config.sample_rate_hz
            or source.format.channels != config.channels
            or source.format.frame_samples != config.frame_samples
        ):
            raise ConfigurationError(
                "Audio source format does not match the device config."
            )
        self.config = config
        self.source = source
        if capture is None:
            self.utterances: QueueUtteranceSink | None = QueueUtteranceSink()
            capture = UtteranceCapture(self.utterances)
        else:
            self.utterances = None
        self.capture = capture
        self.mute = mute if mute is not None else InMemoryMuteControl()
        self.log = detection_log if detection_log is not None else DetectionLog(
            config.detection_log_path
        )
        self.buffer = PreRollBuffer(
            capacity_samples=config.pre_roll_samples,
            channels=config.channels,
            sample_rate_hz=config.sample_rate_hz,
        )
        self.engine = WakeEngine(model if model is not None else build_wake_model(config), config)
        if playback is None:
            playback = SpeakerPlayback(
                NullSpeaker(config.audio_format, queue_ms=config.playback_queue_ms),
                volume=config.volume,
            )
        self.playback = playback
        self.orchestrator = SessionOrchestrator(
            self.capture,
            audio_model=config.audio_model,
            playback=playback,
        )
        if isinstance(self.capture, UtteranceCapture):
            self.capture.bind_session(self.orchestrator)
        self._lock = threading.RLock()
        self.mute.add_listener(self._on_mute_changed)
        if self.mute.is_muted():
            self._on_mute_changed(True)

    def poll(self) -> bool:
        """Read and handle one frame. ``False`` means the source has ended."""
        frame = self.source.read()
        if frame is None:
            return False
        self._handle(frame)
        return True

    def run(self) -> None:
        """Process frames until the source ends or the caller interrupts."""
        while self.poll():
            pass

    def close(self) -> None:
        try:
            self.playback.close()
        finally:
            self.source.close()

    def _handle(self, frame) -> None:
        with self._lock:
            if self.mute.is_muted() or self.orchestrator.is_muted:
                return
            self.buffer.write(frame)
            if self.orchestrator.capture_is_active:
                try:
                    self.capture.on_frame(frame)
                except Exception:
                    # on_frame failed (a frame index or format mismatch).
                    # Cancel drops the utterance, records the failed capture
                    # on the capture log, and returns the session to idle.
                    # Keep reading the microphone.
                    self.orchestrator.cancel()
            event = self.engine.process(frame)
            if event is None:
                return
            try:
                accepted = self.orchestrator.on_wake_accepted(
                    event, self.buffer.snapshot()
                )
            except Exception:
                # CapturePort.start rolls the session back to idle. A cue
                # error can still leave the turn in capturing; that wake
                # was accepted. Keep reading the microphone either way.
                accepted = self.orchestrator.state is DeviceState.CAPTURING
            self.log.record(
                event,
                accepted=accepted,
                audio_model=self.orchestrator.audio_model,
            )

    def _on_mute_changed(self, muted: bool) -> None:
        with self._lock:
            self.orchestrator.apply_mute(muted)
            if muted:
                self.buffer.clear()
                self.engine.reset()
