"""Wire the microphone, pre-roll, wake engine, and session orchestrator.

This is the always-on loop. It is the only reader of the microphone. Audio
stays in the pre-roll and the wake model. After an accepted wake, the
orchestrator receives the decision and a snapshot; later frames are tapped
to capture. The loop does not open a socket.
"""

import threading

from hardware_voice_ai.audio.ring_buffer import PreRollBuffer
from hardware_voice_ai.audio.source import AudioSource
from hardware_voice_ai.capture.port import CapturePort, PendingCapturePort
from hardware_voice_ai.config import DeviceConfig
from hardware_voice_ai.errors import ConfigurationError
from hardware_voice_ai.log.detections import DetectionLog
from hardware_voice_ai.mute.control import InMemoryMuteControl, MuteControl
from hardware_voice_ai.session.orchestrator import SessionOrchestrator
from hardware_voice_ai.wake.engine import WakeEngine
from hardware_voice_ai.wake.factory import build_wake_model
from hardware_voice_ai.wake.model import WakeModel


class WakeRuntime:
    """Always-on on-device wake loop."""

    def __init__(
        self,
        config: DeviceConfig,
        source: AudioSource,
        *,
        model: WakeModel | None = None,
        capture: CapturePort | None = None,
        mute: MuteControl | None = None,
        detection_log: DetectionLog | None = None,
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
        self.capture = capture if capture is not None else PendingCapturePort()
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
        self.orchestrator = SessionOrchestrator(self.capture, audio_model=config.audio_model)
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
        self.source.close()

    def _handle(self, frame) -> None:
        with self._lock:
            if self.mute.is_muted() or self.orchestrator.is_muted:
                return
            self.buffer.write(frame)
            if self.orchestrator.capture_is_active:
                self.capture.on_frame(frame)
            event = self.engine.process(frame)
            if event is None:
                return
            try:
                accepted = self.orchestrator.on_wake_accepted(
                    event, self.buffer.snapshot()
                )
            except Exception:
                # CapturePort.start failed. The orchestrator is back in idle.
                # Keep reading the microphone and record the wake as not accepted.
                accepted = False
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
