"""Wire the microphone, pre-roll, wake engine, capture, and Audio Model.

This is the always-on loop. It is the only reader of the microphone. Audio
stays in the pre-roll and the wake model until an accepted wake. Later frames
are tapped to capture. A local cue is replaced with silence before capture
and wake scoring, and a further wake is ignored while that cue is playing.
If ``on_frame`` raises, the turn is cancelled and this loop keeps reading.
After a successful end-of-utterance, the session moves to ``speaking`` before
the next frame is scored, and the Audio Model runs on its own thread. A wake
during speaking cancels that turn, stops playback, and starts a new capture.
"""

import threading

import numpy as np

from hardware_voice_ai.audio.frame import AudioFrame
from hardware_voice_ai.audio.ring_buffer import PreRollBuffer
from hardware_voice_ai.audio.source import AudioSource
from hardware_voice_ai.audio_model.cues import LocalCue
from hardware_voice_ai.audio_model.events import TurnCancel, TurnOutcome
from hardware_voice_ai.audio_model.factory import build_voice_loop
from hardware_voice_ai.audio_model.loop import TurnResult, VoiceLoop
from hardware_voice_ai.capture.handoff import CapturedUtterance, QueueUtteranceSink
from hardware_voice_ai.capture.port import CapturePort
from hardware_voice_ai.capture.utterance import UtteranceCapture
from hardware_voice_ai.config import DeviceConfig
from hardware_voice_ai.errors import ConfigurationError
from hardware_voice_ai.log.detections import DetectionLog
from hardware_voice_ai.mute.control import InMemoryMuteControl, MuteControl
from hardware_voice_ai.playback.cues import Cue
from hardware_voice_ai.playback.player import SpeakerPlayback
from hardware_voice_ai.playback.source import StubReplyAudio
from hardware_voice_ai.playback.speaker import NullSpeaker
from hardware_voice_ai.session.orchestrator import SessionOrchestrator
from hardware_voice_ai.session.state import DeviceState
from hardware_voice_ai.speaker.port import MemorySpeaker, SpeakerPort
from hardware_voice_ai.wake.engine import WakeEngine
from hardware_voice_ai.wake.factory import build_wake_model
from hardware_voice_ai.wake.model import WakeModel


class WakeRuntime:
    """Always-on on-device wake loop.

    The default capture is ``UtteranceCapture``. This constructor binds it to
    the session orchestrator, so callers do not call ``bind_session``. A
    custom ``CapturePort`` can still be passed for tests. ``utterances`` is
    the queue for the default capture; it stays ``None`` when the caller
    supplies a different port. Only that default queue is run through the
    Audio Model. ``close_capture`` itself only enqueues.
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
        voice: VoiceLoop | None = None,
        speaker: SpeakerPort | None = None,
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
        if voice is not None:
            self.voice = voice
            self.speaker = voice.speaker
        else:
            self.speaker = speaker if speaker is not None else MemorySpeaker()
            bridge = _PlaybackBridge(self.speaker)
            self.voice = build_voice_loop(config, speaker=bridge)
            bridge_voice = self.voice
        self.orchestrator = SessionOrchestrator(
            self.capture,
            audio_model=config.audio_model,
            playback=playback,
            voice=self.voice,
        )
        if voice is None:
            bridge_voice.speaker.bind(self.orchestrator)
        if isinstance(self.capture, UtteranceCapture):
            self.capture.bind_session(self.orchestrator)
        self._lock = threading.RLock()
        self._workers: list[threading.Thread] = []
        self.results: list[TurnResult] = []
        self.mute.add_listener(self._on_mute_changed)
        if self.mute.is_muted():
            self._on_mute_changed(True)

    def poll(self) -> bool:
        """Read and handle one frame. ``False`` means the source has ended."""
        frame = self.source.read()
        if frame is None:
            return False
        armed = self._handle(frame)
        if isinstance(armed, tuple):
            utterance, turn_id, cancel = armed
            if self.orchestrator.owns_turn(turn_id):
                self._start_model(utterance, turn_id, cancel)
        elif isinstance(armed, TurnResult):
            if armed.cue is not None:
                try:
                    self.voice.play_cue(LocalCue(armed.cue))
                except Exception:
                    pass
            with self._lock:
                self.results.append(armed)
        return True

    def run(self) -> None:
        """Process frames until the source ends or the caller interrupts.

        A running Audio Model does not keep this method from returning.
        ``close`` cancels that turn and waits for it.
        """
        while self.poll():
            pass

    def wait_for_turns(self, timeout: float = 2) -> None:
        """Wait until Audio Model turns started so far have finished."""
        for worker in list(self._workers):
            if worker is not threading.current_thread():
                worker.join(timeout=timeout)

    def close(self) -> None:
        try:
            if self.orchestrator.model_is_running:
                self.orchestrator.cancel()
            self.wait_for_turns()
        finally:
            try:
                self.playback.close()
            finally:
                self.source.close()

    def _start_model(
        self,
        utterance: CapturedUtterance,
        turn_id: int,
        cancel: TurnCancel,
    ) -> None:
        """Run one turn off the microphone thread.

        The session is already in ``speaking``. This thread does not move it
        there, so the microphone loop can barge in before the model runs.
        """

        def work() -> None:
            try:
                result = self.orchestrator.run_armed_turn(utterance, turn_id, cancel)
            except Exception:
                result = TurnResult(TurnOutcome.ERROR, message="audio model failed")
            with self._lock:
                self.results.append(result)

        worker = threading.Thread(target=work, name="audio-model", daemon=True)
        with self._lock:
            self._workers.append(worker)
        worker.start()

    def _handle(self, frame) -> TurnResult | tuple[CapturedUtterance, int, TurnCancel] | None:
        with self._lock:
            if self.mute.is_muted() or self.orchestrator.is_muted:
                return None
            # A local cue is tones. Do not score it and do not put it in the
            # utterance. The mic still hears the room; this tap does not keep
            # those frames. Echo cancellation during a spoken reply is later.
            # Speech that opens the follow-up window is the real frame: the
            # listening cue must not hide it, and it is not forwarded twice.
            follow_up = self.orchestrator.follow_up_is_active
            speech = follow_up and self.orchestrator.follow_up_hears_speech(frame)
            heard = frame if speech or not self.playback.is_cue_playing else _without_cue(frame)
            self.buffer.write(heard)
            opened_follow_up = False
            if follow_up:
                if speech:
                    try:
                        opened_follow_up = self.orchestrator.accept_follow_up_speech(
                            self.buffer.snapshot()
                        )
                    except Exception:
                        opened_follow_up = (
                            self.orchestrator.state is DeviceState.CAPTURING
                        )
                    if (
                        not opened_follow_up
                        and self.orchestrator.follow_up_is_active
                    ):
                        self.orchestrator.advance_follow_up(frame.format.frame_samples)
                else:
                    self.orchestrator.advance_follow_up(frame.format.frame_samples)
            if self.orchestrator.capture_is_active and not opened_follow_up:
                try:
                    self.capture.on_frame(heard)
                except Exception:
                    # on_frame failed (a frame index or format mismatch).
                    # Cancel drops the utterance, records the failed capture
                    # on the capture log, and returns the session to idle.
                    # Keep reading the microphone.
                    self.orchestrator.cancel()
            armed = self._arm_queued_utterance()
            event = self.engine.process(heard)
            if event is not None:
                if self.playback.is_cue_playing:
                    self.log.record(
                        event,
                        accepted=False,
                        audio_model=self.orchestrator.audio_model,
                    )
                else:
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
            if isinstance(armed, tuple) and not self.orchestrator.owns_turn(armed[1]):
                return None
            return armed

    def _arm_queued_utterance(
        self,
    ) -> TurnResult | tuple[CapturedUtterance, int, TurnCancel] | None:
        """Enter ``speaking`` before this frame is scored for a new wake.

        ``close_capture`` only enqueues. A custom capture port keeps its own
        sink; this returns nothing in that case.
        """
        if self.utterances is None:
            return None
        if (
            self.orchestrator.state is not DeviceState.CAPTURING
            or self.orchestrator.capture_is_active
        ):
            self.utterances.drain()
            return None
        utterance = self.utterances.take()
        if utterance is None:
            return None
        armed = self.orchestrator.arm_for_model(utterance)
        if isinstance(armed, TurnResult):
            return armed
        turn_id, cancel = armed
        return utterance, turn_id, cancel

    def _on_mute_changed(self, muted: bool) -> None:
        with self._lock:
            self.orchestrator.apply_mute(muted)
            if muted:
                self.buffer.clear()
                self.engine.reset()


class _PlaybackBridge:
    """Record reply audio and play it through the session's speaker.

    ``MemorySpeaker`` stays the stand-in tests inspect. The session plays
    the same chunks with ``play_reply`` and local cues with ``play_cue``.
    ``stop`` does not call back into the session; the orchestrator cuts
    playback itself so a barge-in reason is not overwritten.
    """

    def __init__(self, record: SpeakerPort) -> None:
        self._record = record
        self._session: SessionOrchestrator | None = None
        self._stub: StubReplyAudio | None = None

    def bind(self, session: SessionOrchestrator) -> None:
        self._session = session

    def start_reply(self, sample_rate_hz: int, channels: int) -> None:
        self._record.start_reply(sample_rate_hz, channels)
        stub = StubReplyAudio(sample_rate_hz, channels)
        self._stub = stub
        if self._session is not None:
            self._session.play_reply(stub)

    def write_reply(self, samples: np.ndarray) -> None:
        self._record.write_reply(samples)
        stub = self._stub
        if stub is None:
            return
        try:
            stub.push(samples)
        except RuntimeError:
            return

    def end_reply(self) -> None:
        self._record.end_reply()
        stub = self._stub
        self._stub = None
        if stub is not None:
            stub.finish()

    def stop(self) -> None:
        self._record.stop()
        stub = self._stub
        self._stub = None
        if stub is not None:
            stub.interrupt()
            stub.finish()

    def play_cue(self, cue: str, samples: np.ndarray, sample_rate_hz: int) -> None:
        self._record.play_cue(cue, samples, sample_rate_hz)
        if self._session is None:
            return
        self._session.play_cue(Cue(cue))


def _without_cue(frame: AudioFrame) -> AudioFrame:
    """Same index and format as ``frame``, with silence in place of the cue."""
    return AudioFrame(
        samples=np.zeros_like(frame.samples),
        format=frame.format,
        sample_index=frame.sample_index,
    )
