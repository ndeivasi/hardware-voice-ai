"""Session handoff, mute, and microphone capture."""

import numpy as np
import pytest

from hardware_voice_ai.audio.frame import AudioFormat, AudioSnapshot
from hardware_voice_ai.audio.source import PcmAudioSource
from hardware_voice_ai.capture.port import CaptureStopReason, PendingCapturePort
from hardware_voice_ai.capture.utterance import UtteranceCapture
from hardware_voice_ai.config import DeviceConfig
from hardware_voice_ai.mute.control import InMemoryMuteControl
from hardware_voice_ai.runtime import WakeRuntime
from hardware_voice_ai.session.orchestrator import SessionOrchestrator
from hardware_voice_ai.session.state import (
    ALLOWED_TRANSITIONS,
    DeviceState,
    IllegalTransitionError,
    assert_transition,
)
from hardware_voice_ai.wake.events import WakeEvent
from hardware_voice_ai.wake.model import WakeScore
from hardware_voice_ai.wake.reference import default_template, render_pattern
from tests.support import MemoryLog, RecordingCapture, silence


def _event() -> WakeEvent:
    return WakeEvent(
        wake_word="hey computer",
        model_id="reference-keyword-v1",
        confidence=0.99,
        decision_latency_ms=240,
        winning_channel=0,
        channels=1,
        sample_rate_hz=16000,
        frame_samples=320,
        decision_frame_index=10,
        word_end_frame_index=4,
        word_end_sample_index=1600,
        decision_sample_index=4800,
    )


def _snapshot() -> AudioSnapshot:
    samples = np.zeros((320, 1), dtype=np.float32)
    return AudioSnapshot(
        samples=samples,
        sample_rate_hz=16000,
        start_sample_index=0,
        end_sample_index=320,
    )


def test_architecture_transitions() -> None:
    assert set(ALLOWED_TRANSITIONS) == set(DeviceState)
    assert ALLOWED_TRANSITIONS[DeviceState.IDLE] == frozenset({DeviceState.WOKEN})
    assert ALLOWED_TRANSITIONS[DeviceState.WOKEN] == frozenset(
        {DeviceState.CAPTURING, DeviceState.IDLE}
    )
    assert ALLOWED_TRANSITIONS[DeviceState.CAPTURING] == frozenset(
        {DeviceState.TRANSCRIBING, DeviceState.IDLE}
    )
    assert ALLOWED_TRANSITIONS[DeviceState.TRANSCRIBING] == frozenset(
        {DeviceState.THINKING, DeviceState.IDLE}
    )
    assert ALLOWED_TRANSITIONS[DeviceState.THINKING] == frozenset(
        {DeviceState.SPEAKING, DeviceState.IDLE}
    )
    assert ALLOWED_TRANSITIONS[DeviceState.SPEAKING] == frozenset({DeviceState.IDLE})
    with pytest.raises(IllegalTransitionError):
        assert_transition(DeviceState.IDLE, DeviceState.SPEAKING)
    with pytest.raises(IllegalTransitionError):
        assert_transition(DeviceState.IDLE, DeviceState.CAPTURING)


def test_wake_moves_idle_to_woken_to_capturing() -> None:
    capture = RecordingCapture()
    session = SessionOrchestrator(capture)
    assert session.on_wake_accepted(_event(), _snapshot()) is True
    assert session.state is DeviceState.CAPTURING
    assert capture.requests[0].audio_model.value == "remote"
    assert capture.requests[0].pre_roll.samples.shape == (320, 1)
    assert session.on_wake_accepted(_event(), _snapshot()) is False
    assert len(capture.requests) == 1


def test_failed_handoff_returns_to_idle() -> None:
    class Boom:
        def start(self, request) -> None:
            raise RuntimeError("capture unavailable")

        def on_frame(self, frame) -> None:
            del frame

        def stop(self, reason) -> None:
            del reason

    session = SessionOrchestrator(Boom())
    with pytest.raises(RuntimeError, match="capture unavailable"):
        session.on_wake_accepted(_event(), _snapshot())
    assert session.state is DeviceState.IDLE


def test_failed_capture_start_keeps_listening() -> None:
    """A handoff error returns to idle, records the miss, and the loop continues."""

    class FlakyCapture:
        def __init__(self) -> None:
            self.calls = 0
            self.requests = []

        def start(self, request) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("capture unavailable")
            self.requests.append(request)

        def on_frame(self, frame) -> None:
            del frame

        def stop(self, reason) -> None:
            del reason

    word = render_pattern(default_template())
    gap = silence(int(1.2 * 16000))
    samples = np.concatenate((silence(1600), word, gap, word, silence(8000)))
    config = DeviceConfig()
    capture = FlakyCapture()
    log = MemoryLog()
    runtime = WakeRuntime(
        config,
        PcmAudioSource(samples, config.audio_format),
        capture=capture,
        detection_log=log,
    )
    runtime.run()
    runtime.close()
    assert [accepted for _event, accepted, _model in log.records] == [False, True]
    assert runtime.orchestrator.state is DeviceState.CAPTURING
    assert runtime.orchestrator.is_muted is False
    assert capture.calls == 2
    assert len(capture.requests) == 1


def test_cancel_and_mute_stop_capture_and_return_to_idle() -> None:
    capture = RecordingCapture()
    session = SessionOrchestrator(capture)
    session.on_wake_accepted(_event(), _snapshot())
    session.cancel()
    assert session.state is DeviceState.IDLE
    assert capture.stop_reasons == [CaptureStopReason.CANCEL]
    session.cancel()
    assert capture.stop_reasons == [CaptureStopReason.CANCEL]

    session.on_wake_accepted(_event(), _snapshot())
    session.apply_mute(True)
    assert session.state is DeviceState.IDLE
    assert session.is_muted is True
    assert capture.stop_reasons[-1] is CaptureStopReason.MUTE
    assert session.on_wake_accepted(_event(), _snapshot()) is False
    session.apply_mute(True)
    assert capture.stop_reasons.count(CaptureStopReason.MUTE) == 1
    session.apply_mute(False)
    assert session.is_muted is False
    assert session.state is DeviceState.IDLE
    assert session.on_wake_accepted(_event(), _snapshot()) is True


def test_pending_port_does_not_retain_frames() -> None:
    port = PendingCapturePort()
    fmt = AudioFormat(16000, 1, 320)
    from hardware_voice_ai.audio.frame import AudioFrame
    from hardware_voice_ai.capture.port import CaptureRequest
    from hardware_voice_ai.session.audio_model import AudioModel

    port.start(CaptureRequest(wake=_event(), pre_roll=_snapshot(), audio_model=AudioModel.REMOTE))
    port.on_frame(
        AudioFrame(samples=np.ones((320, 1), dtype=np.float32), format=fmt, sample_index=0)
    )
    assert port.active is True
    assert not hasattr(port, "frames")
    port.stop(CaptureStopReason.CANCEL)
    assert port.active is False


class _CountingModel:
    def __init__(self) -> None:
        self.calls = 0
        self.model_id = "counting"
        self.wake_word = "hey computer"

    def score(self, frame) -> WakeScore:
        del frame
        self.calls += 1
        return WakeScore(confidence=0.0, channel=0)

    def reset(self) -> None:
        return None


def test_mute_drops_audio_and_does_not_score_it() -> None:
    config = DeviceConfig()
    fmt = config.audio_format
    loud = np.full(fmt.frame_samples * 10, 0.8, dtype=np.float32)
    quiet = np.zeros(fmt.frame_samples * 5, dtype=np.float32)
    source = PcmAudioSource(np.concatenate((loud, loud, quiet)), fmt)
    mute = InMemoryMuteControl()
    model = _CountingModel()
    runtime = WakeRuntime(config, source, mute=mute, model=model)
    for _ in range(10):
        assert runtime.poll()
    assert runtime.buffer.snapshot().samples.shape[0] == fmt.frame_samples * 10
    mute.set_muted(True)
    assert runtime.orchestrator.state is DeviceState.IDLE
    assert runtime.buffer.snapshot().samples.shape[0] == 0
    for _ in range(10):
        assert runtime.poll()
    assert model.calls == 10
    assert runtime.buffer.snapshot().samples.shape[0] == 0
    mute.set_muted(False)
    for _ in range(5):
        assert runtime.poll()
    snap = runtime.buffer.snapshot()
    assert snap.samples.shape[0] == fmt.frame_samples * 5
    assert float(np.max(np.abs(snap.samples))) == 0.0
    assert model.calls == 15
    runtime.close()


def test_wake_while_muted_is_ignored_and_later_wake_still_works() -> None:
    template = default_template()
    word = render_pattern(template)
    padding = silence(8000)
    samples = np.concatenate((silence(3200), word, padding, word, padding))
    config = DeviceConfig()
    source = PcmAudioSource(samples, config.audio_format)
    mute = InMemoryMuteControl()
    capture = RecordingCapture()
    runtime = WakeRuntime(config, source, mute=mute, capture=capture)
    mute.set_muted(True)
    # Drop the first word. 3200 + len(word) + 8000 samples, in 320-sample frames.
    muted_frames = (3200 + len(word) + 8000) // config.frame_samples
    for _ in range(muted_frames):
        assert runtime.poll()
    assert capture.requests == []
    assert runtime.orchestrator.state is DeviceState.IDLE
    mute.set_muted(False)
    while runtime.poll():
        pass
    assert len(capture.requests) == 1
    assert runtime.orchestrator.state is DeviceState.CAPTURING
    assert capture.requests[0].pre_roll.start_sample_index >= 3200 + len(word)
    runtime.close()


def test_second_wake_while_capturing_is_not_accepted() -> None:
    template = default_template()
    word = render_pattern(template)
    # Gap covers the 240 ms confirm window plus the 800 ms cooldown.
    gap = silence(int(1.2 * 16000))
    samples = np.concatenate(
        (silence(1600), word, gap, word, gap, word, silence(8000))
    )
    config = DeviceConfig()
    source = PcmAudioSource(samples, config.audio_format)
    capture = RecordingCapture()
    from tests.support import MemoryLog

    log = MemoryLog()
    runtime = WakeRuntime(config, source, capture=capture, detection_log=log)
    accepted = []
    while runtime.poll():
        if len(log.records) > len(accepted):
            accepted.append(log.records[-1][1])
            if len(accepted) == 2:
                assert runtime.orchestrator.state is DeviceState.CAPTURING
                runtime.orchestrator.cancel()
                assert runtime.orchestrator.state is DeviceState.IDLE
    assert accepted == [True, False, True]
    assert [reason for reason in capture.stop_reasons] == [CaptureStopReason.CANCEL]
    assert len(capture.requests) == 2
    assert CaptureStopReason.END_OF_UTTERANCE not in capture.stop_reasons
    runtime.close()


def test_close_capture_stops_the_tap_without_starting_a_model() -> None:
    samples = np.concatenate(
        (silence(1600), render_pattern(default_template()), silence(16000))
    )
    config = DeviceConfig()
    capture = RecordingCapture()
    runtime = WakeRuntime(
        config,
        PcmAudioSource(samples, config.audio_format),
        capture=capture,
    )
    while runtime.orchestrator.state is not DeviceState.CAPTURING:
        assert runtime.poll()
    delivered = len(capture.frames)
    runtime.orchestrator.close_capture()
    assert runtime.orchestrator.state is DeviceState.CAPTURING
    assert runtime.orchestrator.capture_is_active is False
    assert capture.stop_reasons == [CaptureStopReason.END_OF_UTTERANCE]
    while runtime.poll():
        pass
    assert len(capture.frames) == delivered
    assert runtime.orchestrator.state is DeviceState.CAPTURING
    runtime.orchestrator.close_capture()
    assert capture.stop_reasons == [CaptureStopReason.END_OF_UTTERANCE]
    runtime.close()


def test_hardware_mute_after_wake_returns_to_idle() -> None:
    samples = np.concatenate(
        (silence(1600), render_pattern(default_template()), silence(16000))
    )
    config = DeviceConfig()
    mute = InMemoryMuteControl()
    capture = RecordingCapture()
    runtime = WakeRuntime(
        config,
        PcmAudioSource(samples, config.audio_format),
        mute=mute,
        capture=capture,
    )
    while runtime.orchestrator.state is not DeviceState.CAPTURING:
        assert runtime.poll()
    mute.set_muted(True)
    assert runtime.orchestrator.state is DeviceState.IDLE
    assert runtime.orchestrator.capture_is_active is False
    assert capture.stop_reasons == [CaptureStopReason.MUTE]
    assert runtime.buffer.snapshot().samples.shape[0] == 0
    while runtime.poll():
        pass
    assert len(capture.requests) == 1
    assert runtime.orchestrator.state is DeviceState.IDLE
    runtime.close()


def test_source_format_must_match_config() -> None:
    from hardware_voice_ai.errors import ConfigurationError

    config = DeviceConfig()
    fmt = AudioFormat(sample_rate_hz=16000, channels=2, frame_samples=320)
    source = PcmAudioSource(np.zeros((320, 2), dtype=np.float32), fmt)
    with pytest.raises(ConfigurationError, match="format"):
        WakeRuntime(config, source)


def test_default_capture_port_is_utterance_capture() -> None:
    config = DeviceConfig()
    runtime = WakeRuntime(
        config,
        PcmAudioSource(silence(config.frame_samples), config.audio_format),
    )
    assert isinstance(runtime.capture, UtteranceCapture)
    assert runtime.utterances is not None
    runtime.run()
    assert runtime.orchestrator.state is DeviceState.IDLE
    assert runtime.utterances.take() is None
    runtime.close()
