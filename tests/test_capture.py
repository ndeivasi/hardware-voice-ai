"""Utterance capture: pre-roll, endpointing, cancel, and hardware mute."""

from pathlib import Path

import numpy as np
import pytest

from hardware_voice_ai.audio.frame import AudioFormat, AudioFrame, AudioSnapshot
from hardware_voice_ai.capture.endpoint import EndpointConfig
from hardware_voice_ai.capture.handoff import QueueUtteranceSink
from hardware_voice_ai.capture.log import CaptureLog
from hardware_voice_ai.capture.port import (
    CaptureRequest,
    CaptureStopReason,
    PendingCapturePort,
)
from hardware_voice_ai.capture.utterance import CaptureError, UtteranceCapture
from hardware_voice_ai.session.audio_model import AudioModel
from hardware_voice_ai.wake.events import WakeEvent

RATE = 16_000
FRAME = 320
FORMAT = AudioFormat(RATE, 1, FRAME)


class _Session:
    """Stand-in for ``SessionOrchestrator.close_capture``."""

    def __init__(self, capture: UtteranceCapture) -> None:
        self.capture = capture
        self.forward = True
        self.close_calls = 0

    def close_capture(self) -> None:
        if not self.forward:
            return
        self.forward = False
        self.close_calls += 1
        self.capture.stop(CaptureStopReason.END_OF_UTTERANCE)


def _wake(
    *,
    start: int,
    end: int,
    word_end: int,
    channels: int = 1,
    frame_samples: int = FRAME,
    rate: int = RATE,
) -> WakeEvent:
    return WakeEvent(
        wake_word="hey computer",
        model_id="reference-keyword-v1",
        confidence=0.91,
        decision_latency_ms=240,
        winning_channel=0,
        channels=channels,
        sample_rate_hz=rate,
        frame_samples=frame_samples,
        decision_frame_index=end // frame_samples,
        word_end_frame_index=word_end // frame_samples,
        word_end_sample_index=word_end,
        decision_sample_index=end,
    )


def _snapshot(samples: np.ndarray, start: int = 0, rate: int = RATE) -> AudioSnapshot:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return AudioSnapshot(
        samples=array,
        sample_rate_hz=rate,
        start_sample_index=start,
        end_sample_index=start + array.shape[0],
    )


def _request(
    samples: np.ndarray,
    *,
    word_end: int,
    audio_model: AudioModel = AudioModel.REMOTE,
    start: int = 0,
    channels: int | None = None,
    frame_samples: int = FRAME,
    rate: int = RATE,
) -> CaptureRequest:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    channel_count = array.shape[1] if channels is None else channels
    end = start + array.shape[0]
    return CaptureRequest(
        wake=_wake(
            start=start,
            end=end,
            word_end=word_end,
            channels=channel_count,
            frame_samples=frame_samples,
            rate=rate,
        ),
        pre_roll=_snapshot(array, start=start, rate=rate),
        audio_model=audio_model,
    )


def _frame(value: float | np.ndarray, index: int, fmt: AudioFormat = FORMAT) -> AudioFrame:
    if isinstance(value, np.ndarray):
        samples = np.asarray(value, dtype=np.float32)
    else:
        samples = np.full((fmt.frame_samples, fmt.channels), value, dtype=np.float32)
    return AudioFrame(samples=samples, format=fmt, sample_index=index)


def _open(
    samples: np.ndarray,
    *,
    word_end: int,
    endpoint: EndpointConfig | None = None,
    audio_model: AudioModel = AudioModel.REMOTE,
    log: CaptureLog | None = None,
    fmt: AudioFormat = FORMAT,
) -> tuple[UtteranceCapture, QueueUtteranceSink, _Session, CaptureRequest]:
    sink = QueueUtteranceSink()
    capture = UtteranceCapture(sink, endpoint=endpoint, log=log)
    session = _Session(capture)
    capture.bind_session(session)
    request = _request(
        samples,
        word_end=word_end,
        audio_model=audio_model,
        frame_samples=fmt.frame_samples,
        rate=fmt.sample_rate_hz,
    )
    capture.start(request)
    return capture, sink, session, request


def _silence_frames(capture: UtteranceCapture, count: int, index: int, fmt: AudioFormat = FORMAT) -> int:
    for _ in range(count):
        capture.on_frame(_frame(0.0, index, fmt))
        index += fmt.frame_samples
    return index


def test_pending_port_still_discards_frames() -> None:
    port = PendingCapturePort()
    request = _request(np.full(FRAME * 4, 0.1, dtype=np.float32), word_end=FRAME)
    port.start(request)
    port.on_frame(_frame(0.2, request.pre_roll.end_sample_index))
    assert port.active is True
    assert not hasattr(port, "frames")
    port.stop(CaptureStopReason.CANCEL)
    assert port.active is False
    assert port.stop_reasons == [CaptureStopReason.CANCEL]


def test_preroll_including_audio_before_wake_word_end_is_handed_off() -> None:
    pre_roll = np.linspace(-0.5, 0.5, FRAME * 8, dtype=np.float32)
    word_end = FRAME * 5
    pre_roll[0] = -0.33
    pre_roll[word_end - 1] = 0.77
    capture, sink, session, request = _open(pre_roll, word_end=word_end)
    index = request.pre_roll.end_sample_index
    capture.on_frame(_frame(0.4, index))
    session.close_capture()

    utterance = sink.take()
    assert utterance is not None
    assert sink.take() is None
    assert utterance.audio_model is AudioModel.REMOTE
    assert utterance.endpoint == "requested"
    assert utterance.pre_roll_sample_count == pre_roll.shape[0]
    assert utterance.wake_word_end_offset == word_end
    assert utterance.samples.shape == (pre_roll.shape[0] + FRAME, 1)
    np.testing.assert_allclose(utterance.samples[: pre_roll.shape[0], 0], pre_roll)
    assert utterance.samples[0, 0] == pytest.approx(-0.33)
    assert utterance.samples[word_end - 1, 0] == pytest.approx(0.77)
    assert float(utterance.samples[pre_roll.shape[0], 0]) == pytest.approx(0.4)
    assert capture.holds_audio is False
    assert capture.active is False
    session.close_capture()
    assert session.close_calls == 1
    assert sink.take() is None


def test_local_audio_model_is_passed_through_without_running_a_model() -> None:
    capture, sink, session, _request_obj = _open(
        np.full(FRAME * 4, 0.05, dtype=np.float32),
        word_end=FRAME * 2,
        audio_model=AudioModel.LOCAL,
    )
    session.close_capture()
    utterance = sink.take()
    assert utterance is not None
    assert utterance.audio_model is AudioModel.LOCAL
    assert capture.holds_audio is False


def test_trailing_silence_ends_the_utterance_and_keeps_the_preroll() -> None:
    pre_roll = np.zeros(FRAME * 40, dtype=np.float32)
    pre_roll[-FRAME * 10 :] = 0.5
    word_end = FRAME * 32
    capture, sink, session, request = _open(pre_roll, word_end=word_end)
    index = request.pre_roll.end_sample_index
    capture.on_frame(_frame(0.5, index))
    index += FRAME
    assert session.close_calls == 0
    index = _silence_frames(capture, 34, index)
    assert session.close_calls == 0
    assert sink.take() is None
    capture.on_frame(_frame(0.0, index))

    assert session.close_calls == 1
    utterance = sink.take()
    assert utterance is not None
    assert utterance.endpoint == "silence"
    assert utterance.samples.shape[0] == pre_roll.shape[0] + FRAME * 36
    np.testing.assert_allclose(utterance.samples[: pre_roll.shape[0], 0], pre_roll)
    assert utterance.samples[word_end - 1, 0] == pytest.approx(0.5)
    assert capture.holds_audio is False


def test_a_short_pause_does_not_end_the_utterance() -> None:
    pre_roll = np.zeros(FRAME * 20, dtype=np.float32)
    pre_roll[-FRAME * 5 :] = 0.45
    capture, sink, session, request = _open(pre_roll, word_end=FRAME * 16)
    index = request.pre_roll.end_sample_index
    capture.on_frame(_frame(0.45, index))
    index += FRAME
    index = _silence_frames(capture, 10, index)
    capture.on_frame(_frame(0.45, index))
    assert session.close_calls == 0
    assert sink.drain() == []
    assert capture.active is True


def test_leading_silence_ends_when_the_command_never_starts() -> None:
    pre_roll = np.full(FRAME * 40, 0.001, dtype=np.float32)
    capture, sink, session, request = _open(pre_roll, word_end=FRAME * 20)
    index = request.pre_roll.end_sample_index
    index = _silence_frames(capture, 79, index)
    assert session.close_calls == 0
    capture.on_frame(_frame(0.0, index))
    utterance = sink.take()
    assert utterance is not None
    assert utterance.endpoint == "leading_silence"
    assert utterance.pre_roll_sample_count == pre_roll.shape[0]
    np.testing.assert_array_equal(utterance.samples[: pre_roll.shape[0], 0], pre_roll)


def test_speech_after_a_pause_is_kept_until_trailing_silence() -> None:
    config = EndpointConfig(end_silence_ms=40, max_leading_silence_ms=200, max_utterance_ms=2_000)
    pre_roll = np.zeros(FRAME * 10, dtype=np.float32)
    capture, sink, session, request = _open(pre_roll, word_end=FRAME * 4, endpoint=config)
    index = request.pre_roll.end_sample_index
    index = _silence_frames(capture, 3, index)
    assert session.close_calls == 0
    capture.on_frame(_frame(0.4, index))
    index += FRAME
    _silence_frames(capture, 2, index)
    utterance = sink.take()
    assert utterance is not None
    assert utterance.endpoint == "silence"
    assert float(np.max(utterance.samples[pre_roll.shape[0] :])) == pytest.approx(0.4)


def test_room_noise_does_not_count_as_speech_and_a_louder_frame_does() -> None:
    config = EndpointConfig(end_silence_ms=20, max_leading_silence_ms=200, max_utterance_ms=2_000)
    pre_roll = np.full(FRAME * 20, 0.01, dtype=np.float32)
    capture, sink, session, request = _open(pre_roll, word_end=FRAME * 8, endpoint=config)
    index = request.pre_roll.end_sample_index
    capture.on_frame(_frame(0.02, index))
    index += FRAME
    assert session.close_calls == 0
    capture.on_frame(_frame(0.2, index))
    index += FRAME
    capture.on_frame(_frame(0.0, index))
    utterance = sink.take()
    assert utterance is not None
    assert utterance.endpoint == "silence"
    assert float(np.max(utterance.samples)) == pytest.approx(0.2)


def test_loud_noise_floor_ignores_quieter_frames_until_max_time_or_real_speech() -> None:
    config = EndpointConfig(end_silence_ms=20, max_leading_silence_ms=80, max_utterance_ms=2_000)
    pre_roll = np.full(FRAME * 20, 0.2, dtype=np.float32)
    capture, sink, session, request = _open(pre_roll, word_end=FRAME * 10, endpoint=config)
    index = request.pre_roll.end_sample_index
    capture.on_frame(_frame(0.25, index))
    index += FRAME
    assert session.close_calls == 0
    capture.on_frame(_frame(0.9, index))
    index += FRAME
    capture.on_frame(_frame(0.0, index))
    utterance = sink.take()
    assert utterance is not None
    assert utterance.endpoint == "silence"


def test_max_duration_ends_a_continuous_utterance() -> None:
    config = EndpointConfig(end_silence_ms=700, max_leading_silence_ms=1_600, max_utterance_ms=40)
    pre_roll = np.full(FRAME * 8, 0.3, dtype=np.float32)
    capture, sink, session, request = _open(pre_roll, word_end=FRAME * 4, endpoint=config)
    index = request.pre_roll.end_sample_index
    capture.on_frame(_frame(0.3, index))
    assert session.close_calls == 0
    capture.on_frame(_frame(0.3, index + FRAME))
    utterance = sink.take()
    assert utterance is not None
    assert utterance.endpoint == "max_duration"
    assert utterance.samples.shape[0] == pre_roll.shape[0] + FRAME * 2


def test_loudest_channel_keeps_the_array_when_another_channel_is_quiet() -> None:
    fmt = AudioFormat(RATE, 2, FRAME)
    config = EndpointConfig(end_silence_ms=20, max_leading_silence_ms=200, max_utterance_ms=2_000)
    quiet = np.zeros((FRAME * 12, 2), dtype=np.float32)
    quiet[:, 1] = 0.01
    capture, sink, session, request = _open(quiet, word_end=FRAME * 6, endpoint=config, fmt=fmt)
    index = request.pre_roll.end_sample_index
    speech = np.zeros((FRAME, 2), dtype=np.float32)
    speech[:, 1] = 0.45
    capture.on_frame(_frame(speech, index, fmt))
    index += FRAME
    silence = np.zeros((FRAME, 2), dtype=np.float32)
    capture.on_frame(_frame(silence, index, fmt))
    utterance = sink.take()
    assert utterance is not None
    assert utterance.channels == 2
    assert utterance.endpoint == "silence"
    assert float(np.max(utterance.samples[:, 0])) == pytest.approx(0.0)
    assert float(np.max(utterance.samples[:, 1])) == pytest.approx(0.45)


def test_cancel_drops_audio_and_does_not_hand_it_off() -> None:
    capture, sink, session, request = _open(
        np.full(FRAME * 8, 0.2, dtype=np.float32),
        word_end=FRAME * 3,
    )
    index = request.pre_roll.end_sample_index
    capture.on_frame(_frame(0.55, index))
    assert capture.holds_audio is True
    capture.stop(CaptureStopReason.CANCEL)
    assert sink.drain() == []
    assert capture.holds_audio is False
    assert capture.active is False
    capture.on_frame(_frame(0.55, index + FRAME))
    assert sink.drain() == []
    assert session.close_calls == 0
    capture.stop(CaptureStopReason.CANCEL)
    assert sink.drain() == []


def test_hardware_mute_drops_audio_and_a_later_turn_can_start() -> None:
    pre_roll = np.full(FRAME * 8, 0.25, dtype=np.float32)
    capture, sink, session, request = _open(pre_roll, word_end=FRAME * 4)
    index = request.pre_roll.end_sample_index
    capture.on_frame(_frame(0.66, index))
    capture.stop(CaptureStopReason.MUTE)
    assert sink.drain() == []
    assert capture.holds_audio is False
    capture.on_frame(_frame(0.66, index + FRAME))
    assert sink.drain() == []

    capture.start(request)
    assert capture.holds_audio is True
    session.forward = True
    session.close_capture()
    utterance = sink.take()
    assert utterance is not None
    np.testing.assert_allclose(utterance.samples[:, 0], pre_roll)


def test_stop_before_start_is_ignored() -> None:
    sink = QueueUtteranceSink()
    capture = UtteranceCapture(sink)
    capture.stop(CaptureStopReason.MUTE)
    capture.stop(CaptureStopReason.CANCEL)
    assert sink.drain() == []
    assert capture.active is False


def test_frames_before_start_are_ignored() -> None:
    sink = QueueUtteranceSink()
    capture = UtteranceCapture(sink)
    capture.on_frame(_frame(0.4, 0))
    assert capture.holds_audio is False
    assert sink.drain() == []


def test_start_while_active_raises() -> None:
    capture, _sink, _session, request = _open(
        np.full(FRAME * 4, 0.1, dtype=np.float32),
        word_end=FRAME,
    )
    with pytest.raises(CaptureError):
        capture.start(request)


def test_first_frame_must_continue_after_the_decision_sample() -> None:
    capture, sink, _session, request = _open(
        np.full(FRAME * 4, 0.1, dtype=np.float32),
        word_end=FRAME,
    )
    with pytest.raises(CaptureError):
        capture.on_frame(_frame(0.2, request.pre_roll.end_sample_index + FRAME))
    assert sink.drain() == []
    assert capture.active is True
    capture.on_frame(_frame(0.2, request.pre_roll.end_sample_index))
    assert capture.active is True


def test_pre_roll_must_contain_the_wake_word_end() -> None:
    samples = np.full(FRAME * 4, 0.1, dtype=np.float32)
    sink = QueueUtteranceSink()
    capture = UtteranceCapture(sink)
    missing = _request(samples, word_end=0)
    with pytest.raises(CaptureError):
        capture.start(missing)
    too_late = _request(samples, word_end=samples.shape[0] + FRAME)
    with pytest.raises(CaptureError):
        capture.start(too_late)
    assert capture.holds_audio is False


def test_pre_roll_must_end_on_the_decision_sample() -> None:
    samples = np.full(FRAME * 4, 0.1, dtype=np.float32)
    request = _request(samples, word_end=FRAME)
    request = CaptureRequest(
        wake=WakeEvent(
            wake_word=request.wake.wake_word,
            model_id=request.wake.model_id,
            confidence=request.wake.confidence,
            decision_latency_ms=request.wake.decision_latency_ms,
            winning_channel=0,
            channels=1,
            sample_rate_hz=RATE,
            frame_samples=FRAME,
            decision_frame_index=0,
            word_end_frame_index=1,
            word_end_sample_index=FRAME,
            decision_sample_index=request.pre_roll.end_sample_index + FRAME,
        ),
        pre_roll=request.pre_roll,
        audio_model=AudioModel.REMOTE,
    )
    capture = UtteranceCapture(QueueUtteranceSink())
    with pytest.raises(CaptureError):
        capture.start(request)


def test_end_of_utterance_requires_a_bound_session() -> None:
    config = EndpointConfig(end_silence_ms=20, max_leading_silence_ms=40, max_utterance_ms=2_000)
    sink = QueueUtteranceSink()
    capture = UtteranceCapture(sink, endpoint=config)
    request = _request(np.zeros(FRAME * 8, dtype=np.float32), word_end=FRAME * 2)
    capture.start(request)
    index = request.pre_roll.end_sample_index
    with pytest.raises(CaptureError):
        _silence_frames(capture, 2, index)
    assert sink.drain() == []
    assert capture.holds_audio is True


def test_opt_in_log_is_local_metadata_and_mute_does_not_persist_audio(tmp_path: Path) -> None:
    log_path = tmp_path / "capture.log"
    sentinel = 0.424242
    pre_roll = np.full(FRAME * 4, sentinel, dtype=np.float32)
    capture, sink, _session, request = _open(
        pre_roll,
        word_end=FRAME * 2,
        log=CaptureLog(log_path),
    )
    capture.on_frame(_frame(sentinel, request.pre_roll.end_sample_index))
    capture.stop(CaptureStopReason.MUTE)
    text = log_path.read_text(encoding="utf-8")
    assert "event=start" in text
    assert "event=stop" in text
    assert "reason=mute" in text
    assert "handed_off=False" in text
    assert "0.424242" not in text
    assert list(tmp_path.iterdir()) == [log_path]
    assert sink.drain() == []
    with pytest.raises(ValueError):
        CaptureLog("https://example.com/capture.log")


def test_default_log_is_off_and_the_utterance_is_still_handed_off() -> None:
    log = CaptureLog()
    assert log.enabled is False
    assert log.path is None
    capture, sink, session, _request_obj = _open(
        np.full(FRAME * 4, 0.2, dtype=np.float32),
        word_end=FRAME,
        log=log,
    )
    session.close_capture()
    assert sink.take() is not None
    assert capture.holds_audio is False


def _wake_pcm(trailing_samples: int):
    from hardware_voice_ai.audio.source import PcmAudioSource
    from hardware_voice_ai.config import DeviceConfig
    from hardware_voice_ai.wake.reference import default_template, render_pattern
    from tests.support import silence

    config = DeviceConfig()
    samples = np.concatenate(
        (silence(1600), render_pattern(default_template()), silence(trailing_samples))
    )
    return config, PcmAudioSource(samples, config.audio_format)


def test_default_runtime_binds_capture_and_hands_off_the_preroll() -> None:
    """WakeRuntime uses UtteranceCapture and the real orchestrator."""
    from hardware_voice_ai.runtime import WakeRuntime
    from hardware_voice_ai.session.state import DeviceState

    config, source = _wake_pcm(32_000)
    runtime = WakeRuntime(config, source)
    assert isinstance(runtime.capture, UtteranceCapture)
    assert runtime.utterances is not None
    runtime.run()
    runtime.close()

    assert runtime.utterances is not None
    utterance = runtime.utterances.take()
    assert utterance is not None
    assert runtime.utterances.take() is None
    assert utterance.audio_model is AudioModel.REMOTE
    assert utterance.channels == 1
    assert utterance.endpoint in {"silence", "leading_silence"}
    assert utterance.pre_roll_sample_count > 0
    assert 0 < utterance.wake_word_end_offset <= utterance.pre_roll_sample_count
    assert utterance.samples.shape[0] > utterance.pre_roll_sample_count
    preroll = utterance.samples[: utterance.pre_roll_sample_count]
    assert float(np.max(np.abs(preroll))) > 0.0
    assert float(np.max(np.abs(preroll[: utterance.wake_word_end_offset]))) > 0.0
    assert runtime.orchestrator.state is DeviceState.CAPTURING
    assert runtime.orchestrator.capture_is_active is False
    assert runtime.capture.holds_audio is False


class _IndexGapSource:
    """Finite silence. One frame's ``sample_index`` skips ahead by one frame."""

    def __init__(self, frame_count: int, gap_at: int, fmt: AudioFormat) -> None:
        self.format = fmt
        self._frame_count = frame_count
        self._gap_at = gap_at
        self._next = 0
        self._sample_index = 0

    def read(self) -> AudioFrame | None:
        if self._next >= self._frame_count:
            return None
        if self._next == self._gap_at:
            self._sample_index += self.format.frame_samples
        frame = AudioFrame(
            samples=np.zeros(
                (self.format.frame_samples, self.format.channels),
                dtype=np.float32,
            ),
            format=self.format,
            sample_index=self._sample_index,
        )
        self._sample_index += self.format.frame_samples
        self._next += 1
        return frame

    def close(self) -> None:
        return None


def test_on_frame_mismatch_returns_to_idle_and_keeps_listening(tmp_path: Path) -> None:
    """A frame-index CaptureError cancels the turn and the loop keeps reading.

    Uses the real ``WakeRuntime``, ``UtteranceCapture``, and
    ``SessionOrchestrator``. The capture object still raises; the runtime
    is what returns the session to idle.
    """
    from dataclasses import replace

    from hardware_voice_ai.capture.endpoint import EndpointConfig
    from hardware_voice_ai.config import DeviceConfig
    from hardware_voice_ai.runtime import WakeRuntime
    from hardware_voice_ai.session.orchestrator import SessionOrchestrator
    from hardware_voice_ai.session.state import DeviceState
    from tests.support import MemoryLog, ScriptedModel

    config = replace(
        DeviceConfig(),
        accept_threshold=0.6,
        min_positive_ms=120,
        confirm_ms=240,
        cooldown_ms=0,
        max_word_ms=1000,
        end_ratio=0.85,
    )
    word = [0.9] * (config.min_positive_ms // config.frame_ms)
    confirm = [0.0] * (config.confirm_ms // config.frame_ms)
    first_wake = word + confirm
    # The decision lands on the last confirm frame. The next read is on_frame.
    gap_at = len(first_wake)
    second_wake = word + confirm
    trailing = [0.0] * (EndpointConfig().max_leading_silence_ms // config.frame_ms)
    scores = first_wake + [0.0] + second_wake + trailing
    log_path = tmp_path / "capture.log"
    sink = QueueUtteranceSink()
    capture = UtteranceCapture(sink, log=CaptureLog(log_path))
    detection = MemoryLog()
    runtime = WakeRuntime(
        config,
        _IndexGapSource(len(scores), gap_at, config.audio_format),
        model=ScriptedModel(scores=list(scores)),
        capture=capture,
        detection_log=detection,
    )
    assert isinstance(runtime.orchestrator, SessionOrchestrator)
    assert isinstance(runtime.capture, UtteranceCapture)
    assert runtime.capture is capture

    while runtime.orchestrator.state is not DeviceState.CAPTURING:
        assert runtime.poll()
    assert capture.holds_audio is True
    assert runtime.poll() is True
    assert runtime.orchestrator.state is DeviceState.IDLE
    assert runtime.orchestrator.capture_is_active is False
    assert capture.active is False
    assert capture.holds_audio is False
    assert sink.drain() == []
    failed = log_path.read_text(encoding="utf-8").splitlines()
    assert len(failed) == 2
    assert failed[0].startswith("event=start ")
    assert "reason=cancel" in failed[1]
    assert "handed_off=False" in failed[1]
    assert [accepted for _event, accepted, _model in detection.records] == [True]

    runtime.run()
    runtime.close()
    utterance = sink.take()
    assert utterance is not None
    assert sink.take() is None
    assert utterance.endpoint == "leading_silence"
    assert utterance.pre_roll_sample_count > 0
    assert runtime.orchestrator.state is DeviceState.CAPTURING
    assert runtime.orchestrator.capture_is_active is False
    assert capture.holds_audio is False
    assert [accepted for _event, accepted, _model in detection.records] == [True, True]
    finished = log_path.read_text(encoding="utf-8").splitlines()
    assert len(finished) == 4
    assert "reason=end_of_utterance" in finished[-1]
    assert "handed_off=True" in finished[-1]


def test_real_orchestrator_cancel_drops_the_utterance() -> None:
    from hardware_voice_ai.runtime import WakeRuntime
    from hardware_voice_ai.session.state import DeviceState

    config, source = _wake_pcm(32_000)
    runtime = WakeRuntime(config, source)
    assert runtime.utterances is not None
    while runtime.orchestrator.state is not DeviceState.CAPTURING:
        assert runtime.poll()
    runtime.orchestrator.cancel()
    assert runtime.utterances.drain() == []
    assert runtime.orchestrator.state is DeviceState.IDLE
    assert runtime.capture.holds_audio is False
    while runtime.poll():
        pass
    assert runtime.utterances.drain() == []
    runtime.close()


def test_capture_sources_do_not_open_a_network_client() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "hardware_voice_ai" / "capture"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    for banned in ("websocket", "urllib", "http.client", "requests", "speech_to_text", "text_to_speech"):
        assert banned not in text
    assert "import socket" not in text
