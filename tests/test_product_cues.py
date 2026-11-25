"""Product cues and the follow-up window. In-memory audio only."""

import sys
import time
from dataclasses import replace

import numpy as np
import pytest

from hardware_voice_ai.audio.frame import AudioFormat, AudioFrame, AudioSnapshot
from hardware_voice_ai.audio.source import PcmAudioSource
from hardware_voice_ai.capture.endpoint import EndpointConfig
from hardware_voice_ai.capture.handoff import QueueUtteranceSink
from hardware_voice_ai.capture.utterance import UtteranceCapture
from hardware_voice_ai.config import DeviceConfig
from hardware_voice_ai.playback.constants import FOLLOW_UP_WINDOW_MS
from hardware_voice_ai.playback.cues import Cue, render_cue
from hardware_voice_ai.playback.player import SpeakerPlayback
from hardware_voice_ai.playback.source import MemoryReplyAudio
from hardware_voice_ai.playback.speaker import MemorySpeaker, SoundDeviceSpeaker
from hardware_voice_ai.runtime import WakeRuntime
from hardware_voice_ai.session.audio_model import AudioModel
from hardware_voice_ai.session.orchestrator import SessionOrchestrator
from hardware_voice_ai.session.state import DeviceState
from hardware_voice_ai.wake.events import WakeEvent
from tests.support import RecordingCapture, ScriptedModel


def _fmt() -> AudioFormat:
    return AudioFormat(sample_rate_hz=16_000, channels=1, frame_samples=320)


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
        decision_frame_index=4,
        word_end_frame_index=2,
        word_end_sample_index=800,
        decision_sample_index=1600,
    )


def _valid_snapshot() -> AudioSnapshot:
    samples = np.zeros((1600, 1), dtype=np.float32)
    samples[-200:] = 0.2
    return AudioSnapshot(
        samples=samples,
        sample_rate_hz=16000,
        start_sample_index=0,
        end_sample_index=1600,
    )


def _samples_for(ms: int, rate: int = 16_000) -> int:
    return rate * ms // 1000


def _dominant_hz(samples: np.ndarray, rate: int) -> float:
    mono = np.asarray(samples, dtype=np.float64).reshape(-1)
    edge = min(64, mono.size // 4)
    window = mono[edge:-edge] if mono.size > edge * 2 else mono
    window = window - float(np.mean(window))
    spectrum = np.abs(np.fft.rfft(window * np.hanning(window.size)))
    freqs = np.fft.rfftfreq(window.size, d=1.0 / rate)
    return float(freqs[int(np.argmax(spectrum[1:]) + 1)])


def _peak(samples: np.ndarray) -> float:
    return float(np.max(np.abs(samples)))


def _wait_until(predicate, message: str) -> None:
    deadline = time.monotonic() + 2.0
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(message)
        time.sleep(0.005)


def _session(capture=None):
    speaker = MemorySpeaker(_fmt())
    playback = SpeakerPlayback(speaker)
    session = SessionOrchestrator(
        capture if capture is not None else RecordingCapture(),
        playback=playback,
    )
    return session, playback, speaker


def _accept(session: SessionOrchestrator) -> None:
    assert session.on_wake_accepted(_event(), _valid_snapshot()) is True


def _settle(playback: SpeakerPlayback) -> None:
    playback.wait()
    if playback.is_playing:
        playback.wait()


def _finish_reply(session: SessionOrchestrator, playback: SpeakerPlayback, audio: np.ndarray) -> None:
    session.play_reply(MemoryReplyAudio(audio, sample_rate_hz=16_000, channels=1))
    _settle(playback)


def _frame(value: float, index: int) -> AudioFrame:
    return AudioFrame(
        samples=np.full((320, 1), value, dtype=np.float32),
        format=_fmt(),
        sample_index=index,
    )


def _snapshot_ending(end: int, *, loud: bool) -> AudioSnapshot:
    samples = np.zeros((3200, 1), dtype=np.float32)
    if loud:
        samples[-320:] = 0.3
    return AudioSnapshot(
        samples=samples,
        sample_rate_hz=16000,
        start_sample_index=end - 3200,
        end_sample_index=end,
    )


def test_cue_patterns_keep_wake_offline_and_error_and_add_the_new_tones() -> None:
    fmt = _fmt()
    rate = fmt.sample_rate_hz
    rendered = {cue: render_cue(cue, fmt) for cue in Cue}
    assert len({item.tobytes() for item in rendered.values()}) == len(Cue)

    wake = rendered[Cue.WAKE][:, 0]
    first = _samples_for(80)
    gap = _samples_for(20)
    assert wake.size == first + gap + first
    assert _dominant_hz(wake[:first], rate) == pytest.approx(880, abs=20)
    assert _peak(wake[first : first + gap]) == 0.0
    assert _dominant_hz(wake[first + gap :], rate) == pytest.approx(1320, abs=20)
    assert _peak(wake) == pytest.approx(0.25, rel=0.05)
    assert _peak(wake) <= 0.25 + 1e-5

    offline = rendered[Cue.OFFLINE][:, 0]
    low = _samples_for(90)
    pause = _samples_for(40)
    assert offline.size == low + pause + low
    assert _dominant_hz(offline[:low], rate) == pytest.approx(220, abs=20)
    assert _peak(offline[low : low + pause]) == 0.0
    assert _peak(offline) <= 0.25 + 1e-5

    error = rendered[Cue.ERROR][:, 0]
    beep = _samples_for(50)
    rest = _samples_for(30)
    assert error.size == beep * 3 + rest * 2
    assert _dominant_hz(error[:beep], rate) == pytest.approx(660, abs=25)
    assert _peak(error) <= 0.25 + 1e-5

    thinking = rendered[Cue.THINKING]
    assert thinking.shape == (_samples_for(120), 1)
    assert _dominant_hz(thinking, rate) == pytest.approx(440, abs=20)
    assert _peak(thinking) == pytest.approx(0.25, rel=0.05)
    assert _peak(thinking) <= 0.25 + 1e-5

    listening = rendered[Cue.LISTENING]
    assert listening.shape == (_samples_for(120), 1)
    assert _dominant_hz(listening, rate) == pytest.approx(1100, abs=20)
    assert _peak(listening) <= 0.25 + 1e-5
    assert not np.array_equal(listening, thinking)
    assert not np.array_equal(listening, rendered[Cue.WAKE])

    ended = rendered[Cue.FOLLOW_UP_END]
    assert ended.shape == (_samples_for(80), 1)
    assert _dominant_hz(ended, rate) == pytest.approx(330, abs=20)
    assert _peak(ended) == pytest.approx(0.15, rel=0.08)
    assert _peak(ended) <= 0.15 + 1e-5

    stereo = render_cue(Cue.THINKING, AudioFormat(16_000, 2, 320))
    assert stereo.shape == (thinking.shape[0], 2)
    assert np.allclose(stereo[:, 0], stereo[:, 1])


@pytest.mark.parametrize("mode", [AudioModel.LOCAL, AudioModel.REMOTE])
def test_accepted_wake_plays_only_the_wake_cue(mode: AudioModel) -> None:
    speaker = MemorySpeaker(_fmt())
    playback = SpeakerPlayback(speaker)
    session = SessionOrchestrator(RecordingCapture(), audio_model=mode, playback=playback)
    assert session.on_wake_accepted(_event(), _valid_snapshot()) is True
    playback.wait()
    assert session.state is DeviceState.CAPTURING
    assert session.follow_up_is_active is False
    assert not playback.is_playing
    assert np.allclose(speaker.heard(), render_cue(Cue.WAKE, _fmt()))
    time.sleep(0.05)
    assert np.allclose(speaker.heard(), render_cue(Cue.WAKE, _fmt()))
    playback.close()


def test_thinking_cue_plays_once_when_capture_ends_and_stays_capturing() -> None:
    session, playback, speaker = _session()
    _accept(session)
    playback.wait()
    before = len(speaker.heard())
    session.close_capture()
    playback.wait()
    assert session.state is DeviceState.CAPTURING
    assert not playback.is_playing
    thinking = render_cue(Cue.THINKING, _fmt())
    heard = speaker.heard()[before:]
    assert np.allclose(heard, thinking)
    assert heard.shape[0] == thinking.shape[0]
    session.close_capture()
    playback.wait()
    assert np.allclose(speaker.heard()[before:], thinking)
    playback.close()


def test_reply_samples_are_not_mixed_with_the_thinking_cue() -> None:
    session, playback, speaker = _session()
    _accept(session)
    playback.wait()
    session.close_capture()
    reply = np.full(960, 0.55, dtype=np.float32)
    _finish_reply(session, playback, reply)
    heard = speaker.heard()[:, 0]
    found = False
    for start in range(0, len(heard) - len(reply) + 1):
        if np.allclose(heard[start : start + len(reply)], reply):
            found = True
            break
    assert found
    playback.close()


def test_ready_reply_cuts_thinking_before_it_reaches_the_speaker(monkeypatch) -> None:
    class FakeStream:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    class FakeOutput:
        def __init__(self) -> None:
            self.stream: FakeStream | None = None

        def OutputStream(self, **kwargs):
            self.stream = FakeStream(**kwargs)
            return self.stream

    fake = FakeOutput()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    speaker = SoundDeviceSpeaker(_fmt(), queue_ms=80)
    assert fake.stream is not None
    callback = fake.stream.kwargs["callback"]
    playback = SpeakerPlayback(speaker)
    session = SessionOrchestrator(RecordingCapture(), playback=playback)
    _accept(session)
    _wait_until(
        lambda: playback.is_cue_playing and speaker.queued_samples > 0,
        "wake cue was not queued",
    )
    while speaker.queued_samples > 0 or playback.is_playing:
        out = np.zeros((320, 1), dtype=np.float32)
        callback(out, 320, None, None)
    playback.wait()
    played_after_wake = speaker.played_samples
    session.close_capture()
    _wait_until(
        lambda: playback.is_cue_playing and speaker.queued_samples > 0,
        "thinking cue was not held in the queue",
    )
    assert speaker.played_samples == played_after_wake
    reply = np.full(640, 0.4, dtype=np.float32)
    session.play_reply(MemoryReplyAudio(reply, sample_rate_hz=16_000, channels=1))
    assert playback.is_cue_playing is False
    assert session.state is DeviceState.SPEAKING
    collected = []
    for _ in range(8):
        if not playback.is_playing and speaker.queued_samples == 0:
            break
        out = np.zeros((320, 1), dtype=np.float32)
        callback(out, 320, None, None)
        collected.append(out[:, 0].copy())
    playback.wait()
    heard = np.concatenate(collected) if collected else np.zeros(0, dtype=np.float32)
    spoken = heard[np.abs(heard) > 1e-5]
    assert spoken.size == len(reply)
    assert np.allclose(spoken, reply)
    assert speaker.played_samples == played_after_wake + len(reply)
    playback.close()


def test_follow_up_silence_plays_listening_then_end_and_requires_a_wake() -> None:
    session, playback, speaker = _session()
    _accept(session)
    playback.wait()
    session.close_capture()
    playback.wait()
    reply = np.full(320, 0.2, dtype=np.float32)
    _finish_reply(session, playback, reply)
    assert session.state is DeviceState.IDLE
    assert session.follow_up_is_active is True
    listening = render_cue(Cue.LISTENING, _fmt())
    assert np.allclose(speaker.heard()[-len(listening) :], listening)

    rate = 16_000
    window = rate * FOLLOW_UP_WINDOW_MS // 1000
    frame = 320
    almost = window - frame
    session.advance_follow_up(almost)
    assert session.follow_up_is_active is True
    assert np.allclose(speaker.heard()[-len(listening) :], listening)

    session.advance_follow_up(frame)
    playback.wait()
    ended = render_cue(Cue.FOLLOW_UP_END, _fmt())
    assert session.follow_up_is_active is False
    assert session.state is DeviceState.IDLE
    assert np.allclose(speaker.heard()[-len(ended) :], ended)
    assert session.on_wake_accepted(_event(), _valid_snapshot()) is True
    playback.wait()
    assert session.state is DeviceState.CAPTURING
    assert np.allclose(speaker.heard()[-render_cue(Cue.WAKE, _fmt()).shape[0] :], render_cue(Cue.WAKE, _fmt()))
    playback.close()


def test_follow_up_speech_opens_capture_without_the_end_cue_or_wake_pair() -> None:
    sink = QueueUtteranceSink()
    capture = UtteranceCapture(sink)
    session, playback, speaker = _session(capture)
    capture.bind_session(session)
    _accept(session)
    playback.wait()
    session.close_capture()
    playback.wait()
    _finish_reply(session, playback, np.full(320, 0.2, dtype=np.float32))
    assert capture.active is False
    before = len(speaker.heard())
    speech = _snapshot_ending(20_000, loud=True)
    frame = _frame(0.3, 20_000 - 320)
    assert session.follow_up_hears_speech(frame) is True
    assert session.accept_follow_up_speech(speech) is True
    playback.wait()
    assert session.state is DeviceState.CAPTURING
    assert session.follow_up_is_active is False
    added = speaker.heard()[before:]
    ended = render_cue(Cue.FOLLOW_UP_END, _fmt())
    wake = render_cue(Cue.WAKE, _fmt())
    assert added.size == 0 or not _contains(added, ended)
    assert not _contains(added, wake)
    quiet = _frame(0.0, 20_000)
    assert session.follow_up_hears_speech(quiet) is False
    session.advance_follow_up(16_000 * 5)
    playback.wait()
    assert not _contains(speaker.heard()[before:], ended)
    playback.close()


def test_speech_on_the_expiring_frame_does_not_play_follow_up_end() -> None:
    session, playback, speaker = _session()
    _accept(session)
    playback.wait()
    session.close_capture()
    playback.wait()
    _finish_reply(session, playback, np.full(320, 0.2, dtype=np.float32))
    window = 16_000 * FOLLOW_UP_WINDOW_MS // 1000
    session.advance_follow_up(window - 320)
    frame = _frame(0.3, 8_000)
    assert channel_rms_at_least(frame)
    assert session.follow_up_hears_speech(frame) is True
    assert session.accept_follow_up_speech(_snapshot_ending(8_320, loud=True)) is True
    playback.wait()
    assert session.state is DeviceState.CAPTURING
    assert not _contains(speaker.heard(), render_cue(Cue.FOLLOW_UP_END, _fmt()))
    playback.close()


def test_mute_and_stop_and_cancel_skip_follow_up_end() -> None:
    ended = render_cue(Cue.FOLLOW_UP_END, _fmt())

    muted, playback, speaker = _session()
    _accept(muted)
    playback.wait()
    muted.close_capture()
    playback.wait()
    _finish_reply(muted, playback, np.full(320, 0.2, dtype=np.float32))
    muted.apply_mute(True)
    muted.advance_follow_up(16_000 * 6)
    playback.wait()
    assert muted.follow_up_is_active is False
    assert muted.state is DeviceState.IDLE
    assert not _contains(speaker.heard(), ended)
    playback.close()

    stopped, playback, speaker = _session()
    _accept(stopped)
    playback.wait()
    stopped.close_capture()
    playback.wait()
    _finish_reply(stopped, playback, np.full(320, 0.2, dtype=np.float32))
    stopped.stop_playback()
    stopped.advance_follow_up(16_000 * 6)
    playback.wait()
    assert stopped.follow_up_is_active is False
    assert not _contains(speaker.heard(), ended)
    playback.close()

    cancelled, playback, speaker = _session()
    _accept(cancelled)
    playback.wait()
    cancelled.close_capture()
    playback.wait()
    _finish_reply(cancelled, playback, np.full(320, 0.2, dtype=np.float32))
    cancelled.cancel()
    cancelled.advance_follow_up(16_000 * 6)
    playback.wait()
    assert cancelled.follow_up_is_active is False
    assert cancelled.state is DeviceState.IDLE
    assert not _contains(speaker.heard(), ended)
    playback.close()


def test_runtime_follow_up_expires_on_silence_and_accepts_speech() -> None:
    fmt = _fmt()
    scores = [0.9] * 6 + [0.0] * 12
    config = replace(
        DeviceConfig(),
        accept_threshold=0.6,
        min_positive_ms=120,
        confirm_ms=240,
        cooldown_ms=0,
        max_word_ms=1000,
        end_ratio=0.85,
    )
    silence_scores = scores + [0.0] * 400
    silence = np.zeros(320 * len(silence_scores), dtype=np.float32)
    runtime, playback, speaker, capture = _runtime(config, silence, silence_scores)
    try:
        while runtime.orchestrator.state is not DeviceState.CAPTURING:
            assert runtime.poll()
        playback.wait()
        runtime.orchestrator.close_capture()
        playback.wait()
        assert runtime.orchestrator.state is DeviceState.CAPTURING
        reply = np.full(320, 0.2, dtype=np.float32)
        runtime.orchestrator.play_reply(
            MemoryReplyAudio(reply, sample_rate_hz=16_000, channels=1)
        )
        _settle(playback)
        assert runtime.orchestrator.follow_up_is_active is True
        window_frames = (fmt.sample_rate_hz * FOLLOW_UP_WINDOW_MS // 1000) // fmt.frame_samples
        for _ in range(window_frames - 1):
            assert runtime.poll()
        assert runtime.orchestrator.follow_up_is_active is True
        assert not _contains(speaker.heard(), render_cue(Cue.FOLLOW_UP_END, fmt))
        assert runtime.poll()
        playback.wait()
        assert runtime.orchestrator.follow_up_is_active is False
        assert runtime.orchestrator.state is DeviceState.IDLE
        assert np.allclose(
            speaker.heard()[-render_cue(Cue.FOLLOW_UP_END, fmt).shape[0] :],
            render_cue(Cue.FOLLOW_UP_END, fmt),
        )
        assert len(capture.requests) == 1
    finally:
        runtime.close()

    loud_scores = scores + [0.0] * 80
    loud = np.full(320 * len(loud_scores), 0.3, dtype=np.float32)
    runtime, playback, speaker, capture = _runtime(config, loud, loud_scores)
    try:
        while runtime.orchestrator.state is not DeviceState.CAPTURING:
            assert runtime.poll()
        runtime.orchestrator.close_capture()
        playback.wait()
        runtime.orchestrator.play_reply(
            MemoryReplyAudio(np.full(320, 0.2, dtype=np.float32), sample_rate_hz=16_000, channels=1)
        )
        _settle(playback)
        assert runtime.poll()
        playback.wait()
        assert runtime.orchestrator.state is DeviceState.CAPTURING
        assert runtime.orchestrator.follow_up_is_active is False
        assert len(capture.requests) == 2
        assert not _contains(speaker.heard(), render_cue(Cue.FOLLOW_UP_END, fmt))
        wake = render_cue(Cue.WAKE, fmt)
        # The second turn must not replay the wake pair after the listening cue.
        listening = render_cue(Cue.LISTENING, fmt)
        heard = speaker.heard()
        listen_at = _find(heard, listening)
        assert listen_at >= 0
        assert _find(heard[listen_at + len(listening) :], wake) < 0
    finally:
        runtime.close()


def test_follow_up_threshold_matches_the_capture_floor() -> None:
    assert EndpointConfig().min_speech_rms == pytest.approx(0.02)
    session, playback, _speaker = _session()
    _accept(session)
    playback.wait()
    session.close_capture()
    playback.wait()
    _finish_reply(session, playback, np.full(320, 0.2, dtype=np.float32))
    assert session.follow_up_hears_speech(_frame(0.0, 0)) is False
    assert session.follow_up_hears_speech(_frame(0.02, 320)) is True
    playback.close()


def _runtime(config: DeviceConfig, samples: np.ndarray, scores: list[float]):
    speaker = MemorySpeaker(config.audio_format)
    playback = SpeakerPlayback(speaker)
    capture = RecordingCapture()
    runtime = WakeRuntime(
        config,
        PcmAudioSource(samples, config.audio_format),
        model=ScriptedModel(scores=scores),
        capture=capture,
        playback=playback,
    )
    return runtime, playback, speaker, capture


def _contains(haystack: np.ndarray, needle: np.ndarray) -> bool:
    return _find(haystack, needle) >= 0


def _find(haystack: np.ndarray, needle: np.ndarray) -> int:
    hay = np.asarray(haystack, dtype=np.float32).reshape(-1)
    pin = np.asarray(needle, dtype=np.float32).reshape(-1)
    if pin.size == 0 or hay.size < pin.size:
        return -1
    for start in range(0, hay.size - pin.size + 1):
        if np.allclose(hay[start : start + pin.size], pin):
            return start
    return -1


def channel_rms_at_least(frame: AudioFrame) -> bool:
    rms = float(np.sqrt(np.mean(np.square(frame.samples))))
    return rms >= EndpointConfig().min_speech_rms
