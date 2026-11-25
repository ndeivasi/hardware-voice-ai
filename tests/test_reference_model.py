"""Reference keyword model and the live detection path built on it."""

from dataclasses import replace

import numpy as np

from hardware_voice_ai.audio.source import PcmAudioSource
from hardware_voice_ai.config import DeviceConfig
from hardware_voice_ai.runtime import WakeRuntime
from hardware_voice_ai.session.audio_model import AudioModel
from hardware_voice_ai.session.state import DeviceState
from hardware_voice_ai.wake.reference import (
    band_features,
    default_template,
    dominant_band,
    make_template,
    render_pattern,
)
from tests.support import MemoryLog, RecordingCapture, silence


def _pcm(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(list(parts))


def _listen(samples: np.ndarray, config: DeviceConfig | None = None, **runtime_kwargs):
    config = config or DeviceConfig()
    source = PcmAudioSource(samples, config.audio_format)
    capture = runtime_kwargs.pop("capture", None) or RecordingCapture()
    log = runtime_kwargs.pop("log", None) or MemoryLog()
    runtime = WakeRuntime(config, source, capture=capture, detection_log=log, **runtime_kwargs)
    runtime.run()
    runtime.close()
    return runtime, capture, log


def test_pure_tone_is_a_single_band() -> None:
    template = default_template()
    tone = render_pattern(template)[: template.frame_samples]
    features = band_features(
        tone,
        sample_rate_hz=template.sample_rate_hz,
        centers_hz=template.centers_hz,
    )
    assert dominant_band(
        tone,
        sample_rate_hz=template.sample_rate_hz,
        centers_hz=template.centers_hz,
    ) == template.pattern[0]
    assert features[template.pattern[0]] > 0.99
    assert float(features.sum() - features[template.pattern[0]]) < 0.01


def test_aligned_pattern_wakes_inside_the_latency_budget() -> None:
    template = default_template()
    leading = silence(int(0.4 * 16000))
    word = render_pattern(template)
    trailing = silence(int(0.5 * 16000))
    runtime, capture, log = _listen(_pcm(leading, word, trailing))

    assert runtime.orchestrator.state is DeviceState.CAPTURING
    assert len(capture.requests) == 1
    assert log.records[0][1] is True
    event = capture.requests[0].wake
    assert event.wake_word == "hey computer"
    assert event.model_id == "reference-keyword-v1"
    assert event.decision_latency_ms == 240
    assert event.confidence == 0.99
    assert event.winning_channel == 0
    word_end = len(leading) + len(word)
    assert event.word_end_sample_index == word_end

    pre_roll = capture.requests[0].pre_roll
    assert pre_roll.end_sample_index == event.decision_sample_index
    assert pre_roll.offset_of_sample(word_end - 1) is not None
    samples_before_end = word_end - pre_roll.start_sample_index
    assert samples_before_end >= int(0.2 * 16000)
    offset = pre_roll.offset_of_sample(len(leading))
    assert offset is not None
    heard = pre_roll.samples[offset : offset + template.frame_samples, 0]
    assert np.allclose(heard, word[: template.frame_samples], atol=1e-5)

    assert capture.frames
    assert capture.frames[0].sample_index == event.decision_sample_index
    assert capture.stop_reasons == []


def test_trailing_silence_does_not_end_the_utterance() -> None:
    template = default_template()
    samples = _pcm(silence(1600), render_pattern(template), silence(16000))
    _runtime, capture, _log = _listen(samples)
    assert capture.stop_reasons == []
    assert len(capture.frames) > 10


def test_noise_reverse_partial_and_a_held_tone_do_not_wake() -> None:
    template = default_template()
    rate = 16000
    rng = np.random.default_rng(0)
    noise = rng.uniform(-0.2, 0.2, size=rate * 3).astype(np.float32)
    reversed_word = render_pattern(
        make_template(tuple(reversed(template.pattern)), template.wake_word)
    )
    partial = render_pattern(template)[: 8 * template.frame_samples]
    tone = (
        0.5 * np.sin(2 * np.pi * 300 * np.arange(rate * 2) / rate)
    ).astype(np.float32)
    for samples in (
        noise,
        _pcm(silence(3200), reversed_word, silence(8000)),
        _pcm(silence(3200), partial, silence(8000)),
        _pcm(tone, silence(8000)),
    ):
        runtime, capture, log = _listen(samples)
        assert capture.requests == []
        assert log.records == []
        assert runtime.orchestrator.state is DeviceState.IDLE


def test_small_shift_still_wakes_and_a_half_frame_mix_does_not() -> None:
    """A few milliseconds of offset still lands in one band.

    Half a frame mixes two tones, so this stand-in does not treat that as the
    pattern. A spoken-word model is the swap for shift-tolerant detection.
    """
    template = default_template()
    word = render_pattern(template)
    for shift, expected in ((40, 1), (160, 0)):
        shifted = np.concatenate((np.zeros(shift, dtype=np.float32), word))
        samples = _pcm(silence(3200), shifted, silence(8000))
        _runtime, capture, _log = _listen(samples)
        assert len(capture.requests) == expected


def test_quiet_pattern_still_wakes() -> None:
    word = render_pattern(default_template(), amplitude=0.02)
    samples = _pcm(silence(3200), word, silence(8000))
    _runtime, capture, _log = _listen(samples)
    assert len(capture.requests) == 1
    assert capture.requests[0].wake.decision_latency_ms == 240


def test_second_channel_can_carry_the_wake_word() -> None:
    config = replace(DeviceConfig(), channels=2)
    word = render_pattern(default_template(), channels=2, active_channel=1)
    leading = silence(3200, channels=2)
    trailing = silence(8000, channels=2)
    _runtime, capture, _log = _listen(_pcm(leading, word, trailing), config)
    event = capture.requests[0].wake
    assert event.winning_channel == 1
    assert event.channels == 2
    assert capture.requests[0].pre_roll.channels == 2


def test_swapped_template_uses_its_own_word_and_pattern() -> None:
    custom = make_template((1, 4, 2, 6, 0, 3, 5, 7, 1, 4, 6, 2, 7, 3, 5, 0), "okay desk")
    config = replace(
        DeviceConfig(),
        wake_word="okay desk",
        min_positive_ms=280,
    )
    from hardware_voice_ai.wake.factory import build_wake_model

    model = build_wake_model(config, template=custom)
    samples = _pcm(silence(1600), render_pattern(custom), silence(8000))
    runtime, capture, _log = _listen(samples, config, model=model)
    assert capture.requests[0].wake.wake_word == "okay desk"
    assert runtime.orchestrator.audio_model is AudioModel.REMOTE

    # The default "hey computer" pattern is a different sequence.
    other = _pcm(silence(1600), render_pattern(default_template()), silence(8000))
    runtime, capture, _log = _listen(other, config, model=build_wake_model(config, template=custom))
    assert capture.requests == []
    assert runtime.orchestrator.state is DeviceState.IDLE


def test_audio_model_local_is_recorded_on_the_same_wake_path() -> None:
    config = replace(DeviceConfig(), audio_model=AudioModel.LOCAL)
    samples = _pcm(silence(1600), render_pattern(default_template()), silence(8000))
    runtime, capture, log = _listen(samples, config)
    assert capture.requests[0].audio_model is AudioModel.LOCAL
    assert log.records[0][2] == "local"
    assert runtime.orchestrator.state is DeviceState.CAPTURING
    assert runtime.orchestrator.audio_model is AudioModel.LOCAL
