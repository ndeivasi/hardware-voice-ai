"""Wake-engine decisions, independent of a particular acoustic model."""

from dataclasses import replace

from hardware_voice_ai.config import DeviceConfig
from hardware_voice_ai.wake.engine import WakeEngine
from tests.support import ScriptedModel, feed_scores


def _config(**overrides) -> DeviceConfig:
    config = replace(
        DeviceConfig(),
        accept_threshold=0.6,
        min_positive_ms=120,
        confirm_ms=240,
        cooldown_ms=0,
        max_word_ms=1000,
        end_ratio=0.85,
    )
    return replace(config, **overrides) if overrides else config


def test_accepts_after_the_confirm_window() -> None:
    scores = [0.9] * 8 + [0.0] * 12
    model = ScriptedModel(scores=scores, wake_word="hey computer", model_id="scripted")
    events = feed_scores(WakeEngine(model, _config()), scores)
    assert len(events) == 1
    event = events[0]
    assert event.decision_latency_ms == 240
    assert event.confidence == 0.9
    assert event.wake_word == "hey computer"
    assert event.model_id == "scripted"
    assert event.word_end_frame_index == 7
    assert event.decision_frame_index == 19
    assert 200 <= event.decision_latency_ms <= 300
    assert model.resets == 1


def test_spike_does_not_accept() -> None:
    scores = [0.99] + [0.0] * 20
    events = feed_scores(WakeEngine(ScriptedModel(scores=scores), _config()), scores)
    assert events == []


def test_short_burst_below_minimum_does_not_accept() -> None:
    scores = [0.9] * 5 + [0.0] * 12
    events = feed_scores(WakeEngine(ScriptedModel(scores=scores), _config()), scores)
    assert events == []


def test_minimum_duration_accepts() -> None:
    scores = [0.9] * 6 + [0.0] * 12
    events = feed_scores(WakeEngine(ScriptedModel(scores=scores), _config()), scores)
    assert len(events) == 1
    assert events[0].decision_latency_ms == 240


def test_new_peak_during_confirm_moves_the_word_end() -> None:
    scores = [0.9] * 8 + [0.1] * 3 + [0.95] + [0.1] * 12
    events = feed_scores(WakeEngine(ScriptedModel(scores=scores), _config()), scores)
    assert len(events) == 1
    assert events[0].word_end_frame_index == 11
    assert events[0].decision_latency_ms == 240
    assert events[0].confidence == 0.95


def test_cooldown_suppresses_a_second_word() -> None:
    pattern = [0.9] * 8 + [0.0] * 12
    cooldown = [0.99] * 40
    scores = pattern + cooldown + pattern
    events = feed_scores(
        WakeEngine(ScriptedModel(scores=scores), _config(cooldown_ms=800)),
        scores,
    )
    assert len(events) == 2
    assert events[1].decision_frame_index > events[0].decision_frame_index + 40


def test_energy_that_never_ends_is_rejected() -> None:
    scores = [0.99] * 30 + [0.0] * 5 + [0.9] * 8 + [0.0] * 12
    events = feed_scores(
        WakeEngine(ScriptedModel(scores=scores), _config(max_word_ms=200)),
        scores,
    )
    assert len(events) == 1
    assert events[0].confidence == 0.9


def test_single_frame_posterior_accepts_when_minimum_is_one_frame() -> None:
    """Neural wake models often emit one hot frame. The engine allows that."""
    scores = [0.2, 0.92] + [0.0] * 12
    events = feed_scores(
        WakeEngine(ScriptedModel(scores=scores), _config(min_positive_ms=20)),
        scores,
    )
    assert len(events) == 1
    assert events[0].decision_latency_ms == 240
    assert events[0].confidence == 0.92
    assert events[0].word_end_sample_index == 2 * 320
    assert events[0].decision_sample_index - events[0].word_end_sample_index == 12 * 320
