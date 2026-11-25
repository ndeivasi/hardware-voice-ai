"""Opt-in local detection logs never store raw audio."""

import json
from dataclasses import fields

import numpy as np
import pytest

from hardware_voice_ai.config import DeviceConfig
from hardware_voice_ai.errors import ConfigurationError
from hardware_voice_ai.log.detections import LOG_FIELDS, DetectionLog
from hardware_voice_ai.session.audio_model import AudioModel
from hardware_voice_ai.wake.events import WakeEvent


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
        decision_frame_index=30,
        word_end_frame_index=18,
        word_end_sample_index=6080,
        decision_sample_index=9920,
    )


def test_wake_event_has_no_audio_payload() -> None:
    event = _event()
    names = {item.name for item in fields(WakeEvent)}
    assert "samples" not in names
    assert "audio" not in names
    for name in names:
        assert not isinstance(getattr(event, name), (bytes, bytearray, np.ndarray))


def test_log_is_off_unless_a_path_is_set(tmp_path) -> None:
    log = DetectionLog(None)
    assert log.enabled is False
    log.record(_event(), accepted=True, audio_model=AudioModel.REMOTE)
    assert list(tmp_path.iterdir()) == []
    assert DeviceConfig().detection_log_path is None


def test_opt_in_log_is_local_metadata_only(tmp_path) -> None:
    path = tmp_path / "nested" / "detections.jsonl"
    log = DetectionLog(path, clock=lambda: 1_700_000_000.0)
    log.record(_event(), accepted=True, audio_model=AudioModel.REMOTE)
    log.record(_event(), accepted=False, audio_model=AudioModel.LOCAL)
    text = path.read_text(encoding="utf-8")
    lines = text.strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert tuple(first) == LOG_FIELDS
    assert first["event"] == "wake_detection"
    assert first["accepted"] is True
    assert first["timestamp_ms"] == 1_700_000_000_000
    assert first["audio_model"] == "remote"
    assert first["decision_latency_ms"] == 240
    assert second["accepted"] is False
    assert second["audio_model"] == "local"
    assert "samples" not in text
    assert "[" not in text
    assert path.stat().st_size < 2000


def test_runtime_opt_in_log_has_no_audio(tmp_path) -> None:
    from hardware_voice_ai.audio.source import PcmAudioSource
    from hardware_voice_ai.runtime import WakeRuntime
    from hardware_voice_ai.wake.reference import default_template, render_pattern
    from tests.support import silence

    path = tmp_path / "detections.jsonl"
    config = DeviceConfig(detection_log_path=path)
    samples = np.concatenate(
        (silence(1600), render_pattern(default_template()), silence(8000))
    )
    runtime = WakeRuntime(config, PcmAudioSource(samples, config.audio_format))
    runtime.run()
    runtime.close()
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert payload["accepted"] is True
    assert payload["audio_model"] == "remote"
    assert "samples" not in payload
    assert path.stat().st_size < 2000


def test_remote_log_urls_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="local file path"):
        DeviceConfig(detection_log_path="https://example.com/detections.jsonl")


def test_env_detection_log_url_is_rejected_before_path_normalization() -> None:
    url = "https://example.com/detections.jsonl"
    with pytest.raises(ConfigurationError, match="local file path"):
        DeviceConfig.from_env({"HVA_DETECTION_LOG": url})
    with pytest.raises(ConfigurationError, match="local file path"):
        DeviceConfig.from_env({"HVA_DETECTION_LOG": "s3://bucket/detections.jsonl"})
