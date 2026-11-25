"""Device configuration and swappable model / microphone backends."""

import sys
from dataclasses import replace

import numpy as np
import pytest

from hardware_voice_ai.audio.frame import AudioFormat
from hardware_voice_ai.audio.source import (
    PcmAudioSource,
    SoundDeviceAudioSource,
    build_audio_source,
)
from hardware_voice_ai.config import DeviceConfig
from hardware_voice_ai.errors import AudioBackendError, ConfigurationError
from hardware_voice_ai.session.audio_model import AudioModel
from hardware_voice_ai.wake.adapters import OWW_CHUNK_SAMPLES, OpenWakeWordModel
from hardware_voice_ai.wake.defaults import OPENWAKEWORD_MODEL_ID
from hardware_voice_ai.wake.factory import build_wake_model


def test_defaults_match_the_architecture() -> None:
    config = DeviceConfig()
    assert config.audio_model is AudioModel.REMOTE
    assert config.wake_word == "hey computer"
    assert config.wake_model_id == "reference-keyword-v1"
    assert config.channels == 1
    assert config.pre_roll_ms == 800
    assert config.confirm_ms == 240
    assert config.detection_log_path is None
    assert config.volume == 1.0
    assert config.playback_queue_ms == 80
    assert config.audio_format.channels == 1
    assert config.audio_format.frame_samples == 320


def test_audio_model_labels_are_only_local_and_remote() -> None:
    assert {item.value for item in AudioModel} == {"local", "remote"}
    assert DeviceConfig(audio_model="local").audio_model is AudioModel.LOCAL
    with pytest.raises(ConfigurationError):
        DeviceConfig(audio_model="stt")
    with pytest.raises(ConfigurationError):
        DeviceConfig.from_env({"HVA_AUDIO_MODEL": "tts"})


def test_decision_budget_and_preroll_bounds() -> None:
    DeviceConfig(confirm_ms=200)
    DeviceConfig(confirm_ms=300)
    with pytest.raises(ConfigurationError):
        DeviceConfig(confirm_ms=160)
    with pytest.raises(ConfigurationError):
        DeviceConfig(confirm_ms=320)
    with pytest.raises(ConfigurationError):
        DeviceConfig(pre_roll_ms=160)
    with pytest.raises(ConfigurationError):
        DeviceConfig(pre_roll_ms=1200)
    with pytest.raises(ConfigurationError):
        DeviceConfig(channels=0)


def test_from_env_is_opt_in_for_the_log(tmp_path) -> None:
    path = tmp_path / "wake.jsonl"
    config = DeviceConfig.from_env(
        {
            "HVA_AUDIO_MODEL": "local",
            "HVA_CHANNELS": "2",
            "HVA_DETECTION_LOG": str(path),
            "HVA_WAKE_WORD": "hey computer",
        }
    )
    assert config.audio_model is AudioModel.LOCAL
    assert config.channels == 2
    assert config.detection_log_path == path
    assert DeviceConfig.from_env({}).detection_log_path is None


def test_unknown_model_and_word_mismatch_fail_closed() -> None:
    with pytest.raises(ConfigurationError, match="Unknown wake model"):
        build_wake_model(replace(DeviceConfig(), wake_model_id="mystery"))
    with pytest.raises(ConfigurationError, match="does not match"):
        build_wake_model(replace(DeviceConfig(), wake_word="computer"))


def test_extra_model_registry_swaps_the_implementation() -> None:
    sentinel = object()

    def factory(config: DeviceConfig):
        assert config.wake_word == "desk"
        return sentinel

    built = build_wake_model(
        replace(DeviceConfig(), wake_model_id="desk-model", wake_word="desk"),
        extra_models={"desk-model": factory},
    )
    assert built is sentinel


def test_openwakeword_spike_profile_is_required_before_import() -> None:
    with pytest.raises(ConfigurationError, match="min_positive_ms"):
        build_wake_model(
            replace(
                DeviceConfig(),
                wake_model_id=OPENWAKEWORD_MODEL_ID,
                wake_word="hey_jarvis",
            )
        )


def test_openwakeword_missing_extra_is_a_clear_error(monkeypatch) -> None:
    real_import = __import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openwakeword" or name.startswith("openwakeword."):
            raise ImportError("not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", blocked)
    with pytest.raises(AudioBackendError, match="openWakeWord"):
        build_wake_model(
            replace(
                DeviceConfig(),
                wake_model_id=OPENWAKEWORD_MODEL_ID,
                wake_word="hey_jarvis",
                min_positive_ms=20,
            )
        )


class _Runner:
    def __init__(self) -> None:
        self.chunks = []
        self.resets = 0

    def predict(self, pcm: np.ndarray) -> dict[str, float]:
        self.chunks.append(pcm.copy())
        return {"hey_jarvis": 0.42}

    def reset(self) -> None:
        self.resets += 1


def test_openwakeword_adapter_buffers_and_picks_the_louder_channel() -> None:
    fmt = AudioFormat(sample_rate_hz=16000, channels=2, frame_samples=320)
    runner = _Runner()
    model = OpenWakeWordModel(
        "hey_jarvis",
        runner,
        channels=2,
        sample_rate_hz=16000,
    )
    from hardware_voice_ai.audio.frame import AudioFrame

    scores = []
    for index in range(4):
        samples = np.zeros((320, 2), dtype=np.float32)
        samples[:, 1] = 0.25
        frame = AudioFrame(samples=samples, format=fmt, sample_index=index * 320)
        scores.append(model.score(frame))
    assert scores[0].confidence == 0.0
    assert scores[3].confidence == 0.42
    assert scores[3].channel == 1
    assert len(runner.chunks) == 1
    assert runner.chunks[0].shape == (OWW_CHUNK_SAMPLES,)
    assert int(np.max(np.abs(runner.chunks[0]))) > 0
    model.reset()
    assert runner.resets == 1


def test_pcm_source_frames_and_missing_sounddevice(monkeypatch) -> None:
    fmt = AudioFormat(sample_rate_hz=16000, channels=1, frame_samples=320)
    source = build_audio_source(fmt, "pcm", samples=np.ones(320, dtype=np.float32))
    assert isinstance(source, PcmAudioSource)
    frame = source.read()
    assert frame is not None
    assert frame.samples.shape == (320, 1)
    assert source.read() is None
    source.close()

    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(AudioBackendError, match="sounddevice"):
        SoundDeviceAudioSource(fmt)


def test_sounddevice_backend_converts_int16_without_hardware(monkeypatch) -> None:
    class FakeStream:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.started = False
            self.closed = False

        def start(self) -> None:
            self.started = True

        def read(self, n: int):
            channels = self.kwargs["channels"]
            data = np.full((n, channels), 16384, dtype=np.int16)
            return data, False

        def stop(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class FakeSoundDevice:
        def __init__(self) -> None:
            self.stream = None

        def InputStream(self, **kwargs):
            self.stream = FakeStream(**kwargs)
            return self.stream

    fake = FakeSoundDevice()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    fmt = AudioFormat(sample_rate_hz=16000, channels=2, frame_samples=320)
    source = SoundDeviceAudioSource(fmt, device=3)
    assert fake.stream is not None
    assert fake.stream.started is True
    assert fake.stream.kwargs["channels"] == 2
    assert fake.stream.kwargs["device"] == 3
    frame = source.read()
    assert frame.samples.shape == (320, 2)
    assert np.allclose(frame.samples, np.float32(0.5))
    assert frame.sample_index == 0
    source.close()
    assert fake.stream.closed is True
