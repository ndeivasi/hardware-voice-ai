"""Device configuration for on-device wake detection.

The only model choice stored here is Audio Model (``local`` or ``remote``,
default ``remote``). Wake-word identity and the microphone backend are
swappable. Speech-to-text and text-to-speech are not settings.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hardware_voice_ai.audio.frame import AudioFormat
from hardware_voice_ai.errors import ConfigurationError
from hardware_voice_ai.session.audio_model import AudioModel, parse_audio_model
from hardware_voice_ai.wake.defaults import DEFAULT_WAKE_WORD, REFERENCE_MODEL_ID

DECISION_BUDGET_MS = (200, 300)
DEFAULT_CONFIRM_MS = 240
DEFAULT_PRE_ROLL_MS = 800


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    audio_model: AudioModel = AudioModel.REMOTE
    wake_word: str = DEFAULT_WAKE_WORD
    wake_model_id: str = REFERENCE_MODEL_ID
    sample_rate_hz: int = 16_000
    channels: int = 1
    frame_ms: int = 20
    pre_roll_ms: int = DEFAULT_PRE_ROLL_MS
    accept_threshold: float = 0.8
    min_positive_ms: int = 280
    confirm_ms: int = DEFAULT_CONFIRM_MS
    cooldown_ms: int = 800
    max_word_ms: int = 1_000
    end_ratio: float = 0.85
    detection_log_path: Path | None = None
    audio_backend: str = "sounddevice"
    input_device: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.audio_model, str):
            object.__setattr__(self, "audio_model", parse_audio_model(self.audio_model))
        elif not isinstance(self.audio_model, AudioModel):
            raise ConfigurationError("Audio Model must be 'local' or 'remote'.")
        if isinstance(self.detection_log_path, str):
            if "://" in self.detection_log_path:
                raise ConfigurationError("Detection logs must be a local file path.")
            object.__setattr__(self, "detection_log_path", Path(self.detection_log_path))
        if not self.wake_word.strip():
            raise ConfigurationError("wake_word must not be empty.")
        if not self.wake_model_id.strip():
            raise ConfigurationError("wake_model_id must not be empty.")
        if self.sample_rate_hz <= 0:
            raise ConfigurationError("sample_rate_hz must be positive.")
        if self.channels < 1:
            raise ConfigurationError("channels must be at least 1.")
        if self.frame_ms <= 0:
            raise ConfigurationError("frame_ms must be positive.")
        if (self.sample_rate_hz * self.frame_ms) % 1000 != 0:
            raise ConfigurationError(
                "frame_ms must divide the sample rate into a whole number of samples."
            )
        _multiple("pre_roll_ms", self.pre_roll_ms, self.frame_ms)
        _multiple("confirm_ms", self.confirm_ms, self.frame_ms)
        _multiple("min_positive_ms", self.min_positive_ms, self.frame_ms)
        _multiple("cooldown_ms", self.cooldown_ms, self.frame_ms)
        _multiple("max_word_ms", self.max_word_ms, self.frame_ms)
        low, high = DECISION_BUDGET_MS
        if not low <= self.confirm_ms <= high:
            raise ConfigurationError(
                "confirm_ms must sit in the 200–300 ms wake-decision budget."
            )
        if not 200 <= self.pre_roll_ms <= 1000:
            raise ConfigurationError(
                "pre_roll_ms must be a few hundred milliseconds to about a second."
            )
        if self.min_positive_ms < self.frame_ms:
            raise ConfigurationError("min_positive_ms must cover at least one frame.")
        if self.max_word_ms <= self.min_positive_ms:
            raise ConfigurationError("max_word_ms must be longer than min_positive_ms.")
        if not 0.0 < self.accept_threshold <= 1.0:
            raise ConfigurationError("accept_threshold must be in (0, 1].")
        if not 0.0 < self.end_ratio < 1.0:
            raise ConfigurationError("end_ratio must be in (0, 1).")
        if self.cooldown_ms < 0:
            raise ConfigurationError("cooldown_ms must be >= 0.")
        if self.audio_backend not in {"sounddevice", "pcm"}:
            raise ConfigurationError(
                "audio_backend must be 'sounddevice' or 'pcm'."
            )
        if self.detection_log_path is not None and "://" in str(self.detection_log_path):
            raise ConfigurationError("Detection logs must be a local file path.")

    @property
    def frame_samples(self) -> int:
        return self.sample_rate_hz * self.frame_ms // 1000

    @property
    def audio_format(self) -> AudioFormat:
        return AudioFormat(
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            frame_samples=self.frame_samples,
        )

    @property
    def pre_roll_samples(self) -> int:
        return self.sample_rate_hz * self.pre_roll_ms // 1000

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "DeviceConfig":
        values = os.environ if env is None else env
        log_path = values.get("HVA_DETECTION_LOG", "").strip()
        # Path("https://...") collapses "://" to ":/", so reject the raw value first.
        if "://" in log_path:
            raise ConfigurationError("Detection logs must be a local file path.")
        device = values.get("HVA_INPUT_DEVICE", "").strip()
        return cls(
            audio_model=parse_audio_model(values.get("HVA_AUDIO_MODEL", "remote")),
            wake_word=values.get("HVA_WAKE_WORD", DEFAULT_WAKE_WORD),
            wake_model_id=values.get("HVA_WAKE_MODEL", REFERENCE_MODEL_ID),
            sample_rate_hz=_optional_int(values, "HVA_SAMPLE_RATE", 16_000),
            channels=_optional_int(values, "HVA_CHANNELS", 1),
            frame_ms=_optional_int(values, "HVA_FRAME_MS", 20),
            pre_roll_ms=_optional_int(values, "HVA_PRE_ROLL_MS", DEFAULT_PRE_ROLL_MS),
            accept_threshold=_optional_float(values, "HVA_ACCEPT_THRESHOLD", 0.8),
            min_positive_ms=_optional_int(values, "HVA_MIN_POSITIVE_MS", 280),
            confirm_ms=_optional_int(values, "HVA_CONFIRM_MS", DEFAULT_CONFIRM_MS),
            cooldown_ms=_optional_int(values, "HVA_COOLDOWN_MS", 800),
            max_word_ms=_optional_int(values, "HVA_MAX_WORD_MS", 1_000),
            end_ratio=_optional_float(values, "HVA_END_RATIO", 0.85),
            detection_log_path=Path(log_path) if log_path else None,
            audio_backend=values.get("HVA_AUDIO_BACKEND", "sounddevice"),
            input_device=device or None,
        )


def _multiple(name: str, value: int, frame_ms: int) -> None:
    if value < 0 or value % frame_ms != 0:
        raise ConfigurationError(f"{name} must be a non-negative multiple of frame_ms.")


def _optional_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{key} must be an integer.") from exc


def _optional_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{key} must be a number.") from exc
