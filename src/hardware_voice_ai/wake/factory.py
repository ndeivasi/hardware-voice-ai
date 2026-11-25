"""Build a wake model from device configuration."""

from collections.abc import Callable, Mapping

from hardware_voice_ai.config import DeviceConfig
from hardware_voice_ai.errors import AudioBackendError, ConfigurationError
from hardware_voice_ai.wake.defaults import OPENWAKEWORD_MODEL_ID, REFERENCE_MODEL_ID
from hardware_voice_ai.wake.model import WakeModel
from hardware_voice_ai.wake.reference import (
    ReferenceKeywordModel,
    WakeTemplate,
    default_template,
)

WakeModelFactory = Callable[[DeviceConfig], WakeModel]


def build_wake_model(
    config: DeviceConfig,
    *,
    template: WakeTemplate | None = None,
    extra_models: Mapping[str, WakeModelFactory] | None = None,
) -> WakeModel:
    """Return the configured on-device model.

    ``extra_models`` registers additional factories for tests and later model
    packs. Unknown ids fail closed rather than silently using the reference
    pattern.
    """
    if config.wake_model_id == REFERENCE_MODEL_ID:
        return _build_reference(config, template)
    if config.wake_model_id == OPENWAKEWORD_MODEL_ID:
        return _build_openwakeword(config)
    if extra_models and config.wake_model_id in extra_models:
        return extra_models[config.wake_model_id](config)
    known = [REFERENCE_MODEL_ID, OPENWAKEWORD_MODEL_ID]
    if extra_models:
        known.extend(extra_models)
    raise ConfigurationError(
        f"Unknown wake model {config.wake_model_id!r}. Known models: {', '.join(known)}."
    )


def _build_reference(config: DeviceConfig, template: WakeTemplate | None) -> ReferenceKeywordModel:
    chosen = template or default_template()
    if config.wake_word != chosen.wake_word:
        raise ConfigurationError(
            f"Wake word {config.wake_word!r} does not match the reference model "
            f"pattern labeled {chosen.wake_word!r}. Supply a template for that word "
            "or select another wake model."
        )
    if (
        config.sample_rate_hz != chosen.sample_rate_hz
        or config.frame_samples != chosen.frame_samples
    ):
        raise ConfigurationError(
            "reference-keyword-v1 is built for 16000 Hz audio in 20 ms frames. "
            "Select another wake model for a different format."
        )
    pattern_ms = len(chosen.pattern) * config.frame_ms
    if config.min_positive_ms > pattern_ms:
        raise ConfigurationError(
            "min_positive_ms is longer than the reference pattern, so a full "
            "pattern cannot be accepted. Lower min_positive_ms or lengthen the pattern."
        )
    if config.max_word_ms <= pattern_ms:
        raise ConfigurationError(
            "max_word_ms must be longer than the reference pattern."
        )
    return ReferenceKeywordModel(chosen, channels=config.channels)


def _build_openwakeword(config: DeviceConfig) -> WakeModel:
    if config.min_positive_ms > config.frame_ms:
        raise ConfigurationError(
            "openWakeWord posteriors are brief. Set min_positive_ms to one frame "
            f"({config.frame_ms} ms) for this model."
        )
    try:
        from openwakeword.model import Model
    except ImportError as exc:
        raise AudioBackendError(
            "openWakeWord is not installed. Install the 'wake' extra to use "
            "a spoken wake-word model."
        ) from exc
    runner = Model(wakeword_models=[config.wake_word])
    from hardware_voice_ai.wake.adapters import OpenWakeWordModel

    return OpenWakeWordModel(
        config.wake_word,
        runner,
        channels=config.channels,
        sample_rate_hz=config.sample_rate_hz,
    )
