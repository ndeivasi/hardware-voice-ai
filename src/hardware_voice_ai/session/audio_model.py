"""The device's only model setting: Audio Model."""

from enum import StrEnum

from hardware_voice_ai.errors import ConfigurationError


class AudioModel(StrEnum):
    """Where the turn's audio model runs after an accepted wake.

    ``remote`` is the default. ``local`` keeps that turn on the device.
    These are the only labels. There is no separate speech-to-text or
    text-to-speech setting.
    """

    LOCAL = "local"
    REMOTE = "remote"


def parse_audio_model(value: str) -> AudioModel:
    try:
        return AudioModel(value.strip())
    except ValueError as exc:
        raise ConfigurationError(
            "Audio Model must be 'local' or 'remote'."
        ) from exc
