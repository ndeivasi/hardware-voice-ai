"""Configuration, microphone, and audio-model errors."""


class ConfigurationError(ValueError):
    """The device configuration cannot be used as given."""


class AudioBackendError(RuntimeError):
    """The selected microphone or speaker backend cannot be opened."""


class ModelUnavailable(RuntimeError):
    """The selected Audio Model cannot take this turn.

    The message is safe to show: it does not include credentials or audio.
    """
