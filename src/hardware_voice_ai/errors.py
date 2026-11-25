"""Configuration and audio-backend errors."""


class ConfigurationError(ValueError):
    """The device configuration cannot be used as given."""


class AudioBackendError(RuntimeError):
    """The selected microphone backend cannot be opened."""
