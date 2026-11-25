"""Configuration, microphone, and audio-model errors."""


class ConfigurationError(ValueError):
    """The device configuration cannot be used as given."""


class AudioBackendError(RuntimeError):
    """The selected microphone or speaker backend cannot be opened."""


class ModelUnavailable(RuntimeError):
    """The selected Audio Model cannot take this turn.

    The message is safe to show: it does not include credentials or audio.
    """


class RemoteBackendError(RuntimeError):
    """The selected remote speech-to-speech adapter cannot run this turn.

    The wake loop does not raise this. The GPU gateway does, when the
    adapter is missing, needs a GPU, or rejects its configuration.
    """
