"""Selectable remote speech-to-speech adapters.

``local`` and ``remote`` stay the only Audio Model labels. These ids choose
which adapter the gateway runs when the setting is ``remote``. The local
stand-in is not one of them.
"""

from hardware_voice_ai.errors import ConfigurationError

LOCAL_STAND_IN_ID = "reference-audio-v1"
DEFAULT_REMOTE_BACKEND = "moshi"

# Canonical id, in the order a user should see them. Moshi is first because
# it is the default.
REMOTE_BACKENDS = ("moshi", "minicpm-o-4.5", "qwen3-omni")

_ALIASES = {
    "moshi": "moshi",
    "minicpm-o-4.5": "minicpm-o-4.5",
    "minicpm-o": "minicpm-o-4.5",
    "minicpm": "minicpm-o-4.5",
    "qwen3-omni": "qwen3-omni",
    "qwen3-omni-instruct": "qwen3-omni",
}


def parse_remote_backend(value: str) -> str:
    """Return the canonical id for a user-facing backend name."""
    key = value.strip().lower()
    if key == LOCAL_STAND_IN_ID:
        raise ConfigurationError(
            "reference-audio-v1 is the local Audio Model stand-in. "
            "It is not a remote backend."
        )
    try:
        return _ALIASES[key]
    except KeyError as exc:
        raise ConfigurationError(
            "Remote backend must be moshi, minicpm-o-4.5, or qwen3-omni."
        ) from exc
