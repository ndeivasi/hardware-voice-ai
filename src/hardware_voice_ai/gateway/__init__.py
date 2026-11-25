"""GPU-side gateway for the remote Audio Model.

The device still has one Audio Model setting, ``local`` or ``remote``.
This package is the other end of that remote session. It runs one
speech-to-speech adapter. It is not a speech-to-text stage, a separate
language-model call, or a text-to-speech stage.

Importing this package does not open a socket and does not load a model.
"""

from hardware_voice_ai.gateway.catalog import (
    DEFAULT_REMOTE_BACKEND,
    LOCAL_STAND_IN_ID,
    REMOTE_BACKENDS,
    parse_remote_backend,
)

__all__ = [
    "DEFAULT_REMOTE_BACKEND",
    "LOCAL_STAND_IN_ID",
    "REMOTE_BACKENDS",
    "parse_remote_backend",
]
