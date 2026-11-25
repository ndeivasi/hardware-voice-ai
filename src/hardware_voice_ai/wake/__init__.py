"""On-device wake-word types.

Import ``WakeEngine`` from ``hardware_voice_ai.wake.engine`` and
``build_wake_model`` from ``hardware_voice_ai.wake.factory``. Those modules
depend on device configuration, and importing them here would cycle.
"""

from hardware_voice_ai.wake.defaults import (
    DEFAULT_WAKE_WORD,
    OPENWAKEWORD_MODEL_ID,
    REFERENCE_MODEL_ID,
)
from hardware_voice_ai.wake.events import WakeEvent
from hardware_voice_ai.wake.model import WakeModel, WakeScore

__all__ = [
    "DEFAULT_WAKE_WORD",
    "OPENWAKEWORD_MODEL_ID",
    "REFERENCE_MODEL_ID",
    "WakeEvent",
    "WakeModel",
    "WakeScore",
]
