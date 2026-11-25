"""Device session states from the architecture baseline."""

from enum import StrEnum


class DeviceState(StrEnum):
    """Session states.

    The session chain is ``idle → woken → capturing → speaking → idle``.
    Speaker playback enters ``speaking`` from ``capturing`` after the
    microphone tap has closed. Cancel, stop, barge-in, and hardware mute
    return to ``idle``.
    """

    IDLE = "idle"
    WOKEN = "woken"
    CAPTURING = "capturing"
    SPEAKING = "speaking"


# Speaker playback enters speaking from capturing once the tap is closed.
ALLOWED_TRANSITIONS: dict[DeviceState, frozenset[DeviceState]] = {
    DeviceState.IDLE: frozenset({DeviceState.WOKEN}),
    DeviceState.WOKEN: frozenset({DeviceState.CAPTURING, DeviceState.IDLE}),
    DeviceState.CAPTURING: frozenset({DeviceState.SPEAKING, DeviceState.IDLE}),
    DeviceState.SPEAKING: frozenset({DeviceState.IDLE}),
}


class IllegalTransitionError(RuntimeError):
    """A session move that the device state machine does not allow."""


def assert_transition(current: DeviceState, new: DeviceState) -> None:
    allowed = ALLOWED_TRANSITIONS[current]
    if new not in allowed:
        raise IllegalTransitionError(
            f"Cannot move from {current.value} to {new.value}."
        )
