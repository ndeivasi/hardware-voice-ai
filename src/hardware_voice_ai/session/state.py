"""Device session states from the architecture baseline."""

from enum import StrEnum


class DeviceState(StrEnum):
    """``idle → woken → capturing → transcribing → thinking → speaking → idle``.

    Cancel and hardware mute return to ``idle`` from any later state.
    """

    IDLE = "idle"
    WOKEN = "woken"
    CAPTURING = "capturing"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"


# Later issues enter transcribing, thinking, and speaking. The wake path
# uses idle → woken → capturing, then idle on cancel or mute.
ALLOWED_TRANSITIONS: dict[DeviceState, frozenset[DeviceState]] = {
    DeviceState.IDLE: frozenset({DeviceState.WOKEN}),
    DeviceState.WOKEN: frozenset({DeviceState.CAPTURING, DeviceState.IDLE}),
    DeviceState.CAPTURING: frozenset({DeviceState.TRANSCRIBING, DeviceState.IDLE}),
    DeviceState.TRANSCRIBING: frozenset({DeviceState.THINKING, DeviceState.IDLE}),
    DeviceState.THINKING: frozenset({DeviceState.SPEAKING, DeviceState.IDLE}),
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
