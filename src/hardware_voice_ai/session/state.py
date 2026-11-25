"""Device session states from the architecture baseline."""

from enum import StrEnum


class DeviceState(StrEnum):
    """Session states.

    The architecture chain is ``idle → woken → capturing → speaking → idle``.
    ``transcribing`` and ``thinking`` remain in the table from an earlier
    draft until a follow-up removes them. Speaker playback enters
    ``speaking`` from ``capturing`` after the microphone tap has closed.
    Cancel, stop, barge-in, and hardware mute return to ``idle``.
    """

    IDLE = "idle"
    WOKEN = "woken"
    CAPTURING = "capturing"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"


# Speaker playback enters speaking from capturing once the tap is closed.
# transcribing and thinking stay until the enum matches the shorter chain.
ALLOWED_TRANSITIONS: dict[DeviceState, frozenset[DeviceState]] = {
    DeviceState.IDLE: frozenset({DeviceState.WOKEN}),
    DeviceState.WOKEN: frozenset({DeviceState.CAPTURING, DeviceState.IDLE}),
    DeviceState.CAPTURING: frozenset(
        {DeviceState.TRANSCRIBING, DeviceState.SPEAKING, DeviceState.IDLE}
    ),
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
