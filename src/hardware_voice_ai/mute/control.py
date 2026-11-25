"""Hardware mute. The physical control is authoritative."""

from collections.abc import Callable
from typing import Protocol


MuteListener = Callable[[bool], None]


class MuteControl(Protocol):
    """Authoritative microphone mute.

    A later board driver can implement this with a switch or GPIO line.
    ``True`` means the microphone is muted and the device must return to idle.
    While muted, always-on audio is dropped and is not scored.
    """

    def is_muted(self) -> bool: ...

    def add_listener(self, listener: MuteListener) -> None: ...


class InMemoryMuteControl:
    """Test and development stand-in for the physical mute control."""

    def __init__(self, muted: bool = False) -> None:
        self._muted = muted
        self._listeners: list[MuteListener] = []

    def is_muted(self) -> bool:
        return self._muted

    def add_listener(self, listener: MuteListener) -> None:
        self._listeners.append(listener)

    def set_muted(self, muted: bool) -> None:
        if muted == self._muted:
            return
        self._muted = muted
        for listener in tuple(self._listeners):
            listener(muted)
