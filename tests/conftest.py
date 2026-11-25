"""Fixtures shared by the on-device tests. No microphone and no network."""

import sys
from collections.abc import Iterator

import pytest


@pytest.fixture
def hold_thread_switch() -> Iterator[None]:
    """Stretch the GIL switch interval for the duration of one test.

    Python 3.12 usually preempts a tight loop and runs the playback thread.
    Python 3.13 does not, while that loop stays inside one switch interval.
    A cue then stays "playing" and a reply stays unqueued only because the
    thread has not been scheduled. Holding the interval makes 3.12 follow
    that schedule.
    """
    interval = sys.getswitchinterval()
    sys.setswitchinterval(30)
    try:
        yield
    finally:
        sys.setswitchinterval(interval)
