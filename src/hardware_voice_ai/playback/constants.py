"""Playback timing budget from the architecture baseline.

These are design goals, not measurements on a board. The hardware queue is
capped below the stop budget so a stop or barge-in can quiet the speaker
without waiting for the rest of the reply.
"""

STOP_LATENCY_BUDGET_MS = 200
DEFAULT_QUEUE_MS = 80
