"""On-device speaker playback.

The session orchestrator plays spoken-reply audio and local cues through
``SpeakerPlayback``. The speaker itself is swappable. This package does
not run the Audio Model and does not open a cloud session.
"""

from hardware_voice_ai.playback.constants import DEFAULT_QUEUE_MS, STOP_LATENCY_BUDGET_MS
from hardware_voice_ai.playback.cues import Cue, render_cue
from hardware_voice_ai.playback.player import (
    PlaybackResult,
    PlaybackStatus,
    PlaybackStopReason,
    SpeakerPlayback,
)
from hardware_voice_ai.playback.source import MemoryReplyAudio, ReplyAudio, StubReplyAudio
from hardware_voice_ai.playback.speaker import (
    MemorySpeaker,
    NullSpeaker,
    SoundDeviceSpeaker,
    Speaker,
    build_speaker,
)

__all__ = [
    "DEFAULT_QUEUE_MS",
    "STOP_LATENCY_BUDGET_MS",
    "Cue",
    "MemoryReplyAudio",
    "MemorySpeaker",
    "NullSpeaker",
    "PlaybackResult",
    "PlaybackStatus",
    "PlaybackStopReason",
    "ReplyAudio",
    "SoundDeviceSpeaker",
    "Speaker",
    "SpeakerPlayback",
    "StubReplyAudio",
    "build_speaker",
    "render_cue",
]
