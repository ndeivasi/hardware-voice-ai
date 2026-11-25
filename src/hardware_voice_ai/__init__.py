"""On-device wake word, microphone capture, and speaker playback.

Always-on microphone audio stays in this process. A detection notifies the
session orchestrator. Spoken replies and local cues play on the device
speaker. This package does not run an Audio Model and it does not open a
network connection.
"""

__version__ = "0.1.0"
