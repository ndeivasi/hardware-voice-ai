"""On-device wake word, capture, speaker playback, and one Audio Model.

Always-on microphone audio stays in this process. A detection notifies the
session orchestrator. After capture, the Audio Model (``local`` or
``remote``) produces the spoken reply. Playback and local cues stay on the
device speaker. Speech-to-text and text-to-speech are not separate stages.
"""

__version__ = "0.1.0"
