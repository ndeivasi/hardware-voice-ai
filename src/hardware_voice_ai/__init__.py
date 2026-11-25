"""On-device wake-word detection for the hardware voice assistant.

Always-on microphone audio stays in this process. A detection notifies the
session orchestrator. It does not open a network connection and it does not
hand audio to a separate speech-to-text stage.
"""

__version__ = "0.1.0"
