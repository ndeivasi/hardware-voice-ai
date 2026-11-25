"""Stand-in ids for the Audio Model. Not a cloud vendor."""

REFERENCE_LOCAL_MODEL_ID = "reference-audio-v1"

# Failure thresholds, not the architecture's 2-second first-audio goal.
# A hung model must give up; a slow but live reply should still be able to start.
DEFAULT_FIRST_AUDIO_TIMEOUT_S = 8.0
DEFAULT_TURN_TIMEOUT_S = 30.0

# Safety cap so a bad handoff cannot stream a long buffer off the device.
MAX_UTTERANCE_S = 30.0
