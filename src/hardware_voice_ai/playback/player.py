"""Play spoken replies and local cues on a swappable speaker.

The session orchestrator calls ``play_reply``, ``play_cue``, ``set_volume``,
``stop``, and ``status``. A reply stream can start before the rest of the
audio has arrived. Stop, barge-in, and hardware mute discard the queued
tail instead of playing it out.

This module does not run an Audio Model and does not open a network connection.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from hardware_voice_ai.errors import ConfigurationError
from hardware_voice_ai.playback.cues import Cue, render_cue
from hardware_voice_ai.playback.pcm import apply_volume
from hardware_voice_ai.playback.source import MemoryReplyAudio, ReplyAudio
from hardware_voice_ai.playback.speaker import Speaker


class PlaybackStatus(StrEnum):
    """Whether the current sound is still going, finished, or was cut."""

    IDLE = "idle"
    PLAYING = "playing"
    COMPLETED = "completed"
    STOPPED = "stopped"


class PlaybackStopReason(StrEnum):
    """Why playback is not currently running, or ``none`` while it is."""

    NONE = "none"
    COMPLETED = "completed"
    STOP = "stop"
    BARGE_IN = "barge_in"
    MUTE = "mute"
    REPLACED = "replaced"


@dataclass(frozen=True, slots=True)
class PlaybackResult:
    """Completion or stop outcome for one sound.

    ``samples_played`` have left the hardware queue. ``samples_dropped``
    were still queued when playback was cut and were not played.
    ``completed`` means the reply or cue finished. ``stopped`` means it
    was cut. ``playing`` is the in-progress view from ``status``.
    """

    status: PlaybackStatus
    reason: PlaybackStopReason
    samples_played: int
    samples_dropped: int


FinishedCallback = Callable[[PlaybackResult], None]


class SpeakerPlayback:
    """Local playback engine behind the session orchestrator.

    One sound plays at a time. A new reply or cue stops the previous one.
    Volume applies to the next block written, including a reply already
    in progress. The speaker's queue, not a stored file, is what a stop
    discards.
    """

    def __init__(self, speaker: Speaker, *, volume: float = 1.0) -> None:
        self.speaker = speaker
        self._volume = _require_volume(volume)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._suppressed = False
        self._thread: threading.Thread | None = None
        self._source: ReplyAudio | None = None
        self._on_finished: object | None = None
        self._status = PlaybackStatus.IDLE
        self._reason = PlaybackStopReason.NONE
        self._error: BaseException | None = None
        self._played_base = 0
        self._dropped_base = 0
        self._cue = False
        self._result = PlaybackResult(
            status=PlaybackStatus.IDLE,
            reason=PlaybackStopReason.NONE,
            samples_played=0,
            samples_dropped=0,
        )

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._status is PlaybackStatus.PLAYING

    @property
    def is_cue_playing(self) -> bool:
        """True while a local wake, offline, or error cue is still going.

        This stays true through ``flush``, until the speaker queue is quiet
        or a stop discards it. A spoken reply does not set it.
        """
        with self._lock:
            return self._status is PlaybackStatus.PLAYING and self._cue

    @property
    def volume(self) -> float:
        with self._lock:
            return self._volume

    def play_reply(
        self,
        source: ReplyAudio,
        on_finished: FinishedCallback | None = None,
    ) -> PlaybackResult:
        """Start ``source`` now. Later chunks are played as they arrive.

        Returns the current status. A short reply may already be
        ``completed`` by the time this returns. ``on_finished`` is called
        from the playback thread with the final ``PlaybackResult`` and
        must not call back into ``play_reply`` on this same object.
        """
        self._require_source(source)
        self.stop(PlaybackStopReason.REPLACED)
        return self._start(source, on_finished, cue=False)

    def play_cue(self, cue: Cue) -> PlaybackResult:
        """Play one local cue. The samples are rendered here, not fetched."""
        if not isinstance(cue, Cue):
            raise ConfigurationError("cue must be wake, offline, or error.")
        rendered = render_cue(cue, self.speaker.format)
        source = MemoryReplyAudio(
            rendered,
            sample_rate_hz=self.speaker.format.sample_rate_hz,
            channels=self.speaker.format.channels,
        )
        self.stop(PlaybackStopReason.REPLACED)
        return self._start(source, None, cue=True)

    def set_suppressed(self, suppressed: bool) -> None:
        """Refuse new playback while hardware mute is held.

        The orchestrator also stops the current sound. This flag stops a
        start that races with that mute.
        """
        with self._lock:
            self._suppressed = suppressed

    def set_volume(self, volume: float) -> None:
        """Set gain in ``[0, 1]``. ``0`` is silence. ``1`` is unity gain."""
        checked = _require_volume(volume)
        with self._lock:
            self._volume = checked

    def stop(self, reason: PlaybackStopReason = PlaybackStopReason.STOP) -> PlaybackResult:
        """Cut playback and discard audio still in the speaker queue.

        The queued tail is the only audio that can still be stopped. Its
        duration is capped by the speaker queue, which is under the
        200 ms design goal. Already-played samples stay played.
        """
        if reason is PlaybackStopReason.NONE or reason is PlaybackStopReason.COMPLETED:
            raise ConfigurationError("stop needs a stop reason.")
        thread: threading.Thread | None = None
        source: ReplyAudio | None = None
        with self._lock:
            if self._status is not PlaybackStatus.PLAYING:
                return self._result
            self._stop.set()
            self._reason = reason
            thread = self._thread
            source = self._source
        self.speaker.drop_queued()
        _interrupt(source)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._lock:
            if self._status is PlaybackStatus.PLAYING:
                self.speaker.drop_queued()
                self._mark(PlaybackStatus.STOPPED, reason)
            return self._result

    def status(self) -> PlaybackResult:
        """Current outcome. ``playing`` until the sound completes or stops."""
        with self._lock:
            if self._status is PlaybackStatus.PLAYING:
                return self._counts(PlaybackStatus.PLAYING, PlaybackStopReason.NONE)
            return self._result

    def wait(self, timeout: float = 2.0) -> PlaybackResult:
        """Block until the current sound's thread exits."""
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError("playback did not finish")
        with self._lock:
            error = self._error
            if self._status is PlaybackStatus.PLAYING:
                result = self._counts(PlaybackStatus.PLAYING, PlaybackStopReason.NONE)
            else:
                result = self._result
        if error is not None:
            raise error
        return result

    def close(self) -> None:
        self.stop(PlaybackStopReason.STOP)
        self.speaker.close()

    def _start(
        self,
        source: ReplyAudio,
        on_finished: FinishedCallback | None,
        *,
        cue: bool,
    ) -> PlaybackResult:
        with self._lock:
            if self._suppressed:
                return muted_result()
            self._stop.clear()
            self._error = None
            self._cue = cue
            self._source = source
            self._on_finished = on_finished
            self._played_base = self.speaker.played_samples
            self._dropped_base = self.speaker.dropped_samples
            self._status = PlaybackStatus.PLAYING
            self._reason = PlaybackStopReason.NONE
            self._result = self._counts(PlaybackStatus.PLAYING, PlaybackStopReason.NONE)
            thread = threading.Thread(
                target=self._run,
                name="speaker-playback",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        return self.status()

    def _run(self) -> None:
        error: BaseException | None = None
        try:
            frame = self.speaker.format.frame_samples
            while not self._stop.is_set():
                source = self._source
                if source is None:
                    break
                chunk = source.read(frame)
                if chunk is None:
                    break
                if len(chunk) == 0:
                    continue
                gained = apply_volume(chunk, self.volume)
                if self._stop.is_set():
                    break
                self.speaker.write(gained)
        except Exception as exc:
            error = exc
        # Flush outside the player lock. Stop and mute must still be able to
        # drop the queue while this wait runs, and status stays PLAYING until
        # the speaker is quiet or the tail is discarded.
        if error is not None or self._stop.is_set():
            self.speaker.drop_queued()
        else:
            self.speaker.flush()
        callback: FinishedCallback | None = None
        result: PlaybackResult
        with self._lock:
            if error is not None:
                self._error = error
            if self._status is PlaybackStatus.PLAYING:
                if error is not None or self._stop.is_set():
                    self.speaker.drop_queued()
                    reason = (
                        self._reason if self._stop.is_set() else PlaybackStopReason.STOP
                    )
                    self._mark(PlaybackStatus.STOPPED, reason)
                else:
                    self._mark(PlaybackStatus.COMPLETED, PlaybackStopReason.COMPLETED)
            result = self._result
            finished = self._on_finished
            self._on_finished = None
            if finished is not None:
                callback = finished
        if callback is not None:
            callback(result)

    def _require_source(self, source: ReplyAudio) -> None:
        fmt = self.speaker.format
        if source.sample_rate_hz != fmt.sample_rate_hz or source.channels != fmt.channels:
            raise ConfigurationError(
                "Reply audio format does not match the speaker "
                f"({fmt.sample_rate_hz} Hz, {fmt.channels} channels)."
            )

    def _mark(self, status: PlaybackStatus, reason: PlaybackStopReason) -> None:
        self._status = status
        self._reason = reason
        self._result = self._counts(status, reason)

    def _counts(self, status: PlaybackStatus, reason: PlaybackStopReason) -> PlaybackResult:
        played = self.speaker.played_samples - self._played_base
        dropped = self.speaker.dropped_samples - self._dropped_base
        return PlaybackResult(
            status=status,
            reason=reason,
            samples_played=max(0, played),
            samples_dropped=max(0, dropped),
        )


def _require_volume(volume: float) -> float:
    if isinstance(volume, bool) or not isinstance(volume, (int, float)):
        raise ConfigurationError("volume must be in [0, 1].")
    value = float(volume)
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError("volume must be in [0, 1].")
    return value


def _interrupt(source: ReplyAudio | None) -> None:
    if source is None:
        return
    source.interrupt()


def muted_result() -> PlaybackResult:
    """Status used when hardware mute refuses playback."""
    return PlaybackResult(
        status=PlaybackStatus.STOPPED,
        reason=PlaybackStopReason.MUTE,
        samples_played=0,
        samples_dropped=0,
    )

