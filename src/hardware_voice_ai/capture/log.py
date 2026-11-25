"""Opt-in local metadata for capture. Raw audio is never written."""

from pathlib import Path


class CaptureLog:
    """Append-only text log of capture events.

    The log stays off unless a local file path is provided. Each line is one
    event (start, stop, and the endpoint cause). Sample values are not accepted
    and are not formatted. A path that looks like a URL is rejected so this
    cannot be pointed at a network location by accident.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None or path == "":
            self._path: Path | None = None
            return
        text = str(path)
        if "://" in text:
            raise ValueError("Capture logs must be a local file path.")
        self._path = Path(text)

    @property
    def enabled(self) -> bool:
        return self._path is not None

    @property
    def path(self) -> Path | None:
        return self._path

    def write(self, message: str) -> None:
        if self._path is None:
            return
        if "\n" in message or "\r" in message:
            raise ValueError("Capture log entries must be a single line.")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
