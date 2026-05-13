"""JSON Lines telemetry logger."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from pupil_tracker.logging_config import get_logger

_LOGGER = get_logger("telemetry")


class JsonlLogger:
    """Write non-video telemetry events as one JSON object per line."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")
        _LOGGER.debug("opened JSONL telemetry log at %s", path)

    def write_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        """Write a JSON-serializable telemetry event."""

        event = {
            "event_type": event_type,
            "timestamp": time.time(),
            "payload": dict(payload),
        }
        line = self._serialize_event(event)
        self._file.write(f"{line}\n")
        self._file.flush()

    def close(self) -> None:
        """Close the underlying log file."""

        if not self._file.closed:
            self._file.close()
            _LOGGER.debug("closed JSONL telemetry log at %s", self._path)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @staticmethod
    def _serialize_event(event: Mapping[str, Any]) -> str:
        try:
            return json.dumps(event, sort_keys=True)
        except TypeError as error:
            msg = "telemetry event payload must be JSON serializable"
            raise TypeError(msg) from error
