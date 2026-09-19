"""Logging setup: readable console output by default, or one-JSON-object-per-line when
`json_format=True` (or the `ROXROOM_JSON_LOGS` env var is set) -- structured logs with
whatever `extra={...}` fields a call site attaches (trace_id, participant_id, per-stage
latencies, ...), for the STT-final -> gate -> route -> LLM first token -> TTS first
byte -> audio published trace this project's M8 milestone is built around (see
obs/trace.py). JSON mode is opt-in rather than default so interactive dev use
(`run_bot.py`, `run_transcriber.py`) keeps human-readable console output.
"""
from __future__ import annotations

import json
import logging
import os

_STANDARD_LOG_RECORD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime"}


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": round(record.created, 6),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_RECORD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: int = logging.INFO, json_format: bool | None = None) -> None:
    if json_format is None:
        json_format = bool(os.environ.get("ROXROOM_JSON_LOGS", "").strip())

    handler = logging.StreamHandler()
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
