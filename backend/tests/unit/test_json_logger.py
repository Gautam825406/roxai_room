import json
import logging

from roxroom.obs.json_logger import JSONFormatter


def _make_record(msg: str = "hello", extra: dict | None = None) -> logging.LogRecord:
    logger = logging.getLogger("test.json_logger")
    record = logger.makeRecord(
        logger.name, logging.INFO, __file__, 1, msg, (), None, extra=extra or {}
    )
    return record


def test_formats_valid_json():
    formatter = JSONFormatter()
    line = formatter.format(_make_record("hello world"))
    data = json.loads(line)  # must not raise
    assert data["message"] == "hello world"
    assert data["level"] == "INFO"
    assert data["logger"] == "test.json_logger"
    assert "ts" in data


def test_extra_fields_are_included():
    formatter = JSONFormatter()
    line = formatter.format(_make_record("latency breakdown", extra={"trace_id": "abc123", "total_ms": 842.5}))
    data = json.loads(line)
    assert data["trace_id"] == "abc123"
    assert data["total_ms"] == 842.5


def test_standard_log_record_attrs_are_not_duplicated_as_noise():
    formatter = JSONFormatter()
    line = formatter.format(_make_record("hello"))
    data = json.loads(line)
    # internal stdlib bookkeeping fields should not leak into the JSON payload
    for noisy_key in ("msg", "args", "levelno", "pathname", "exc_text", "stack_info"):
        assert noisy_key not in data


def test_exception_info_is_included_when_present():
    formatter = JSONFormatter()
    logger = logging.getLogger("test.json_logger")
    try:
        raise ValueError("boom")
    except ValueError:
        record = logger.makeRecord(
            logger.name, logging.ERROR, __file__, 1, "failed", (), __import__("sys").exc_info()
        )
    data = json.loads(formatter.format(record))
    assert "ValueError" in data["exc_info"]
    assert "boom" in data["exc_info"]
