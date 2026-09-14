"""Structured JSON logging.

Decision -> Reason -> Alternative -> Trade-off
- Decision: one-line JSON logs to stdout, request_id injected via contextvars.
- Reason: containers collect stdout; JSON is queryable in Application Insights /
  Loki / CloudWatch without regex parsing. contextvars means the request id does
  not have to be threaded through every function signature.
- Alternative: plain text logs + a parsing rule in the collector.
- Trade-off: JSON is harder to read by eye locally; `jq` fixes that.

PII rule enforced here: `log_event` accepts explicit fields only. The customer's
raw question is never logged, only its length and a non-reversible hash, so that
duplicate traffic can still be spotted in aggregate.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "thread", "threadName", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # uvicorn keeps its own handlers otherwise, producing double / unstructured lines
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = [handler]
        lg.propagate = False


def text_fingerprint(text: str) -> str:
    """Non-reversible short hash. Lets us count repeats without storing the text."""
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:12]
