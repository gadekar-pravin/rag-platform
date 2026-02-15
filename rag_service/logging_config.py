"""Structured JSON logging for Cloud Run compatibility.

Configures python-json-logger for GCP Cloud Logging severity mapping
and request correlation via trace IDs.
"""

from __future__ import annotations

import logging
import os
import uuid

from pythonjsonlogger.json import JsonFormatter

# GCP severity mapping: Python log levels -> Cloud Logging severity strings
_GCP_SEVERITY = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}


class GCPJsonFormatter(JsonFormatter):
    """JSON formatter that maps Python log levels to GCP severity."""

    def add_fields(
        self,
        log_record: dict[str, object],
        record: logging.LogRecord,
        message_dict: dict[str, object],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["severity"] = _GCP_SEVERITY.get(record.levelname, record.levelname)
        log_record.pop("levelname", None)


def setup_logging(*, level: str = "INFO") -> None:
    """Configure structured JSON logging when on Cloud Run, plain text locally."""
    is_cloud_run = bool(os.getenv("K_SERVICE"))

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler()
    if is_cloud_run:
        handler.setFormatter(GCPJsonFormatter(
            fmt="%(message)s %(name)s %(funcName)s %(lineno)d",
            rename_fields={"message": "message", "name": "logger"},
        ))
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d  %(message)s",
            datefmt="%H:%M:%S",
        ))

    root.addHandler(handler)


def generate_request_id() -> str:
    """Generate a unique request ID for trace correlation."""
    return uuid.uuid4().hex[:16]
