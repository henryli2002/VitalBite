import logging
import sys
import os
import json
import time
import uuid
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in (
                "msg",
                "args",
                "exc_info",
                "exc_text",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "name",
                "stack_info",
            ):
                log_data[key] = value

        return json.dumps(log_data)


class PlainFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(
            "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s"
        )

    def format(self, record: logging.LogRecord) -> str:
        record.request_id = request_id_var.get() or "N/A"
        return super().format(record)


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(PlainFormatter())
    logger.addHandler(console_handler)

    log_dir = os.getenv("LOG_DIR", "./logs")
    log_max_size = int(os.getenv("LOG_MAX_SIZE", "10")) * 1024 * 1024
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    log_format_str = os.getenv("LOG_FORMAT", "json")

    if os.getenv("LOG_TO_FILE", "true").lower() == "true":
        try:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = RotatingFileHandler(
                os.path.join(log_dir, f"{name}.log"),
                maxBytes=log_max_size,
                backupCount=log_backup_count,
            )
            file_handler.setLevel(log_level)

            if log_format_str == "json":
                file_handler.setFormatter(JsonFormatter())
            else:
                file_handler.setFormatter(PlainFormatter())

            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to setup file handler: {e}")

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    return setup_logger(name)


def set_request_id(request_id: Optional[str] = None) -> str:
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    request_id_var.set(request_id)
    return request_id


def clear_request_id() -> None:
    request_id_var.set(None)


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    extra: Optional[Dict[str, Any]] = None,
):
    extra = extra or {}
    extra["request_id"] = request_id_var.get()
    logger.log(level, message, extra=extra)


logger = setup_logger("wabi_agent")


def log_trace(
    node_name: str,
    provider: str,
    model_name: str,
    latency_ms: float,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    status: str = "success",
    error_msg: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
):
    trace_data = {
        "event": "llm_trace",
        "request_id": request_id_var.get(),
        "node": node_name,
        "provider": provider,
        "model": model_name,
        "latency_ms": round(latency_ms, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": (input_tokens or 0) + (output_tokens or 0)
        if input_tokens or output_tokens
        else None,
        "status": status,
        "timestamp": time.time(),
    }

    if error_msg:
        trace_data["error"] = error_msg

    if extra_meta:
        trace_data.update(extra_meta)

    logger.info(json.dumps(trace_data))
