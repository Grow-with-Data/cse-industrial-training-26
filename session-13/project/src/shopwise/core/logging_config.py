"""Logging for the server. New in v10.

    from shopwise.core.logging_config import setup_logging, get_logger
    setup_logging()                       # once, at startup (main.py's lifespan)
    log = get_logger(__name__)            # in any module
    log.info("thread created", extra={"thread_id": tid})

WHY NOT print()
    A terminal script prints; a server logs. A log line has a time, a level and
    the module it came from, and the server decides where it goes (the terminal
    today, journald on the session 16 server). print() has none of that.

THE REQUEST ID
    Every request gets an id (api/middleware.py). It is stored in a context
    variable for the life of the request, and RequestIdFilter stamps it on every
    log line written while that request runs. When a user reports "it failed",
    the problem-details response they saw carries the same id: search the log
    for it and you have every line that request wrote, and nothing else.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar

from shopwise.core.config import settings

# "-" outside a request (startup, shutdown, background work).
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

FORMAT = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"


class RequestIdFilter(logging.Filter):
    """Adds `request_id` to every record, so FORMAT can print it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger once. Safe to call twice (the reloader does)."""
    root = logging.getLogger()
    if any(isinstance(f, RequestIdFilter) for h in root.handlers for f in h.filters):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(FORMAT))
    handler.addFilter(RequestIdFilter())
    root.addHandler(handler)
    root.setLevel(level or settings.LOG_LEVEL)
    # Libraries that are chatty at INFO: one line per HTTP call to the model.
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
