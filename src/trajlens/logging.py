"""Structured logging configuration for trajlens.

Call configure_logging() once at process startup (the CLI does this).
All internal code uses structlog.get_logger() — never print(), never the
stdlib logging module directly.

Secret redaction is enforced here: any log record whose key matches the
redaction pattern has its value replaced with '<redacted>'. This is the
mitigation for T6 (secret leakage) in the threat model.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import structlog

# Keys whose values must never appear in logs, regardless of level.
# Matches: token, secret, password, authorization, bearer, api_key (case-insensitive).
_SECRET_KEY_PATTERN: re.Pattern[str] = re.compile(
    r"token|secret|password|authorization|bearer|api_key",
    re.IGNORECASE,
)

_REDACTED: str = "<redacted>"


def _redact_secrets(
    logger: Any,
    method: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Structlog processor that redacts secret values in log records.

    Operates on the event_dict in-place before rendering. Any key whose name
    matches _SECRET_KEY_PATTERN has its value replaced with '<redacted>'.
    This is unconditional — it runs at every log level.
    """
    for key in list(event_dict.keys()):
        if _SECRET_KEY_PATTERN.search(key):
            event_dict[key] = _REDACTED
    return event_dict


def configure_logging(*, level: str = "WARNING", force_plain: bool = False) -> None:
    """Configure structlog for the trajlens process.

    Args:
        level: Minimum log level to emit (DEBUG, INFO, WARNING, ERROR).
               CLI default is WARNING so normal runs are quiet.
        force_plain: If True, use plain text output even in a TTY.
                     Used in tests to keep output deterministic.
    """
    log_level = getattr(logging, level.upper(), logging.WARNING)

    # Configure stdlib logging as the backend (structlog wraps it)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
    )

    use_colors = sys.stderr.isatty() and not force_plain

    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _redact_secrets,  # must run before any renderer
        structlog.processors.StackInfoRenderer(),
    ]

    if use_colors:
        renderer: Any = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *shared_processors,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)
