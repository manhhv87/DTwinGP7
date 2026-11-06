"""
blind_console.py
────────────────
What the console of a blinded run shows: only lines that read the same in every arm.

In a blinded campaign the person who places the parts also scores the outcome, so
the console must not tell the arms apart. Session 1 of E2 (e2chuan-1, 23 Oct 2026)
kept warnings and errors on the console, and 14 lines of the form "pick refused
(depth_geometry_reject): commanded top face X mm, depth sees Y mm" appeared there,
all in the four blocks of the plane arm: a command that did not come from depth,
which reads as the plane mode.

Under install_blind_console the console handlers show a log record only when

  * it carries the extra "console": a string replaces the message on the console
    (the log file keeps the message), True shows the message as it is. The code
    marks this way the placement prompts, the controller and servo safety lines,
    and every refusal before motion, as one neutral line per trial;
  * or it is a warning or worse from the controller side (HARDWARE_LOGGERS), which
    knows nothing of the depth mode, the model or the outcome of a trial.

Everything else, refusal reasons, commanded heights, depth readings, the depth mode
itself, reaches the log file only. Lines printed with print() and the input()
prompts are not log records and are not filtered.
"""
from __future__ import annotations

import logging

HARDWARE_LOGGERS = (
    "src.orchestrator.backends",
    "src.orchestrator.digital_twin",
    "src.orchestrator.telemetry",
)


class BlindConsoleFilter(logging.Filter):
    """Pass a record marked for the console, or a controller-side warning."""

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "console", None):
            return True
        return (record.levelno >= logging.WARNING
                and record.name.startswith(HARDWARE_LOGGERS))


class BlindConsoleFormatter(logging.Formatter):
    """The replacement text of a marked record, else 'LEVEL: message'."""

    def format(self, record: logging.LogRecord) -> str:
        text = getattr(record, "console", None)
        if isinstance(text, str):
            return text
        if text:
            return record.getMessage()
        return f"{record.levelname}: {record.getMessage()}"


def console_handlers(root: logging.Logger | None = None) -> list[logging.Handler]:
    """The stream handlers of a logger that write to a console, not to a file."""
    root = root or logging.getLogger()
    return [h for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)]


def install_blind_console(root: logging.Logger | None = None) -> int:
    """Put the console handlers of ``root`` (default the root logger) on the blinded
    filter and formatter. File handlers are left as they are. Returns the number of
    handlers changed."""
    handlers = console_handlers(root)
    for h in handlers:
        h.setLevel(logging.INFO)
        h.addFilter(BlindConsoleFilter())
        h.setFormatter(BlindConsoleFormatter())
    return len(handlers)
