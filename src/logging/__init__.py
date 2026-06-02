"""
logging — Logs pick-and-place experiment results.

Note: this package is named `logging` but does NOT shadow Python's standard
`logging` — other modules doing `import logging` still get the stdlib (absolute import).

Public API:
    TrialLogger, FIELDNAMES
"""
from .logger import FIELDNAMES, TrialLogger

__all__ = ["TrialLogger", "FIELDNAMES"]
