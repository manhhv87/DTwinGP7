"""
logging — Ghi log kết quả thí nghiệm pick-and-place.

Lưu ý: package này tên `logging` nhưng KHÔNG che khuất `logging` chuẩn của
Python — các module khác `import logging` vẫn nhận stdlib (absolute import).

Public API:
    TrialLogger, FIELDNAMES
"""
from .logger import FIELDNAMES, TrialLogger

__all__ = ["TrialLogger", "FIELDNAMES"]
