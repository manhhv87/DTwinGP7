"""
exceptions.py
─────────────
Custom exceptions for the cell config module.
"""
from __future__ import annotations


class CellError(Exception):
    """Base exception for all cell config errors."""


class InvalidConfigError(CellError):
    """Invalid YAML config file.

    Includes:
        - YAML syntax error
        - Schema validation failure (Pydantic)
        - Logical inconsistency (e.g. parent_frame does not exist)
    """


class MissingMeshError(CellError):
    """Mesh file (STL/OBJ/...) referenced in config but not found on disk."""

    def __init__(self, mesh_path: str, item_name: str | None = None):
        msg = f"Mesh file not found: {mesh_path}"
        if item_name:
            msg += f" (referenced by '{item_name}')"
        super().__init__(msg)
        self.mesh_path = mesh_path
        self.item_name = item_name
