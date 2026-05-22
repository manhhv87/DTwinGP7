"""
exceptions.py
─────────────
Custom exceptions cho cell config module.
"""
from __future__ import annotations


class CellError(Exception):
    """Base exception cho mọi lỗi cell config."""


class InvalidConfigError(CellError):
    """File YAML config không hợp lệ.

    Bao gồm:
        - YAML syntax error
        - Schema validation failure (Pydantic)
        - Logical inconsistency (vd parent_frame không tồn tại)
    """


class MissingMeshError(CellError):
    """File mesh (STL/OBJ/...) được tham chiếu trong config nhưng không tồn tại."""

    def __init__(self, mesh_path: str, item_name: str | None = None):
        msg = f"Mesh file not found: {mesh_path}"
        if item_name:
            msg += f" (referenced by '{item_name}')"
        super().__init__(msg)
        self.mesh_path = mesh_path
        self.item_name = item_name
