"""
cell — Cell configuration schema (Pydantic).

Cell config được dùng cho:
  - Viewport rendering (Open3D mirror dựng worktable + camera frame)
  - Robot base pose cho kinematics
  - Hand-eye calibration sim mode
  - Object templates cho perception mock

Public API:
    CellConfig: Pydantic schema cho cell config

Exceptions:
    CellError, InvalidConfigError, MissingMeshError

Helpers:
    make_homogeneous, matrix_to_rpy_xyz
"""
from .cell_models import CellConfig
from .exceptions import (
    CellError,
    InvalidConfigError,
    MissingMeshError,
)
from .pose_utils import make_homogeneous, matrix_to_rpy_xyz

__version__ = "2.0.0"
__all__ = [
    "CellConfig",
    "CellError",
    "InvalidConfigError",
    "MissingMeshError",
    "make_homogeneous",
    "matrix_to_rpy_xyz",
]
