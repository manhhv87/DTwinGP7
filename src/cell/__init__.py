"""
cell — Module quản lý cell RoboDK theo paradigm "code-first".

Public API:
    CellConfig: Pydantic schema cho cell config
    CellLoader: Class dựng cell từ config
    
Exceptions:
    CellError, RoboDKConnectionError, InvalidConfigError,
    MissingMeshError, MissingRobotError, RobotConnectionError

Helpers:
    make_homogeneous, matrix_to_rpy_xyz
"""
from .cell_models import CellConfig
from .cell_loader import CellLoader
from .exceptions import (
    CellError,
    InvalidConfigError,
    MissingMeshError,
    MissingRobotError,
    RoboDKConnectionError,
    RobotConnectionError,
)
from .pose_utils import make_homogeneous, matrix_to_rpy_xyz

__version__ = "1.0.0"
__all__ = [
    "CellConfig",
    "CellLoader",
    "CellError",
    "InvalidConfigError",
    "MissingMeshError",
    "MissingRobotError",
    "RoboDKConnectionError",
    "RobotConnectionError",
    "make_homogeneous",
    "matrix_to_rpy_xyz",
]
