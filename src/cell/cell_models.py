"""
cell_models.py
──────────────
Pydantic models cho validation cell configuration.

Mỗi field trong YAML được map sang một Pydantic field với:
  - Type checking
  - Range validation
  - Cross-field consistency checks

Sử dụng:
    from cell_models import CellConfig
    config = CellConfig.from_yaml('config/cell_layout.yaml')

CLI:
    python -m src.cell.cell_models validate config/cell_layout.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ───── Validation constants ─────
MAX_DISTANCE_MM = 5000.0
ANGLE_MIN = -360.0
ANGLE_MAX = 360.0
NUM_JOINTS = 6


# ───── Sub-models ─────


class PoseConfig(BaseModel):
    """Pose 6-DOF: translation + rotation."""

    xyz_mm: Tuple[float, float, float] = Field(
        ..., description="Translation (x, y, z) in mm"
    )
    rpy_deg: Tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="Rotation (roll, pitch, yaw) in degrees"
    )

    @field_validator("xyz_mm")
    @classmethod
    def _validate_xyz(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        for axis, val in zip("xyz", v):
            if abs(val) > MAX_DISTANCE_MM:
                raise ValueError(
                    f"{axis}={val:.1f}mm vượt giới hạn ±{MAX_DISTANCE_MM}mm. "
                    f"Kiểm tra lại đơn vị (mm không phải m)."
                )
        return v

    @field_validator("rpy_deg")
    @classmethod
    def _validate_rpy(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        for axis, val in zip("rpy", v):
            if not ANGLE_MIN <= val <= ANGLE_MAX:
                raise ValueError(
                    f"{axis}={val}° ngoài range [{ANGLE_MIN}, {ANGLE_MAX}]"
                )
        return v


class MetadataConfig(BaseModel):
    """Optional metadata cho versioning."""

    version: str = "0.0.0"
    last_modified: Optional[str] = None
    author: Optional[str] = None
    notes: Optional[str] = None


class RobotConfig(BaseModel):
    """Cấu hình robot."""

    name: str
    source: Literal["library", "file"] = "library"
    library_name: Optional[str] = None
    file_path: Optional[str] = None
    pose: PoseConfig
    home_joints_deg: List[float] = Field(..., min_length=NUM_JOINTS, max_length=NUM_JOINTS)

    @model_validator(mode="after")
    def _validate_source(self) -> "RobotConfig":
        if self.source == "library" and not self.library_name:
            raise ValueError("source='library' requires library_name")
        if self.source == "file" and not self.file_path:
            raise ValueError("source='file' requires file_path")
        return self

    @field_validator("home_joints_deg")
    @classmethod
    def _validate_joints(cls, v: List[float]) -> List[float]:
        for i, j in enumerate(v):
            if not ANGLE_MIN <= j <= ANGLE_MAX:
                raise ValueError(f"home_joints_deg[{i}]={j}° ngoài range")
        return v


class WorktableConfig(BaseModel):
    """Bàn làm việc."""

    mesh: str
    pose: PoseConfig
    color_rgb: Tuple[float, float, float] = (0.6, 0.6, 0.7)

    @field_validator("color_rgb")
    @classmethod
    def _validate_color(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        for c, val in zip("rgb", v):
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"color_rgb[{c}]={val} phải trong [0, 1]")
        return v


class CameraMountConfig(BaseModel):
    """Giàn lắp camera (optional)."""

    mesh: str
    pose: PoseConfig


class CameraIntrinsics(BaseModel):
    fov_deg: float = Field(..., gt=0, lt=180)
    focal_length_mm: float = Field(..., gt=0)
    size_px: Tuple[int, int]

    @field_validator("size_px")
    @classmethod
    def _validate_size(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        for dim, val in zip(["width", "height"], v):
            if val < 100 or val > 8192:
                raise ValueError(f"size_px {dim}={val} ngoài range hợp lý [100, 8192]")
        return v


class CameraConfig(BaseModel):
    type: Literal["virtual", "real"] = "virtual"
    model: Optional[str] = None
    pose: PoseConfig
    intrinsics: Optional[CameraIntrinsics] = None


class GripperConfig(BaseModel):
    """Cấu hình end-effector tool."""

    name: str
    mesh: Optional[str] = None
    tcp_offset_xyz_mm: Tuple[float, float, float]
    tcp_offset_rpy_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class FrameConfig(BaseModel):
    """Reference frame."""

    name: str
    pose: PoseConfig
    parent: Optional[str] = None


class ObjectConfig(BaseModel):
    """Object template (vật cần gắp)."""

    name: str
    mesh: str
    visible: bool = False
    parent_frame: Optional[str] = None


class RobotConnectionConfig(BaseModel):
    """Cấu hình kết nối robot thật."""

    enabled: bool = False
    ip: Optional[str] = None
    port: int = 80
    driver: str = "Motoman"
    max_speed_percent: float = Field(default=30.0, ge=1.0, le=100.0)
    acceleration_percent: float = Field(default=50.0, ge=1.0, le=100.0)

    @model_validator(mode="after")
    def _check_ip_if_enabled(self) -> "RobotConnectionConfig":
        if self.enabled and not self.ip:
            raise ValueError("enabled=true requires 'ip' field")
        return self


# ───── Top-level model ─────


class CellConfig(BaseModel):
    """Top-level cell configuration."""

    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    robot: RobotConfig
    worktable: WorktableConfig
    camera: CameraConfig
    gripper: GripperConfig
    camera_mount: Optional[CameraMountConfig] = None
    frames: List[FrameConfig] = Field(default_factory=list)
    objects: List[ObjectConfig] = Field(default_factory=list)
    robot_connection: RobotConnectionConfig = Field(default_factory=RobotConnectionConfig)

    @model_validator(mode="after")
    def _validate_frame_references(self) -> "CellConfig":
        """Đảm bảo parent_frame, parent_frame references đều tồn tại."""
        frame_names = {f.name for f in self.frames}

        for f in self.frames:
            if f.parent and f.parent not in frame_names:
                raise ValueError(
                    f"Frame '{f.name}' references parent '{f.parent}' nhưng frame này không tồn tại"
                )

        for obj in self.objects:
            if obj.parent_frame and obj.parent_frame not in frame_names:
                raise ValueError(
                    f"Object '{obj.name}' references parent_frame '{obj.parent_frame}' nhưng frame này không tồn tại"
                )

        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CellConfig":
        """Load và validate từ file YAML."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config không tồn tại: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)


# ───── CLI ─────


def _cli_validate(path: str) -> int:
    """Validate config file. Return exit code (0 = OK, 1 = fail)."""
    try:
        config = CellConfig.from_yaml(path)
        print(f"✓ Config hợp lệ: {path}")
        print(f"  Version: {config.metadata.version}")
        print(f"  Robot: {config.robot.name}")
        print(f"  Frames: {[f.name for f in config.frames]}")
        print(f"  Objects: {[o.name for o in config.objects]}")
        print(f"  Real robot mode: {config.robot_connection.enabled}")
        return 0
    except Exception as e:
        print(f"✗ Config lỗi: {path}")
        print(f"  {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    # Console Windows cp1252 không in được ✓/✗ → ép stdout sang UTF-8.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    if len(sys.argv) < 3 or sys.argv[1] != "validate":
        print("Usage: python -m src.cell.cell_models validate <config.yaml>")
        sys.exit(2)
    sys.exit(_cli_validate(sys.argv[2]))
