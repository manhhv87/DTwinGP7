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

import math
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
    # Runtime visibility state (Show/Hide từ cell tree). Persist để khi
    # reload cell, hiển thị nguyên trạng. Key = component_visibility_key
    # ("robot", "object::name", "frame::name"…), value = bool.
    visibility_state: Optional[dict] = None


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
    color_rgb: Tuple[float, float, float] = (0.66, 0.67, 0.66)  # RAL 7035 light grey

    @field_validator("color_rgb")
    @classmethod
    def _validate_color(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        for c, val in zip("rgb", v):
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"color_rgb[{c}]={val} phải trong [0, 1]")
        return v


class FloorConfig(BaseModel):
    """Sàn nhà (optional) — tấm phẳng làm mặt phẳng tham chiếu cho cả cell."""

    mesh: str
    pose: PoseConfig
    color_rgb: Tuple[float, float, float] = (0.50, 0.52, 0.55)  # bê tông/epoxy xám

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


class PedestalConfig(BaseModel):
    """Pedestal nâng robot lên (optional, dùng cho cell công nghiệp)."""

    mesh: str
    pose: PoseConfig
    color_rgb: Tuple[float, float, float] = (0.28, 0.29, 0.31)  # RAL 7016 anthracite

    @field_validator("color_rgb")
    @classmethod
    def _validate_color(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        for c, val in zip("rgb", v):
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"color_rgb[{c}]={val} phải trong [0, 1]")
        return v


class CameraIntrinsics(BaseModel):
    """Tham số quang học pinhole (chuẩn OpenCV/RealSense + tiện ích FOV).

    Hai cách khai (đều optional, có thể kết hợp):
      (a) fx/fy/cx/cy pixel — từ camera THẬT (D455/OpenCV), ưu tiên cho tính FOV.
      (b) fov_deg/focal_length_mm — camera ẢO/sim kiểu RoboDK.
    size_px luôn có (mặc định 1280×720).
    """

    size_px: Tuple[int, int] = (1280, 720)
    # Pinhole thật (pixel) — từ D455/OpenCV. Optional.
    fx: Optional[float] = Field(default=None, gt=0)
    fy: Optional[float] = Field(default=None, gt=0)
    cx: Optional[float] = Field(default=None, ge=0)
    cy: Optional[float] = Field(default=None, ge=0)
    # Mô tả kiểu sim/RoboDK — Optional.
    fov_deg: Optional[float] = Field(default=None, gt=0, lt=180)
    focal_length_mm: Optional[float] = Field(default=None, gt=0)

    @field_validator("size_px")
    @classmethod
    def _validate_size(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        for dim, val in zip(["width", "height"], v):
            if val < 100 or val > 8192:
                raise ValueError(f"size_px {dim}={val} ngoài range hợp lý [100, 8192]")
        return v

    def hfov_vfov_deg(self) -> Tuple[float, float]:
        """Trả (hfov, vfov) độ — dùng dựng frustum. Ưu tiên fx/fy; thiếu thì
        từ fov_deg (suy vfov theo tỉ lệ khung); cuối cùng fallback D455 87°."""
        w, h = self.size_px

        def _vfov_from_hfov(hfov_deg: float) -> float:
            return math.degrees(
                2.0 * math.atan(math.tan(math.radians(hfov_deg) / 2.0) * h / w))

        if self.fx and self.fy:
            hfov = math.degrees(2.0 * math.atan(w / (2.0 * self.fx)))
            vfov = math.degrees(2.0 * math.atan(h / (2.0 * self.fy)))
            return hfov, vfov
        hfov = float(self.fov_deg) if self.fov_deg else 87.0  # D455 RGB mặc định
        return hfov, _vfov_from_hfov(hfov)


class CameraConfig(BaseModel):
    type: Literal["virtual", "real"] = "virtual"
    model: Optional[str] = None
    # Kiểu lắp camera (thuật ngữ robot-vision): eye_to_hand = cố định nhìn vào
    # vùng làm việc; eye_in_hand = gắn trên flange/tool, di chuyển theo robot.
    mount: Literal["eye_to_hand", "eye_in_hand"] = "eye_to_hand"
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
    pose: Optional[PoseConfig] = None       # offset so với parent_frame; None = trùng gốc


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
    """Top-level cell configuration.

    Chỉ `robot` là bắt buộc (cần URDFRobot để FK/IK chạy). Worktable, camera,
    gripper, floor, pedestal, camera_mount đều OPTIONAL — user add qua Cell
    Editor UI khi cần, hoặc khai báo trong YAML.
    """

    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    robot: RobotConfig
    worktable: Optional[WorktableConfig] = None
    camera: Optional[CameraConfig] = None
    gripper: Optional[GripperConfig] = None
    floor: Optional[FloorConfig] = None
    camera_mount: Optional[CameraMountConfig] = None
    robot_pedestal: Optional[PedestalConfig] = None
    frames: List[FrameConfig] = Field(default_factory=list)
    objects: List[ObjectConfig] = Field(default_factory=list)
    # Danh sách lớp vật thể của BÀI TOÁN (do người dùng định nghĩa) — dùng cho
    # dán nhãn dataset + hiển thị. Detection thật lấy tên lớp từ chính model YOLO.
    object_classes: List[str] = Field(
        default_factory=lambda: ["tray", "bottle", "cup", "bolt"])
    robot_connection: RobotConnectionConfig = Field(default_factory=RobotConnectionConfig)

    @field_validator("object_classes")
    @classmethod
    def _validate_classes(cls, v: List[str]) -> List[str]:
        """Strip + bỏ rỗng + khử trùng (giữ thứ tự). Rỗng → default 4 lớp."""
        seen: set[str] = set()
        out: List[str] = []
        for name in v:
            s = str(name).strip()
            if s and s not in seen:
                seen.add(s); out.append(s)
        return out or ["tray", "bottle", "cup", "bolt"]

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

    def to_yaml(self, path: str | Path) -> None:
        """Dump config sang YAML file giữ toàn bộ field (kể cả floor,
        camera_mount, robot_pedestal nếu có). Pydantic dump tự convert
        tuple→list. Format đầu ra:
          • Top-level: block style (mỗi key 1 dòng) — dễ scan.
          • List ngắn (xyz, rpy, joints, color): flow style `[a, b, c]` —
            khớp style input YAML user thường viết, gọn 3 dòng → 1 dòng.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(exclude_none=True, mode="python")

        # Custom dumper: list/tuple of primitives ngắn (<=8 phần tử) ⇒ flow
        # style. Pydantic dump giữ tuple cho Field Tuple[...] (xyz_mm, rpy_deg,
        # color_rgb) nên cần register cho cả list lẫn tuple.
        class _CompactDumper(yaml.SafeDumper):
            pass

        def _seq_repr(dumper, value):
            seq = list(value)
            if (len(seq) <= 8
                    and all(isinstance(v, (int, float, bool, str))
                            and not (isinstance(v, str) and "\n" in v)
                            for v in seq)):
                return dumper.represent_sequence(
                    "tag:yaml.org,2002:seq", seq, flow_style=True)
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", seq, flow_style=False)

        _CompactDumper.add_representer(list, _seq_repr)
        _CompactDumper.add_representer(tuple, _seq_repr)

        with path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, Dumper=_CompactDumper, sort_keys=False,
                      indent=2, allow_unicode=True,
                      default_flow_style=False)


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
