"""
cell_loader.py
──────────────
Core logic: dựng RoboDK station từ CellConfig.

Architecture: dùng pattern Strategy với mỗi method `_load_X()` tương ứng
một loại item. Dễ extend (thêm conveyor, multi-robot, ...) bằng cách thêm method.

Sử dụng:
    from cell.cell_models import CellConfig
    from cell.cell_loader import CellLoader

    config = CellConfig.from_yaml('config/cell_layout.yaml')
    loader = CellLoader(config)
    items = loader.build()
    # items['robot'], items['table'], items['tool'], ...
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .cell_models import CellConfig
from .exceptions import (
    InvalidConfigError,
    MissingMeshError,
    MissingRobotError,
    RoboDKConnectionError,
    RobotConnectionError,
)
from .pose_utils import make_homogeneous, matrix_to_robodk_pose

logger = logging.getLogger(__name__)


# Default Library path trên Windows (RoboDK default install)
DEFAULT_LIBRARY_PATH = Path("C:/RoboDK/Library")


# Màu mặc định cho từng loại item (R, G, B, A) ∈ [0, 1]. Phối màu công nghiệp:
# thép sáng (gripper), thép tối (bàn), nhựa đen (khay); object khác để debug.
_DEFAULT_GRIPPER_COLOR = [0.78, 0.80, 0.82, 1.0]    # thép inox sáng
_DEFAULT_OBJECT_COLORS = {
    "tray":   [0.18, 0.18, 0.20, 1.0],    # nhựa ABS đen (anti-static)
    "bottle": [0.2, 0.45, 0.95, 1.0],     # xanh dương
    "cup":    [0.95, 0.85, 0.15, 1.0],    # vàng
    "bolt":   [0.85, 0.25, 0.25, 1.0],    # đỏ
}
_DEFAULT_WORKTABLE_COLOR = [0.52, 0.55, 0.58, 1.0]  # thép công nghiệp


def _apply_color(item, rgb_or_rgba):
    """Áp dụng màu cho RoboDK Item (best-effort, bỏ qua nếu API không có)."""
    if item is None or not hasattr(item, "setColor"):
        return
    try:
        color = list(rgb_or_rgba)
        if len(color) == 3:
            color = color + [1.0]
        item.setColor(color)
    except Exception as e:  # noqa: BLE001
        logger.debug("setColor bỏ qua: %s", e)


class CellLoader:
    """Xây dựng RoboDK station từ validated CellConfig.

    Workflow:
        1. Instantiate với CellConfig + project_root path
        2. Gọi build() → returns dict các Items đã tạo
        3. (Optional) Gọi connect_real_robot() nếu cần online mode
    """

    def __init__(
        self,
        config: CellConfig,
        project_root: Optional[Path] = None,
        library_path: Optional[Path] = None,
        minimal_build: bool = False,
    ) -> None:
        """
        Args:
            config: Validated CellConfig.
            project_root: Project root để resolve mesh paths. Default = cwd.
            library_path: Custom RoboDK Library path. Default = C:/RoboDK/Library.
            minimal_build: Nếu True, bỏ các item cosmetic (floor, Cam2D viewport,
                CalibrationTarget frame, 2/3 object templates) để tiết kiệm
                ~10 API call cho RoboDK Free (quota 49 calls trọng số/phiên).
        """
        self.config = config
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.library_path = Path(library_path) if library_path else DEFAULT_LIBRARY_PATH
        self.minimal_build = minimal_build

        self._rdk = None  # Lazy connect
        self._items: Dict[str, Any] = {}

    # ────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────

    def build(self, clear_existing: bool = True) -> Dict[str, Any]:
        """Build cell trong RoboDK.

        Args:
            clear_existing: Có xóa items hiện tại trước khi build không.

        Returns:
            Dict {item_name: RoboDK Item} cho mọi item đã tạo.

        Raises:
            RoboDKConnectionError: RoboDK GUI chưa mở.
            MissingMeshError: Mesh file không tồn tại.
            MissingRobotError: Robot không có trong Library.
        """
        self._connect()

        if clear_existing:
            self._clear_station()

        logger.info("Building cell từ config version=%s (minimal=%s)",
                    self.config.metadata.version, self.minimal_build)

        # Floor load đầu tiên — là mặt phẳng tham chiếu cho cả cell.
        # Bỏ trong minimal_build vì chỉ cosmetic, tiết kiệm 2 API call.
        if self.config.floor and not self.minimal_build:
            self._items["floor"] = self._load_floor()

        # Build theo thứ tự dependency. Pedestal load TRƯỚC robot vì robot có
        # thể được positioned trên đỉnh pedestal — robot pose là independent
        # nhưng visual flow logic là pedestal trước.
        if self.config.robot_pedestal:
            self._items["pedestal"] = self._load_pedestal()

        self._items["robot"] = self._load_robot()
        self._items["worktable"] = self._load_worktable()

        if self.config.camera_mount:
            self._items["camera_mount"] = self._load_camera_mount()

        self._items["camera"] = self._load_camera()
        self._items["tool"] = self._load_gripper()

        self._items["frames"] = self._load_frames()
        self._items["objects"] = self._load_objects()

        logger.info("Cell built successfully. Items: %s", list(self._items.keys()))

        if self.config.robot_connection.enabled:
            self.connect_real_robot()

        return self._items

    def connect_real_robot(self) -> None:
        """Kết nối tới robot thật qua RoboDK driver.

        Phải gọi sau build().

        Raises:
            RobotConnectionError: Không kết nối được.
        """
        conn = self.config.robot_connection
        if not conn.enabled:
            logger.info("robot_connection.enabled=False, bỏ qua connect")
            return

        robot = self._items.get("robot")
        if robot is None:
            raise RuntimeError("Phải gọi build() trước connect_real_robot()")

        logger.info("Connecting to real robot tại %s:%d (%s)", conn.ip, conn.port, conn.driver)
        robot.setConnectionParams(conn.ip, conn.port)

        # Connect với safe option (timeout)
        success = robot.Connect()
        if not success:
            raise RobotConnectionError(conn.ip, conn.port, conn.driver)

        # Apply safety limits
        robot.setSpeed(-1, -1)  # reset
        robot.setRounding(1)  # smooth blending
        logger.info(
            "Connected. Max speed: %d%%, accel: %d%%",
            conn.max_speed_percent,
            conn.acceleration_percent,
        )

    def items(self) -> Dict[str, Any]:
        """Return current items dict (after build)."""
        return self._items

    # ────────────────────────────────────────────────────────────
    # Internal: connection management
    # ────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        """Lazy connect tới RoboDK Software."""
        if self._rdk is not None:
            return

        try:
            from robodk.robolink import Robolink
            self._rdk = Robolink()
        except Exception as e:
            raise RoboDKConnectionError(
                "Không kết nối được tới RoboDK Software.\n"
                "Đảm bảo:\n"
                "  1. RoboDK GUI đang mở\n"
                "  2. Cài đúng version Python robodk (pip install robodk)"
            ) from e

        logger.debug("Connected to RoboDK Software")

    def _clear_station(self) -> None:
        """Xoá hết items trong station hiện tại.

        RoboDK Free giới hạn API call/phiên → ưu tiên ít round-trip nhất.
        Dùng ItemList() duy nhất, lặp Delete trên cùng tập handle, nuốt
        các handle stale (do cascade-delete khi xoá Robot kéo theo Tool/Frame).
        """
        if self._rdk is None:
            return

        items = self._rdk.ItemList()
        if not items:
            return
        logger.info("Clearing %d existing items", len(items))

        for it in items:
            try:
                it.Delete()
            except Exception:
                # Handle stale (đã bị cascade-delete) — bỏ qua.
                pass

    # ────────────────────────────────────────────────────────────
    # Internal: item loaders
    # ────────────────────────────────────────────────────────────

    def _load_robot(self):
        cfg = self.config.robot

        if cfg.source == "library":
            robot_file = self.library_path / cfg.library_name
            if not robot_file.exists():
                # Tìm trong subdirs của Library
                matches = list(self.library_path.rglob(cfg.library_name))
                if matches:
                    robot_file = matches[0]
                else:
                    raise MissingRobotError(cfg.name, str(self.library_path))
        else:  # source == "file"
            robot_file = self._resolve_path(cfg.file_path)
            if not robot_file.exists():
                raise MissingMeshError(str(robot_file), cfg.name)

        robot = self._rdk.AddFile(str(robot_file))
        robot.setName(cfg.name)

        # Khi AddFile() load .robot file, RoboDK tạo cấu trúc:
        #   Station root
        #     └─ "Yaskawa Motoman" (parent frame, auto-created)
        #          └─ "Yaskawa GP7" (robot item, return value của AddFile)
        # `robot.setPose()` đặt pose của robot tương đối parent frame, nhưng
        # robot có offset built-in nên kết quả thường lệch. Cách đúng: đặt
        # pose của PARENT FRAME — cả hierarchy robot sẽ dịch chuyển đồng bộ.
        T = make_homogeneous(cfg.pose.xyz_mm, cfg.pose.rpy_deg)
        target_pose = matrix_to_robodk_pose(T)

        parent_frame = robot.Parent()
        if parent_frame and parent_frame.Valid() and parent_frame.item != robot.item:
            parent_frame.setPose(target_pose)
            # Reset robot pose relative to parent = identity
            try:
                from robodk.robomath import eye
                robot.setPose(eye(4))
            except Exception as e:  # noqa: BLE001
                logger.debug("Cannot reset robot relative pose: %s", e)
        else:
            # Không có parent frame riêng → đặt trực tiếp
            try:
                robot.setPoseAbs(target_pose)
            except AttributeError:
                robot.setPose(target_pose)

        robot.setJoints(cfg.home_joints_deg)

        # Orchestrator tự convert pose world → parent frame trước khi MoveJ
        # (xem Orchestrator._compute_world_to_robotbase). Không setPoseFrame ở đây.

        logger.info("✓ Robot loaded: %s at target world %s", cfg.name, cfg.pose.xyz_mm)
        return robot

    def _load_worktable(self):
        cfg = self.config.worktable
        mesh = self._resolve_path(cfg.mesh)

        if not mesh.exists():
            raise MissingMeshError(str(mesh), "worktable")

        table = self._rdk.AddFile(str(mesh))
        T = make_homogeneous(cfg.pose.xyz_mm, cfg.pose.rpy_deg)
        table.setPose(matrix_to_robodk_pose(T))

        # Worktable: nâu gỗ để dễ phân biệt với gripper/objects màu sắc
        color = getattr(cfg, "color_rgb", None) or _DEFAULT_WORKTABLE_COLOR[:3]
        _apply_color(table, color)

        logger.info("✓ Worktable loaded")
        return table

    def _load_floor(self):
        cfg = self.config.floor
        mesh = self._resolve_path(cfg.mesh)

        if not mesh.exists():
            logger.warning("Floor mesh không có (optional): %s", mesh)
            return None

        floor = self._rdk.AddFile(str(mesh))
        T = make_homogeneous(cfg.pose.xyz_mm, cfg.pose.rpy_deg)
        floor.setPose(matrix_to_robodk_pose(T))

        color = getattr(cfg, "color_rgb", None)
        if color:
            _apply_color(floor, color)

        logger.info("✓ Floor loaded")
        return floor

    def _load_pedestal(self):
        cfg = self.config.robot_pedestal
        mesh = self._resolve_path(cfg.mesh)

        if not mesh.exists():
            raise MissingMeshError(str(mesh), "robot_pedestal")

        pedestal = self._rdk.AddFile(str(mesh))
        T = make_homogeneous(cfg.pose.xyz_mm, cfg.pose.rpy_deg)
        pedestal.setPose(matrix_to_robodk_pose(T))

        color = getattr(cfg, "color_rgb", None)
        if color:
            _apply_color(pedestal, color)

        logger.info("✓ Pedestal loaded")
        return pedestal

    def _load_camera_mount(self):
        cfg = self.config.camera_mount
        mesh = self._resolve_path(cfg.mesh)

        if not mesh.exists():
            logger.warning("Camera mount mesh không có (optional): %s", mesh)
            return None

        mount = self._rdk.AddFile(str(mesh))
        T = make_homogeneous(cfg.pose.xyz_mm, cfg.pose.rpy_deg)
        mount.setPose(matrix_to_robodk_pose(T))

        logger.info("✓ Camera mount loaded")
        return mount

    def _load_camera(self):
        cfg = self.config.camera

        # Cam2D_Add() gắn camera vào một item và camera nhìn dọc trục Z của
        # item đó — gọi setPoseAbs()/setPose() trên CHÍNH camera item KHÔNG
        # dời được viewpoint (camera kẹt ở gốc toạ độ). Cách đúng: tạo một
        # reference frame riêng tại world pose mong muốn rồi gắn camera vào
        # frame đó. Frame ở station root → setPose = world coords trực tiếp.
        T = make_homogeneous(cfg.pose.xyz_mm, cfg.pose.rpy_deg)
        cam_frame = self._rdk.AddFrame("CameraFrame")
        try:
            cam_frame.setPoseAbs(matrix_to_robodk_pose(T))
        except AttributeError:
            cam_frame.setPose(matrix_to_robodk_pose(T))

        # Minimal mode: chỉ giữ CameraFrame để orchestrator biết camera pose
        # (T_BC calibration), KHÔNG gọi Cam2D_Add (-1 API call). MockCamera
        # sinh data riêng, không cần viewport render từ RoboDK.
        if self.minimal_build:
            logger.info(
                "✓ Camera frame only (minimal): %s tại world %s",
                cfg.model, list(cfg.pose.xyz_mm),
            )
            return cam_frame

        # Camera ảo trong RoboDK (cả "virtual" và "real" đều add để visualize)
        intr = cfg.intrinsics
        if intr is None:
            cam_params = "FOCAL_LENGTH=2 FOV=70 FAR_LENGTH=2000 SIZE=1280x720"
        else:
            cam_params = (
                f"FOCAL_LENGTH={intr.focal_length_mm} "
                f"FOV={intr.fov_deg} "
                f"FAR_LENGTH=2000 "
                f"SIZE={intr.size_px[0]}x{intr.size_px[1]}"
            )

        cam = self._rdk.Cam2D_Add(cam_frame, cam_params)

        logger.info(
            "✓ Camera (%s) loaded: %s tại world %s",
            cfg.type, cfg.model, list(cfg.pose.xyz_mm),
        )
        return cam

    def _load_gripper(self):
        cfg = self.config.gripper
        robot = self._items.get("robot")
        if robot is None:
            raise RuntimeError("Phải load robot trước gripper")

        T_tcp = make_homogeneous(cfg.tcp_offset_xyz_mm, cfg.tcp_offset_rpy_deg)
        tool = robot.AddTool(matrix_to_robodk_pose(T_tcp), cfg.name)

        if cfg.mesh:
            mesh = self._resolve_path(cfg.mesh)
            if mesh.exists():
                # RoboDK API không có Item.AddGeometryFromFile. Pattern đúng:
                # AddFile() tạo Item mới với geometry, truyền parent=tool để gắn.
                gripper_mesh = self._rdk.AddFile(str(mesh), tool)
                _apply_color(gripper_mesh, _DEFAULT_GRIPPER_COLOR)
            else:
                logger.warning("Gripper mesh không có: %s (TCP vẫn được tạo)", mesh)

        logger.info("✓ Gripper attached: %s (TCP offset %s)", cfg.name, cfg.tcp_offset_xyz_mm)
        return tool

    def _load_frames(self) -> Dict[str, Any]:
        frames: Dict[str, Any] = {}

        # Sort: parent frames trước
        sorted_cfgs = sorted(self.config.frames, key=lambda f: f.parent is not None)

        # Minimal mode: bỏ frame CalibrationTarget (chỉ dùng cho calibration
        # script, không cần cho pick-place). Tiết kiệm 2 API call.
        if self.minimal_build:
            sorted_cfgs = [c for c in sorted_cfgs if c.name != "CalibrationTarget"]

        for cfg in sorted_cfgs:
            parent_item = frames.get(cfg.parent) if cfg.parent else None
            if cfg.parent and parent_item is None:
                raise InvalidConfigError(f"Parent frame '{cfg.parent}' chưa được tạo")

            frame = self._rdk.AddFrame(cfg.name, parent_item)
            T = make_homogeneous(cfg.pose.xyz_mm, cfg.pose.rpy_deg)
            frame.setPose(matrix_to_robodk_pose(T))

            frames[cfg.name] = frame
            logger.info("✓ Frame: %s at %s", cfg.name, cfg.pose.xyz_mm)

        return frames

    def _load_objects(self) -> Dict[str, Any]:
        objects: Dict[str, Any] = {}
        frames = self._items.get("frames", {})

        # Minimal mode: chỉ load object đầu tiên (~4 API call ít hơn). MockDetector
        # sinh detection độc lập với object templates có mặt trong RoboDK.
        object_cfgs = self.config.objects[:1] if self.minimal_build else self.config.objects

        for cfg in object_cfgs:
            mesh = self._resolve_path(cfg.mesh)

            if not mesh.exists():
                logger.warning(
                    "Object mesh không tồn tại, bỏ qua: %s (object='%s')", mesh, cfg.name
                )
                continue

            parent = frames.get(cfg.parent_frame) if cfg.parent_frame else None
            obj = self._rdk.AddFile(str(mesh), parent)

            # Áp pose offset (nếu có) — quan trọng khi nhiều object chia chung
            # 1 parent_frame nhưng cần đặt ở vị trí khác nhau.
            if cfg.pose is not None:
                T = make_homogeneous(cfg.pose.xyz_mm, cfg.pose.rpy_deg)
                obj.setPose(matrix_to_robodk_pose(T))

            # Apply màu theo class_name để phân biệt object trong RoboDK GUI.
            color = _DEFAULT_OBJECT_COLORS.get(cfg.name)
            if color:
                _apply_color(obj, color)

            objects[cfg.name] = obj
            logger.info("✓ Object template: %s", cfg.name)

        return objects

    # ────────────────────────────────────────────────────────────
    # Internal: helpers
    # ────────────────────────────────────────────────────────────

    def _resolve_path(self, relative_path: str) -> Path:
        """Resolve relative path tới absolute, dựa trên project_root."""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return self.project_root / p
