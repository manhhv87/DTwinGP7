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
    ) -> None:
        """
        Args:
            config: Validated CellConfig.
            project_root: Project root để resolve mesh paths. Default = cwd.
            library_path: Custom RoboDK Library path. Default = C:/RoboDK/Library.
        """
        self.config = config
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.library_path = Path(library_path) if library_path else DEFAULT_LIBRARY_PATH

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

        logger.info("Building cell từ config version=%s", self.config.metadata.version)

        # Build theo thứ tự dependency
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
        """Xóa hết items trong station hiện tại."""
        if self._rdk is None:
            return

        items = self._rdk.ItemList()
        logger.info("Clearing %d existing items", len(items))
        for item in items:
            try:
                item.Delete()
            except Exception as e:
                logger.warning("Không xóa được item: %s", e)

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

        # Set pose
        T = make_homogeneous(cfg.pose.xyz_mm, cfg.pose.rpy_deg)
        robot.setPose(matrix_to_robodk_pose(T))

        # Set home joints
        robot.setJoints(cfg.home_joints_deg)

        logger.info("✓ Robot loaded: %s", cfg.name)
        return robot

    def _load_worktable(self):
        cfg = self.config.worktable
        mesh = self._resolve_path(cfg.mesh)

        if not mesh.exists():
            raise MissingMeshError(str(mesh), "worktable")

        table = self._rdk.AddFile(str(mesh))
        table.setName("Worktable")

        T = make_homogeneous(cfg.pose.xyz_mm, cfg.pose.rpy_deg)
        table.setPose(matrix_to_robodk_pose(T))
        table.setColor(list(cfg.color_rgb))

        logger.info("✓ Worktable loaded")
        return table

    def _load_camera_mount(self):
        cfg = self.config.camera_mount
        mesh = self._resolve_path(cfg.mesh)

        if not mesh.exists():
            logger.warning("Camera mount mesh không có (optional): %s", mesh)
            return None

        mount = self._rdk.AddFile(str(mesh))
        mount.setName("CameraMount")
        T = make_homogeneous(cfg.pose.xyz_mm, cfg.pose.rpy_deg)
        mount.setPose(matrix_to_robodk_pose(T))

        logger.info("✓ Camera mount loaded")
        return mount

    def _load_camera(self):
        cfg = self.config.camera
        parent = self._items.get("worktable") or self._items.get("robot")

        # Camera ảo trong RoboDK (cả "virtual" và "real" đều add để visualize)
        intr = cfg.intrinsics
        if intr is None:
            # Default
            cam_params = "FOCAL_LENGTH=2 FOV=70 FAR_LENGTH=2000 SIZE=1280x720"
        else:
            cam_params = (
                f"FOCAL_LENGTH={intr.focal_length_mm} "
                f"FOV={intr.fov_deg} "
                f"FAR_LENGTH=2000 "
                f"SIZE={intr.size_px[0]}x{intr.size_px[1]}"
            )

        cam = self._rdk.Cam2D_Add(parent, cam_params)
        cam.setName(f"{cfg.model or 'Camera'}_{cfg.type}")

        T = make_homogeneous(cfg.pose.xyz_mm, cfg.pose.rpy_deg)
        cam.setPose(matrix_to_robodk_pose(T))

        logger.info("✓ Camera (%s) loaded: %s", cfg.type, cfg.model)
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
                tool.AddGeometryFromFile(str(mesh))
            else:
                logger.warning("Gripper mesh không có: %s (TCP vẫn được tạo)", mesh)

        logger.info("✓ Gripper attached: %s (TCP offset %s)", cfg.name, cfg.tcp_offset_xyz_mm)
        return tool

    def _load_frames(self) -> Dict[str, Any]:
        frames: Dict[str, Any] = {}

        # Sort: parent frames trước
        sorted_cfgs = sorted(self.config.frames, key=lambda f: f.parent is not None)

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

        for cfg in self.config.objects:
            mesh = self._resolve_path(cfg.mesh)

            if not mesh.exists():
                logger.warning(
                    "Object mesh không tồn tại, bỏ qua: %s (object='%s')", mesh, cfg.name
                )
                continue

            parent = frames.get(cfg.parent_frame) if cfg.parent_frame else None
            obj = self._rdk.AddFile(str(mesh), parent)
            obj.setName(f"object_{cfg.name}_template")
            obj.setVisible(cfg.visible)

            objects[cfg.name] = obj
            logger.info("✓ Object template: %s (visible=%s)", cfg.name, cfg.visible)

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
