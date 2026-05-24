"""
open3d_gui_sim_robot.py
───────────────────────
SimRobot + Open3D **GUI** viewport (O3DVisualizer / Filament) — navigate giống
RoboDK hơn hẳn legacy Visualizer.

Vì sao thêm cái này (bên cạnh open3d_sim_robot.py legacy):
  - Legacy `o3d.visualization.Visualizer` (OpenGL) chỉ có arcball thô, xoay quanh
    1 điểm `lookat` cố định, bước zoom/pan scale theo bounding box → "rất khó
    xoay/đưa góc nhìn ra-vào".
  - `O3DVisualizer` (Filament) có arcball mượt, ánh sáng IBL + skybox sẵn, panel
    settings, và camera điều khiển tự nhiên như RoboDK (trái=xoay, phải=pan,
    lăn=zoom). Render qua GPU Filament.

KHÁC BIỆT KIẾN TRÚC QUAN TRỌNG so với legacy:
  - GUI Filament BẮT BUỘC chạy event loop trên MAIN thread qua
    `gui.Application.instance.run()` (blocking). Không thể "pump" thủ công như
    legacy. → Orchestrator (thí nghiệm) phải chạy ở WORKER thread, GUI ở main.
  - Vì vậy: scripts/03 dựng robot này (main thread), start worker chạy
    `orch.run_n_trials(...)`, rồi main thread gọi `robot.run_gui()` (block tới
    khi user đóng cửa sổ). MoveJ/MoveL (gọi từ worker) KHÔNG đụng GUI trực tiếp
    mà `post_to_main_thread` các transform → an toàn thread.
  - Animation = đổi TRANSFORM của geometry (set_geometry_transform), KHÔNG ghi
    lại vertex như legacy → nhẹ GPU, mượt. Mỗi link mesh add 1 lần ở frame
    link-local, sau đó chỉ cập nhật ma trận 4x4 world (mm→m).

Render dùng chung FK `link_frames_urdf` (đã verify match RoboDK SolveFK 0.00mm)
và tái dùng hằng số mesh từ open3d_sim_robot (1 nguồn sự thật).

open3d import LAZY trong __init__ → import module này không cần open3d.
"""
from __future__ import annotations

import logging
import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..kinematics.urdf_chain import URDFRobot, gp7_urdf, link_frames_urdf
from ..sim_robot import SimRobot

logger = logging.getLogger(__name__)


# ───── Filament stderr filter ─────
# Open3D's Filament renderer in một cảnh báo "Camera preconditions not met"
# qua C-stderr (FD 2) khi nó tự gọi setProjection ở frame render đầu tiên
# (trước khi widget có viewport size). Cảnh báo vô hại (Filament dùng default
# projection rồi user code override sau ở tick #3) nhưng làm bẩn output.
#
# Python không bắt được vì in qua libc stderr, không đi qua sys.stderr. Cách
# DUY NHẤT chặn: dup2 FD 2 sang pipe + thread filter line-based, drop dòng
# match pattern Filament, forward phần còn lại về terminal thật.
_FILAMENT_NOISE = (
    re.compile(rb"FCamera::setProjection"),
    re.compile(rb"Camera preconditions not met"),
)
_stderr_filter_installed = False


def _install_filament_stderr_filter() -> None:
    """Cài filter cho FD 2 (idempotent, 1 lần / process).

    Pipe stderr qua daemon thread; drop line khớp `_FILAMENT_NOISE` + drop
    blank line ngay sau đó (Filament thường đệm bằng newline trên/dưới). Mọi
    output khác (Python logging, traceback…) đi qua nguyên vẹn.
    """
    global _stderr_filter_installed
    if _stderr_filter_installed:
        return
    try:
        sys.stderr.flush()
        real_fd = os.dup(2)                            # giữ ref tới terminal thật
        r_fd, w_fd = os.pipe()
        os.dup2(w_fd, 2)                               # FD 2 → write end of pipe
        os.close(w_fd)
    except OSError as e:                               # noqa: BLE001
        logger.debug("Không cài được Filament stderr filter: %s", e)
        return

    def _pump() -> None:
        # Đọc từng dòng từ pipe; nếu match noise pattern → drop + cũng drop
        # blank line ngay sau. Còn lại ghi thẳng ra FD terminal thật.
        drop_next_blank = False
        try:
            with os.fdopen(r_fd, "rb", buffering=0) as r:
                buf = b""
                while True:
                    chunk = r.read(4096)
                    if not chunk:
                        return
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        stripped = line.strip()
                        if drop_next_blank and not stripped:
                            drop_next_blank = False
                            continue
                        if any(p.search(line) for p in _FILAMENT_NOISE):
                            drop_next_blank = True
                            continue
                        drop_next_blank = False
                        try:
                            os.write(real_fd, line + b"\n")
                        except OSError:
                            return
        except Exception:                              # noqa: BLE001
            pass

    threading.Thread(
        target=_pump, name="filament-stderr-filter", daemon=True,
    ).start()
    _stderr_filter_installed = True


# ── GP7 mesh constants (shared with potential mirror-only renderers) ──
_LINK_COLORS = {
    "link_S": (0.90, 0.55, 0.10),
    "link_L": (0.20, 0.45, 0.85),
    "link_U": (0.20, 0.60, 0.30),
    "link_R": (0.85, 0.20, 0.20),
    "link_B": (0.50, 0.35, 0.70),
    "link_T": (0.95, 0.75, 0.10),
    "link_flange": (0.60, 0.60, 0.60),
    "link_tool0": (0.15, 0.15, 0.15),
}

# Yaskawa Motoman blue — vivid ROYAL blue như RoboDK render Yaskawa-GP7.robot.
# RGB(45, 100, 225). Trước đây dùng pure blue (20,25,230) R/G gần 0 → luminance
# rất thấp → dưới ánh sáng scene mờ thì chìm thành gần-đen. Royal blue này giữ
# độ bão hoà nhưng có đủ luminance để nổi rõ, kết hợp với _setup_lighting() boost
# sun + IBL cho robot sáng đúng kiểu RoboDK (vẫn có shading khối 3D).
_YASKAWA_BLUE = (0.176, 0.392, 0.882)

# ros-industrial GP7 visual meshes (motoman_gp7_support, noetic-devel).
# Tải sẵn vào models/gp7_links/. STL trong MÉT, không scale.
# base_link mesh có origin ở mounting surface (đáy robot), còn `link_frames_urdf`
# trả base_link = J1 axis (cao hơn 330mm theo URDF joint_1_s). → bù -0.330m Z.
_GP7_MESH_MAP: tuple[tuple[str, str, tuple[float, float, float]], ...] = (
    ("base_link", "gp7_base_link.stl", (0.0, 0.0, -0.330)),
    ("link_S",    "gp7_link_1_s.stl",  (0.0, 0.0, 0.0)),
    ("link_L",    "gp7_link_2_l.stl",  (0.0, 0.0, 0.0)),
    ("link_U",    "gp7_link_3_u.stl",  (0.0, 0.0, 0.0)),
    ("link_R",    "gp7_link_4_r.stl",  (0.0, 0.0, 0.0)),
    ("link_B",    "gp7_link_5_b.stl",  (0.0, 0.0, 0.0)),
    ("link_T",    "gp7_link_6_t.stl",  (0.0, 0.0, 0.0)),
)


def _seg_box(next_mm, thick: float = 0.07):
    """(size_m, center_m) cho khối bao đoạn (0,0,0)→next trong link frame (mét)."""
    dx, dy, dz = (v / 1000.0 for v in next_mm)
    return ((abs(dx) + thick, abs(dy) + thick, abs(dz) + thick),
            (dx / 2.0, dy / 2.0, dz / 2.0))


def _gp7_box_specs(model: URDFRobot):
    """List (link_name, size_m, center_m) cho 6 link + flange (fallback primitive)."""
    j = model.joints
    return [
        ("link_S", *_seg_box(j[1].origin_mm)),
        ("link_L", *_seg_box(j[2].origin_mm)),
        ("link_U", *_seg_box(j[3].origin_mm)),
        ("link_R", (0.07, 0.07, 0.07), (0.0, 0.0, 0.0)),
        ("link_B", (0.07, 0.07, 0.07), (0.0, 0.0, 0.0)),
        ("link_T", *_seg_box(model.flange_xyz_mm)),
        ("link_flange", (0.06, 0.06, 0.06), (0.0, 0.0, 0.0)),
    ]


def _rgba(rgb, a: float = 1.0) -> list[float]:
    return [float(rgb[0]), float(rgb[1]), float(rgb[2]), a]


class O3DGuiSimRobot(SimRobot):
    """SimRobot + viewport O3DVisualizer (Filament). Drop-in cho `robot=`.

    Khác Open3DSimRobot: GUI chạy trên main thread (`run_gui()`), thí nghiệm chạy
    worker thread. Mọi cập nhật hình học từ worker đi qua `post_to_main_thread`.
    """

    supports_cartesian_pose: bool = True

    def __init__(
        self,
        base_xyz: tuple[float, float, float] = (0.0, 0.0, 630.0),
        home_joints: list[float] | None = None,
        cell_config: Any = None,
        project_root: Any = None,
        minimal_build: bool = False,
        anim_duration_s: float = 0.6,
        fps: int = 60,
        window_size: tuple[int, int] = (1280, 800),
        **sim_kwargs: Any,
    ) -> None:
        super().__init__(home_joints=home_joints, base_xyz=base_xyz, **sim_kwargs)
        self._minimal_build = minimal_build
        self._model = gp7_urdf(base_xyz_mm=tuple(base_xyz))
        self._anim_steps = max(1, int(anim_duration_s * fps))
        self._anim_dt = anim_duration_s / self._anim_steps
        self._open = True
        self._running = False
        self._cam_done = False                        # camera set lazy ở tick sau render
        self._cam_tick = 0                            # đếm tick để defer setup_camera

        # Object follow-gripper state (world_T 4x4 ở MÉT cho mỗi object)
        self._objects: dict[str, dict[str, Any]] = {}
        self._grasped_name: str | None = None
        self._grasp_offset_m: np.ndarray | None = None

        # geom_name → frame_key của link nó bám theo (vd "gripper" → "link_tool0")
        self._dynamic: dict[str, str] = {}

        import open3d as o3d                       # lazy import
        import open3d.visualization.gui as gui
        import open3d.visualization.rendering as rendering
        self._o3d, self._gui, self._rendering = o3d, gui, rendering

        gui.Application.instance.initialize()
        self._vis = o3d.visualization.O3DVisualizer(
            "GP7 Digital Twin — Open3D GUI (Filament)", window_size[0], window_size[1])
        self._vis.show_settings = False               # panel ẩn cho gọn (menu mở lại được)
        # KHÔNG dùng show_axes built-in: nó auto-scale triad theo bounding box
        # của scene, mà dome gradient bán kính 25m kéo bbox lên ~50m → 3 trục
        # X/Y/Z dài khổng lồ bắn xuyên hết màn hình. Thay bằng triad nhỏ cố
        # định 0.3m ở gốc world (giống RoboDK) — xem _add_world_axes().
        self._vis.show_axes = False
        # Background fallback dark navy (sát đáy gradient) — khi user zoom vượt
        # ra ngoài dome sẽ thấy màu này thay vì trắng tinh.
        for fn in (
            lambda: self._vis.show_skybox(False),
            lambda: self._vis.set_background([0.031, 0.031, 0.133, 1.0], None),
        ):
            try:
                fn()
            except Exception:                         # noqa: BLE001
                pass
        # Dark navy gradient dome (RoboDK Free style): top dark → bottom
        # purple-blue. Inverted sphere + defaultUnlit + vertex colors.
        self._add_gradient_dome()
        # Boost ánh sáng cho defaultLit surfaces (robot/gripper/table) — nếu
        # không robot màu xanh sẽ chìm vào nền dark navy như RoboDK render.
        self._setup_lighting()
        # Triad nhỏ cố định ở gốc world (thay show_axes auto-scale khổng lồ).
        self._add_world_axes()

        # Dựng hình (add 1 lần; animate bằng transform)
        self._has_meshes = self._load_links(project_root)
        self._load_gripper(cell_config, project_root)
        self._add_static_scene(cell_config, project_root)

        # Pose home ban đầu (gọi trực tiếp — đang ở main thread, app chưa run).
        # KHÔNG set camera ở đây: event loop chưa chạy → widget size=0×0 → Filament
        # "Camera preconditions not met" + bỏ qua projection. Camera set lazy trong
        # _do() ở lần post đầu (lúc đó loop đã chạy, viewport đã có kích thước hợp lệ).
        self._apply_frames(self._joints, threadsafe=False)

        gui.Application.instance.add_window(self._vis)
        try:
            self._vis.set_on_close(self._on_close)
        except Exception:                             # noqa: BLE001
            pass
        # CONTINUOUS RENDER LOOP: mặc định event loop chỉ vẽ lại khi có sự kiện
        # chuột → robot "đứng im, phải rê chuột mới nhúc nhích". Bật animation mode
        # để loop tick liên tục và tự vẽ lại mỗi frame (~vsync) → robot tự chạy mượt.
        try:
            self._vis.set_on_animation_tick(self._on_anim_tick)
            self._vis.is_animating = True
        except Exception as e:                        # noqa: BLE001
            logger.debug("Không bật được animation tick: %s", e)
        logger.info("O3DVisualizer mở — GP7 base=%s (mesh=%s)",
                    list(base_xyz), "ros-industrial" if self._has_meshes else "primitive")
        logger.info("Điều khiển chuột: trái-kéo=xoay | phải-kéo=di chuyển | "
                    "lăn=zoom | (menu Settings để đổi Mouse Controls/ánh sáng)")

    # ── Materials ─────────────────────────────────────────────────────
    def _mat(self, rgba: list[float]):
        m = self._rendering.MaterialRecord()
        m.shader = "defaultLit"
        m.base_color = rgba
        return m

    def _setup_lighting(
        self,
        sun_intensity: float = 120000.0,
        ibl_intensity: float = 90000.0,
    ) -> None:
        """Tăng sáng cho mọi defaultLit surface (robot/gripper/table/object).

        Vì sao cần: O3DVisualizer + skybox-off + post-processing-off → exposure
        thấp, robot defaultLit chìm thành gần-đen trên nền dark navy (eff. ~0.3×
        base color). RoboDK render robot SÁNG rõ, vẫn có shading khối 3D.

        2 nguồn sáng:
          - Sun (directional): tạo shading khối 3D. Hướng từ phía camera
            (-X,-Y,trên) chiếu xuống → mặt robot hướng người xem được sáng.
          - IBL / indirect (ambient all-around): nâng sáng đều cả mặt khuất nắng
            để robot không có vùng đen tuyền.

        Hai hằng số intensity là KNOB chỉnh sáng — tăng/giảm nếu robot quá
        tối/cháy sáng (sàn + dome dùng defaultUnlit nên KHÔNG bị ảnh hưởng).
        """
        try:
            self._vis.set_ibl_intensity(ibl_intensity)
        except Exception as e:                          # noqa: BLE001
            logger.debug("set_ibl_intensity lỗi: %s", e)
        try:
            raw = self._vis.scene.scene                 # low-level rendering.Scene
            sun_dir = np.array([0.45, 0.45, -0.77], dtype=np.float32)
            raw.set_sun_light(
                sun_dir,
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
                sun_intensity,
            )
            raw.enable_sun_light(True)
            raw.set_indirect_light_intensity(ibl_intensity)
        except Exception as e:                          # noqa: BLE001
            logger.debug("setup_lighting (sun/indirect) lỗi: %s", e)

    def _add_world_axes(self, size_m: float = 0.3) -> None:
        """Triad nhỏ cố định ở gốc world (X=đỏ, Y=lục, Z=lam) — giống RoboDK.

        Thay cho `show_axes` built-in vốn scale theo bounding box scene; với
        dome 25m thì trục dài khổng lồ. create_coordinate_frame có sẵn vertex
        color cho 3 mũi tên → dùng defaultUnlit để hiện đúng màu, không bị
        scene lighting làm tối.
        """
        try:
            axes = self._o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=size_m, origin=[0.0, 0.0, 0.0])
            mat = self._rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.base_color = [1.0, 1.0, 1.0, 1.0]
            self._vis.add_geometry("__world_axes", axes, mat)
        except Exception as e:                          # noqa: BLE001
            logger.debug("Không thêm được world axes: %s", e)

    def _add_gradient_dome(
        self,
        radius_m: float = 25.0,
        n_lat: int = 32,
        n_lon: int = 64,
        top_rgb: tuple[int, int, int] = (5, 5, 28),       # near-black navy (top)
        bot_rgb: tuple[int, int, int] = (95, 65, 175),    # vivid purple-blue (bot)
        ease_power: float = 0.45,                          # push dark up
    ) -> None:
        """Bao scene bằng quả cầu lớn LẬT với vertex-color gradient theo trục Z.

        Dark navy gradient (RoboDK Free style): top gần đen-navy → bottom
        purple-blue. Ease 0.45 → top 60% gần như uniform dark, chỉ chuyển sang
        purple ở 1/3 dưới (match RoboDK).

        Camera đứng BÊN TRONG cầu, mặt trong là front-face → user thấy gradient
        mọi hướng nhìn. Bán kính 25m: vừa đủ để workspace ~2m không thấy edge
        cong, vừa đủ để camera 3/4 góc thấy gradient rõ.

        defaultUnlit shader: vertex colors hiện đúng RGB, không bị scene
        lighting ảnh hưởng.
        """
        try:
            n_lat = max(8, n_lat)
            n_lon = max(8, n_lon)
            phi = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
            theta = np.linspace(0.0, 2.0 * np.pi, n_lon)
            phis, thetas = np.meshgrid(phi, theta, indexing="ij")
            x = radius_m * np.cos(phis) * np.cos(thetas)
            y = radius_m * np.cos(phis) * np.sin(thetas)
            z = radius_m * np.sin(phis)
            verts = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

            tris = []
            for i in range(n_lat - 1):
                for j in range(n_lon - 1):
                    a = i * n_lon + j
                    b = a + 1
                    c = a + n_lon
                    d = c + 1
                    tris.append([a, c, b])               # winding CW (mặt trong)
                    tris.append([b, c, d])
            tris = np.asarray(tris, dtype=np.int32)

            # Vertex colors theo Z + easing → top stays dark longer (≈ RoboDK).
            # z_norm = 0 ở bottom, 1 ở top
            z_flat = verts[:, 2]
            z_norm = (z_flat - z_flat.min()) / (z_flat.max() - z_flat.min() + 1e-9)
            t = z_norm ** ease_power                     # đẩy dark lên top
            top = np.asarray(top_rgb, dtype=np.float32) / 255.0
            bot = np.asarray(bot_rgb, dtype=np.float32) / 255.0
            colors = (1.0 - t[:, None]) * bot + t[:, None] * top
            colors = np.clip(colors, 0.0, 1.0)

            mesh = self._o3d.geometry.TriangleMesh()
            mesh.vertices = self._o3d.utility.Vector3dVector(verts.astype(np.float64))
            mesh.triangles = self._o3d.utility.Vector3iVector(tris)
            mesh.vertex_colors = self._o3d.utility.Vector3dVector(
                colors.astype(np.float64))

            mat = self._rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.base_color = [1.0, 1.0, 1.0, 1.0]
            self._vis.add_geometry("__bg_dome", mesh, mat)

            # Tắt post-processing (bloom + tonemap) trên scene view — gradient
            # vertex color sẽ hiện đúng RGB, không bị tonemap compress về
            # màu trung tính làm "gradient phẳng".
            try:
                self._vis.scene.view.set_post_processing(False)
            except Exception as e:                       # noqa: BLE001
                logger.debug("set_post_processing(False) lỗi: %s", e)

            logger.debug("Gradient dome added (r=%.0fm, %d×%d)",
                         radius_m, n_lat, n_lon)
        except Exception as e:                          # noqa: BLE001
            logger.warning("Không thêm được gradient dome: %s — dùng solid bg",
                           e)

    # ── Dựng hình ─────────────────────────────────────────────────────
    def _load_links(self, project_root) -> bool:
        """Add 7 STL ros-industrial ở frame link-local; animate bằng transform.

        Mesh frame ≡ URDF link frame nên transform = link_frames_urdf output
        (mm→m). base_link bù -0.330m Z trong local (xem _GP7_MESH_MAP). Thiếu
        thư mục/mesh → fallback khối hộp primitive.
        """
        root = Path(project_root) if project_root else Path(".")
        mesh_dir = root / "models" / "gp7_links"
        loaded = 0
        if mesh_dir.exists():
            for key, fname, off in _GP7_MESH_MAP:
                path = mesh_dir / fname
                if not path.exists():
                    continue
                try:
                    m = self._o3d.io.read_triangle_mesh(str(path))
                    if m.is_empty():
                        continue
                    if any(off):
                        m.translate(list(off))
                    m.compute_vertex_normals()
                    self._vis.add_geometry(key, m, self._mat(_rgba(_YASKAWA_BLUE)))
                    self._dynamic[key] = key
                    loaded += 1
                except Exception as e:                # noqa: BLE001
                    logger.debug("Lỗi load mesh %s: %s", path, e)
        if loaded:
            logger.info("O3DVisualizer: %d/%d GP7 link mesh (ros-industrial)",
                        loaded, len(_GP7_MESH_MAP))
            return True
        # Fallback primitive: box ở frame link-local, animate bằng transform.
        for name, size, center in _gp7_box_specs(self._model):
            box = self._make_local_box(size, center)
            self._vis.add_geometry(name, box, self._mat(_rgba(
                _LINK_COLORS.get(name, (0.6, 0.6, 0.6)))))
            self._dynamic[name] = name
        return False

    def _make_local_box(self, size, center):
        sx, sy, sz = size
        m = self._o3d.geometry.TriangleMesh.create_box(sx, sy, sz)
        verts = np.asarray(m.vertices) + np.array(
            [center[0] - sx / 2, center[1] - sy / 2, center[2] - sz / 2])
        m.vertices = self._o3d.utility.Vector3dVector(verts)
        m.compute_vertex_normals()
        return m

    def _load_gripper(self, cell_config, project_root):
        """Gripper STL ở frame tool0-local, bám theo link_tool0 (xem legacy)."""
        if cell_config is None or not getattr(cell_config, "gripper", None):
            return
        cfg = cell_config.gripper
        if not getattr(cfg, "mesh", None):
            return
        root = Path(project_root) if project_root else Path(".")
        path = Path(cfg.mesh)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            logger.warning("Gripper mesh không có: %s — bỏ qua", path)
            return
        try:
            m = self._o3d.io.read_triangle_mesh(str(path))
            if m.is_empty():
                return
            m.scale(0.001, center=(0, 0, 0))          # mm → m
            m.translate([0.0, 0.0, 0.1])              # palm sát flange, bars ra tip
            m.compute_vertex_normals()
            self._vis.add_geometry("gripper", m, self._mat([0.78, 0.78, 0.80, 1.0]))
            self._dynamic["gripper"] = "link_tool0"
        except Exception as e:                        # noqa: BLE001
            logger.debug("Lỗi load gripper STL: %s", e)

    def _add_static_scene(self, cell_config, project_root):
        o3d = self._o3d
        # Sàn lớn ~4×3m, flat plane (2 triangles) với texture lát gạch xám —
        # thay vì solid color, tạo cảm giác floor thật của xưởng / studio.
        try:
            gx0, gx1, gy0, gy1 = -1.20, 2.80, -1.50, 1.50
            z = -0.001                                  # ngay dưới z=0 để tránh z-fighting

            verts = np.array([
                [gx0, gy0, z],                          # 0: BL
                [gx1, gy0, z],                          # 1: BR
                [gx1, gy1, z],                          # 2: TR
                [gx0, gy1, z],                          # 3: TL
            ], dtype=np.float64)
            tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
            # Triangle UVs: per-vertex per-triangle (6 UVs cho 2 tris). UV [0,1]
            # ánh xạ tới texture toàn sàn — texture đã chứa pattern lát gạch đầy đủ.
            tri_uvs = np.array([
                [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
                [0.0, 0.0], [1.0, 1.0], [0.0, 1.0],
            ], dtype=np.float64)

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(tris)
            mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)
            mesh.compute_vertex_normals()

            tex = self._make_floor_tile_texture(
                floor_w_m=(gx1 - gx0), floor_h_m=(gy1 - gy0),
                tile_size_m=0.40,
            )
            mat = self._rendering.MaterialRecord()
            # defaultUnlit: texture hiện đúng RGB design, KHÔNG bị scene
            # lighting (ambient + directional) darken xuống → tile sáng đúng
            # như ý muốn trên gradient dome dark.
            mat.shader = "defaultUnlit"
            mat.base_color = [1.0, 1.0, 1.0, 1.0]       # 1× với texture albedo
            if tex is not None:
                mat.albedo_img = tex
            else:
                mat.base_color = [0.78, 0.78, 0.80, 1.0]   # fallback gray sáng
            self._vis.add_geometry("ground", mesh, mat)
        except Exception as e:                        # noqa: BLE001
            logger.debug("Không thêm được ground: %s", e)
        self._add_cell_meshes(cell_config, project_root)

    def _make_floor_tile_texture(
        self,
        floor_w_m: float,
        floor_h_m: float,
        tile_size_m: float = 0.40,
        px_per_tile: int = 96,
    ):
        """Texture lát gạch xám: tile light gray + grout line đậm hơn, alternate
        shade kiểu checkerboard nhẹ.

        Texture phủ TOÀN sàn (UV [0,1] → cả floor), nên kích thước pixel = số
        tile × px_per_tile theo từng trục.
        """
        try:
            nx = max(1, int(round(floor_w_m / tile_size_m)))
            ny = max(1, int(round(floor_h_m / tile_size_m)))
            tex_w = nx * px_per_tile
            tex_h = ny * px_per_tile

            # Base: gạch trắng sáng (tile chính) — nổi RẤT rõ trên gradient
            # dome dark, robot màu nóng dễ thấy đứng trên.
            light = np.array([240, 240, 244], dtype=np.uint8)   # gần trắng
            # Alternate shade nhẹ (checker) — chỉ khác light ~10 đơn vị
            dark  = np.array([225, 225, 230], dtype=np.uint8)
            # Grout (đường ron giữa các ô) — gray sáng, nét vẽ rõ nhưng không chát
            grout = np.array([160, 160, 168], dtype=np.uint8)
            line_w = max(2, px_per_tile // 28)            # ~3px grout (mảnh hơn)

            img = np.broadcast_to(light, (tex_h, tex_w, 3)).copy()

            # Checker alternate shade
            for i in range(ny):
                for j in range(nx):
                    if (i + j) % 2 == 1:
                        y0 = i * px_per_tile + line_w
                        y1 = (i + 1) * px_per_tile - line_w
                        x0 = j * px_per_tile + line_w
                        x1 = (j + 1) * px_per_tile - line_w
                        if x0 < x1 and y0 < y1:
                            img[y0:y1, x0:x1] = dark

            # Grout lines (ngang + dọc)
            for i in range(ny + 1):
                y = i * px_per_tile
                img[max(0, y - line_w // 2):min(tex_h, y + line_w // 2 + 1), :] = grout
            for j in range(nx + 1):
                x = j * px_per_tile
                img[:, max(0, x - line_w // 2):min(tex_w, x + line_w // 2 + 1)] = grout

            return self._o3d.geometry.Image(img)
        except Exception as e:                          # noqa: BLE001
            logger.debug("Floor tile texture lỗi: %s", e)
            return None

    def _add_cell_meshes(self, cell_config, project_root):
        if cell_config is None:
            return
        o3d = self._o3d
        root = Path(project_root) if project_root else Path(".")

        def _static(name, mesh_rel, xyz_mm, rpy_deg, rgb):
            try:
                path = Path(mesh_rel)
                if not path.is_absolute():
                    path = root / path
                if not path.exists():
                    return
                m = o3d.io.read_triangle_mesh(str(path))
                if m.is_empty():
                    return
                m.scale(0.001, center=(0, 0, 0))
                R = m.get_rotation_matrix_from_xyz([math.radians(a) for a in rpy_deg])
                m.rotate(R, center=(0, 0, 0))
                m.translate([xyz_mm[0] / 1000.0, xyz_mm[1] / 1000.0, xyz_mm[2] / 1000.0])
                m.compute_vertex_normals()
                self._vis.add_geometry(name, m, self._mat(_rgba(rgb)))
            except Exception as e:                    # noqa: BLE001
                logger.debug("Bỏ qua mesh '%s': %s", mesh_rel, e)

        for attr, drgb in (("worktable", [0.52, 0.55, 0.58]),
                           ("robot_pedestal", [0.40, 0.40, 0.40]),
                           ("camera_mount", [0.50, 0.50, 0.50])):
            item = getattr(cell_config, attr, None)
            if item is not None and getattr(item, "mesh", None):
                rgb = getattr(item, "color_rgb", None) or drgb
                _static(attr, item.mesh, item.pose.xyz_mm, item.pose.rpy_deg, rgb)

        frames = {f.name: f for f in getattr(cell_config, "frames", [])}
        all_objs = list(getattr(cell_config, "objects", []))
        objs = all_objs[:1] if self._minimal_build else all_objs
        for obj in objs:
            if not getattr(obj, "mesh", None):
                continue
            pxyz = list(frames[obj.parent_frame].pose.xyz_mm) \
                if getattr(obj, "parent_frame", None) in frames else [0.0, 0.0, 0.0]
            off = list(obj.pose.xyz_mm) if getattr(obj, "pose", None) else [0, 0, 0]
            world_xyz_mm = [pxyz[k] + off[k] for k in range(3)]
            self._register_object(obj.name, obj.mesh, world_xyz_mm, root,
                                  rgb=[0.80, 0.75, 0.20])

    def _register_object(self, name, mesh_rel, world_xyz_mm, root, rgb):
        """Object STL như dynamic geometry — transform world cập nhật mỗi frame."""
        try:
            path = Path(mesh_rel)
            if not path.is_absolute():
                path = root / path
            if not path.exists():
                return
            m = self._o3d.io.read_triangle_mesh(str(path))
            if m.is_empty():
                return
            m.scale(0.001, center=(0, 0, 0))          # mm → m, ở object-local frame
            m.compute_vertex_normals()
            self._vis.add_geometry(name, m, self._mat(_rgba(rgb)))
            world_T = np.eye(4)
            world_T[:3, 3] = [v / 1000.0 for v in world_xyz_mm]
            self._vis.scene.set_geometry_transform(name, world_T)
            # Lưu initial_world_T để reset_scene() đầu mỗi trial — tránh object
            # "trôi" dần qua nhiều trial (khay đã gắp + thả không tự quay về
            # vị trí pick template).
            self._objects[name] = {
                "world_T": world_T.copy(),
                "initial_world_T": world_T.copy(),
            }
        except Exception as e:                        # noqa: BLE001
            logger.debug("Bỏ qua object '%s': %s", name, e)

    # ── Render / animate ──────────────────────────────────────────────
    def _object_updates(self, T_tool0_mm: np.ndarray | None):
        """List (name, world_T_m) cho mọi object; grasped bám tool0."""
        if not self._objects:
            return []
        if self._grasped_name and self._grasp_offset_m is not None \
                and self._grasped_name in self._objects and T_tool0_mm is not None:
            T_tool_m = T_tool0_mm.copy()
            T_tool_m[:3, 3] = T_tool0_mm[:3, 3] / 1000.0
            self._objects[self._grasped_name]["world_T"] = T_tool_m @ self._grasp_offset_m
        return [(name, o["world_T"]) for name, o in self._objects.items()]

    def _apply_frames(self, joints_deg, threadsafe: bool) -> None:
        """Tính transform world cho mọi geom rồi áp (trực tiếp hoặc post lên GUI).

        Skip nếu cửa sổ đã đóng (`_open=False`) — tránh post_to_main_thread
        treo khi Application đang teardown.
        """
        if not self._open:
            return
        frames = dict(link_frames_urdf(
            self._model, [math.radians(q) for q in joints_deg]))
        updates: list[tuple[str, np.ndarray]] = []
        for geom, fkey in self._dynamic.items():
            T = frames.get(fkey)
            if T is None:
                continue
            Tm = T.copy()
            Tm[:3, 3] = T[:3, 3] / 1000.0             # mm → m
            updates.append((geom, Tm))
        updates += self._object_updates(frames.get("link_tool0"))

        def _do():
            scene = self._vis.scene
            for name, Tm in updates:
                try:
                    scene.set_geometry_transform(name, Tm)
                except Exception:                     # noqa: BLE001
                    pass
            # Camera setup KHÔNG làm ở đây nữa: post_to_main_thread có thể fire
            # trước khi widget hoàn thành layout đầu tiên → Filament
            # `setProjection` thấy aspect ratio = 0/0 → in cảnh báo
            # "Camera preconditions not met" + dùng default projection. Camera
            # giờ được set lazy trong `_on_anim_tick` khi content_rect đã có
            # kích thước thật (xem `_on_anim_tick`).
            try:
                self._vis.post_redraw()
            except Exception:                         # noqa: BLE001
                pass

        if threadsafe:
            self._gui.Application.instance.post_to_main_thread(self._vis, _do)
        else:
            _do()

    def _animate(self, start_deg: list[float], end_deg: list[float]) -> None:
        if not self._open:
            return
        n = min(len(start_deg), len(end_deg))
        for s in range(1, self._anim_steps + 1):
            if not self._open:
                return
            t = s / self._anim_steps
            frame = [start_deg[k] + (end_deg[k] - start_deg[k]) * t for k in range(n)]
            self._apply_frames(frame, threadsafe=True)
            time.sleep(self._anim_dt)

    def _on_anim_tick(self, *args):
        """Tick liên tục khi is_animating=True → luôn vẽ lại (robot tự chạy).

        Trả REDRAW mỗi frame: cost vsync-capped, không đáng kể với cảnh này, và
        đảm bảo các transform vừa post từ worker hiện ngay, không cần rê chuột.

        Đây cũng là chỗ ĐÚNG để set camera lần đầu — NHƯNG phải đợi đủ vài
        tick. Lý do: `content_rect` trả ngay kích thước cấu hình (1280×800) từ
        constructor, KHÔNG phản ánh state thực của Filament view (vốn chỉ có
        viewport hợp lệ SAU khi render frame đầu). Tick fires TRƯỚC render mỗi
        frame; render đầu chỉ chạy SAU tick #1 → tick #2 trở đi mới đảm bảo
        view đã sized. Đợi tick ≥3 cho chắc (cộng thêm 1 frame buffer).
        """
        if not self._cam_done:
            self._cam_tick += 1
            if self._cam_tick >= 3:                   # 2 render frames đã chạy
                self._cam_done = True
                self._setup_camera()
        return self._o3d.visualization.O3DVisualizer.TickResult.REDRAW

    def _setup_camera(self):
        """Khung nhìn 3/4 ban đầu; fallback reset_camera_to_default.

        Robot làm việc về phía +X (bàn, vật, camera đều ở X≈700mm). Đặt camera ở
        phía +X nhìn ngược về -X → thấy MẶT TRƯỚC robot hướng ra người xem
        ("quay ra ngoài"), tay với về phía mình — giống góc nhìn RoboDK. Đổi dấu
        thành phần Y (-1.5 ↔ +1.5) nếu muốn xoay sang đường chéo bên kia.
        """
        try:
            center = np.array([0.35, 0.0, 0.50])
            eye = center + np.array([1.8, -1.5, 0.95])
            self._vis.setup_camera(60.0, center, eye, [0.0, 0.0, 1.0])
        except Exception as e:                        # noqa: BLE001
            logger.debug("setup_camera lỗi (%s) — dùng default", e)
            try:
                self._vis.reset_camera_to_default()
            except Exception:                         # noqa: BLE001
                pass

    # ── Real-mode mirror API ──────────────────────────────────────────
    def mirror_state(self, joints_deg: list[float]) -> None:
        """Render external joints (degrees) — không animation.

        Dùng cho real-mode digital twin: HSE backend poll Joints @10Hz từ
        YRC1000, DigitalTwinMirror callback gọi method này để render state THẬT
        của robot lên viewport (thread-safe qua post_to_main_thread).
        """
        if not self._open:
            return
        self._joints = list(joints_deg)
        self._apply_frames(list(joints_deg), threadsafe=True)

    # ── Override motion ───────────────────────────────────────────────
    def MoveJ(self, target: Any) -> None:
        start = list(self._joints)
        super().MoveJ(target)
        self._animate(start, list(self._joints))

    def MoveL(self, target: Any) -> None:
        start = list(self._joints)
        try:
            super().MoveL(target)
        finally:
            self._animate(start, list(self._joints))

    def timer(self, seconds: float) -> None:
        """Worker chỉ ngủ — GUI tự render trên main thread."""
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            if not self._open:
                return
            time.sleep(0.02)

    # ── Lifecycle ─────────────────────────────────────────────────────
    def run_gui(self, message: str = "") -> None:
        """BLOCKING — chạy event loop GUI trên MAIN thread tới khi user đóng."""
        if message:
            logger.info(message)
        self._running = True
        try:
            self._gui.Application.instance.run()      # block tới khi đóng cửa sổ
        finally:
            self._open = False
            self._running = False

    def _on_close(self) -> bool:
        """User đóng cửa sổ → set flag để worker abort sớm."""
        self._open = False
        # KHÔNG gọi Application.quit() ở đây — run() đang trên main thread sẽ
        # tự return sau khi cửa sổ cuối cùng đóng. Quit từ event handler có thể
        # deadlock với event loop đang chạy.
        return True                                   # cho phép đóng

    def spin(self, message: str = "") -> None:
        """Tương thích interface: nếu gọi từ worker, chờ tới khi cửa sổ đóng."""
        if message:
            logger.info(message)
        while self._open:
            time.sleep(0.05)

    def disconnect(self) -> None:
        """Tear-down: ép Application quit để Filament thả C-level threads.

        Gọi SAU khi run_gui() đã return (window đã đóng). Lý do explicit quit:
        Filament có background threads (asset loader, renderer pool) không tự
        chết khi run() return — Python process không exit. Application.quit()
        ép cleanup.
        """
        self._open = False
        try:
            inst = self._gui.Application.instance
            if inst is not None:
                inst.quit()
        except Exception as e:                          # noqa: BLE001
            logger.debug("Application.quit() lỗi: %s", e)

    def Stop(self) -> None:
        logger.debug("O3DGuiSimRobot.Stop() no-op")

    # ── Grasp tracking (object follows gripper) ───────────────────────
    def setDO(self, index: int, value: int) -> None:
        super().setDO(index, value)
        if not self._objects:
            return
        if value:
            self._grasp_nearest_object()
        else:
            self._release_object()

    def _current_tool0_world_mm(self) -> np.ndarray | None:
        try:
            frames = dict(link_frames_urdf(
                self._model, [math.radians(q) for q in self._joints]))
            return frames.get("link_tool0")
        except Exception:                             # noqa: BLE001
            return None

    def _grasp_nearest_object(self) -> None:
        T_tool0 = self._current_tool0_world_mm()
        if T_tool0 is None:
            return
        tool0_pos_m = T_tool0[:3, 3] / 1000.0
        best_name, best_d = None, float("inf")
        for name, obj in self._objects.items():
            d = float(np.linalg.norm(obj["world_T"][:3, 3] - tool0_pos_m))
            if d < best_d:
                best_name, best_d = name, d
        if best_name is None or best_d > 0.25:
            logger.debug("Grasp close nhưng không có object gần tool0 (<25cm)")
            return
        T_tool_m = T_tool0.copy()
        T_tool_m[:3, 3] = tool0_pos_m
        self._grasp_offset_m = np.linalg.inv(T_tool_m) @ self._objects[best_name]["world_T"]
        self._grasped_name = best_name
        logger.info("O3DVisualizer: grasped '%s' (d=%.3fm) — sẽ theo gripper",
                    best_name, best_d)

    def _release_object(self) -> None:
        if self._grasped_name:
            pos = self._objects[self._grasped_name]["world_T"][:3, 3].round(3).tolist()
            logger.info("O3DVisualizer: released '%s' tại %s", self._grasped_name, pos)
        self._grasped_name = None
        self._grasp_offset_m = None

    # ── Scene reset (gọi từ Orchestrator đầu mỗi trial) ───────────────
    def reset_scene(self) -> None:
        """Reset mọi object về `initial_world_T` + clear grasped state.

        Gọi đầu mỗi trial qua `Orchestrator.run_one_cycle` (duck-type
        `hasattr(robot, 'reset_scene')`). Tránh tình trạng khay sau khi
        gắp-thả ở vị trí PLACE không tự quay về PICK template để trial kế
        tiếp gắp lại — mock detection trả pose template gốc, nhưng visual
        mesh nằm lệch → robot "gắp vào không khí".
        """
        self._grasped_name = None
        self._grasp_offset_m = None
        for name, obj in self._objects.items():
            init_T = obj.get("initial_world_T")
            if init_T is None:
                continue
            obj["world_T"] = init_T.copy()
            try:
                self._gui.Application.instance.post_to_main_thread(
                    self._vis,
                    lambda n=name, T=init_T.copy(): (
                        self._vis.scene.set_geometry_transform(n, T)
                        if self._vis is not None else None
                    ),
                )
            except Exception as e:                          # noqa: BLE001
                logger.debug("reset_scene set_geometry_transform lỗi '%s': %s", name, e)
