"""
gp7_app.py
──────────
Unified GP7 Digital Twin GUI — **1 window**, 3D scene on the left + side panel
with tabs on the right. Uses `gui.Window + SceneWidget + TabControl` instead of
O3DVisualizer (which is a closed window that cannot host a side panel;
multi-window is not stable on Windows).

Tabs:
  * **Jog**: joint sliders θ1..θ6, Home/Zero, TCP pose readout, Tool/Reference
    frame combo, Cartesian jog ±X/Y/Z trans/rot, gripper Open/Close, jog frame
    + step size.
  * **View**: bg dome on/off, world axes on/off, ground tiles on/off, camera
    preset (Iso / Front / Top / Side).
  * **Cell**: list frames + objects in cell config (readonly readout).
  * **Run**: run demo motion sequence (jog through several poses) — verify
    dynamic scene + worker thread mirroring state into scene.

Sim-only. For real experiments use `scripts/03_run_experiment.py`.

Launcher: `python scripts/15_app.py`.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..backends.inform_codegen import InformJobBuilder
from ..kinematics.inverse_kinematics import inverse_kinematics_seeded
from ..kinematics.urdf_chain import URDFRobot, gp7_urdf, link_frames_urdf
from .control_panel import (
    _build_ref_frames,
    _build_tool_frames,
    _matrix_to_xyz_rpy_deg,
    _rotation_about_axis_3x3,
    _xyz_rpy_to_matrix,
)
from .open3d_gui_sim_robot import (
    _GP7_MESH_MAP,
    _LINK_COLORS,
    _YASKAWA_BLUE,
    _gp7_box_specs,
    _install_filament_stderr_filter,
    _rgba,
)
from .program_model import Instruction

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────

class GP7App:
    """Unified GP7 Digital Twin app — 1 window, scene + tabbed panel."""

    AXIS_NAMES = ("X", "Y", "Z")
    _AXIS_VEC = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    JOG_FRAMES = ("Tool Frame", "Reference Frame", "Base")

    # Camera presets (eye_offset, up). center = robot mid-Z (0.35, 0, 0.5).
    _CAM_PRESETS: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Iso":   (np.array([1.8, -1.5, 0.95]),  np.array([0.0, 0.0, 1.0])),
        "Front": (np.array([2.2,  0.0,  0.30]), np.array([0.0, 0.0, 1.0])),
        "Side":  (np.array([0.0, -2.5,  0.50]), np.array([0.0, 0.0, 1.0])),
        "Top":   (np.array([0.01, 0.01, 3.0]),  np.array([1.0, 0.0, 0.0])),
    }

    # ── Visual theme constants ────────────────────────────────────────
    # Light-on-dark scheme; pose values get RoboDK-like color-coded backgrounds.
    COLOR_HEADER       = (0.55, 0.85, 1.00)             # accent light blue
    COLOR_SUB          = (0.75, 0.78, 0.85)             # subdued sub-label
    COLOR_STATUS_INFO  = (0.85, 0.85, 0.90)
    COLOR_STATUS_OK    = (0.45, 0.95, 0.55)
    COLOR_STATUS_WARN  = (1.00, 0.78, 0.35)
    COLOR_STATUS_ERR   = (1.00, 0.45, 0.45)
    # XYZ + RxRyRz pose-value background colors (RoboDK convention).
    AXIS_BG_RGB = (
        (0.96, 0.55, 0.60),                              # X — pink/red
        (0.55, 0.92, 0.55),                              # Y — green
        (0.55, 0.70, 0.98),                              # Z — blue
        (0.55, 0.92, 0.98),                              # Rx — cyan
        (0.96, 0.65, 0.96),                              # Ry — magenta
        (0.96, 0.96, 0.55),                              # Rz — yellow
    )
    AXIS_FG_RGB = (0.04, 0.04, 0.06)                     # dark text on light bg

    def __init__(
        self,
        cell_config: Any,
        project_root: str | Path,
        window_size: tuple[int, int] = (1500, 920),
    ) -> None:
        _install_filament_stderr_filter()

        # Lazy import — Open3D is heavy; only import when the app opens.
        import open3d as o3d
        import open3d.visualization.gui as gui
        import open3d.visualization.rendering as rendering
        self._o3d = o3d
        self._gui = gui
        self._rendering = rendering

        self._cell_config = cell_config
        self._project_root = Path(project_root)

        # Robot model + state
        base_xyz = tuple(cell_config.robot.pose.xyz_mm) \
            if getattr(cell_config.robot, "pose", None) else (0.0, 0.0, 630.0)
        self._base_xyz = base_xyz
        self._model = gp7_urdf(base_xyz_mm=base_xyz)
        self._home_joints = list(cell_config.robot.home_joints_deg) \
            if getattr(cell_config.robot, "home_joints_deg", None) else [0.0] * 6
        self._joints = list(self._home_joints)

        # Frames
        self._tool_frames = _build_tool_frames(cell_config)
        self._ref_frames = _build_ref_frames(cell_config)
        self._tool_idx = len(self._tool_frames) - 1 if len(self._tool_frames) > 1 else 0
        self._ref_idx = 0
        self._jog_frame_idx = 0
        self._jog_step_mm = 10.0
        self._jog_step_deg = 5.0

        # Object follow-gripper state
        self._objects: dict[str, dict[str, Any]] = {}
        self._grasped_name: str | None = None
        self._grasp_offset_m: np.ndarray | None = None
        self._dynamic: dict[str, str] = {}            # geom → frame_key

        # Worker control
        self._demo_worker: threading.Thread | None = None
        self._demo_stop = threading.Event()

        # Panel visibility — hidden by default; use menu View > Show ... to toggle.
        self._panel_visible = False
        self._program_visible = False

        # Program state — RoboDK-style instruction list
        self._program: list[Instruction] = []
        self._prog_worker: threading.Thread | None = None
        self._prog_stop = threading.Event()

        # ─── GUI ───
        gui.Application.instance.initialize()
        self._window = gui.Application.instance.create_window(
            "GP7 Digital Twin - Integrated App", window_size[0], window_size[1])

        self._build_menu()
        self._build_scene_widget()
        self._build_left_panel()
        self._build_program_panel()
        self._build_status_bar()
        self._window.set_on_layout(self._on_layout)
        self._window.set_on_close(self._on_close)

        # Initial pose + readouts
        self._apply_joints(self._joints, animate=False)
        self._refresh_pose_readout()

        logger.info("GP7App: window opened %dx%d, base=%s, home=%s",
                    window_size[0], window_size[1], list(base_xyz),
                    [round(q, 1) for q in self._home_joints])

    # ══════════════════════════════════════════════════════════════════
    # Scene
    # ══════════════════════════════════════════════════════════════════

    def _build_scene_widget(self) -> None:
        gui = self._gui
        rendering = self._rendering
        self._scene_widget = gui.SceneWidget()
        self._scene_widget.scene = rendering.Open3DScene(self._window.renderer)
        # Bg fallback (at dome bottom) — when the user zooms past the dome boundary.
        try:
            self._scene_widget.scene.set_background([0.031, 0.031, 0.133, 1.0])
            self._scene_widget.scene.show_skybox(False)
        except Exception as e:                         # noqa: BLE001
            logger.debug("set_background/show_skybox error: %s", e)

        self._add_gradient_dome()
        self._setup_lighting()
        self._dome_visible = True
        self._axes_visible = True
        self._floor_visible = True

        self._has_meshes = self._load_robot_links()
        self._load_gripper()
        self._add_cell_meshes()
        self._add_floor()
        self._add_world_axes()

        self._setup_camera_preset("Iso")
        self._window.add_child(self._scene_widget)

    def _mat_lit(self, rgba: list[float]):
        m = self._rendering.MaterialRecord()
        m.shader = "defaultLit"
        m.base_color = rgba
        return m

    def _mat_unlit(self, rgba: list[float] = (1.0, 1.0, 1.0, 1.0), tex=None):
        m = self._rendering.MaterialRecord()
        m.shader = "defaultUnlit"
        m.base_color = list(rgba)
        if tex is not None:
            m.albedo_img = tex
        return m

    def _add_gradient_dome(self) -> None:
        """Inverted 25m sphere enclosing the scene, gradient top dark→bottom purple-blue."""
        try:
            n_lat, n_lon = 32, 64
            radius_m = 25.0
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
                    tris.append([a, c, b])
                    tris.append([b, c, d])
            tris = np.asarray(tris, dtype=np.int32)

            z_flat = verts[:, 2]
            z_norm = (z_flat - z_flat.min()) / (z_flat.max() - z_flat.min() + 1e-9)
            t = z_norm ** 0.45
            top = np.array([5.0, 5.0, 28.0]) / 255.0
            bot = np.array([95.0, 65.0, 175.0]) / 255.0
            colors = (1.0 - t[:, None]) * bot + t[:, None] * top
            colors = np.clip(colors, 0.0, 1.0)

            mesh = self._o3d.geometry.TriangleMesh()
            mesh.vertices = self._o3d.utility.Vector3dVector(verts.astype(np.float64))
            mesh.triangles = self._o3d.utility.Vector3iVector(tris)
            mesh.vertex_colors = self._o3d.utility.Vector3dVector(
                colors.astype(np.float64))
            self._scene_widget.scene.add_geometry("__bg_dome", mesh,
                                                  self._mat_unlit())
            try:
                self._scene_widget.scene.view.set_post_processing(False)
            except Exception:                          # noqa: BLE001
                pass
        except Exception as e:                         # noqa: BLE001
            logger.warning("Failed to add gradient dome: %s", e)

    def _add_floor(self) -> None:
        """4×3m grey tile floor — flat plane + procedural texture."""
        try:
            gx0, gx1, gy0, gy1 = -1.20, 2.80, -1.50, 1.50
            z = -0.001
            verts = np.array([
                [gx0, gy0, z], [gx1, gy0, z], [gx1, gy1, z], [gx0, gy1, z],
            ], dtype=np.float64)
            tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
            tri_uvs = np.array([
                [0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1],
            ], dtype=np.float64)
            mesh = self._o3d.geometry.TriangleMesh()
            mesh.vertices = self._o3d.utility.Vector3dVector(verts)
            mesh.triangles = self._o3d.utility.Vector3iVector(tris)
            mesh.triangle_uvs = self._o3d.utility.Vector2dVector(tri_uvs)
            mesh.compute_vertex_normals()

            tex = self._make_floor_tile_texture(gx1 - gx0, gy1 - gy0, 0.40)
            self._scene_widget.scene.add_geometry(
                "__ground", mesh, self._mat_unlit(tex=tex))
        except Exception as e:                         # noqa: BLE001
            logger.debug("Failed to add ground: %s", e)

    def _make_floor_tile_texture(self, w_m, h_m, tile_size_m: float, px_per_tile: int = 96):
        try:
            nx = max(1, int(round(w_m / tile_size_m)))
            ny = max(1, int(round(h_m / tile_size_m)))
            tex_w = nx * px_per_tile
            tex_h = ny * px_per_tile
            light = np.array([240, 240, 244], dtype=np.uint8)
            dark = np.array([225, 225, 230], dtype=np.uint8)
            grout = np.array([160, 160, 168], dtype=np.uint8)
            line_w = max(2, px_per_tile // 28)
            img = np.broadcast_to(light, (tex_h, tex_w, 3)).copy()
            for i in range(ny):
                for j in range(nx):
                    if (i + j) % 2 == 1:
                        y0 = i * px_per_tile + line_w
                        y1 = (i + 1) * px_per_tile - line_w
                        x0 = j * px_per_tile + line_w
                        x1 = (j + 1) * px_per_tile - line_w
                        if x0 < x1 and y0 < y1:
                            img[y0:y1, x0:x1] = dark
            for i in range(ny + 1):
                yp = i * px_per_tile
                img[max(0, yp - line_w // 2):min(tex_h, yp + line_w // 2 + 1), :] = grout
            for j in range(nx + 1):
                xp = j * px_per_tile
                img[:, max(0, xp - line_w // 2):min(tex_w, xp + line_w // 2 + 1)] = grout
            return self._o3d.geometry.Image(img)
        except Exception as e:                         # noqa: BLE001
            logger.debug("Floor tile texture error: %s", e)
            return None

    def _add_world_axes(self, size_m: float = 0.3) -> None:
        try:
            axes = self._o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=size_m, origin=[0.0, 0.0, 0.0])
            self._scene_widget.scene.add_geometry("__world_axes", axes,
                                                  self._mat_unlit())
        except Exception as e:                         # noqa: BLE001
            logger.debug("Failed to add world axes: %s", e)

    def _setup_lighting(self) -> None:
        try:
            raw = self._scene_widget.scene.scene
            sun_dir = np.array([0.45, 0.45, -0.77], dtype=np.float32)
            raw.set_sun_light(sun_dir,
                              np.array([1.0, 1.0, 1.0], dtype=np.float32),
                              120000.0)
            raw.enable_sun_light(True)
            raw.set_indirect_light_intensity(90000.0)
        except Exception as e:                         # noqa: BLE001
            logger.debug("setup_lighting error: %s", e)

    def _load_robot_links(self) -> bool:
        """Load 7 ros-industrial GP7 STL meshes or fall back to primitives."""
        mesh_dir = self._project_root / "models" / "gp7_links"
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
                    self._scene_widget.scene.add_geometry(
                        key, m, self._mat_lit(_rgba(_YASKAWA_BLUE)))
                    self._dynamic[key] = key
                    loaded += 1
                except Exception as e:                 # noqa: BLE001
                    logger.debug("Error loading mesh %s: %s", path, e)
        if loaded:
            logger.info("GP7App: %d/%d GP7 link mesh (ros-industrial)",
                        loaded, len(_GP7_MESH_MAP))
            return True
        # Fallback primitive boxes
        for name, size, center in _gp7_box_specs(self._model):
            sx, sy, sz = size
            m = self._o3d.geometry.TriangleMesh.create_box(sx, sy, sz)
            verts = np.asarray(m.vertices) + np.array(
                [center[0] - sx / 2, center[1] - sy / 2, center[2] - sz / 2])
            m.vertices = self._o3d.utility.Vector3dVector(verts)
            m.compute_vertex_normals()
            self._scene_widget.scene.add_geometry(
                name, m, self._mat_lit(_rgba(_LINK_COLORS.get(name, (0.6, 0.6, 0.6)))))
            self._dynamic[name] = name
        return False

    def _load_gripper(self) -> None:
        cfg = getattr(self._cell_config, "gripper", None)
        if cfg is None or not getattr(cfg, "mesh", None):
            return
        path = Path(cfg.mesh)
        if not path.is_absolute():
            path = self._project_root / path
        if not path.exists():
            return
        try:
            m = self._o3d.io.read_triangle_mesh(str(path))
            if m.is_empty():
                return
            m.scale(0.001, center=(0, 0, 0))
            m.translate([0.0, 0.0, 0.1])
            m.compute_vertex_normals()
            self._scene_widget.scene.add_geometry(
                "gripper", m, self._mat_lit([0.78, 0.78, 0.80, 1.0]))
            self._dynamic["gripper"] = "link_tool0"
        except Exception as e:                         # noqa: BLE001
            logger.debug("Error loading gripper: %s", e)

    def _add_cell_meshes(self) -> None:
        cfg = self._cell_config
        for attr, drgb in (("worktable", [0.52, 0.55, 0.58]),
                           ("robot_pedestal", [0.40, 0.40, 0.40]),
                           ("camera_mount", [0.50, 0.50, 0.50])):
            item = getattr(cfg, attr, None)
            if item is None or not getattr(item, "mesh", None):
                continue
            rgb = getattr(item, "color_rgb", None) or drgb
            self._add_static(attr, item.mesh, item.pose.xyz_mm,
                             item.pose.rpy_deg, rgb)
        frames = {f.name: f for f in getattr(cfg, "frames", []) or []}
        for obj in getattr(cfg, "objects", []) or []:
            if not getattr(obj, "mesh", None):
                continue
            pxyz = list(frames[obj.parent_frame].pose.xyz_mm) \
                if getattr(obj, "parent_frame", None) in frames else [0, 0, 0]
            off = list(obj.pose.xyz_mm) if getattr(obj, "pose", None) else [0, 0, 0]
            world_xyz_mm = [pxyz[k] + off[k] for k in range(3)]
            self._register_object(obj.name, obj.mesh, world_xyz_mm,
                                  rgb=[0.80, 0.75, 0.20])

    def _add_static(self, name, mesh_rel, xyz_mm, rpy_deg, rgb) -> None:
        try:
            path = Path(mesh_rel)
            if not path.is_absolute():
                path = self._project_root / path
            if not path.exists():
                return
            m = self._o3d.io.read_triangle_mesh(str(path))
            if m.is_empty():
                return
            m.scale(0.001, center=(0, 0, 0))
            R = m.get_rotation_matrix_from_xyz([math.radians(a) for a in rpy_deg])
            m.rotate(R, center=(0, 0, 0))
            m.translate([xyz_mm[0] / 1000.0, xyz_mm[1] / 1000.0, xyz_mm[2] / 1000.0])
            m.compute_vertex_normals()
            self._scene_widget.scene.add_geometry(name, m, self._mat_lit(_rgba(rgb)))
        except Exception as e:                         # noqa: BLE001
            logger.debug("Skipping '%s': %s", mesh_rel, e)

    def _register_object(self, name, mesh_rel, world_xyz_mm, rgb) -> None:
        try:
            path = Path(mesh_rel)
            if not path.is_absolute():
                path = self._project_root / path
            if not path.exists():
                return
            m = self._o3d.io.read_triangle_mesh(str(path))
            if m.is_empty():
                return
            m.scale(0.001, center=(0, 0, 0))
            m.compute_vertex_normals()
            self._scene_widget.scene.add_geometry(name, m, self._mat_lit(_rgba(rgb)))
            world_T = np.eye(4)
            world_T[:3, 3] = [v / 1000.0 for v in world_xyz_mm]
            self._scene_widget.scene.set_geometry_transform(name, world_T)
            self._objects[name] = {
                "world_T": world_T.copy(),
                "initial_world_T": world_T.copy(),
            }
        except Exception as e:                         # noqa: BLE001
            logger.debug("Skipping object '%s': %s", name, e)

    def _setup_camera_preset(self, name: str) -> None:
        eye_offset, up = self._CAM_PRESETS.get(name, self._CAM_PRESETS["Iso"])
        center = np.array([0.35, 0.0, 0.50])
        eye = center + eye_offset
        try:
            bbox = self._o3d.geometry.AxisAlignedBoundingBox(
                [-2.0, -2.0, 0.0], [3.0, 2.0, 2.0])
            self._scene_widget.setup_camera(60.0, bbox, center)
            self._scene_widget.look_at(center, eye, up)
        except Exception as e:                         # noqa: BLE001
            logger.debug("setup_camera_preset error: %s", e)

    # ══════════════════════════════════════════════════════════════════
    # Animation (joint → scene transform)
    # ══════════════════════════════════════════════════════════════════

    def _apply_joints(self, joints_deg: list[float], animate: bool = False) -> None:
        """Set joints + update scene transforms. animate=True performs a short slerp."""
        if animate:
            self._animate_to(joints_deg)
            return
        self._joints = list(joints_deg)
        self._render_scene(joints_deg)
        self._refresh_joint_sliders()
        self._refresh_pose_readout()

    def _render_scene(self, joints_deg: list[float]) -> None:
        if not hasattr(self, "_scene_widget"):
            return
        frames = dict(link_frames_urdf(
            self._model, [math.radians(q) for q in joints_deg]))
        scene = self._scene_widget.scene
        for geom, fkey in self._dynamic.items():
            T = frames.get(fkey)
            if T is None:
                continue
            Tm = T.copy()
            Tm[:3, 3] = T[:3, 3] / 1000.0
            try:
                scene.set_geometry_transform(geom, Tm)
            except Exception:                          # noqa: BLE001
                pass
        # Object follow gripper
        T_tool0_mm = frames.get("link_tool0")
        if T_tool0_mm is not None:
            for name, obj in self._objects.items():
                if (self._grasped_name == name and self._grasp_offset_m is not None):
                    T_tool_m = T_tool0_mm.copy()
                    T_tool_m[:3, 3] = T_tool0_mm[:3, 3] / 1000.0
                    obj["world_T"] = T_tool_m @ self._grasp_offset_m
                try:
                    scene.set_geometry_transform(name, obj["world_T"])
                except Exception:                      # noqa: BLE001
                    pass

    def _animate_to(self, end_deg: list[float], steps: int = 36, dt: float = 0.02) -> None:
        """Slerp animation from current joints → end_deg, posting each frame to the main thread."""
        start = list(self._joints)
        n = min(len(start), len(end_deg))
        for s in range(1, steps + 1):
            t = s / steps
            frame = [start[k] + (end_deg[k] - start[k]) * t for k in range(n)]
            self._post_render(frame)
            time.sleep(dt)
        self._joints = list(end_deg)
        self._gui.Application.instance.post_to_main_thread(
            self._window, self._refresh_joint_sliders)
        self._gui.Application.instance.post_to_main_thread(
            self._window, self._refresh_pose_readout)

    def _post_render(self, joints_deg: list[float]) -> None:
        """Post _render_scene to the main thread (thread-safe call from worker)."""
        self._gui.Application.instance.post_to_main_thread(
            self._window, lambda j=list(joints_deg): self._render_scene(j))

    # ══════════════════════════════════════════════════════════════════
    # Menu bar / Toolbar / Left panel (RoboDK-style layout)
    # ══════════════════════════════════════════════════════════════════

    # Menu IDs (top-level menus: File / View / Camera / Robot / Tools / Help)
    _MID_FILE_EXIT = 1
    _MID_VIEW_DOME = 10
    _MID_VIEW_FLOOR = 11
    _MID_VIEW_AXES = 12
    _MID_VIEW_PANEL = 13
    _MID_CAM_ISO = 20
    _MID_CAM_FRONT = 21
    _MID_CAM_SIDE = 22
    _MID_CAM_TOP = 23
    _MID_ROBOT_HOME = 30
    _MID_ROBOT_ZERO = 31
    _MID_ROBOT_RESET = 32
    _MID_ROBOT_DEMO = 33
    _MID_TOOLS_CELLINFO = 40
    _MID_HELP_ABOUT = 50
    # Program menu
    _MID_PROG_SHOW   = 60
    _MID_PROG_PLAY   = 61
    _MID_PROG_STOP   = 62
    _MID_PROG_SAVE   = 63
    _MID_PROG_LOAD   = 64
    _MID_PROG_EXPORT = 65
    _MID_PROG_CLEAR  = 66

    def _build_menu(self) -> None:
        """5 top-level menus grouped by function:

            File   View   Robot   Run   Help

          • View   = all visual settings (camera submenu + scene toggles + panel)
          • Robot  = motion + cell information (same "robot+cell context")
          • Run    = time-based execution actions (demo, reset scene after demo)
        """
        gui = self._gui

        file_m = gui.Menu()
        file_m.add_item("Exit", self._MID_FILE_EXIT)

        # ── View: visual settings ──
        # 3 submenus: Camera (viewpoint) + Visibility (3D toggles) + 1 panel item.
        camera_sub = gui.Menu()
        camera_sub.add_item("Iso",   self._MID_CAM_ISO)
        camera_sub.add_item("Front", self._MID_CAM_FRONT)
        camera_sub.add_item("Side",  self._MID_CAM_SIDE)
        camera_sub.add_item("Top",   self._MID_CAM_TOP)

        # Visibility submenu — groups the 3 scene toggles (dome / floor / axes)
        # to avoid cluttering the top-level View menu.
        vis_sub = gui.Menu()
        vis_sub.add_item("Background dome", self._MID_VIEW_DOME)
        vis_sub.set_checked(self._MID_VIEW_DOME, True)
        vis_sub.add_item("Floor tiles", self._MID_VIEW_FLOOR)
        vis_sub.set_checked(self._MID_VIEW_FLOOR, True)
        vis_sub.add_item("World axes", self._MID_VIEW_AXES)
        vis_sub.set_checked(self._MID_VIEW_AXES, True)
        self._vis_menu = vis_sub                       # ref for set_checked

        view_m = gui.Menu()
        view_m.add_menu("Camera",     camera_sub)
        view_m.add_menu("Visibility", vis_sub)
        view_m.add_separator()
        view_m.add_item("Controls panel", self._MID_VIEW_PANEL)
        view_m.set_checked(self._MID_VIEW_PANEL, False)
        self._view_menu = view_m
        self._geom_visible = {
            self._MID_VIEW_DOME: True,
            self._MID_VIEW_FLOOR: True,
            self._MID_VIEW_AXES: True,
        }

        # ── Robot: motion + cell context ──
        robot_m = gui.Menu()
        robot_m.add_item("Home", self._MID_ROBOT_HOME)
        robot_m.add_item("Zero", self._MID_ROBOT_ZERO)
        robot_m.add_separator()
        robot_m.add_item("Cell info...", self._MID_TOOLS_CELLINFO)

        # ── Run: runtime/execution ──
        run_m = gui.Menu()
        run_m.add_item("Demo motion (start/stop)", self._MID_ROBOT_DEMO)
        run_m.add_item("Reset scene (restore objects)", self._MID_ROBOT_RESET)

        # ── Program: RoboDK-style instruction list ──
        prog_m = gui.Menu()
        prog_m.add_item("Program panel", self._MID_PROG_SHOW)
        prog_m.set_checked(self._MID_PROG_SHOW, False)
        prog_m.add_separator()
        prog_m.add_item("Play", self._MID_PROG_PLAY)
        prog_m.add_item("Stop", self._MID_PROG_STOP)
        prog_m.add_separator()
        prog_m.add_item("Clear all", self._MID_PROG_CLEAR)
        prog_m.add_item("Save .json...", self._MID_PROG_SAVE)
        prog_m.add_item("Load .json...", self._MID_PROG_LOAD)
        prog_m.add_item("Export .JBI (Yaskawa INFORM)...", self._MID_PROG_EXPORT)
        self._prog_menu = prog_m

        help_m = gui.Menu()
        help_m.add_item("About...", self._MID_HELP_ABOUT)

        menubar = gui.Menu()
        menubar.add_menu("File",    file_m)
        menubar.add_menu("View",    view_m)
        menubar.add_menu("Robot",   robot_m)
        menubar.add_menu("Run",     run_m)
        menubar.add_menu("Program", prog_m)
        menubar.add_menu("Help",    help_m)
        gui.Application.instance.menubar = menubar

        w = self._window
        w.set_on_menu_item_activated(self._MID_FILE_EXIT, lambda: w.close())
        w.set_on_menu_item_activated(
            self._MID_VIEW_DOME,
            lambda: self._menu_toggle_geom("__bg_dome", self._MID_VIEW_DOME))
        w.set_on_menu_item_activated(
            self._MID_VIEW_FLOOR,
            lambda: self._menu_toggle_geom("__ground", self._MID_VIEW_FLOOR))
        w.set_on_menu_item_activated(
            self._MID_VIEW_AXES,
            lambda: self._menu_toggle_geom("__world_axes", self._MID_VIEW_AXES))
        w.set_on_menu_item_activated(self._MID_VIEW_PANEL, self._on_toggle_panel)
        for mid, name in ((self._MID_CAM_ISO, "Iso"),
                          (self._MID_CAM_FRONT, "Front"),
                          (self._MID_CAM_SIDE, "Side"),
                          (self._MID_CAM_TOP, "Top")):
            w.set_on_menu_item_activated(
                mid, (lambda n=name: self._setup_camera_preset(n)))
        w.set_on_menu_item_activated(self._MID_ROBOT_HOME,  self._on_home)
        w.set_on_menu_item_activated(self._MID_ROBOT_ZERO,  self._on_zero)
        w.set_on_menu_item_activated(self._MID_ROBOT_RESET, self._on_reset_scene)
        w.set_on_menu_item_activated(self._MID_ROBOT_DEMO,  self._on_demo_toggle)
        w.set_on_menu_item_activated(self._MID_TOOLS_CELLINFO, self._show_cell_info)
        w.set_on_menu_item_activated(self._MID_HELP_ABOUT, self._show_about)
        # Program menu
        w.set_on_menu_item_activated(self._MID_PROG_SHOW,   self._on_toggle_program)
        w.set_on_menu_item_activated(self._MID_PROG_PLAY,   self._on_prog_play)
        w.set_on_menu_item_activated(self._MID_PROG_STOP,   self._on_prog_stop)
        w.set_on_menu_item_activated(self._MID_PROG_CLEAR,  self._on_prog_clear)
        w.set_on_menu_item_activated(self._MID_PROG_SAVE,   self._on_prog_save_dlg)
        w.set_on_menu_item_activated(self._MID_PROG_LOAD,   self._on_prog_load_dlg)
        w.set_on_menu_item_activated(self._MID_PROG_EXPORT, self._on_prog_export_dlg)

    def _menu_toggle_geom(self, geom_name: str, mid: int) -> None:
        new = not self._geom_visible.get(mid, True)
        self._geom_visible[mid] = new
        self._toggle_geom(geom_name, new)
        # Item lives in submenu View > Visibility (not in the view_m root).
        try:
            self._vis_menu.set_checked(mid, new)
        except Exception:                              # noqa: BLE001
            pass
        self._set_status(f"{geom_name.lstrip('_')}: {'on' if new else 'off'}")

    def _on_toggle_panel(self) -> None:
        """Toggle side panel visibility via menu View > Controls panel."""
        self._panel_visible = not self._panel_visible
        self._left_panel.visible = self._panel_visible
        try:
            self._view_menu.set_checked(self._MID_VIEW_PANEL, self._panel_visible)
        except Exception:                              # noqa: BLE001
            pass
        try:
            self._window.set_needs_layout()
        except Exception:                              # noqa: BLE001
            self._window.post_redraw()
        self._set_status(
            "Controls panel: " + ("shown" if self._panel_visible else "hidden"),
            level="ok")

    def _on_toggle_program(self) -> None:
        """Toggle program panel visibility via menu Program > Program panel."""
        self._program_visible = not self._program_visible
        self._prog_panel.visible = self._program_visible
        try:
            self._prog_menu.set_checked(self._MID_PROG_SHOW, self._program_visible)
        except Exception:                              # noqa: BLE001
            pass
        try:
            self._window.set_needs_layout()
        except Exception:                              # noqa: BLE001
            self._window.post_redraw()
        self._set_status(
            "Program panel: " + ("shown" if self._program_visible else "hidden"),
            level="ok")

    # ── Program: data ops ──
    def _refresh_program_list(self) -> None:
        if not self._program:
            self._prog_list.set_items(["(empty)"])
            return
        self._prog_list.set_items(
            [f"{i+1:>2}. {ins.describe()}" for i, ins in enumerate(self._program)])

    def _on_prog_add_movej(self) -> None:
        self._program.append(Instruction(type="MoveJ", joints=list(self._joints)))
        self._refresh_program_list()
        self._set_status(f"Program +MoveJ (now {len(self._program)} steps)", "ok")

    def _on_prog_add_movel(self) -> None:
        T = self._current_tool_world()
        if T is None:
            return
        x, y, z, rx, ry, rz = _matrix_to_xyz_rpy_deg(T)
        self._program.append(Instruction(type="MoveL",
                                          tcp_pose=[x, y, z, rx, ry, rz]))
        self._refresh_program_list()
        self._set_status(f"Program +MoveL (now {len(self._program)} steps)", "ok")

    def _on_prog_add_gripper(self, close: bool) -> None:
        self._program.append(Instruction(type="SetGripper", gripper_close=close))
        self._refresh_program_list()
        self._set_status(
            f"Program +Gripper {'CLOSE' if close else 'OPEN'} "
            f"(now {len(self._program)} steps)", "ok")

    def _on_prog_add_wait(self) -> None:
        secs = float(self._prog_wait_secs.double_value)
        self._program.append(Instruction(type="Wait", wait_seconds=secs))
        self._refresh_program_list()
        self._set_status(
            f"Program +Wait {secs:.2f}s (now {len(self._program)} steps)", "ok")

    def _on_prog_delete(self) -> None:
        idx = int(getattr(self._prog_list, "selected_index", -1))
        if 0 <= idx < len(self._program):
            del self._program[idx]
            self._refresh_program_list()
            self._set_status(f"Deleted step {idx+1}", "ok")

    def _on_prog_move_up(self) -> None:
        idx = int(getattr(self._prog_list, "selected_index", -1))
        if 0 < idx < len(self._program):
            self._program[idx-1], self._program[idx] = \
                self._program[idx], self._program[idx-1]
            self._refresh_program_list()
            try: self._prog_list.selected_index = idx - 1
            except Exception: pass

    def _on_prog_move_down(self) -> None:
        idx = int(getattr(self._prog_list, "selected_index", -1))
        if 0 <= idx < len(self._program) - 1:
            self._program[idx+1], self._program[idx] = \
                self._program[idx], self._program[idx+1]
            self._refresh_program_list()
            try: self._prog_list.selected_index = idx + 1
            except Exception: pass

    def _on_prog_clear(self) -> None:
        self._program.clear()
        self._refresh_program_list()
        self._set_status("Program cleared", "ok")

    # ── Program: execution ──
    def _on_prog_play(self) -> None:
        if self._prog_worker is not None and self._prog_worker.is_alive():
            self._set_status("Program already running", "warn")
            return
        if not self._program:
            self._set_status("Program empty", "warn")
            return
        self._prog_stop.clear()
        self._set_status(f"Program: running ({len(self._program)} steps)", "ok")
        self._prog_worker = threading.Thread(
            target=self._play_program_loop, daemon=True)
        self._prog_worker.start()

    def _on_prog_stop(self) -> None:
        if self._prog_worker is not None and self._prog_worker.is_alive():
            self._prog_stop.set()
            self._set_status("Program: stopping...", "warn")

    def _post_status(self, msg: str, level: str = "info") -> None:
        self._gui.Application.instance.post_to_main_thread(
            self._window, lambda m=msg, l=level: self._set_status(m, level=l))

    def _play_program_loop(self) -> None:
        n = len(self._program)
        try:
            for i, ins in enumerate(self._program):
                if self._prog_stop.is_set():
                    self._post_status("Program: stopped", "warn")
                    return
                self._post_status(f"Step {i+1}/{n}: {ins.describe()}")
                if ins.type == "MoveJ":
                    self._animate_to(list(ins.joints), steps=40, dt=0.025)
                elif ins.type == "MoveL":
                    sol = self._solve_movel(ins.tcp_pose)
                    if sol is None:
                        self._post_status(f"Step {i+1}: IK fail, abort", "err")
                        return
                    self._animate_to(sol, steps=40, dt=0.025)
                elif ins.type == "SetGripper":
                    self._gui.Application.instance.post_to_main_thread(
                        self._window,
                        lambda c=ins.gripper_close: self._toggle_gripper(c))
                    time.sleep(0.25)
                elif ins.type == "Wait":
                    deadline = time.monotonic() + max(0.0, ins.wait_seconds)
                    while time.monotonic() < deadline:
                        if self._prog_stop.is_set(): return
                        time.sleep(0.05)
            self._post_status(f"Program: done ({n} steps)", "ok")
        except Exception as e:                             # noqa: BLE001
            self._post_status(f"Program error: {e}", "err")

    def _solve_movel(self, tcp_pose_6: list[float]) -> list[float] | None:
        """IK seeded URDF cho TCP target pose (X,Y,Z mm + Rx,Ry,Rz deg, world).
        Returns joints in degrees, or None on fail.
        """
        T_target = _xyz_rpy_to_matrix(*tcp_pose_6)
        T_flange_tool = self._tool_frames[self._tool_idx][1]
        T_target_tool0 = T_target @ np.linalg.inv(T_flange_tool)
        q_init = [math.radians(q) for q in self._joints]
        sol = inverse_kinematics_seeded(
            self._model, T_target_tool0, q_init,
            tol_mm=0.5, tol_rad=1e-3, max_iter=100)
        return [math.degrees(q) for q in sol] if sol is not None else None

    # ── Program: file ops (Save / Load JSON, Export .JBI) ──
    def _on_prog_save_dlg(self) -> None:
        gui = self._gui
        dlg = gui.FileDialog(
            gui.FileDialog.SAVE, "Save program (.json)", self._window.theme)
        dlg.add_filter(".json", "Program (JSON)")
        dlg.set_on_done(self._prog_save_to)
        dlg.set_on_cancel(self._window.close_dialog)
        self._window.show_dialog(dlg)

    def _prog_save_to(self, path: str) -> None:
        self._window.close_dialog()
        try:
            data = [ins.to_dict() for ins in self._program]
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._set_status(f"Saved {len(data)} steps to {Path(path).name}", "ok")
        except Exception as e:                             # noqa: BLE001
            self._set_status(f"Save failed: {e}", "err")

    def _on_prog_load_dlg(self) -> None:
        gui = self._gui
        dlg = gui.FileDialog(
            gui.FileDialog.OPEN, "Load program (.json)", self._window.theme)
        dlg.add_filter(".json", "Program (JSON)")
        dlg.set_on_done(self._prog_load_from)
        dlg.set_on_cancel(self._window.close_dialog)
        self._window.show_dialog(dlg)

    def _prog_load_from(self, path: str) -> None:
        self._window.close_dialog()
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self._program = [Instruction.from_dict(d) for d in data]
            self._refresh_program_list()
            self._set_status(
                f"Loaded {len(self._program)} steps from {Path(path).name}", "ok")
        except Exception as e:                             # noqa: BLE001
            self._set_status(f"Load failed: {e}", "err")

    def _on_prog_export_dlg(self) -> None:
        if not self._program:
            self._set_status("Program empty — nothing to export", "warn")
            return
        gui = self._gui
        dlg = gui.FileDialog(
            gui.FileDialog.SAVE, "Export Yaskawa INFORM .JBI",
            self._window.theme)
        dlg.add_filter(".JBI", "Yaskawa INFORM (.JBI)")
        dlg.set_on_done(self._prog_export_to)
        dlg.set_on_cancel(self._window.close_dialog)
        self._window.show_dialog(dlg)

    def _prog_export_to(self, path: str) -> None:
        self._window.close_dialog()
        try:
            stem = Path(path).stem[:32].replace(" ", "_") or "PROG"
            builder = InformJobBuilder(name=stem)
            pos_idx = 0
            for i, ins in enumerate(self._program):
                if ins.type == "MoveJ":
                    pname = f"P{pos_idx:03d}"
                    builder.add_position(pname, list(ins.joints))
                    builder.movj(pname)
                    pos_idx += 1
                elif ins.type == "MoveL":
                    sol = self._solve_movel(ins.tcp_pose)
                    if sol is None:
                        self._set_status(
                            f"Export aborted: IK fail at step {i+1}", "err")
                        return
                    pname = f"P{pos_idx:03d}"
                    builder.add_position(pname, sol)
                    builder.movl(pname)
                    pos_idx += 1
                elif ins.type == "SetGripper":
                    builder.dout(1, on=ins.gripper_close)  # OT#(1) = gripper
                elif ins.type == "Wait":
                    builder.timer(max(0.0, ins.wait_seconds))
            Path(path).write_bytes(builder.render().encode("utf-8"))
            self._set_status(
                f"Exported {pos_idx} positions to {Path(path).name}", "ok")
        except Exception as e:                             # noqa: BLE001
            self._set_status(f"Export .JBI failed: {e}", "err")

    def _show_about(self) -> None:
        gui = self._gui
        em = self._window.theme.font_size
        dlg = gui.Dialog("About")
        v = gui.Vert(int(0.3*em), gui.Margins(em, em, em, em))
        v.add_child(gui.Label("GP7 Digital Twin"))
        v.add_child(gui.Label("Yaskawa GP7 + Open3D Filament viewport"))
        v.add_child(gui.Label("Sim-only standalone app"))
        v.add_fixed(int(0.5 * em))
        ok = gui.Button("OK")
        ok.set_on_clicked(self._window.close_dialog)
        v.add_child(ok)
        dlg.add_child(v)
        self._window.show_dialog(dlg)

    def _build_status_bar(self) -> None:
        """Thin status bar at the bottom of the window — single line, does not obscure the scene."""
        gui = self._gui
        em = self._window.theme.font_size
        pad = int(0.3 * em)
        self._status_bar = gui.Horiz(int(0.4*em), gui.Margins(pad, pad, pad, pad))
        self._status_bar.add_child(self._make_sub("Status:"))
        self._status = gui.Label("Ready")
        self._status.text_color = gui.Color(*self.COLOR_STATUS_INFO)
        self._status_bar.add_child(self._status)
        self._status_bar.add_stretch()
        self._window.add_child(self._status_bar)

    def _build_left_panel(self) -> None:
        """Side panel: Joint Jog + Cartesian Jog only. Uses plain Vert (NOT
        CollapsableVert) because Open3D Vert mis-calculates the preferred-height
        of CollapsableVert → sections overlap when there are many children.
        Tree/Cell info moved to menu Tools > Cell info (Dialog) to keep the panel compact.
        """
        gui = self._gui
        em = self._window.theme.font_size
        pad = int(0.5 * em)
        self._left_panel = gui.Vert(int(0.3*em), gui.Margins(pad, pad, pad, pad))

        # ── Joint axis jog section ──
        self._left_panel.add_child(self._make_header("Joint axis jog"))
        joint_box = gui.Vert(int(0.2 * em),
                             gui.Margins(int(0.4*em), 0, 0, int(0.3*em)))
        self._build_joint_jog_into(joint_box, em)
        self._left_panel.add_child(joint_box)

        # ── Cartesian jog section ──
        self._left_panel.add_child(self._make_header("Cartesian jog"))
        cart_box = gui.Vert(int(0.2 * em),
                            gui.Margins(int(0.4*em), 0, 0, int(0.3*em)))
        self._build_cartesian_jog_into(cart_box, em)
        self._left_panel.add_child(cart_box)

        # Hidden by default — user clicks "Show controls" on the toolbar to show.
        self._left_panel.visible = self._panel_visible
        self._window.add_child(self._left_panel)

    def _build_program_panel(self) -> None:
        """Right panel: instruction list + Play/Save/Load/Export — RoboDK-style.

        Add / Delete / Up / Down operate on `self._program`. Play runs a worker
        thread sequentially through MoveJ → MoveL → SetGripper → Wait.
        Export .JBI uses InformJobBuilder (already available).
        """
        gui = self._gui
        em = self._window.theme.font_size
        pad = int(0.5 * em)
        self._prog_panel = gui.Vert(int(0.3*em), gui.Margins(pad, pad, pad, pad))

        self._prog_panel.add_child(self._make_header("Program"))

        # Instruction list
        self._prog_list = gui.ListView()
        self._prog_list.set_items(["(empty)"])
        self._prog_panel.add_child(self._prog_list)

        # Add row
        self._prog_panel.add_child(self._make_sub("Add (captures current pose/state)"))
        for row_btns in (
            (
                ("+ MoveJ", self._on_prog_add_movej),
                ("+ MoveL", self._on_prog_add_movel),
            ),
            (
                ("+ Grip OPEN",  lambda: self._on_prog_add_gripper(False)),
                ("+ Grip CLOSE", lambda: self._on_prog_add_gripper(True)),
            ),
        ):
            row = gui.Horiz(int(0.2 * em))
            for label, cb in row_btns:
                btn = gui.Button(label); btn.set_on_clicked(cb)
                row.add_child(btn)
            self._prog_panel.add_child(row)

        # Wait with editable seconds
        row = gui.Horiz(int(0.2 * em))
        row.add_child(gui.Label("Wait"))
        self._prog_wait_secs = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._prog_wait_secs.set_limits(0.0, 600.0)
        self._prog_wait_secs.double_value = 0.5
        row.add_child(self._prog_wait_secs)
        row.add_child(gui.Label("s"))
        add_wait = gui.Button("+ Wait")
        add_wait.set_on_clicked(self._on_prog_add_wait)
        row.add_child(add_wait)
        self._prog_panel.add_child(row)

        # Edit row
        self._prog_panel.add_fixed(int(0.2 * em))
        self._prog_panel.add_child(self._make_sub("Edit selected"))
        row = gui.Horiz(int(0.2 * em))
        for label, cb in (
            ("Delete",  self._on_prog_delete),
            ("Move up", self._on_prog_move_up),
            ("Move dn", self._on_prog_move_down),
        ):
            btn = gui.Button(label); btn.set_on_clicked(cb)
            row.add_child(btn)
        self._prog_panel.add_child(row)

        # Play/Stop
        self._prog_panel.add_fixed(int(0.2 * em))
        self._prog_panel.add_child(self._make_sub("Run"))
        row = gui.Horiz(int(0.2 * em))
        play_btn = gui.Button("Play"); play_btn.set_on_clicked(self._on_prog_play)
        stop_btn = gui.Button("Stop"); stop_btn.set_on_clicked(self._on_prog_stop)
        row.add_child(play_btn); row.add_child(stop_btn)
        self._prog_panel.add_child(row)

        # File ops
        self._prog_panel.add_fixed(int(0.2 * em))
        self._prog_panel.add_child(self._make_sub("File"))
        row = gui.Horiz(int(0.2 * em))
        for label, cb in (
            ("Save .json", self._on_prog_save_dlg),
            ("Load .json", self._on_prog_load_dlg),
        ):
            btn = gui.Button(label); btn.set_on_clicked(cb)
            row.add_child(btn)
        self._prog_panel.add_child(row)
        row = gui.Horiz(int(0.2 * em))
        exp_btn = gui.Button("Export .JBI")
        exp_btn.set_on_clicked(self._on_prog_export_dlg)
        clr_btn = gui.Button("Clear all")
        clr_btn.set_on_clicked(self._on_prog_clear)
        row.add_child(exp_btn); row.add_child(clr_btn)
        self._prog_panel.add_child(row)

        self._prog_panel.visible = self._program_visible
        self._window.add_child(self._prog_panel)

    def _show_cell_info(self) -> None:
        """Dialog showing cell config information (Robot / Frames / Objects)."""
        gui = self._gui
        em = self._window.theme.font_size
        dlg = gui.Dialog("Cell info")
        v = gui.Vert(int(0.25 * em), gui.Margins(em, em, em, em))

        v.add_child(self._make_header("Robot"))
        v.add_child(gui.Label(
            f"  Name:        {self._cell_config.robot.name}\n"
            f"  Base xyz:    {list(self._base_xyz)} mm\n"
            f"  Home joints: "
            + ", ".join(f"{q:+.2f}" for q in self._home_joints) + " deg"))
        v.add_fixed(int(0.4 * em))

        v.add_child(self._make_header(
            f"Reference frames ({len(self._ref_frames)})"))
        for name, T in self._ref_frames:
            x, y, z, rx, ry, rz = _matrix_to_xyz_rpy_deg(T)
            v.add_child(gui.Label(
                f"  - {name}: xyz=({x:.0f},{y:.0f},{z:.0f}) "
                f"rpy=({rx:.0f},{ry:.0f},{rz:.0f})"))
        v.add_fixed(int(0.4 * em))

        objs = list(getattr(self._cell_config, "objects", []) or [])
        v.add_child(self._make_header(f"Objects ({len(objs)})"))
        for o in objs:
            xyz = list(o.pose.xyz_mm) if getattr(o, "pose", None) else [0, 0, 0]
            parent = getattr(o, "parent_frame", "-")
            v.add_child(gui.Label(f"  - {o.name} ({parent}) xyz={xyz}"))

        v.add_fixed(int(0.5 * em))
        ok = gui.Button("Close")
        ok.set_on_clicked(self._window.close_dialog)
        v.add_child(ok)
        dlg.add_child(v)
        self._window.show_dialog(dlg)

    def _build_joint_jog_into(self, parent, em) -> None:
        """6 joint rows, each row has ONLY 'J#' + a full-width slider → perfectly
        uniform left/right slider positions.

        Min/max are NOT shown inline (Open3D uses a proportional font → " -65" and
        "-170" have different pixel widths → slider edges would be misaligned).
        Range info is available via:
          • Slider tooltip (hover → "J1 range: -170 to 170 deg")
          • Menu Robot > Cell info → dialog listing full home + ranges
        Current value is shown natively inside the slider track — no external
        value label needed.
        """
        gui = self._gui
        self._joint_sliders = []
        for i, joint in enumerate(self._model.joints):
            jmin = math.degrees(joint.joint_min)
            jmax = math.degrees(joint.joint_max)
            row = gui.Horiz(int(0.4 * em))
            jlbl = gui.Label(f" J{i+1} ")
            tip = f"J{i+1} range: {jmin:+.0f} to {jmax:+.0f} deg"
            try: jlbl.tooltip = tip
            except Exception: pass                     # noqa: BLE001
            row.add_child(jlbl)
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(jmin, jmax)
            slider.double_value = self._joints[i]
            try: slider.tooltip = tip
            except Exception: pass                     # noqa: BLE001
            slider.set_on_value_changed(
                (lambda v, idx=i: self._on_joint_slider(idx, float(v))))
            row.add_child(slider)
            self._joint_sliders.append(slider)
            parent.add_child(row)

    def _build_cartesian_jog_into(self, parent, em) -> None:
        """Cartesian jog — form-style with a 2-column VGrid to align labels and widgets.

        VGrid 2-col → column 1 holds Labels, width = max(preferred) → "Translate"
        sets the standard → all column-2 widgets start at the same X position.
        Completely fixes the misalignment of the previous Horiz-based rows.
        """
        gui = self._gui

        # ── Top form: Tool + Ref combos ──
        top_form = gui.VGrid(2, int(0.4 * em), gui.Margins(0, 0, 0, 0))
        top_form.add_child(gui.Label("Tool"))
        self._tool_combo = gui.Combobox()
        for n, _T in self._tool_frames: self._tool_combo.add_item(n)
        self._tool_combo.selected_index = self._tool_idx
        self._tool_combo.set_on_selection_changed(self._on_tool_changed)
        top_form.add_child(self._tool_combo)

        top_form.add_child(gui.Label("Ref"))
        self._ref_combo = gui.Combobox()
        for n, _T in self._ref_frames: self._ref_combo.add_item(n)
        self._ref_combo.selected_index = self._ref_idx
        self._ref_combo.set_on_selection_changed(self._on_ref_changed)
        top_form.add_child(self._ref_combo)
        parent.add_child(top_form)
        parent.add_fixed(int(0.3 * em))

        # ── Pose readouts (collapsable) ──
        tcp_sec = self._make_section(
            parent, "TCP pose (Tool / Reference)", open_default=True)
        self._tcp_pose_labels = self._add_pose_row(tcp_sec)

        more_sec = self._make_section(
            parent, "More poses (Tool / Flange + Ref / Base)",
            open_default=False)
        more_sec.add_child(self._make_sub("Tool w.r.t. Flange"))
        self._tool_pose_labels = self._add_pose_row(more_sec)
        more_sec.add_fixed(int(0.3 * em))
        more_sec.add_child(self._make_sub("Reference w.r.t. Base"))
        self._ref_pose_labels = self._add_pose_row(more_sec)

        parent.add_fixed(int(0.3 * em))

        # ── Bottom form: VGrid 2-col cho Jog inputs ──
        # Col 1 = Labels (width = max "Translate"), col 2 = widgets.
        # → all combos / button-rows start at the same X position.
        bot_form = gui.VGrid(2, int(0.4 * em), gui.Margins(0, 0, 0, 0))

        bot_form.add_child(gui.Label("Jog in"))
        self._jog_frame_combo = gui.Combobox()
        for n in self.JOG_FRAMES: self._jog_frame_combo.add_item(n)
        self._jog_frame_combo.selected_index = self._jog_frame_idx
        self._jog_frame_combo.set_on_selection_changed(self._on_jog_frame_changed)
        bot_form.add_child(self._jog_frame_combo)

        bot_form.add_child(gui.Label("Step"))
        step_row = gui.Horiz(int(0.2 * em))
        self._step_mm = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._step_mm.set_limits(0.1, 500.0); self._step_mm.double_value = self._jog_step_mm
        self._step_mm.set_on_value_changed(
            lambda v: setattr(self, "_jog_step_mm", float(v)))
        step_row.add_child(self._step_mm)
        step_row.add_child(gui.Label("mm"))
        self._step_deg = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._step_deg.set_limits(0.1, 90.0); self._step_deg.double_value = self._jog_step_deg
        self._step_deg.set_on_value_changed(
            lambda v: setattr(self, "_jog_step_deg", float(v)))
        step_row.add_child(self._step_deg)
        step_row.add_child(gui.Label("deg"))
        bot_form.add_child(step_row)

        bot_form.add_child(gui.Label("Translate"))
        bot_form.add_child(self._build_jog_buttons_only(em, self._on_translate))

        bot_form.add_child(gui.Label("Rotate"))
        bot_form.add_child(self._build_jog_buttons_only(em, self._on_rotate))

        bot_form.add_child(gui.Label("Gripper"))
        grip_row = gui.Horiz(int(0.2 * em))
        bo = gui.Button("Open"); bo.set_on_clicked(lambda: self._toggle_gripper(False))
        bc = gui.Button("Close"); bc.set_on_clicked(lambda: self._toggle_gripper(True))
        grip_row.add_child(bo); grip_row.add_child(bc)
        bot_form.add_child(grip_row)

        parent.add_child(bot_form)

    def _build_jog_btn_row(self, em, label, callback):
        gui = self._gui
        row = gui.Horiz(int(0.2 * em))
        row.add_child(gui.Label(label))
        for axis_idx, name in enumerate(self.AXIS_NAMES):
            for sign, sym in ((-1, "-"), (+1, "+")):
                btn = gui.Button(f"{sym}{name}")
                btn.set_on_clicked((lambda a=axis_idx, s=sign: callback(a, s)))
                row.add_child(btn)
            row.add_fixed(int(0.2 * em))
        return row

    def _build_jog_buttons_only(self, em, callback):
        """6 buttons ±X/Y/Z, no Label inside (Label is already in VGrid col 1)."""
        gui = self._gui
        row = gui.Horiz(int(0.2 * em))
        for axis_idx, name in enumerate(self.AXIS_NAMES):
            for sign, sym in ((-1, "-"), (+1, "+")):
                btn = gui.Button(f"{sym}{name}")
                btn.set_on_clicked((lambda a=axis_idx, s=sign: callback(a, s)))
                row.add_child(btn)
            row.add_fixed(int(0.15 * em))
        return row

    def _make_header(self, text: str, color_rgb=None):
        """Header label with accent color (light blue by default) — used for sections."""
        gui = self._gui
        lbl = gui.Label(text)
        rgb = color_rgb or self.COLOR_HEADER
        lbl.text_color = gui.Color(*rgb)
        return lbl

    def _make_sub(self, text: str):
        gui = self._gui
        lbl = gui.Label(text)
        lbl.text_color = gui.Color(*self.COLOR_SUB)
        return lbl

    def _make_section(self, parent, title: str, open_default: bool = True):
        """Custom collapsable section (Open3D CollapsableVert has a layout bug
        when nested inside a Vert with many children — manually build Button +
        Vert.visible toggle for precise control).

        Returns the inner Vert — caller calls add_child(...) on it.
        """
        gui = self._gui
        em = self._window.theme.font_size
        state = {"open": open_default}
        btn = gui.Button(("[-] " if open_default else "[+] ") + title)
        content = gui.Vert(int(0.2 * em),
                           gui.Margins(int(0.6*em), 0, 0, int(0.2*em)))
        content.visible = open_default

        def _toggle():
            state["open"] = not state["open"]
            content.visible = state["open"]
            btn.text = ("[-] " if state["open"] else "[+] ") + title
            try:
                self._window.set_needs_layout()
            except Exception:                          # noqa: BLE001
                self._window.post_redraw()

        btn.set_on_clicked(_toggle)
        parent.add_child(btn)
        parent.add_child(content)
        return content

    def _add_pose_row(self, parent, label_text: str = ""):
        """Pose readout = 2 SIMPLE Labels (XYZ row + RxRyRz row).

        The previous approach (6 cells with axis+value, VGrid or Horiz) would
        wrap when values were long. A single Label per row NEVER wraps because
        Label renders at its intrinsic text width — Open3D does not clip Label text.
        Trade-off: loses per-axis text color. Gain: COMPACT + NO wrap.
        """
        gui = self._gui
        if label_text:
            parent.add_child(self._make_sub(label_text))
        xyz = gui.Label("   X    +0.00     Y    +0.00     Z    +0.00")
        rpy = gui.Label("  Rx    +0.00    Ry    +0.00    Rz    +0.00")
        parent.add_child(xyz)
        parent.add_child(rpy)
        return (xyz, rpy)

    # ══════════════════════════════════════════════════════════════════
    # Callbacks
    # ══════════════════════════════════════════════════════════════════

    def _on_joint_slider(self, idx: int, value_deg: float) -> None:
        j = list(self._joints); j[idx] = value_deg
        self._apply_joints(j, animate=False)
        self._set_status(f"J{idx+1} = {value_deg:+.1f} deg")

    def _on_home(self) -> None:
        threading.Thread(target=self._animate_to, args=(list(self._home_joints),),
                         daemon=True).start()
        self._set_status("Move: Home", level="ok")

    def _on_zero(self) -> None:
        threading.Thread(target=self._animate_to, args=([0.0] * 6,),
                         daemon=True).start()
        self._set_status("Move: Zero", level="ok")

    def _on_tool_changed(self, _text, idx):
        self._tool_idx = int(idx); self._refresh_pose_readout()

    def _on_ref_changed(self, _text, idx):
        self._ref_idx = int(idx); self._refresh_pose_readout()

    def _on_jog_frame_changed(self, _text, idx):
        self._jog_frame_idx = int(idx)
        self._set_status(f"Jog frame = {self.JOG_FRAMES[idx]}")

    def _on_translate(self, axis_idx: int, sign: int) -> None:
        T_tool = self._current_tool_world()
        if T_tool is None: return
        axis_world = self._jog_axis_world(axis_idx, T_tool)
        T_target = T_tool.copy()
        T_target[:3, 3] = T_tool[:3, 3] + axis_world * (sign * self._jog_step_mm)
        self._apply_cartesian_target(T_target,
            f"T{'+' if sign>0 else '-'}{self.AXIS_NAMES[axis_idx]} "
            f"{self._jog_step_mm:.1f}mm")

    def _on_rotate(self, axis_idx: int, sign: int) -> None:
        T_tool = self._current_tool_world()
        if T_tool is None: return
        axis_world = self._jog_axis_world(axis_idx, T_tool)
        R_step = _rotation_about_axis_3x3(axis_world,
            sign * math.radians(self._jog_step_deg))
        T_target = T_tool.copy()
        T_target[:3, :3] = R_step @ T_tool[:3, :3]
        self._apply_cartesian_target(T_target,
            f"R{'+' if sign>0 else '-'}{self.AXIS_NAMES[axis_idx]} "
            f"{self._jog_step_deg:.1f} deg")

    def _toggle_gripper(self, close: bool) -> None:
        if close:
            self._grasp_nearest_object()
        else:
            self._release_object()
        self._render_scene(self._joints)
        self._set_status(f"Gripper {'CLOSE' if close else 'OPEN'}")

    def _on_reset_scene(self) -> None:
        self._grasped_name = None
        self._grasp_offset_m = None
        for name, obj in self._objects.items():
            obj["world_T"] = obj["initial_world_T"].copy()
        self._render_scene(self._joints)
        self._set_status("Scene reset")

    def _toggle_geom(self, name: str, visible: bool) -> None:
        try:
            self._scene_widget.scene.show_geometry(name, bool(visible))
        except Exception as e:                         # noqa: BLE001
            logger.debug("toggle %s error: %s", name, e)

    def _on_sun_intensity(self, val: float) -> None:
        try:
            self._scene_widget.scene.scene.set_sun_light(
                np.array([0.45, 0.45, -0.77], dtype=np.float32),
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
                float(val))
        except Exception:                              # noqa: BLE001
            pass

    def _on_ibl_intensity(self, val: float) -> None:
        try:
            self._scene_widget.scene.scene.set_indirect_light_intensity(float(val))
        except Exception:                              # noqa: BLE001
            pass

    def _on_demo_toggle(self) -> None:
        if self._demo_worker is not None and self._demo_worker.is_alive():
            self._demo_stop.set()
            self._set_status("Demo: stopping...")
            return
        self._demo_stop.clear()
        self._set_status("Demo: running", level="ok")
        self._demo_worker = threading.Thread(target=self._demo_loop, daemon=True)
        self._demo_worker.start()

    def _demo_loop(self) -> None:
        """Worker: cycle through 4 poses, ~1s each, loop until stop."""
        poses = [
            list(self._home_joints),
            [30, -30, 30, 0, 30, 0],
            [-30, 20, -40, 90, -30, 45],
            [0, 50, -80, 0, 60, 0],
        ]
        i = 0
        try:
            while not self._demo_stop.is_set():
                self._animate_to(poses[i % len(poses)], steps=40, dt=0.025)
                i += 1
                for _ in range(20):
                    if self._demo_stop.is_set(): break
                    time.sleep(0.05)
        finally:
            self._gui.Application.instance.post_to_main_thread(
                self._window,
                lambda: setattr(self._status, "text", "Demo: idle"))

    # ══════════════════════════════════════════════════════════════════
    # State helpers
    # ══════════════════════════════════════════════════════════════════

    def _refresh_joint_sliders(self) -> None:
        if not hasattr(self, "_joint_sliders"):
            return
        for i, s in enumerate(self._joint_sliders):
            s.double_value = self._joints[i]
            # Value shows in slider track natively; no external label needed.

    def _refresh_pose_readout(self) -> None:
        if not hasattr(self, "_tool_pose_labels"):
            return
        self._fill_pose_row(self._tool_pose_labels,
                            self._tool_frames[self._tool_idx][1])
        self._fill_pose_row(self._ref_pose_labels,
                            self._ref_frames[self._ref_idx][1])
        T_tool = self._current_tool_world()
        if T_tool is not None:
            T_world_base = np.eye(4); T_world_base[:3, 3] = self._base_xyz
            T_world_ref = T_world_base @ self._ref_frames[self._ref_idx][1]
            T_ref_tool = np.linalg.inv(T_world_ref) @ T_tool
            self._fill_pose_row(self._tcp_pose_labels, T_ref_tool)

    def _fill_pose_row(self, labels, T) -> None:
        x, y, z, rx, ry, rz = _matrix_to_xyz_rpy_deg(T)
        labels[0].text = f"   X {x:+8.2f}     Y {y:+8.2f}     Z {z:+8.2f}"
        labels[1].text = f"  Rx {rx:+8.2f}    Ry {ry:+8.2f}    Rz {rz:+8.2f}"

    def _set_status(self, msg: str, level: str = "info") -> None:
        color = {
            "info": self.COLOR_STATUS_INFO,
            "ok":   self.COLOR_STATUS_OK,
            "warn": self.COLOR_STATUS_WARN,
            "err":  self.COLOR_STATUS_ERR,
        }.get(level, self.COLOR_STATUS_INFO)
        try:
            self._status.text = msg
            self._status.text_color = self._gui.Color(*color)
        except Exception:                              # noqa: BLE001
            pass

    def _current_tool_world(self) -> np.ndarray | None:
        try:
            q_rad = [math.radians(q) for q in self._joints]
            T_tool0 = link_frames_urdf(self._model, q_rad)
            T_world_tool0 = dict(T_tool0).get("link_tool0")
            if T_world_tool0 is None:
                return None
            return T_world_tool0 @ self._tool_frames[self._tool_idx][1]
        except Exception as e:                         # noqa: BLE001
            self._set_status(f"FK error: {e}", level="err")
            return None

    def _jog_axis_world(self, axis_idx, T_world_tool):
        unit = self._AXIS_VEC[axis_idx]
        frame = self.JOG_FRAMES[self._jog_frame_idx]
        if frame == "Tool Frame":
            return T_world_tool[:3, :3] @ unit
        if frame == "Reference Frame":
            T_world_base = np.eye(4); T_world_base[:3, 3] = self._base_xyz
            T_world_ref = T_world_base @ self._ref_frames[self._ref_idx][1]
            return T_world_ref[:3, :3] @ unit
        return unit

    def _apply_cartesian_target(self, T_world_tool_target, source: str) -> None:
        T_flange_tool = self._tool_frames[self._tool_idx][1]
        T_world_tool0_target = T_world_tool_target @ np.linalg.inv(T_flange_tool)
        q_init = [math.radians(q) for q in self._joints]
        sol = inverse_kinematics_seeded(
            self._model, T_world_tool0_target, q_init,
            tol_mm=0.5, tol_rad=1e-3, max_iter=100)
        if sol is None:
            self._set_status(f"IK fail: {source}", level="err")
            return
        self._apply_joints([math.degrees(q) for q in sol], animate=False)
        self._set_status(source)

    # ── Gripper grasp/release ─────────────────────────────────────────
    def _grasp_nearest_object(self) -> None:
        T_tool0 = self._tool0_world()
        if T_tool0 is None or not self._objects:
            return
        tool0_pos_m = T_tool0[:3, 3] / 1000.0
        best_name, best_d = None, float("inf")
        for name, obj in self._objects.items():
            d = float(np.linalg.norm(obj["world_T"][:3, 3] - tool0_pos_m))
            if d < best_d:
                best_name, best_d = name, d
        if best_name is None or best_d > 0.25:
            return
        T_tool_m = T_tool0.copy(); T_tool_m[:3, 3] = tool0_pos_m
        self._grasp_offset_m = np.linalg.inv(T_tool_m) @ self._objects[best_name]["world_T"]
        self._grasped_name = best_name
        logger.info("GP7App: grasped '%s' d=%.3fm", best_name, best_d)

    def _release_object(self) -> None:
        if self._grasped_name:
            logger.info("GP7App: released '%s'", self._grasped_name)
        self._grasped_name = None
        self._grasp_offset_m = None

    def _tool0_world(self) -> np.ndarray | None:
        try:
            q_rad = [math.radians(q) for q in self._joints]
            return dict(link_frames_urdf(self._model, q_rad)).get("link_tool0")
        except Exception:                              # noqa: BLE001
            return None

    # ══════════════════════════════════════════════════════════════════
    # Layout + lifecycle
    # ══════════════════════════════════════════════════════════════════

    def _on_layout(self, _ctx) -> None:
        """Menu bar auto. 3-column content layout:

            ┌─[Controls?]──Scene──[Program?]─┐
            │                                │
            └─Status bar────────────────────┘

        Both panels are hidden by default; toggle via menu View / Program.
        """
        r = self._window.content_rect
        em = self._window.theme.font_size
        sb_h = int(2.0 * em)
        left_w  = max(420, int(28 * em)) if self._panel_visible   else 0
        right_w = max(420, int(28 * em)) if self._program_visible else 0
        content_h = r.height - sb_h

        if self._panel_visible:
            self._left_panel.frame = self._gui.Rect(
                r.x, r.y, left_w, content_h)
        if self._program_visible:
            self._prog_panel.frame = self._gui.Rect(
                r.x + r.width - right_w, r.y, right_w, content_h)
        self._scene_widget.frame = self._gui.Rect(
            r.x + left_w, r.y,
            max(100, r.width - left_w - right_w), content_h)
        self._status_bar.frame = self._gui.Rect(
            r.x, r.y + content_h, r.width, sb_h)

    def _on_close(self) -> bool:
        self._demo_stop.set()
        return True

    def run(self) -> None:
        try:
            self._gui.Application.instance.run()
        finally:
            try:
                self._gui.Application.instance.quit()
            except Exception:                          # noqa: BLE001
                pass
