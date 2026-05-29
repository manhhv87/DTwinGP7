"""
render_doc_figures.py
─────────────────────
Sinh ảnh minh hoạ scene 3D (offscreen, KHÔNG mở cửa sổ) cho tài liệu:
robot GP7, cell đầy đủ, camera frustum, quỹ đạo pick-place. Render bằng
pyvista off-screen — dùng đúng mesh + transform của app (`_GP7_MESH_MAP`,
`link_frames_urdf`, scale mm→m). Tái tạo ảnh bất cứ lúc nào:

    python scripts/render_doc_figures.py            # → docs/figures/*.png

Ảnh dùng trong README, GIOI_THIEU_PHAN_MEM.md, HUONG_DAN_GUI.md.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root → import src

from src.cell import CellConfig
from src.orchestrator.kinematics.urdf_chain import (
    gp7_urdf, link_frames_urdf, forward_kinematics_urdf)
from src.orchestrator.kinematics import interpolate_joints
from src.orchestrator.kinematics.pieper_gp7 import (
    inverse_kinematics_pieper_gp7_nearest)
from src.orchestrator.viewports.control_panel import _xyz_rpy_to_matrix
from src.orchestrator.viewports.open3d_gui_sim_robot import (
    _GP7_MESH_MAP, _YASKAWA_BLUE)

pv.OFF_SCREEN = True

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
MESH_DIR = ROOT / "models" / "gp7_links"
WINDOW = (1000, 780)

_OBJ_COLORS = {"tray": "#b0b4ba", "bottle": "#2e9e5b",
               "cup": "#2f6fd0", "bolt": "#e08a1e"}


def _autotrim(path, bg_rgb=None, pad=12, thresh=30, min_ink=6):
    """Cắt viền nền quanh ảnh dựa trên MẬT ĐỘ pixel theo hàng/cột.

    Dùng profile (bỏ qua viền/scrollbar mảnh chạy hết cạnh — thứ làm
    `getbbox` thất bại). bg_rgb=None → lấy màu góc dưới-phải làm nền (panel
    nền tối); (255,255,255) cho ảnh scene nền trắng.
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.int16)
    if bg_rgb is None:
        bg_rgb = img.getpixel((img.width - 3, img.height - 3))
    diff = np.abs(arr - np.array(bg_rgb, dtype=np.int16)).max(axis=2)
    mask = diff > thresh
    row_ink, col_ink = mask.sum(axis=1), mask.sum(axis=0)
    rows = np.where(row_ink > max(min_ink, int(0.012 * img.width)))[0]
    cols = np.where(col_ink > max(min_ink, int(0.012 * img.height)))[0]
    if len(rows) == 0 or len(cols) == 0:
        return
    l = max(0, int(cols[0]) - pad); t = max(0, int(rows[0]) - pad)
    r = min(img.width, int(cols[-1]) + pad + 1)
    b = min(img.height, int(rows[-1]) + pad + 1)
    img.crop((l, t, r, b)).save(path)


def _read(path: Path):
    try:
        return pv.read(str(path)) if path.exists() else None
    except Exception:
        return None


def add_robot(pl, model, q_rad, color=None):
    """Thêm 7 link mesh GP7 vào plotter ở tư thế q_rad (radian)."""
    frames = dict(link_frames_urdf(model, q_rad))
    for key, fname, off in _GP7_MESH_MAP:
        m = _read(MESH_DIR / fname)
        if m is None:
            continue
        if any(off):
            m.translate([off[0], off[1], off[2]], inplace=True)
        T = frames.get(key)
        if T is not None:
            Tm = np.array(T, float).copy()
            Tm[:3, 3] = Tm[:3, 3] / 1000.0          # mm → m
            m.transform(Tm, inplace=True)
        pl.add_mesh(m, color=list(color or _YASKAWA_BLUE), smooth_shading=True)


def add_static(pl, mesh_rel, xyz_mm, rpy_deg, color, opacity=1.0):
    m = _read(ROOT / mesh_rel)
    if m is None:
        return
    m.points *= 0.001                               # mm → m
    T = _xyz_rpy_to_matrix(*xyz_mm, *rpy_deg)
    T[:3, 3] /= 1000.0
    m.transform(T, inplace=True)
    pl.add_mesh(m, color=color, smooth_shading=True, opacity=opacity)


def add_cell(pl, cfg, with_objects=True):
    """Thêm sàn/bàn/pedestal/giàn camera (+ vật) từ CellConfig."""
    for attr, default_col in (("floor", [0.82, 0.82, 0.86]),
                              ("worktable", [0.52, 0.55, 0.58]),
                              ("robot_pedestal", [0.4, 0.4, 0.4]),
                              ("camera_mount", [0.30, 0.32, 0.35])):
        c = getattr(cfg, attr, None)
        if c is None:
            continue
        col = list(getattr(c, "color_rgb", default_col) or default_col)
        add_static(pl, c.mesh, c.pose.xyz_mm, c.pose.rpy_deg, col)
    if with_objects:
        frame_xy = {f.name: f.pose.xyz_mm for f in cfg.frames}
        for o in cfg.objects:
            base = frame_xy.get(o.parent_frame, (0.0, 0.0, 0.0))
            off = o.pose.xyz_mm if o.pose else (0.0, 0.0, 0.0)
            world = [base[i] + off[i] for i in range(3)]
            add_static(pl, o.mesh, world, (0, 0, 0),
                       _OBJ_COLORS.get(o.name, "#cccccc"))


def add_frustum(pl, cam, table_top_mm=500.0):
    """Vẽ frustum (nón nhìn) camera xuống mặt bàn — wireframe + mặt ảnh mờ."""
    if cam is None:
        return
    cx, cy, cz = cam.pose.xyz_mm
    hfov, vfov = (cam.intrinsics.hfov_vfov_deg()
                  if cam.intrinsics else (87.0, 56.0))
    depth = (cz - table_top_mm)
    hw = depth * math.tan(math.radians(hfov) / 2.0)
    hh = depth * math.tan(math.radians(vfov) / 2.0)
    apex = np.array([cx, cy, cz]) / 1000.0
    corners_mm = [(cx - hw, cy - hh, table_top_mm),
                  (cx + hw, cy - hh, table_top_mm),
                  (cx + hw, cy + hh, table_top_mm),
                  (cx - hw, cy + hh, table_top_mm)]
    corners = [np.array(c) / 1000.0 for c in corners_mm]
    for c in corners:
        pl.add_mesh(pv.Line(apex, c), color="#1565c0", line_width=2)
    rect = corners + [corners[0]]
    for a, b in zip(rect, rect[1:]):
        pl.add_mesh(pv.Line(a, b), color="#1565c0", line_width=2)
    plane = pv.Quadrilateral(np.array(corners))
    pl.add_mesh(plane, color="#1565c0", opacity=0.12)


def tcp_path(model, q_waypoints_rad):
    """Nội suy joint → FK → list điểm TCP (m) cho polyline quỹ đạo."""
    samples = interpolate_joints(q_waypoints_rad, dt=0.05,
                                 max_joint_speed_deg_s=60.0)
    pts = []
    for s in samples:
        T = forward_kinematics_urdf(model, s.joints_rad)
        pts.append(T[:3, 3] / 1000.0)
    return np.array(pts)


def _save(pl, name, cpos, zoom=1.0):
    pl.set_background("white")
    pl.camera_position = cpos
    if zoom != 1.0:
        pl.camera.zoom(zoom)
    pl.add_axes()
    pl.screenshot(str(OUT / name))
    pl.close()
    _autotrim(OUT / name, bg_rgb=(255, 255, 255))     # cắt nền trắng
    print("[ok]", name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = CellConfig.from_yaml(str(ROOT / "config" / "cell_layout.yaml"))
    base = tuple(float(v) for v in cfg.robot.pose.xyz_mm)
    model = gp7_urdf(base_xyz_mm=base)
    home = [math.radians(q) for q in cfg.robot.home_joints_deg]

    iso = [(2.6, -2.6, 2.2), (0.5, 0.0, 0.45), (0, 0, 1)]
    cell_cam = [(2.9, -2.9, 2.4), (0.6, 0.0, 0.55), (0, 0, 1)]

    # 1) Robot GP7 (home pose) — tập trung robot, VẪN GIỮ nền sàn
    pl = pv.Plotter(off_screen=True, window_size=WINDOW)
    add_robot(pl, model, home)
    add_static(pl, "models/floor.stl", cfg.floor.pose.xyz_mm,
               cfg.floor.pose.rpy_deg, [0.85, 0.85, 0.88], opacity=0.6)
    robot_cam = [(2.5, -2.5, 1.85), (0.15, 0.0, 0.62), (0, 0, 1)]
    _save(pl, "gp7_robot.png", robot_cam, zoom=1.12)

    # 2) Cell đầy đủ (robot + bàn + pedestal + giàn camera + vật)
    pl = pv.Plotter(off_screen=True, window_size=WINDOW)
    add_robot(pl, model, home)
    add_cell(pl, cfg)
    _save(pl, "cell_overview.png", cell_cam)

    # 3) Camera frustum trên cell
    pl = pv.Plotter(off_screen=True, window_size=WINDOW)
    add_robot(pl, model, home)
    add_cell(pl, cfg)
    add_frustum(pl, getattr(cfg, "camera", None))
    _save(pl, "camera_frustum.png", cell_cam)

    # 4) Quỹ đạo pick-place (TCP path)
    grasp = _xyz_rpy_to_matrix(700, -120, 560, 180, 0, 0)   # trên tray
    lift = _xyz_rpy_to_matrix(700, -120, 700, 180, 0, 0)
    place = _xyz_rpy_to_matrix(700, 120, 560, 180, 0, 0)
    place_lift = _xyz_rpy_to_matrix(700, 120, 700, 180, 0, 0)
    seq = [home]
    seed = home
    for T in (lift, grasp, lift, place_lift, place, place_lift):
        q = inverse_kinematics_pieper_gp7_nearest(model, T, seed)
        if q is not None:
            seq.append(list(q)); seed = q
    pl = pv.Plotter(off_screen=True, window_size=WINDOW)
    add_robot(pl, model, seq[2] if len(seq) > 2 else home)   # tư thế grasp
    add_cell(pl, cfg)
    if len(seq) >= 2:
        path = tcp_path(model, seq)
        pl.add_mesh(pv.Spline(path, 400), color="#d81b60", line_width=4)
        pl.add_mesh(pv.PolyData(path[::8]), color="#d81b60",
                    point_size=6, render_points_as_spheres=True)
    _save(pl, "pick_place_path.png", cell_cam)

    print("Done ->", OUT)


def capture_panels(out=OUT):
    """Chụp panel GUI THẬT (menu/Cell/Program/Camera/Controls) bằng widget.grab().

    Mở app GP7AppQt trên màn hình vài giây (cần OpenGL desktop — không chạy
    được trong sandbox headless). Viewport VTK không nằm trong widget.grab()
    nên dùng các ảnh scene 3D ở main() cho phần viewport.
    """
    import time
    from PyQt6.QtWidgets import QApplication
    from src.orchestrator.viewports.gp7_app_qt import GP7AppQt
    from src.orchestrator.viewports.qt_theme import apply_dark_theme

    out.mkdir(parents=True, exist_ok=True)
    cfg = CellConfig.from_yaml(str(ROOT / "config" / "cell_layout.yaml"))
    app = QApplication.instance() or QApplication([])
    apply_dark_theme(app)
    win = GP7AppQt(cell_config=cfg, project_root=ROOT)
    win.resize(1320, 900)
    win.show()

    def pump(n=40, dt=0.03):
        for _ in range(n):
            app.processEvents(); time.sleep(dt)

    pump(60)
    win.menuBar().grab().save(str(out / "panel_menubar.png"))
    _autotrim(out / "panel_menubar.png")             # cắt khoảng trống bên phải
    print("[ok] panel_menubar.png")
    for dock, nm in ((getattr(win, "_cell_tree_dock", None), "panel_cell.png"),
                     (getattr(win, "_program_dock", None), "panel_program.png"),
                     (getattr(win, "_camera_dock", None), "panel_camera.png"),
                     (getattr(win, "_jog_dock", None), "panel_controls.png")):
        if dock is None:
            continue
        try:
            win._open_dock_tab(dock, True); dock.raise_(); pump(18)
        except Exception:
            pass
        dock.grab().save(str(out / nm))
        _autotrim(out / nm)                          # cắt nền trống dưới panel
        print("[ok]", nm)
    app.quit()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Render ảnh minh hoạ cho tài liệu")
    ap.add_argument("--panels", action="store_true",
                    help="Chụp panel GUI thật (mở app GP7AppQt — cần OpenGL desktop)")
    args = ap.parse_args()
    if args.panels:
        capture_panels()
    else:
        main()
