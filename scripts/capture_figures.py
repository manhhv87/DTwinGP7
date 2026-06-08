"""
capture_figures.py
──────────────────
Tự động chụp một số ảnh holder cho docs/HUONG_DAN_GUI.md.

CHẠY TRÊN MÁY CÓ MÀN HÌNH / GPU (OpenGL). Không chạy được trong phiên headless
không có OpenGL — viewport 3D (pyvistaqt) cần context cửa sổ thật.

    .venv\\Scripts\\python.exe scripts\\capture_figures.py

Cửa sổ app sẽ bật lên trong giây lát rồi tự đóng; ảnh lưu vào docs/figures/.
Ảnh được chụp ở ĐỘ PHÂN GIẢI CAO (panel/dialog render 2×, viewport scale 2× +
SSAA), tự cắt nền và crop → hiển thị nét ở 1024px. Xem dòng SUMMARY cuối cùng.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

try:                                   # Windows console hay là cp1252 → ép UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                      # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

from PyQt6.QtCore import QPoint, Qt, QTimer                    # noqa: E402
from PyQt6.QtGui import QGuiApplication, QPainter, QPixmap     # noqa: E402
from PyQt6.QtWidgets import QApplication                       # noqa: E402

# Hệ số siêu lấy mẫu: render panel/viewport ở 2× rồi hiển thị nhỏ hơn → nét.
HIRES = 2

from src.orchestrator.viewports.gp7_app_qt import GP7AppQt    # noqa: E402
from src.orchestrator.viewports.qt_theme import apply_dark_theme  # noqa: E402

done: list[str] = []
todo: list[str] = []


def _grab_hires(widget, scale=HIRES):
    """Render widget Qt ở độ phân giải gấp `scale` lần (nét khi hiển thị 1024px).
    Dùng QPainter scale thay vì grab() native (vốn chỉ bằng kích thước widget)."""
    sz = widget.size()
    pm = QPixmap(sz.width() * scale, sz.height() * scale)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.scale(scale, scale)
    widget.render(p)
    p.end()
    return pm


def _smart_crop(path, edge_thr=30, exclude_corner=0.16, margin_frac=0.06, max_w=2048):
    """Cắt ảnh viewport về đúng vùng có nội dung (bỏ nền gradient + orientation widget
    góc dưới-trái), dùng bbox theo biên cạnh + percentile để bỏ điểm nhiễu lẻ."""
    try:
        from PIL import Image, ImageFilter
        im = Image.open(path).convert("RGB"); W, H = im.size
        g = np.asarray(im.convert("L").filter(ImageFilter.FIND_EDGES))
        mask = g > edge_thr
        cw, ch = int(exclude_corner * W), int(exclude_corner * H)
        mask[H - ch:, :cw] = False
        mask[:3, :] = mask[-3:, :] = mask[:, :3] = mask[:, -3:] = False
        ys, xs = np.where(mask)
        if len(xs) < 30:
            return
        x0, x1 = np.percentile(xs, [0.1, 99.9])
        y0, y1 = np.percentile(ys, [0.1, 99.9])
        mx, my = int(margin_frac * (x1 - x0)), int(margin_frac * (y1 - y0))
        box = (int(max(0, x0 - mx)), int(max(0, y0 - my)),
               int(min(W, x1 + mx)), int(min(H, y1 + my)))
        out = im.crop(box)
        if out.width > max_w:
            out = out.resize((max_w, int(out.height * max_w / out.width)),
                             Image.LANCZOS)
        out.save(path)
    except Exception:                                         # noqa: BLE001
        pass


def _pump(app, ms=400):
    end = time.time() + ms / 1000.0
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def _shot_view(win, name):
    try:
        win._plotter.render()
        win._plotter.screenshot(str(FIG / name), scale=HIRES)
        _smart_crop(str(FIG / name))
        done.append(name)
    except Exception as e:                                     # noqa: BLE001
        todo.append(f"{name} (viewport lỗi: {e})")


def _shot_widget(app, win, attr, name):
    try:
        d = getattr(win, attr, None)
        if d is None:
            todo.append(f"{name} (không có {attr})"); return
        d.setVisible(True); d.raise_(); _pump(app, 250)
        w = d.widget()
        pm = _grab_hires(w) if w is not None else None
        if pm is None or pm.isNull() or pm.width() < 5:
            todo.append(f"{name} (grab rỗng)"); return
        pm.save(str(FIG / name)); done.append(name)
    except Exception as e:                                     # noqa: BLE001
        todo.append(f"{name} (lỗi: {e})")


def main() -> int:
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    win = GP7AppQt(cell_config=None, project_root=ROOT, program_path=None)
    win.resize(1500, 950); win.show()
    _pump(app, 800)

    win._load_robot_gp7()
    _pump(app, 800)
    try:
        win._plotter.enable_anti_aliasing("ssaa")              # viewport mượt, hết răng cưa
    except Exception:                                          # noqa: BLE001
        pass
    try:
        win._plotter.view_isometric()
    except Exception:                                          # noqa: BLE001
        pass
    _pump(app, 400)

    # cell_overview: toàn cảnh app (menu + panel + viewport + status). Phải dùng
    # grabWindow (chụp màn hình thật) để lấy được viewport OpenGL; maximize cửa sổ
    # → nhiều pixel → nét ở 1024. (widget.grab/render cho viewport đen.)
    try:
        d = getattr(win, "_jog_dock", None)
        if d is not None:
            d.setVisible(True); d.raise_()
        win.showMaximized(); _pump(app, 900)
        win._plotter.render(); _pump(app, 500)
        pm = QGuiApplication.primaryScreen().grabWindow(int(win.winId()))
        if not pm.isNull() and pm.width() > 5:
            pm.save(str(FIG / "cell_overview.png")); done.append("cell_overview.png")
        else:
            todo.append("cell_overview.png (grabWindow rỗng)")
    except Exception as e:                                     # noqa: BLE001
        todo.append(f"cell_overview.png (lỗi: {e})")

    # Viewport: góc nhìn iso của cell
    _shot_view(win, "viewport_navigation.png")

    # Viewport: reach envelope (outer + inner) ở chế độ tool
    try:
        win._on_workspace_changed("tool"); _pump(app, 600)
        _shot_view(win, "workspace_envelope.png")
        win._on_workspace_changed("none"); _pump(app, 200)
    except Exception as e:                                     # noqa: BLE001
        todo.append(f"workspace_envelope.png (lỗi: {e})")

    # Panel jog: grab cả panel rồi crop thành 3 vùng (theo tỉ lệ → bền với kích thước)
    def _grab_jog_crops():
        try:
            d = win._jog_dock
            d.setVisible(True); d.raise_(); _pump(app, 300)
            pm = _grab_hires(d.widget())
            if pm.isNull() or pm.width() < 5:
                todo.append("panel_jog_* (grab rỗng)"); return
            tmp = FIG / "_jog_full_tmp.png"; pm.save(str(tmp))
            try:
                from PIL import Image
            except Exception:                                 # noqa: BLE001
                # Không có PIL → lưu full panel vào cả 3 tên (bạn tự crop)
                for n in ("panel_jog_cartesian.png", "panel_workspace_frames.png",
                          "panel_jog_joint.png"):
                    pm.save(str(FIG / n)); done.append(n + " (full, cần crop)")
                tmp.unlink(missing_ok=True); return
            im = Image.open(str(tmp)); W, H = im.size
            crops = {
                "panel_jog_cartesian.png":    (0, 0, W, int(0.617 * H)),
                "panel_workspace_frames.png": (int(0.289 * W), int(0.270 * H),
                                               W, int(0.615 * H)),
                "panel_jog_joint.png":        (0, int(0.619 * H), W, H),
            }
            for name, box in crops.items():
                im.crop(box).save(str(FIG / name)); done.append(name)
            tmp.unlink(missing_ok=True)
        except Exception as e:                                # noqa: BLE001
            todo.append(f"panel_jog_* (lỗi: {e})")

    _grab_jog_crops()
    _shot_widget(app, win, "_experiment_dock", "panel_digital_twin.png")

    # ── Viewport có trạng thái scene: pose robot + bật triad + đặt camera ──
    def _set_frames(keys):
        try:
            for cb in win._frame_checks.values():
                cb.setChecked(False)
            for k in keys:
                cb = win._frame_checks.get(k)
                if cb is not None:
                    cb.setChecked(True)
            _pump(app, 300)
        except Exception:                                     # noqa: BLE001
            pass

    def _shot_view_pose(name, joints, frame_keys, cam):
        try:
            win._apply_joints_main(joints); _pump(app, 350)
            _set_frames(frame_keys)
            if cam == "iso":
                win._plotter.view_isometric()
                win._plotter.reset_camera()
            elif cam == "tool":
                T = win._current_tool_world()
                if T is not None:
                    p = np.asarray(T[:3, 3], float) / 1000.0   # mm → m (scene mét)
                    cpos = p + np.array([0.45, -0.45, 0.30])
                    win._plotter.camera.focal_point = tuple(float(v) for v in p)
                    win._plotter.camera.position = tuple(float(v) for v in cpos)
                    win._plotter.camera.up = (0.0, 0.0, 1.0)
                else:
                    win._plotter.view_isometric(); win._plotter.reset_camera()
            elif cam == "base":                                # focus đế robot + triad
                b = np.asarray(win._base_xyz, float) / 1000.0
                fp = b + np.array([0.0, 0.0, 0.12])
                win._plotter.camera.focal_point = tuple(float(v) for v in fp)
                win._plotter.camera.position = tuple(
                    float(v) for v in (fp + np.array([0.55, -0.55, 0.28])))
                win._plotter.camera.up = (0.0, 0.0, 1.0)
            _pump(app, 300)
            win._plotter.render(); _pump(app, 250)
            win._plotter.screenshot(str(FIG / name), scale=HIRES)
            _smart_crop(str(FIG / name)); done.append(name)
            _set_frames([])                                    # dọn triad cho ảnh sau
        except Exception as e:                                # noqa: BLE001
            todo.append(f"{name} (lỗi: {e})")

    # tool_tcp: tư thế làm việc + triad Tool + Flange, camera cận TCP
    _shot_view_pose("tool_tcp.png", [40.0, -30.0, 20.0, 0.0, 50.0, 0.0],
                    ["tool", "flange"], cam="tool")
    # reference_frame: triad Base (0) + Ref. Frame, camera focus đế robot
    _shot_view_pose("reference_frame.png", [30.0, -20.0, 15.0, 0.0, 40.0, 0.0],
                    ["base", "ref"], cam="base")

    # ── Dialog modal: mở bằng trigger (gọi .exec()) rồi grab + đóng qua QTimer ──
    def _grab_modal(trigger, name, delay_ms=900):
        state = {"ok": False}

        def _do():
            w = QApplication.activeModalWidget() or QApplication.activeWindow()
            try:
                if w is not None and w is not win:
                    pm = _grab_hires(w)
                    if not pm.isNull() and pm.width() > 5:
                        pm.save(str(FIG / name)); state["ok"] = True
                    w.close()
            except Exception:                                 # noqa: BLE001
                pass
        QTimer.singleShot(delay_ms, _do)
        try:
            trigger()
        except Exception as e:                                # noqa: BLE001
            todo.append(f"{name} (lỗi: {e})"); return
        done.append(name) if state["ok"] else todo.append(f"{name} (không grab được)")

    # dlg_configurations: cần pose có nghiệm IK → đặt tư thế làm việc trước
    win._apply_joints_main([30.0, -20.0, 15.0, 0.0, 40.0, 0.0]); _pump(app, 300)
    _grab_modal(win._show_configurations_dlg, "dlg_configurations.png")

    # dlg_hse_connection: dialog cấu hình HSE
    _grab_modal(win._on_show_connection_settings, "dlg_hse_connection.png")

    # dlg_add_frame: dialog Add Frame (Name / Parent / Pose)
    _grab_modal(win._show_add_frame_dlg, "dlg_add_frame.png")

    # teach_on_surface: dựng target trên sàn (prompt=False → không treo), đưa robot
    # tới đó + bật triad Tool, chụp → minh hoạ tool vuông góc bề mặt.
    def _cap_teach():
        try:
            pt = np.array([400.0, 0.0, 0.0])                   # mm, điểm trên sàn trước robot
            z = np.array([0.0, 0.0, -1.0])                     # Z_tcp đâm xuống sàn
            rx = np.array([1.0, 0.0, 0.0])
            if abs(float(rx @ z)) > 0.99:
                rx = np.array([0.0, 1.0, 0.0])
            x = rx - float(rx @ z) * z; x /= np.linalg.norm(x)
            y = np.cross(z, x)
            T = np.eye(4); T[:3, 0] = x; T[:3, 1] = y; T[:3, 2] = z; T[:3, 3] = pt
            name = win._teach_target_from_matrix(T, "SURF_DEMO", prompt=False)
            if name and name in win._targets:
                win._apply_joints_main(win._targets[name]["joints"]); _pump(app, 300)
            for cb in win._frame_checks.values():
                cb.setChecked(False)
            win._frame_checks["tool"].setChecked(True); _pump(app, 300)
            win._plotter.view_isometric(); win._plotter.reset_camera()
            try:
                win._plotter.set_focus((pt / 1000.0).tolist())
            except Exception:                                 # noqa: BLE001
                pass
            _pump(app, 250); win._plotter.render(); _pump(app, 250)
            win._plotter.screenshot(str(FIG / "teach_on_surface.png"), scale=HIRES)
            _smart_crop(str(FIG / "teach_on_surface.png"))
            done.append("teach_on_surface.png")
            for cb in win._frame_checks.values():
                cb.setChecked(False)
        except Exception as e:                                # noqa: BLE001
            todo.append(f"teach_on_surface.png (lỗi: {e})")
    _cap_teach()

    # ── Direct-control (Phase 1/2) UI: Robot menu + safety dialogs ──────────
    def _grab_menu(title, name):
        """Pop up a top-level menu by title and grab it (for documenting new menu
        actions like Send-pose / Live-jog)."""
        try:
            mb = win.menuBar()
            menu = next((a.menu() for a in mb.actions()
                         if a.text().replace("&", "") == title and a.menu()), None)
            if menu is None:
                todo.append(f"{name} (menu '{title}' không thấy)"); return
            menu.popup(win.mapToGlobal(QPoint(20, 50))); _pump(app, 300)
            pm = _grab_hires(menu); menu.close()
            if pm.isNull() or pm.width() < 5:
                todo.append(f"{name} (grab rỗng)"); return
            pm.save(str(FIG / name)); done.append(name)
        except Exception as e:                                 # noqa: BLE001
            todo.append(f"{name} (lỗi: {e})")

    # menu_robot: menu Robot mở, thấy "Send current pose…" + "Live jog…"
    _grab_menu("Robot", "menu_robot.png")
    # dlg_send_pose: hộp thoại an toàn Phase-1 (gửi pose hiện tại xuống robot thật)
    _grab_modal(win._on_send_pose_to_robot, "dlg_send_pose.png")
    # dlg_live_jog: hộp thoại an toàn Phase-2 (bật live jog streaming)
    _grab_modal(lambda: win._act_live_jog.setChecked(True), "dlg_live_jog.png")
    win._act_live_jog.blockSignals(True)        # đảm bảo toggle về OFF sau khi chụp
    win._act_live_jog.setChecked(False)
    win._act_live_jog.blockSignals(False)
    # dlg_robot_params: bảng tham số robot (URDF joints) — đã co gọn theo nội dung
    _grab_modal(win._show_parameters_dlg, "dlg_robot_params.png")

    print("\n=== SUMMARY ===", flush=True)
    print("Đã chụp:", ", ".join(done) or "(không có)")
    print("Còn thủ công / lỗi:")
    for t in todo:
        print("  -", t)
    print(f"\nĐộ phân giải: panel/dialog render {HIRES}×, viewport scale={HIRES} + SSAA "
          "→ nét khi hiển thị 1024px. panel_jog_* tự crop thành 3 vùng.")
    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
