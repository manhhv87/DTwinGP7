"""
check_workspace.py
──────────────────
Kiểm tra WorkSpace (reach envelope hình "quả thận" + lõm R233) của Robot Panel có
ĐÚNG không, bằng cách so **mesh envelope** (cách app dựng: J1 áp giải tích) với **FK
độc lập** (quét cả J1 thật rồi đo bán kính với tới theo từng hướng) và với các mốc
**datasheet HW1483944 Fig 5-3(b)**. Hai phương pháp khác nhau — nếu trùng thì đúng.

Chạy:  .venv\\Scripts\\python.exe scripts\\check_workspace.py
Không cần OpenGL (chỉ tính hình học, không mở cửa sổ).
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np

try:                                   # Windows console hay là cp1252 → ép UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                      # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.orchestrator.viewports.gp7_app_qt import GP7AppQt
from src.orchestrator.kinematics.urdf_chain import gp7_urdf, link_frames_urdf

TOL_MM = 80.0          # sai số cho phép giữa mesh và FK (robot ~900mm → ~9%)
AZ = (0, 45, 90, 135, 180, 225, 270, 315)

# Mốc datasheet GP7 (HW1483944 Fig 5-3(b)), P-point = tâm cổ tay, rel mặt sàn đế.
DS_R927 = 927.0        # tầm với ngang ngoài
DS_TOP = 1217.0        # đỉnh envelope
DS_BOTTOM = -476.0     # đáy envelope
DS_R233 = 233.0        # bán kính lõm trong
BASE_Z = 330.0         # cao trục J2(L) so với sàn → z_relFloor = z_FK + 330


def _build_meshes(mode: str = "wrist"):
    m = gp7_urdf(base_xyz_mm=(0.0, 0.0, 0.0))
    stub = types.SimpleNamespace(
        _model=m, _base_xyz=(0.0, 0.0, 0.0),
        _tool_frames=[("flange", np.eye(4))], _tool_idx=0,
        _reach_envelope_cache={})
    outer, inner = GP7AppQt._compute_reach_envelope_mesh.__get__(stub)(mode)
    return m, outer, inner


def _fk_maxr(model, az_deg, j2_filter) -> float:
    """Bán kính ngang với tới xa nhất tại 1 hướng — quét J1,J2,J3 thật (ground truth)."""
    jl = [(j.joint_min, j.joint_max) for j in model.joints]
    best = 0.0
    for q1 in np.linspace(jl[0][0], jl[0][1], 120):
        for qL in np.linspace(jl[1][0], jl[1][1], 22):
            if not j2_filter(qL):
                continue
            for qU in np.linspace(jl[2][0], jl[2][1], 22):
                p = dict(link_frames_urdf(model, [q1, qL, qU, 0, 0, 0]))["link_B"][:3, 3]
                r = math.hypot(p[0], p[1])
                a = math.degrees(math.atan2(p[1], p[0]))
                if abs(((a - az_deg + 180) % 360) - 180) < 6 and r > best:
                    best = r
    return best


def _fk_minr_band(model, zlo_mm, zhi_mm) -> float:
    """Bán kính ngang với tới GẦN trục nhất trong dải độ cao (FK z, chưa +330)."""
    jl = [(j.joint_min, j.joint_max) for j in model.joints]
    best = 1e9
    for q1 in np.linspace(jl[0][0], jl[0][1], 24):
        for qL in np.linspace(jl[1][0], jl[1][1], 30):
            for qU in np.linspace(jl[2][0], jl[2][1], 36):
                p = dict(link_frames_urdf(model, [q1, qL, qU, 0, 0, 0]))["link_B"][:3, 3]
                if zlo_mm <= p[2] <= zhi_mm:
                    r = math.hypot(p[0], p[1])
                    if 1.0 <= r < best:
                        best = r
    return best if best < 1e9 else 0.0


def _mesh_maxr(mesh, az_deg) -> float:
    P = mesh.points
    a = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
    r = np.hypot(P[:, 0], P[:, 1]) * 1000.0
    sel = r[np.abs(((a - az_deg + 180) % 360) - 180) < 6]
    return float(sel.max()) if len(sel) else 0.0


def _mesh_minr_band(mesh, zlo_mm, zhi_mm) -> float:
    P = mesh.points
    z = P[:, 2] * 1000.0
    r = np.hypot(P[:, 0], P[:, 1]) * 1000.0
    sel = r[(z >= zlo_mm) & (z <= zhi_mm)]
    return float(sel.min()) if len(sel) else 0.0


def main() -> int:
    m, kidney, inner = _build_meshes("wrist")
    print(f"GP7 WorkSpace check — envelope đơn (kidney + lõm R233), tol ±{TOL_MM:.0f} mm\n")

    # 1) Vỏ ngoài: mesh vs FK theo 8 hướng
    print(f"{'azimuth':>8} | {'OUTER mesh':>10} {'FK(all)':>8} {'Δ':>6}")
    print("-" * 40)
    ok = True
    for az in AZ:
        om, of = _mesh_maxr(kidney, az), _fk_maxr(m, az, lambda q: True)
        do = abs(om - of)
        ok &= do <= TOL_MM
        print(f"{az:7}° | {om:10.0f} {of:8.0f} {do:6.0f}")

    # 2) Đối chiếu mốc datasheet (z theo mặt sàn đế = z_FK + 330)
    P = kidney.points
    z_rel = P[:, 2] * 1000.0 + BASE_Z
    top, bottom = float(z_rel.max()), float(z_rel.min())
    rmax = float((np.hypot(P[:, 0], P[:, 1]) * 1000.0).max())
    # lõm R233: rmin trong dải giữa (raw FK z quanh 0) ; FK đối chứng
    void_mesh = _mesh_minr_band(kidney, -40.0, 40.0)
    void_fk = _fk_minr_band(m, -40.0, 40.0)
    checks = [
        ("Tầm ngang ngoài (R927)", rmax, DS_R927, 30.0),
        ("Đỉnh envelope", top, DS_TOP, 30.0),
        ("Đáy envelope", bottom, DS_BOTTOM, 30.0),
        ("Lõm trong (R233)", void_mesh, DS_R233, 50.0),
    ]
    print("\nĐối chiếu datasheet HW1483944 Fig 5-3(b):")
    print(f"{'mốc':>26} | {'mesh':>7} {'datasheet':>10} {'Δ':>6}")
    print("-" * 56)
    ds_ok = True
    for name, val, ds, tol in checks:
        d = abs(val - ds)
        ds_ok &= d <= tol
        print(f"{name:>26} | {val:7.0f} {ds:10.0f} {d:6.0f}")
    print(f"  (lõm FK đối chứng = {void_fk:.0f} mm)")

    # 3) Tính chất hình học: envelope là MỘT mặt (không còn mặt inner riêng);
    #    lõm nằm trong vỏ; vỏ móp ở phía sau.
    single = inner is None
    dent_inside = void_mesh < rmax - 400
    rear_dent = _mesh_maxr(kidney, 0) > _mesh_maxr(kidney, 180) + 50
    print("\nTính chất:")
    print(f"  Envelope đơn (inner=None)?       {'CÓ' if single else 'KHÔNG'}")
    print(f"  Lõm nằm trong vỏ ngoài?          {'CÓ' if dent_inside else 'KHÔNG'}")
    print(f"  Vỏ móp ở phía sau?               {'CÓ' if rear_dent else 'KHÔNG'}")

    verdict = ok and ds_ok and single and dent_inside and rear_dent
    print("\n" + ("✅ PASS — envelope khớp FK + datasheet."
                  if verdict else
                  "❌ FAIL — có sai lệch, xem bảng trên."))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
