"""
check_workspace.py
──────────────────
Kiểm tra WorkSpace (outer/inner) của Robot Panel có ĐÚNG không, bằng cách so
**mesh envelope** (cách app dựng: J1 áp giải tích) với **FK độc lập** (quét cả J1
thật rồi đo bán kính với tới tối đa theo từng hướng). Hai phương pháp khác nhau —
nếu trùng thì envelope là đúng.

Chạy:  .venv\\Scripts\\python.exe scripts\\check_workspace.py
Không cần OpenGL (chỉ tính hình học, không mở cửa sổ).
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.orchestrator.viewports.gp7_app_qt import GP7AppQt
from src.orchestrator.kinematics.urdf_chain import gp7_urdf, link_frames_urdf

TOL_MM = 80.0          # sai số cho phép giữa mesh và FK (robot ~900mm → ~9%)
AZ = (0, 45, 90, 135, 180, 225, 270, 315)


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


def _mesh_maxr(mesh, az_deg) -> float:
    P = mesh.points
    a = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
    r = np.hypot(P[:, 0], P[:, 1]) * 1000.0
    sel = r[np.abs(((a - az_deg + 180) % 360) - 180) < 6]
    return float(sel.max()) if len(sel) else 0.0


def main() -> int:
    m, outer, inner = _build_meshes("wrist")
    print(f"GP7 WorkSpace check — tolerance ±{TOL_MM:.0f} mm "
          f"(datasheet reach ≈ 927 mm)\n")
    print(f"{'azimuth':>8} | {'OUTER mesh':>10} {'FK(all)':>8} {'Δ':>6} | "
          f"{'INNER mesh':>10} {'FK(J2<0)':>9} {'Δ':>6}")
    print("-" * 72)
    ok = True
    for az in AZ:
        om, of = _mesh_maxr(outer, az), _fk_maxr(m, az, lambda q: True)
        im, ff = _mesh_maxr(inner, az), _fk_maxr(m, az, lambda q: q < 0)
        do, di = abs(om - of), abs(im - ff)
        ok &= (do <= TOL_MM and di <= TOL_MM)
        print(f"{az:7}° | {om:10.0f} {of:8.0f} {do:6.0f} | "
              f"{im:10.0f} {ff:9.0f} {di:6.0f}")

    # Kiểm tính chất: outer móp phía sau, inner móp phía trước (ngược nhau)
    o_front, o_rear = _mesh_maxr(outer, 0), _mesh_maxr(outer, 180)
    i_front, i_rear = _mesh_maxr(inner, 0), _mesh_maxr(inner, 180)
    dent_ok = (o_front > o_rear + 50) and (i_rear > i_front + 50)
    print("\nĐặc tính móp (dent):")
    print(f"  OUTER  trước={o_front:.0f}  sau={o_rear:.0f}  → móp ở {'SAU' if o_front>o_rear else 'TRƯỚC'}")
    print(f"  INNER  trước={i_front:.0f}  sau={i_rear:.0f}  → móp ở {'TRƯỚC' if i_rear>i_front else 'SAU'}")
    print(f"  Hai chỗ móp đối diện nhau? {'CÓ' if dent_ok else 'KHÔNG'}")

    verdict = ok and dent_ok
    print("\n" + ("✅ PASS — outer & inner khớp FK, móp đúng hướng."
                  if verdict else
                  "❌ FAIL — có sai lệch vượt ngưỡng, xem bảng trên."))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
