"""
urdf_gen.py
───────────
Generate URDF for GP7 from `kinematics.urdf_chain.gp7_urdf()` for loading into PyBullet.

PyBullet reads URDF only. This module builds URDF text directly from the origins + axes in
`gp7_urdf()` — parameters verified to match RoboDK SolveFK 0.00mm/0.00° via
`scripts/13_verify_vs_robodk.py`. As a result the robot in PyBullet moves exactly
as the joints commanded by the orchestrator (no sign/offset convention drift).

v1: visual = primitive box geometry following the vector between consecutive joint origins
→ recognizable arm shape. Kinematics are determined by <origin>/<axis>,
NOT by visuals → later replace <box> with <mesh filename="...stl"> from
ros-industrial/motoman without changing kinematics at all.

Module PURE TEXT — no pybullet dependency. Testable via string.
"""
from __future__ import annotations

from pathlib import Path

from ..kinematics.urdf_chain import URDFRobot, gp7_urdf

_MM_TO_M = 0.001

# Colors for 6 links (RGBA) — cosmetic only.
_LINK_COLORS = [
    (0.90, 0.55, 0.10, 1.0),   # S — orange
    (0.20, 0.45, 0.85, 1.0),   # L — blue
    (0.20, 0.60, 0.30, 1.0),   # U — green
    (0.85, 0.20, 0.20, 1.0),   # R — red
    (0.50, 0.35, 0.70, 1.0),   # B — purple
    (0.95, 0.75, 0.10, 1.0),   # T — yellow
]


def _inertial(mass: float = 0.2) -> str:
    """Minimal inertial block — sufficient to suppress PyBullet URDF parser warnings.
    We use resetJointState (kinematic mode) so values do not affect rendering.
    """
    return (
        '    <inertial>'
        '<origin xyz="0 0 0" rpy="0 0 0"/>'
        f'<mass value="{mass}"/>'
        '<inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>'
        '</inertial>\n'
    )


def _box_visual(size_m, center_m, rgba, matname: str) -> str:
    sx, sy, sz = size_m
    cx, cy, cz = center_m
    r, g, b, a = rgba
    return (
        f'    <visual>'
        f'<origin xyz="{cx:.5f} {cy:.5f} {cz:.5f}" rpy="0 0 0"/>'
        f'<geometry><box size="{sx:.5f} {sy:.5f} {sz:.5f}"/></geometry>'
        f'<material name="{matname}"><color rgba="{r} {g} {b} {a}"/></material>'
        f'</visual>\n'
    )


def _segment_box(dx: float, dy: float, dz: float, thick: float = 0.07):
    """Box (size, center) enclosing the segment from (0,0,0) → (dx,dy,dz) [m] in link frame."""
    sx = abs(dx) + thick
    sy = abs(dy) + thick
    sz = abs(dz) + thick
    return (sx, sy, sz), (dx / 2.0, dy / 2.0, dz / 2.0)


def build_gp7_urdf_xml(model: URDFRobot | None = None) -> str:
    """Build URDF text for GP7 from URDFRobot.

    Joint chain matches `forward_kinematics_urdf`:
        base_link → joint_S → link_S → joint_L → ... → link_T
                  → joint_flange (fixed) → link_flange
                  → joint_tool0 (fixed, rpy) → link_tool0 (gripper)

    base_xyz_mm is NOT baked into the URDF — the caller spawns the body at
    basePosition (PyBullet) so the robot stands on a pedestal in world space.
    URDF describes the robot at the J1 origin.
    """
    model = model or gp7_urdf()
    joints = model.joints
    n = len(joints)

    out: list[str] = ['<?xml version="1.0"?>', '<robot name="gp7_primitive">']

    # ── base_link: base block + "stand" dropping 330mm (fills pedestal→J1 gap) ──
    out.append('  <link name="base_link">')
    out.append(_box_visual((0.16, 0.16, 0.12), (0, 0, 0.0),
                           (0.30, 0.30, 0.30, 1.0), "mat_base"))
    out.append(_box_visual((0.13, 0.13, 0.34), (0, 0, -0.165),
                           (0.35, 0.35, 0.35, 1.0), "mat_stand"))
    out.append(_inertial())
    out.append('  </link>')

    prev_link = "base_link"
    for i, j in enumerate(joints):
        link = f"link_{j.name}"
        ox, oy, oz = (v * _MM_TO_M for v in j.origin_mm)
        ax, ay, az = j.axis

        out.append(f'  <joint name="joint_{j.name}" type="revolute">')
        out.append(f'    <parent link="{prev_link}"/>')
        out.append(f'    <child link="{link}"/>')
        out.append(f'    <origin xyz="{ox:.5f} {oy:.5f} {oz:.5f}" rpy="0 0 0"/>')
        out.append(f'    <axis xyz="{ax:.1f} {ay:.1f} {az:.1f}"/>')
        out.append(
            f'    <limit lower="{j.joint_min:.4f}" upper="{j.joint_max:.4f}" '
            f'effort="100" velocity="3.14"/>'
        )
        out.append('  </joint>')

        # Visual: segment from this link frame to the next joint origin (or flange for last link)
        nxt_mm = joints[i + 1].origin_mm if i < n - 1 else model.flange_xyz_mm
        dx, dy, dz = (v * _MM_TO_M for v in nxt_mm)
        size, ctr = _segment_box(dx, dy, dz)
        out.append(f'  <link name="{link}">')
        out.append(_box_visual(size, ctr, _LINK_COLORS[i % len(_LINK_COLORS)],
                               f"mat_{j.name}"))
        out.append(_inertial())
        out.append('  </link>')
        prev_link = link

    # ── flange (fixed) ──
    fx, fy, fz = (v * _MM_TO_M for v in model.flange_xyz_mm)
    out.append('  <joint name="joint_flange" type="fixed">')
    out.append(f'    <parent link="{prev_link}"/>')
    out.append('    <child link="link_flange"/>')
    out.append(f'    <origin xyz="{fx:.5f} {fy:.5f} {fz:.5f}" rpy="0 0 0"/>')
    out.append('  </joint>')
    out.append('  <link name="link_flange">')
    out.append(_box_visual((0.06, 0.06, 0.06), (0, 0, 0),
                           (0.60, 0.60, 0.60, 1.0), "mat_flange"))
    out.append(_inertial())
    out.append('  </link>')

    # ── tool0 (fixed, with rpy) + gripper primitive attached to tool0 +Z (~TCP 100mm) ──
    rr, pp, yy = model.tool0_rpy_rad
    out.append('  <joint name="joint_tool0" type="fixed">')
    out.append('    <parent link="link_flange"/>')
    out.append('    <child link="link_tool0"/>')
    out.append(f'    <origin xyz="0 0 0" rpy="{rr:.6f} {pp:.6f} {yy:.6f}"/>')
    out.append('  </joint>')
    out.append('  <link name="link_tool0">')
    out.append(_box_visual((0.09, 0.04, 0.10), (0, 0, 0.05),
                           (0.15, 0.15, 0.15, 1.0), "mat_tool"))
    out.append(_inertial())
    out.append('  </link>')

    out.append('</robot>')
    return "\n".join(out) + "\n"


def ensure_gp7_urdf(project_root, model: URDFRobot | None = None) -> Path:
    """Write GP7 URDF to `models/gp7/gp7_primitive.urdf` and return the path.

    Always regenerates (cheap) so the URDF never drifts from `urdf_chain.py`.
    """
    out_path = Path(project_root) / "models" / "gp7" / "gp7_primitive.urdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_gp7_urdf_xml(model), encoding="utf-8")
    return out_path
