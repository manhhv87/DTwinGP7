"""
urdf_chain.py
─────────────
Forward kinematics theo URDF chain (translation + rotation) thay vì Modified DH.

Lý do: URDF chain trực tiếp ánh xạ datasheet vật lý của robot (joint origins +
axes), tránh sai sót khi convert sang DH convention. Match exactly với
ros-industrial/motoman GP7 URDF + RoboDK GP7 model.

Mỗi joint trong URDF có:
  - origin xyz: vị trí joint axis trong PARENT link frame (mm)
  - axis xyz:   hướng joint axis (đơn vị vector, có thể là negative ±1)

FK chain:
    T = base
    For each joint i:
        T = T · Translate(origin_i) · Rotate(axis_i, q_i)

`q_i` là joint angle theo Yaskawa convention (positive = theo axis direction
trong URDF). Negative axis trong URDF (vd (-1,0,0) cho J4) đã include sign.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class URDFJoint:
    """1 joint trong URDF chain (revolute)."""

    name: str
    origin_mm: tuple[float, float, float]    # xyz trong parent frame
    axis: tuple[float, float, float]         # unit vector (có dấu)
    joint_min: float                         # radian
    joint_max: float                         # radian


@dataclass(frozen=True)
class URDFRobot:
    """Robot model dùng URDF chain.

    Args:
        joints: Joint chain (revolute only).
        base_xyz_mm: Robot base position trong world frame (mm).
        base_rpy_rad: Robot base orientation (radian, XYZ-fixed).
        tool_offset_mm: TCP offset từ flange theo Z (mm) — backwards compat.
        flange_xyz_mm: Fixed offset từ joint_6 → flange link (URDF fixed joint).
            GP7: (80, 0, 0).
        tool0_rpy_rad: Tool0 frame rotation tương đối flange (URDF tool0 joint).
            GP7: (π, -π/2, 0).
    """

    name: str
    joints: tuple[URDFJoint, ...]
    base_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tool_offset_mm: float = 0.0
    flange_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tool0_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def num_joints(self) -> int:
        return len(self.joints)


def _axis_angle_to_matrix(axis: tuple[float, float, float], q: float) -> np.ndarray:
    """Rotation matrix 4x4 từ axis-angle (Rodrigues)."""
    x, y, z = axis
    norm = np.sqrt(x * x + y * y + z * z)
    if norm < 1e-12:
        return np.eye(4)
    return _axis_angle_unit((x / norm, y / norm, z / norm), q)


def _axis_angle_unit(axis_unit: tuple[float, float, float], q: float) -> np.ndarray:
    """Rodrigues 4x4 từ axis ĐÃ chuẩn hoá (đơn vị) + góc q.

    Tách khỏi `_axis_angle_to_matrix` để FK truyền sẵn axis đã normalize (cache
    trong `_urdf_consts`) → khỏi tính `sqrt` + chia mỗi joint mỗi call. Với unit
    axis, kết quả bit-identical với `_axis_angle_to_matrix(axis, q)`.
    """
    x, y, z = axis_unit
    if x == 0.0 and y == 0.0 and z == 0.0:
        return np.eye(4)
    c, s = np.cos(q), np.sin(q)
    C = 1 - c
    return np.array([
        [c + x * x * C,      x * y * C - z * s,   x * z * C + y * s,  0],
        [y * x * C + z * s,  c + y * y * C,       y * z * C - x * s,  0],
        [z * x * C - y * s,  z * y * C + x * s,   c + z * z * C,      0],
        [0,                  0,                   0,                  1],
    ])


@lru_cache(maxsize=None)
def _urdf_consts(model: "URDFRobot"):
    """Hằng số FK không phụ thuộc joint angle — cache 1 lần cho mỗi model.

    Gồm: base transform, axis ĐÃ normalize của từng joint, ma trận
    `_translate(origin)` của từng joint (cố định), và các transform cuối
    (flange/tool0/tool_offset) hoặc None nếu không áp dụng. Mọi giá trị y hệt
    bản dựng-lại-mỗi-call cũ; chỉ khác là không tính lại nữa.
    """
    base = _base_transform(model)
    axes: list[tuple[float, float, float]] = []
    trans: list[np.ndarray] = []
    for j in model.joints:
        x, y, z = j.axis
        norm = float(np.sqrt(x * x + y * y + z * z))
        axes.append((x / norm, y / norm, z / norm) if norm >= 1e-12
                    else (0.0, 0.0, 0.0))
        trans.append(_translate(j.origin_mm))
    flange = (_translate(model.flange_xyz_mm)
              if any(v != 0.0 for v in model.flange_xyz_mm) else None)
    tool0 = (_rpy_to_matrix(model.tool0_rpy_rad)
             if any(v != 0.0 for v in model.tool0_rpy_rad) else None)
    tooloff = (_translate((0.0, 0.0, model.tool_offset_mm))
               if model.tool_offset_mm != 0.0 else None)
    return base, tuple(axes), tuple(trans), flange, tool0, tooloff


def _translate(xyz: tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = xyz
    return T


def _base_transform(model: URDFRobot) -> np.ndarray:
    """Base frame transform từ base_xyz_mm + base_rpy_rad."""
    roll, pitch, yaw = model.base_rpy_rad
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr,  0],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr,  0],
        [-sp,      cp * sr,                 cp * cr,                 0],
        [0,        0,                       0,                       1],
    ])
    R[:3, 3] = model.base_xyz_mm
    return R


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF RPY (intrinsic XYZ = Rz·Ry·Rx) → 4x4 rotation matrix."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    R = np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr,  0],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr,  0],
        [-sp,      cp * sr,                 cp * cr,                 0],
        [0,        0,                       0,                       1],
    ])
    return R


def forward_kinematics_urdf(
    model: URDFRobot,
    joints_rad: list[float] | tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    """FK theo URDF chain. Trả về 4x4 pose của tool0 frame."""
    joints = np.asarray(joints_rad, dtype=float).flatten()
    if len(joints) != model.num_joints():
        raise ValueError(
            f"Expected {model.num_joints()} joints, got {len(joints)}"
        )

    base, axes, trans, flange, tool0, tooloff = _urdf_consts(model)
    T = base
    for i in range(len(joints)):
        T = T @ trans[i] @ _axis_angle_unit(axes[i], joints[i])

    # Fixed flange offset → tool0 rotation → backward-compat tool_offset (Z),
    # giữ NGUYÊN thứ tự matmul cũ nên bit-identical.
    if flange is not None:
        T = T @ flange
    if tool0 is not None:
        T = T @ tool0
    if tooloff is not None:
        T = T @ tooloff
    return T


def joint_positions_urdf(
    model: URDFRobot,
    joints_rad: list[float] | tuple[float, ...] | np.ndarray,
) -> list[np.ndarray]:
    """List origin của từng joint trong world frame."""
    joints = np.asarray(joints_rad, dtype=float).flatten()
    base, axes, trans, _flange, _tool0, tooloff = _urdf_consts(model)
    positions: list[np.ndarray] = []
    T = base
    positions.append(T[:3, 3].copy())

    for i in range(len(joints)):
        T = T @ trans[i] @ _axis_angle_unit(axes[i], joints[i])
        positions.append(T[:3, 3].copy())

    if tooloff is not None:
        T = T @ tooloff
        positions.append(T[:3, 3].copy())
    return positions


def fk_with_joint_frames_urdf(
    model: URDFRobot,
    joints_rad: list[float] | tuple[float, ...] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FK + collect joint origin position + axis direction (WORLD frame)
    cho analytical Jacobian.

    Returns:
        T_tcp:    (4,4) end-effector pose (tool0 frame) — same as forward_kinematics_urdf.
        p_joints: (n, 3) world origin của mỗi joint AT THE MOMENT axis rotation applies.
        z_joints: (n, 3) world-frame axis vector của mỗi joint (đã normalize).

    Analytical Jacobian column i cho revolute joint:
        J[:3, i] = z_joints[i] × (p_tcp - p_joints[i])
        J[3:, i] = z_joints[i]
    """
    joints = np.asarray(joints_rad, dtype=float).flatten()
    n = len(joints)
    base, axes, trans, flange, tool0, tooloff = _urdf_consts(model)
    p_joints = np.zeros((n, 3))
    z_joints = np.zeros((n, 3))
    T = base
    for i in range(n):
        # Apply translation tới joint i origin TRƯỚC khi rotate
        T = T @ trans[i]
        # Lúc này T[:3, 3] = joint i origin, T[:3, :3] = parent frame của axis_local
        p_joints[i] = T[:3, 3]
        z_joints[i] = T[:3, :3] @ np.asarray(axes[i])
        # Apply joint rotation
        T = T @ _axis_angle_unit(axes[i], joints[i])
    # Apply flange + tool0 + tool_offset như forward_kinematics_urdf
    if flange is not None:
        T = T @ flange
    if tool0 is not None:
        T = T @ tool0
    if tooloff is not None:
        T = T @ tooloff
    return T, p_joints, z_joints


def _axis_angle_unit_batch(axis_unit: tuple[float, float, float],
                            q_batch: np.ndarray) -> np.ndarray:
    """Batched Rodrigues: (N,) angles → (N, 4, 4) rotation matrices.

    axis FIXED, q VARIES per batch — phù hợp với URDF joint i (cùng axis cho
    mọi q_batch[k, i]). Tránh per-element Python loop.
    """
    x, y, z = axis_unit
    if x == 0.0 and y == 0.0 and z == 0.0:
        N = q_batch.shape[0]
        return np.broadcast_to(np.eye(4), (N, 4, 4)).copy()
    c = np.cos(q_batch)                 # (N,)
    s = np.sin(q_batch)
    C = 1.0 - c
    N = q_batch.shape[0]
    R = np.zeros((N, 4, 4))
    R[:, 0, 0] = c + x * x * C
    R[:, 0, 1] = x * y * C - z * s
    R[:, 0, 2] = x * z * C + y * s
    R[:, 1, 0] = y * x * C + z * s
    R[:, 1, 1] = c + y * y * C
    R[:, 1, 2] = y * z * C - x * s
    R[:, 2, 0] = z * x * C - y * s
    R[:, 2, 1] = z * y * C + x * s
    R[:, 2, 2] = c + z * z * C
    R[:, 3, 3] = 1.0
    return R


def fk_with_joint_frames_batch_urdf(
    model: URDFRobot, q_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched FK: solve N FK problems song song bằng numpy batched matmul.

    Args:
        q_batch: (N, num_joints) joint angles.

    Returns:
        T_tcp:    (N, 4, 4) end-effector poses.
        p_joints: (N, n, 3) joint origin in world frame.
        z_joints: (N, n, 3) joint axis in world frame.

    Performance: 1 batched call cho N=30 problems ~150µs, vs 30 × 50µs sequential
    = 1.5ms. Speedup ~10× cho N=30, scale linear với N.
    """
    q_batch = np.asarray(q_batch, dtype=float)
    if q_batch.ndim != 2:
        raise ValueError(f"q_batch phải 2D (N, n), got shape {q_batch.shape}")
    N, n = q_batch.shape
    if n != model.num_joints():
        raise ValueError(f"Expected (N, {model.num_joints()}), got {q_batch.shape}")
    base, axes, trans, flange, tool0, tooloff = _urdf_consts(model)

    # Init T as base broadcasted to N copies
    T = np.broadcast_to(base, (N, 4, 4)).copy()
    p_joints = np.zeros((N, n, 3))
    z_joints = np.zeros((N, n, 3))

    for i in range(n):
        # Apply translation (constant cho mọi batch element)
        T = T @ trans[i]                            # (N,4,4) @ (4,4) → (N,4,4)
        p_joints[:, i, :] = T[:, :3, 3]
        # Axis ở world frame = T[:3, :3] @ axis_local
        z_joints[:, i, :] = T[:, :3, :3] @ np.asarray(axes[i])
        # Batched rotation by q_batch[:, i]
        Rot = _axis_angle_unit_batch(axes[i], q_batch[:, i])   # (N, 4, 4)
        T = T @ Rot                                  # (N,4,4) @ (N,4,4) → (N,4,4)

    if flange is not None:
        T = T @ flange
    if tool0 is not None:
        T = T @ tool0
    if tooloff is not None:
        T = T @ tooloff
    return T, p_joints, z_joints


def link_frames_urdf(
    model: URDFRobot,
    joints_rad: list[float] | tuple[float, ...] | np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    """Frame world (4x4) của từng link theo URDF chain — cho viewport vẽ khối.

    Trả về list (link_name, T_world) theo thứ tự:
        base_link, link_S, link_L, link_U, link_R, link_B, link_T,
        link_flange (nếu có flange offset), link_tool0 (nếu có tool0 rpy).

    T_world tính bằng mm (như origin_mm). Dùng chung công thức với
    `forward_kinematics_urdf` nên frame tool0 trùng khớp tuyệt đối.
    """
    joints = np.asarray(joints_rad, dtype=float).flatten()
    if len(joints) != model.num_joints():
        raise ValueError(
            f"Expected {model.num_joints()} joints, got {len(joints)}"
        )

    base, axes, trans, flange, tool0, _tooloff = _urdf_consts(model)
    frames: list[tuple[str, np.ndarray]] = []
    T = base
    frames.append(("base_link", T.copy()))
    for i, joint in enumerate(model.joints):
        T = T @ trans[i] @ _axis_angle_unit(axes[i], joints[i])
        frames.append((f"link_{joint.name}", T.copy()))

    if flange is not None:
        T = T @ flange
        frames.append(("link_flange", T.copy()))
    if tool0 is not None:
        T = T @ tool0
        frames.append(("link_tool0", T.copy()))
    return frames


# ───── GP7 chain từ ros-industrial/motoman noetic-devel ─────


def gp7_urdf(
    base_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    base_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tool_offset_mm: float = 0.0,
) -> URDFRobot:
    """Yaskawa GP7 chain trích từ ros-industrial/motoman gp7_macro.xacro.

    Source: github.com/ros-industrial/motoman noetic-devel branch
            motoman_gp7_support/urdf/gp7_macro.xacro
    """
    deg = np.deg2rad
    # NOTE: ROS URDF có joint_1_s origin (0, 0, 0.33) — base_link to J1 axis.
    # RoboDK GP7 model coi "robot frame" = J1 axis (không phải base_link/floor).
    # Để match RoboDK SolveFK, set origin J1 = (0,0,0). Caller dùng base_xyz_mm
    # để position robot trong world (vd (0,0,630) cho pedestal).
    #
    # Joint limits = query trực tiếp từ RoboDK Yaskawa-GP7.robot (robot.JointLimits(),
    # scripts/13_verify_vs_robodk.py) → khớp 1:1 controller thật. Trước đây R chỉ
    # ±180° làm IK fail ở home (J4=-181.93°); L/U cũng lệch. Đã sửa cho khớp RoboDK.
    joints = (
        URDFJoint(name="S", origin_mm=(0.0, 0.0, 0.0),
                  axis=(0.0, 0.0, 1.0),
                  joint_min=deg(-170), joint_max=deg(170)),
        URDFJoint(name="L", origin_mm=(40.0, 0.0, 0.0),
                  axis=(0.0, 1.0, 0.0),
                  joint_min=deg(-65),  joint_max=deg(145)),
        URDFJoint(name="U", origin_mm=(0.0, 0.0, 445.0),
                  axis=(0.0, -1.0, 0.0),
                  joint_min=deg(-116), joint_max=deg(255)),
        URDFJoint(name="R", origin_mm=(440.0, 0.0, 40.0),
                  axis=(-1.0, 0.0, 0.0),
                  joint_min=deg(-190), joint_max=deg(190)),
        URDFJoint(name="B", origin_mm=(0.0, 0.0, 0.0),
                  axis=(0.0, -1.0, 0.0),
                  joint_min=deg(-135), joint_max=deg(135)),
        URDFJoint(name="T", origin_mm=(0.0, 0.0, 0.0),
                  axis=(-1.0, 0.0, 0.0),
                  joint_min=deg(-360), joint_max=deg(360)),
    )
    return URDFRobot(
        name="Yaskawa GP7",
        joints=joints,
        base_xyz_mm=base_xyz_mm,
        base_rpy_rad=base_rpy_rad,
        tool_offset_mm=tool_offset_mm,
        flange_xyz_mm=(80.0, 0.0, 0.0),                # URDF fixed joint 6t-flange
        tool0_rpy_rad=(np.pi, -np.pi / 2, 0.0),         # URDF fixed joint flange-tool0
    )
