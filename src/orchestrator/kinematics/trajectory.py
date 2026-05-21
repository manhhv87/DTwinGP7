"""
trajectory.py
─────────────
Joint trajectory interpolation + safety checks (joint limit, self-collision).

Pure-numpy → predict offline trước khi gửi command thật cho robot. Tốc độ
~10ms cho trajectory 100 sample → đủ cho per-trial online preview (UC2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dh_model import RobotDHModel
from .forward_kinematics import joint_positions


@dataclass
class TrajectorySample:
    """1 sample tại thời điểm `t`."""

    t: float                          # giây từ start
    joints_rad: list[float]           # joint angles (radian)


def interpolate_joints(
    waypoints: list[list[float]],
    dt: float = 0.05,
    max_joint_speed_deg_s: float = 30.0,
) -> list[TrajectorySample]:
    """Linear interpolate joint waypoints, sample mỗi `dt` giây.

    Time tự động tính từ max-joint-speed: segment dài (max joint movement) /
    speed = duration. Đảm bảo speed cap → predict đúng motion robot thật.

    Args:
        waypoints: List joint configs (mỗi cái = 6 float radian).
        dt: Sample interval (giây). Default 50ms → 20Hz, đủ smooth.
        max_joint_speed_deg_s: Tốc độ joint giả định để tính duration.

    Returns:
        List TrajectorySample sắp xếp theo t tăng.
    """
    if len(waypoints) < 2:
        raise ValueError("Cần ít nhất 2 waypoints để interpolate")
    if dt <= 0:
        raise ValueError(f"dt phải > 0, got {dt}")

    speed_rad_s = np.deg2rad(max_joint_speed_deg_s)
    samples: list[TrajectorySample] = []
    t = 0.0

    for i in range(len(waypoints) - 1):
        start = np.asarray(waypoints[i], dtype=float)
        end = np.asarray(waypoints[i + 1], dtype=float)
        max_dq = float(np.max(np.abs(end - start)))
        if max_dq < 1e-6:
            continue
        duration = max_dq / speed_rad_s
        n_steps = max(1, int(np.ceil(duration / dt)))
        for k in range(n_steps):
            alpha = k / n_steps
            joints = (start * (1 - alpha) + end * alpha).tolist()
            samples.append(TrajectorySample(t=t, joints_rad=joints))
            t += duration / n_steps

    # Final waypoint exact
    samples.append(TrajectorySample(t=t, joints_rad=list(waypoints[-1])))
    return samples


def check_joint_limits(
    model: RobotDHModel,
    samples: list[TrajectorySample],
) -> list[tuple[int, int, float]]:
    """Verify mọi sample nằm trong joint limit của model.

    Returns:
        List violations: (sample_idx, joint_idx, value_rad). Rỗng = OK.
    """
    violations: list[tuple[int, int, float]] = []
    for s_idx, sample in enumerate(samples):
        for j_idx, (q, link) in enumerate(zip(sample.joints_rad, model.links)):
            if not (link.joint_min <= q <= link.joint_max):
                violations.append((s_idx, j_idx, float(q)))
    return violations


# Sphere radii (mm) cho self-collision check. Mỗi joint là 1 sphere center;
# radius xấp xỉ kích thước link tại vị trí đó. Conservative — over-estimate
# hơn under-estimate. Tune nếu false positive nhiều.
GP7_LINK_SPHERE_RADII_MM: tuple[float, ...] = (
    100.0,    # base
    120.0,    # J1 (shoulder mount)
    100.0,    # J2 (upper arm)
    90.0,     # J3 (elbow)
    70.0,     # J4 (forearm)
    60.0,     # J5 (wrist)
    50.0,     # J6 (flange)
    40.0,     # TCP / gripper
)


def check_self_collision_spheres(
    model: RobotDHModel,
    samples: list[TrajectorySample],
    radii_mm: tuple[float, ...] = GP7_LINK_SPHERE_RADII_MM,
    min_non_adjacent_gap: int = 3,
) -> list[tuple[int, int, int, float]]:
    """Sphere-based self-collision check giữa joint sphere không adjacent.

    Pattern phổ biến: mỗi joint origin = 1 sphere. Hai sphere không kề
    (cách >= `min_non_adjacent_gap` link) overlap → coi như collision.

    Args:
        model: Robot DH model.
        samples: Trajectory để check.
        radii_mm: Sphere radius cho từng joint (kể cả base + TCP).
        min_non_adjacent_gap: Tối thiểu cách bao nhiêu link để check. Default 3
            (J1 vs J4, J2 vs J5, ...) — gap nhỏ luôn close vì link ngắn, false
            positive cao. Tăng nếu DH có link rất ngắn (vd wrist GP7).

    Returns:
        List (sample_idx, joint_i, joint_j, distance_mm). Rỗng = OK.
    """
    violations: list[tuple[int, int, int, float]] = []
    for s_idx, sample in enumerate(samples):
        positions = joint_positions(model, sample.joints_rad)
        n = len(positions)
        # Lấy radii khớp số position; nếu thiếu thì pad với last value.
        radii = list(radii_mm[:n]) + [radii_mm[-1]] * max(0, n - len(radii_mm))

        for i in range(n):
            for j in range(i + min_non_adjacent_gap, n):
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                if dist < radii[i] + radii[j]:
                    violations.append((s_idx, i, j, dist))
    return violations
