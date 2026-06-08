"""
trajectory.py
─────────────
Joint trajectory interpolation + safety checks (joint limit, self-collision).

Pure-numpy → predict offline before sending real commands to the robot. Runtime
~10ms for a 100-sample trajectory → sufficient for per-trial online preview (UC2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dh_model import RobotDHModel
from .forward_kinematics import joint_positions, joint_positions_batch


@dataclass
class TrajectorySample:
    """One sample at time `t`."""

    t: float                          # seconds from start
    joints_rad: list[float]           # joint angles (radian)


def interpolate_joints(
    waypoints: list[list[float]],
    dt: float = 0.05,
    max_joint_speed_deg_s: float = 30.0,
) -> list[TrajectorySample]:
    """Linearly interpolate joint waypoints, sampled every `dt` seconds.

    Time is computed automatically from max-joint-speed: segment length
    (max joint movement) / speed = duration. Enforces the speed cap so the
    predicted motion matches the real robot.

    Args:
        waypoints: List of joint configs (each = 6 floats in radians).
        dt: Sample interval (seconds). Default 50ms → 20Hz, smooth enough.
        max_joint_speed_deg_s: Assumed joint speed used to compute duration.

    Returns:
        List of TrajectorySample sorted by increasing t.
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints to interpolate")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if max_joint_speed_deg_s <= 0:
        # speed 0 → duration = max_dq/0 = inf → int(ceil(inf)) raises OverflowError;
        # negative → bogus 1-step negative-time trajectory. Fail clearly instead.
        raise ValueError(
            f"max_joint_speed_deg_s must be > 0, got {max_joint_speed_deg_s}")

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
    """Verify that every sample lies within the model's joint limits.

    Returns:
        List of violations: (sample_idx, joint_idx, value_rad). Empty = OK.
    """
    # Polymorphic: URDFRobot uses `.joints`, RobotDHModel uses `.links`
    link_attr = getattr(model, "joints", None) or getattr(model, "links", None)
    violations: list[tuple[int, int, float]] = []
    for s_idx, sample in enumerate(samples):
        for j_idx, (q, link) in enumerate(zip(sample.joints_rad, link_attr)):
            if not (link.joint_min <= q <= link.joint_max):
                violations.append((s_idx, j_idx, float(q)))
    return violations


# Sphere radii (mm) for self-collision check. Each joint is one sphere center;
# radius approximates the link size at that position. Conservative — over-estimate
# rather than under-estimate. Tune if false positives are frequent.
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
    """Sphere-based self-collision check between non-adjacent joint spheres.

    Common pattern: each joint origin = one sphere. Two non-adjacent spheres
    (separated by >= `min_non_adjacent_gap` links) that overlap are flagged as a collision.

    Args:
        model: Robot DH model.
        samples: Trajectory to check.
        radii_mm: Sphere radius for each joint (including base + TCP).
        min_non_adjacent_gap: Minimum link separation before checking. Default 3
            (J1 vs J4, J2 vs J5, ...) — smaller gaps are always close due to short
            links, causing high false-positive rates. Increase for DH models with
            very short links (e.g. GP7 wrist).

    Returns:
        List of (sample_idx, joint_i, joint_j, distance_mm). Empty = OK.
    """
    if not samples:
        return []

    # Batch FK for all samples at once (vectorized matmul) instead of N separate calls —
    # positions[p] is an array (N,3), bit-identical to joint_positions for each sample.
    joints_batch = np.array([s.joints_rad for s in samples], dtype=float)
    positions = joint_positions_batch(model, joints_batch)
    n = len(positions)
    # Match radii to position count; pad with the last value if too short.
    radii = list(radii_mm[:n]) + [radii_mm[-1]] * max(0, n - len(radii_mm))
    # (N, n, 3): positions of all joints for all samples.
    P = np.stack(positions, axis=1)

    violations: list[tuple[int, int, int, float]] = []
    for i in range(n):
        for j in range(i + min_non_adjacent_gap, n):
            thr = radii[i] + radii[j]
            diff = P[:, i, :] - P[:, j, :]
            # Vectorized SQUARED distance as a coarse filter (superset):
            # threshold is widened to ensure NO pair with dist < thr is missed (ULP error).
            dist_sq = np.einsum("kc,kc->k", diff, diff)
            thr_sq_pad = (thr * (1.0 + 1e-9) + 1e-6) ** 2
            for k in np.nonzero(dist_sq < thr_sq_pad)[0]:
                # Confirm with the EXACT original formula (np.linalg.norm + < comparison)
                # → violation list + dist values are bit-identical to the original.
                dist = float(np.linalg.norm(P[k, i] - P[k, j]))
                if dist < thr:
                    violations.append((int(k), i, j, dist))
    # Sort by (sample, i, j) — matches the exact order of the original nested loop.
    violations.sort(key=lambda t: (t[0], t[1], t[2]))
    return violations
