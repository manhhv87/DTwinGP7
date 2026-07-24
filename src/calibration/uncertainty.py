"""
uncertainty.py
──────────────
Bootstrap uncertainty of the hand-eye calibration (paper §3.4 → Eq. (1)).

The anchored-DR sampler needs a per-parameter sigma, not just a point estimate
of T_BC. We bootstrap: resample the calibration pose pairs WITH replacement
B times, re-solve hand-eye each time, and report the empirical standard
deviations of the translation components (mm) and of the rotation, expressed
as a rotation-vector deviation from the mean rotation (deg, per axis + total).

Output JSON (config/calibration/T_base_camera_sigma.json) is consumed by
src/synthgen/scene_sampler.load_sigma_json.

cv2 is used only through solve_hand_eye (lazy inside); the statistics are
pure numpy.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .hand_eye_solver import solve_hand_eye

logger = logging.getLogger(__name__)

DEFAULT_N_BOOT = 200
# A resample is discarded when it has fewer distinct poses than this — with
# too few distinct motions AX=XB is ill-conditioned and the solve is garbage.
MIN_DISTINCT_POSES = 5


def _matrix_to_rotvec_deg(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → rotation vector (axis * angle), degrees."""
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_a))
    if angle < 1e-12:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        # Near-180°: extract the axis from R + I (rank-1). Rare for bootstrap
        # deviations from the mean, but handle it rather than divide by ~0.
        M = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(M), 0.0, None))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
    else:
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / (2.0 * np.sin(angle))
    return np.rad2deg(angle * axis)


def _mean_rotation(Rs: list[np.ndarray]) -> np.ndarray:
    """Chordal mean of rotation matrices via SVD projection."""
    M = np.sum(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return R


def bootstrap_hand_eye(
    poses_gripper2base: list[np.ndarray],
    poses_target2cam: list[np.ndarray],
    method: str = "park",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap the hand-eye solve → sigma per calibrated parameter.

    Args:
        poses_gripper2base / poses_target2cam: The calibration pose pairs
            (same lists CalibrationSession collected, mm units).
        method: Hand-eye method (default "park" — see solver notes on Tsai).
        n_boot: Number of bootstrap resamples (paper uses B=200).
        seed: RNG seed for reproducibility.

    Returns:
        Dict with T_ref (full-set solve), sigma_trans_mm [3], sigma_rot_deg [3],
        sigma_rot_total_deg, n_boot, n_valid, n_poses, method, seed.

    Raises:
        ValueError: Mismatched inputs or every resample failed.
    """
    n = len(poses_gripper2base)
    if n != len(poses_target2cam):
        raise ValueError("pose list length mismatch")

    # Reference solve on the full set (this is the T_BC the cell actually uses).
    T_ref = solve_hand_eye(poses_gripper2base, poses_target2cam, method=method)

    rng = np.random.default_rng(seed)
    ts: list[np.ndarray] = []
    Rs: list[np.ndarray] = []
    n_failed = 0
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        if len(set(idx.tolist())) < MIN_DISTINCT_POSES:
            n_failed += 1
            continue
        g2b = [poses_gripper2base[i] for i in idx]
        t2c = [poses_target2cam[i] for i in idx]
        try:
            T = solve_hand_eye(g2b, t2c, method=method)
        except Exception as e:  # noqa: BLE001 — cv2 can fail on degenerate sets
            logger.debug("Bootstrap resample failed: %s", e)
            n_failed += 1
            continue
        ts.append(T[:3, 3].copy())
        Rs.append(T[:3, :3].copy())

    n_valid = len(ts)
    if n_valid < 10:
        raise ValueError(
            f"Only {n_valid}/{n_boot} bootstrap resamples solved — "
            f"calibration set too small/degenerate for uncertainty estimation"
        )

    t_arr = np.asarray(ts)
    sigma_trans = t_arr.std(axis=0, ddof=1)

    R_mean = _mean_rotation(Rs)
    rotvecs = np.asarray([_matrix_to_rotvec_deg(R_mean.T @ R) for R in Rs])
    sigma_rot = rotvecs.std(axis=0, ddof=1)
    sigma_rot_total = float(np.linalg.norm(rotvecs, axis=1).std(ddof=1))

    result = {
        "source": "bootstrap",
        "method": method,
        "n_poses": n,
        "n_boot": int(n_boot),
        "n_valid": n_valid,
        "seed": int(seed),
        "T_ref_mm": T_ref.tolist(),
        "sigma_trans_mm": sigma_trans.round(4).tolist(),
        "sigma_rot_deg": sigma_rot.round(5).tolist(),
        "sigma_rot_total_deg": round(sigma_rot_total, 5),
    }
    logger.info(
        "Bootstrap hand-eye (%s, B=%d, valid=%d): sigma_t=%s mm, sigma_r=%s deg",
        method, n_boot, n_valid,
        sigma_trans.round(3).tolist(), sigma_rot.round(4).tolist(),
    )
    return result


def save_sigma_json(result: dict[str, Any], path: str | Path) -> Path:
    """Write the bootstrap result where the synthgen sampler expects it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Calibration sigma saved → %s", p)
    return p
