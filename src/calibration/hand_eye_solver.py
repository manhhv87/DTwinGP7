"""
hand_eye_solver.py
──────────────────
Solves the hand-eye calibration problem for an EYE-TO-HAND setup.

Setup: D455 camera fixed on a gantry; ChArUco board attached to the end-effector.
Goal: find T_BC = camera-to-base matrix such that
    p_base = T_BC @ p_camera

⚠ IMPORTANT CONVENTION:
cv2.calibrateHandEye() is designed for eye-IN-hand. For eye-TO-hand you must
pass base2gripper = inverse(gripper2base). The output is then the camera in the
base frame = T_BC. This is a common convention pitfall — this solver handles the
inversion internally; callers only need to pass gripper2base as returned by
robot.Pose().

Solving logic is pure-numpy; cv2 is used only for calibrateHandEye (lazy import).
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Minimum number of poses required to solve. 25+ with diverse rotations recommended.
MIN_POSES = 10


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transformation matrix."""
    T = np.asarray(T, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _opencv_method(name: str):
    """Map method name → cv2 constant."""
    import cv2

    table = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"Method '{name}' is not supported. Choose from: {list(table)}")
    return table[key]


def validate_poses(poses_gripper2base: list[np.ndarray]) -> None:
    """Check pose count and rotation diversity of a pose set.

    Calibration becomes ill-conditioned when poses differ only in translation.

    Raises:
        ValueError: If there are too few poses.
    """
    n = len(poses_gripper2base)
    if n < MIN_POSES:
        raise ValueError(
            f"At least {MIN_POSES} poses required (25+ ideal), got {n}."
        )

    # Measure rotation spread as the max PAIRWISE angular deviation (not just vs the
    # first pose) — an outlier first pose could otherwise mask a rotation-deficient
    # set (all the rest mutually near-identical).
    Rs = [T[:3, :3] for T in poses_gripper2base]
    angles = []
    for i in range(len(Rs)):
        for j in range(i + 1, len(Rs)):
            dR = Rs[i].T @ Rs[j]
            cos_a = np.clip((np.trace(dR) - 1) / 2, -1, 1)
            angles.append(np.degrees(np.arccos(cos_a)))
    if angles and max(angles) < 15.0:
        logger.warning(
            "Pose rotations are too similar (max deviation %.1f°). "
            "Calibration may be ill-conditioned — add poses with end-effector rotation ±30°.",
            max(angles),
        )


def solve_hand_eye(
    poses_gripper2base: list[np.ndarray],
    poses_target2cam: list[np.ndarray],
    method: str = "park",
) -> np.ndarray:
    """Solve eye-to-hand hand-eye calibration → returns T_BC (camera-to-base).

    ⚠ Do NOT use "tsai" for this setup: the camera looks straight down, so T_BC
    contains a ~180° rotation component — exactly the singularity of the Tsai-Lenz
    parameterization → incorrect result. Park (1994) and Daniilidis (1998) have no
    such singularity and are the recommended defaults.

    Args:
        poses_gripper2base: List of 4x4 matrices — end-effector in base frame
            (i.e. robot.Pose()). Translation units are arbitrary but must be
            CONSISTENT with poses_target2cam.
        poses_target2cam: List of 4x4 matrices — ChArUco board in camera frame
            (from OpenCV solvePnP on ChArUco corners).
        method: "park" (default), "horaud", "daniilidis", "andreff", "tsai".

    Returns:
        T_BC 4x4 — camera in base frame, same units as input.

    Raises:
        ValueError: Insufficient poses or mismatched list lengths.
    """
    import cv2

    if len(poses_gripper2base) != len(poses_target2cam):
        raise ValueError(
            f"Pose count mismatch: gripper2base={len(poses_gripper2base)}, "
            f"target2cam={len(poses_target2cam)}"
        )
    validate_poses(poses_gripper2base)

    # EYE-TO-HAND: invert gripper2base → base2gripper.
    base2gripper = [invert_transform(T) for T in poses_gripper2base]
    R_b2g = [T[:3, :3] for T in base2gripper]
    t_b2g = [T[:3, 3] for T in base2gripper]
    R_t2c = [T[:3, :3] for T in poses_target2cam]
    t_t2c = [T[:3, 3] for T in poses_target2cam]

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_b2g, t_b2g, R_t2c, t_t2c, method=_opencv_method(method)
    )

    T_BC = np.eye(4)
    T_BC[:3, :3] = R_cam2base
    T_BC[:3, 3] = np.asarray(t_cam2base).flatten()
    logger.info(
        "Hand-eye solved (%s, %d poses). Camera at %s",
        method, len(poses_gripper2base), T_BC[:3, 3].round(2),
    )
    return T_BC


def touch_test_error(
    points_camera: list[np.ndarray],
    points_base_truth: list[np.ndarray],
    T_BC: np.ndarray,
) -> dict[str, float]:
    """Evaluate calibration error using points with known coordinates.

    For each point in camera frame, predict its base-frame coordinate via T_BC
    and compare with the measured base-frame ground truth.

    Args:
        points_camera: List of (x, y, z) points in camera frame.
        points_base_truth: List of corresponding points measured in base frame.
        T_BC: Hand-eye matrix to evaluate.

    Returns:
        Dict {mean_mm, max_mm, rms_mm} of distance errors.
    """
    if len(points_camera) != len(points_base_truth):
        raise ValueError("Number of camera and base points must be equal")
    if not points_camera:
        raise ValueError("No evaluation points provided (empty list) — "
                         "cannot compute touch-test error")

    errors = []
    for p_cam, p_truth in zip(points_camera, points_base_truth):
        p_h = np.append(np.asarray(p_cam, dtype=float), 1.0)
        p_pred = (T_BC @ p_h)[:3]
        errors.append(float(np.linalg.norm(p_pred - np.asarray(p_truth))))

    errors = np.array(errors)
    return {
        "mean_mm": float(errors.mean()),
        "max_mm": float(errors.max()),
        "rms_mm": float(np.sqrt(np.mean(errors**2))),
    }
