"""
capture_calibration.py
──────────────────────
Collect hand-eye calibration data: detect ChArUco board and pair with
robot pose, then call the solver.

ChArUco estimator uses cv2 (lazy-import). API follows OpenCV 4.8+
(CharucoDetector + matchImagePoints + solvePnP).

Workflow (see section 6.3 of documentation):
    1. Robot holds ChArUco board on end-effector
    2. Move robot to 25-30 diverse poses (rotation ±30°)
    3. Each pose: capture_pose(rgb, robot_pose) records one pair
    4. solve() → T_BC
"""
from __future__ import annotations

import logging

import numpy as np

from .hand_eye_solver import solve_hand_eye

logger = logging.getLogger(__name__)


class CharucoBoardEstimator:
    """Detect ChArUco board and estimate pose T_target2cam.

    Args:
        squares_xy: Number of squares (cols, rows). Default (7, 5).
        square_length_m: Square side length (meters). Default 0.04.
        marker_length_m: ArUco marker side length (meters). Default 0.03.
        dictionary: ArUco dictionary name. Default "DICT_4X4_50".
    """

    def __init__(
        self,
        squares_xy: tuple[int, int] = (7, 5),
        square_length_m: float = 0.04,
        marker_length_m: float = 0.03,
        dictionary: str = "DICT_4X4_50",
    ) -> None:
        import cv2  # lazy import

        self._cv2 = cv2
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dictionary)
        )
        self.board = cv2.aruco.CharucoBoard(
            squares_xy, square_length_m, marker_length_m, self.aruco_dict
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.square_length_m = square_length_m

    def estimate_pose(
        self,
        gray: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        min_corners: int = 10,
    ) -> np.ndarray | None:
        """Estimate board → camera pose (T_target2cam 4x4, **mm**).

        Unit note: the ChArUco board is created with square_length in meters
        (OpenCV convention), so solvePnP returns tvec in meters. Convert
        to mm for consistency with the full pipeline (robot.Pose() returns mm, T_BC mm).

        Returns:
            4x4 matrix in mm units, or None if insufficient corners detected.
        """
        cv2 = self._cv2
        ch_corners, ch_ids, _, _ = self.detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < min_corners:
            return None

        obj_pts, img_pts = self.board.matchImagePoints(ch_corners, ch_ids)
        if obj_pts is None or len(obj_pts) < min_corners:
            return None

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, camera_matrix, dist_coeffs
        )
        if not ok:
            return None

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten() * 1000.0       # meters → mm
        return T


class CalibrationSession:
    """Store pose pairs and solve hand-eye calibration.

    Logic is decoupled from camera/robot I/O → testable by feeding poses directly.
    """

    def __init__(self, estimator: CharucoBoardEstimator | None = None) -> None:
        self.estimator = estimator
        self.poses_gripper2base: list[np.ndarray] = []  # T_E^B (mm)
        self.poses_target2cam: list[np.ndarray] = []     # T_W^C (mm)

    def add_pose_pair(
        self,
        T_gripper2base: np.ndarray,
        T_target2cam: np.ndarray,
    ) -> int:
        """Add a known pose pair directly. Returns the total number of pairs."""
        self.poses_gripper2base.append(np.asarray(T_gripper2base, dtype=float))
        self.poses_target2cam.append(np.asarray(T_target2cam, dtype=float))
        return len(self.poses_gripper2base)

    def capture_pose(
        self,
        gray: np.ndarray,
        T_gripper2base: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> bool:
        """Detect board in image and pair with robot pose.

        Args:
            gray: Grayscale image containing ChArUco board.
            T_gripper2base: End-effector pose in base frame (translation in **mm**,
                consistent with estimate_pose's tvec*1000 and solve_hand_eye). NOT
                meters — feeding metres here gives a 1000x-wrong hand-eye result.
            camera_matrix: 3x3 intrinsic matrix.
            dist_coeffs: Distortion coefficients.

        Returns:
            True if a pose pair was captured, False if board not detected.
        """
        if self.estimator is None:
            raise RuntimeError("CalibrationSession requires an estimator for capture_pose")

        T_target2cam = self.estimator.estimate_pose(
            gray, camera_matrix, dist_coeffs
        )
        if T_target2cam is None:
            logger.info("Insufficient corners detected — try a different pose")
            return False

        self.add_pose_pair(T_gripper2base, T_target2cam)
        logger.info("Recorded pose #%d", len(self.poses_gripper2base))
        return True

    def solve(self, method: str = "park") -> np.ndarray:
        """Solve hand-eye from collected pose pairs. Returns T_BC (mm).

        Default "park" (NOT "tsai") because the downward-facing camera causes T_BC
        to rotate ~180°, hitting the Tsai-Lenz singularity — see notes in solve_hand_eye.
        """
        return solve_hand_eye(
            self.poses_gripper2base, self.poses_target2cam, method=method
        )

    @property
    def num_poses(self) -> int:
        """Number of pose pairs collected."""
        return len(self.poses_gripper2base)
