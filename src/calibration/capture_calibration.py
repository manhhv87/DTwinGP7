"""
capture_calibration.py
──────────────────────
Thu thập dữ liệu hand-eye calibration: phát hiện ChArUco board + ghép với
pose robot, sau đó gọi solver.

ChArUco estimator dùng cv2 (lazy-import). API theo OpenCV 4.8+
(CharucoDetector + matchImagePoints + solvePnP).

Quy trình (xem mục 6.3 tài liệu):
    1. Robot giữ ChArUco board trên end-effector
    2. Di chuyển robot tới 25-30 pose đa dạng (rotation ±30°)
    3. Mỗi pose: capture_pose(rgb, robot_pose) ghi lại 1 cặp
    4. solve() → T_BC
"""
from __future__ import annotations

import logging

import numpy as np

from .hand_eye_solver import solve_hand_eye

logger = logging.getLogger(__name__)


class CharucoBoardEstimator:
    """Phát hiện ChArUco board và ước lượng pose T_target2cam.

    Args:
        squares_xy: Số ô (cols, rows). Mặc định (7, 5).
        square_length_m: Cạnh ô vuông (mét). Mặc định 0.04.
        marker_length_m: Cạnh marker ArUco (mét). Mặc định 0.03.
        dictionary: Tên từ điển ArUco. Mặc định "DICT_4X4_50".
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
        """Ước lượng pose board → camera (T_target2cam 4x4, mét).

        Returns:
            Ma trận 4x4, hoặc None nếu không phát hiện đủ corner.
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
        T[:3, 3] = tvec.flatten()
        return T


class CalibrationSession:
    """Lưu trữ các cặp pose và giải hand-eye.

    Tách logic khỏi I/O camera/robot → test được bằng cách feed pose trực tiếp.
    """

    def __init__(self, estimator: CharucoBoardEstimator | None = None) -> None:
        self.estimator = estimator
        self.poses_gripper2base: list[np.ndarray] = []  # T_E^B (mét)
        self.poses_target2cam: list[np.ndarray] = []     # T_W^C (mét)

    def add_pose_pair(
        self,
        T_gripper2base: np.ndarray,
        T_target2cam: np.ndarray,
    ) -> int:
        """Thêm trực tiếp một cặp pose đã biết. Trả về tổng số cặp."""
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
        """Phát hiện board trong ảnh + ghép pose robot.

        Args:
            gray: Ảnh xám chứa ChArUco board.
            T_gripper2base: Pose end-effector trong base (mét), từ robot.Pose().
            camera_matrix: Ma trận nội 3x3.
            dist_coeffs: Hệ số méo.

        Returns:
            True nếu thu được cặp pose, False nếu không phát hiện board.
        """
        if self.estimator is None:
            raise RuntimeError("CalibrationSession cần estimator để capture_pose")

        T_target2cam = self.estimator.estimate_pose(
            gray, camera_matrix, dist_coeffs
        )
        if T_target2cam is None:
            logger.info("Không phát hiện đủ corner — thử pose khác")
            return False

        self.add_pose_pair(T_gripper2base, T_target2cam)
        logger.info("Đã ghi pose #%d", len(self.poses_gripper2base))
        return True

    def solve(self, method: str = "tsai") -> np.ndarray:
        """Giải hand-eye từ các cặp pose đã thu. Trả về T_BC (mét)."""
        return solve_hand_eye(
            self.poses_gripper2base, self.poses_target2cam, method=method
        )

    @property
    def num_poses(self) -> int:
        """Số cặp pose đã thu."""
        return len(self.poses_gripper2base)
