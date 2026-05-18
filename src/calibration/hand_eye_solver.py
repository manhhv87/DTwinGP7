"""
hand_eye_solver.py
──────────────────
Giải bài toán hand-eye calibration cho setup EYE-TO-HAND.

Setup: camera D455 cố định trên giàn, ChArUco board gắn trên end-effector.
Mục tiêu: tìm T_BC = ma trận camera-trong-base (camera-to-base) sao cho
    p_base = T_BC @ p_camera

⚠ QUY ƯỚC QUAN TRỌNG:
cv2.calibrateHandEye() được thiết kế cho eye-IN-hand. Với eye-TO-hand phải
truyền base2gripper = inverse(gripper2base). Khi đó output chính là camera
trong base frame = T_BC. Đây là lỗi quy ước thường gặp — solver này xử lý
sẵn việc nghịch đảo, người dùng chỉ cần truyền gripper2base như robot.Pose()
trả về.

Logic giải là pure-numpy; cv2 chỉ dùng cho calibrateHandEye và lazy-import.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Số pose tối thiểu để giải. Khuyến nghị 25+ với rotation đa dạng.
MIN_POSES = 10


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Nghịch đảo ma trận biến đổi đồng nhất 4x4."""
    T = np.asarray(T, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _opencv_method(name: str):
    """Map tên phương pháp → hằng số cv2."""
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
        raise ValueError(f"Phương pháp '{name}' không hỗ trợ. Chọn: {list(table)}")
    return table[key]


def validate_poses(poses_gripper2base: list[np.ndarray]) -> None:
    """Kiểm tra số lượng + tính đa dạng rotation của tập pose.

    Calibration sẽ ill-conditioned nếu các pose chỉ khác nhau về translation.

    Raises:
        ValueError: Nếu quá ít pose.
    """
    n = len(poses_gripper2base)
    if n < MIN_POSES:
        raise ValueError(
            f"Cần tối thiểu {MIN_POSES} pose (lý tưởng 25+), chỉ có {n}."
        )

    # Đo độ phân tán rotation: góc lệch giữa pose đầu và các pose còn lại.
    R0 = poses_gripper2base[0][:3, :3]
    angles = []
    for T in poses_gripper2base[1:]:
        dR = R0.T @ T[:3, :3]
        cos_a = np.clip((np.trace(dR) - 1) / 2, -1, 1)
        angles.append(np.degrees(np.arccos(cos_a)))
    if angles and max(angles) < 15.0:
        logger.warning(
            "Rotation các pose quá giống nhau (max lệch %.1f°). "
            "Calibration dễ ill-conditioned — thêm pose xoay end-effector ±30°.",
            max(angles),
        )


def solve_hand_eye(
    poses_gripper2base: list[np.ndarray],
    poses_target2cam: list[np.ndarray],
    method: str = "park",
) -> np.ndarray:
    """Giải hand-eye eye-to-hand → trả về T_BC (camera-to-base).

    ⚠ KHÔNG dùng "tsai" cho setup này: camera nhìn thẳng xuống nên T_BC
    chứa thành phần xoay ~180°, đúng điểm kỳ dị của tham số hóa Tsai-Lenz
    → kết quả sai. Park (1994) và Daniilidis (1998) không có điểm kỳ dị này
    và là lựa chọn mặc định khuyến nghị.

    Args:
        poses_gripper2base: List ma trận 4x4 — end-effector trong base frame
            (chính là robot.Pose()). Đơn vị translation tuỳ ý nhưng phải
            ĐỒNG NHẤT với poses_target2cam.
        poses_target2cam: List ma trận 4x4 — ChArUco board trong camera frame
            (từ OpenCV solvePnP trên ChArUco corners).
        method: "park" (mặc định), "horaud", "daniilidis", "andreff", "tsai".

    Returns:
        T_BC 4x4 — camera trong base frame, cùng đơn vị với input.

    Raises:
        ValueError: Số pose không đủ hoặc 2 list lệch độ dài.
    """
    import cv2

    if len(poses_gripper2base) != len(poses_target2cam):
        raise ValueError(
            f"Số pose lệch: gripper2base={len(poses_gripper2base)}, "
            f"target2cam={len(poses_target2cam)}"
        )
    validate_poses(poses_gripper2base)

    # EYE-TO-HAND: nghịch đảo gripper2base → base2gripper.
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
        "Hand-eye giải xong (%s, %d pose). Camera tại %s",
        method, len(poses_gripper2base), T_BC[:3, 3].round(2),
    )
    return T_BC


def touch_test_error(
    points_camera: list[np.ndarray],
    points_base_truth: list[np.ndarray],
    T_BC: np.ndarray,
) -> dict[str, float]:
    """Đánh giá sai số calibration bằng các điểm đã biết toạ độ.

    Với mỗi điểm trong camera frame, dự đoán toạ độ base qua T_BC rồi so
    với toạ độ base đo thực tế.

    Args:
        points_camera: List điểm (x, y, z) trong camera frame.
        points_base_truth: List điểm tương ứng đo trong base frame.
        T_BC: Ma trận hand-eye cần đánh giá.

    Returns:
        Dict {mean_mm, max_mm, rms_mm} sai số khoảng cách.
    """
    if len(points_camera) != len(points_base_truth):
        raise ValueError("Số điểm camera và base phải bằng nhau")

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
