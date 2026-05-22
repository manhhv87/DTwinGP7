"""
inverse_kinematics.py
─────────────────────
Numerical inverse kinematics qua Damped Least Squares (DLS) — pure numpy.

KHÔNG phụ thuộc RoboDK. Cho phép HSE backend real mode chạy mà KHÔNG cần
RoboDK API call cho mỗi MoveJ → bypass RoboDK Free quota hoàn toàn.

Thuật toán DLS:
    q_{k+1} = q_k + J^T (J J^T + λ^2 I)^{-1} · error

trong đó:
    - J: 6x6 spatial Jacobian (numerical finite-difference từ FK)
    - error: 6-vector [position (mm), rotation (rad axis-angle)]
    - λ: damping factor (default 0.1) — tránh singularity instability

Performance: ~1-5ms/call trên CPU modern cho GP7 (6-DOF, ≤100 iter).
Accuracy: tol position 0.5mm, orientation 1e-3 rad ~ 0.06°.

Convergence:
    - q_init gần solution → 5-15 iter
    - q_init xa solution → có thể không converge → return None
    - Singular config (J rank < 6) → damping cứu, vẫn converge nhưng chậm
"""
from __future__ import annotations

import numpy as np

from .dh_model import RobotDHModel
from .forward_kinematics import forward_kinematics
from .urdf_chain import URDFRobot, forward_kinematics_urdf


def _fk(model, q):
    """Polymorphic FK: URDFRobot hoặc RobotDHModel."""
    if isinstance(model, URDFRobot):
        return forward_kinematics_urdf(model, q)
    return forward_kinematics(model, q)


def _jacobian_numerical(
    model, q: np.ndarray, eps: float = 1e-6, T0: np.ndarray | None = None,
) -> np.ndarray:
    """Spatial Jacobian 6x6 qua finite differences.

    Args:
        T0: FK(q) đã tính sẵn (tránh tính lại — caller thường vừa tính cho
            convergence check). None → tự tính. Truyền vào giữ kết quả y hệt.

    Returns:
        J: shape (6, num_joints). Rows 0-2 = linear velocity (mm/rad),
           rows 3-5 = angular velocity (rad/rad). Cột i = đạo hàm theo joint i.
    """
    n = model.num_joints()
    if T0 is None:
        T0 = _fk(model, q)
    p0 = T0[:3, 3]
    R0 = T0[:3, :3]

    J = np.zeros((6, n))
    for i in range(n):
        q_plus = q.copy()
        q_plus[i] += eps
        T1 = _fk(model, q_plus)
        # Position derivative
        J[:3, i] = (T1[:3, 3] - p0) / eps
        # Orientation derivative qua skew approximation:
        #   R1 = R0 · (I + eps·skew(ω)) → skew(ω) ≈ (R0^T · R1 - I) / eps
        # Body-frame angular velocity → convert sang world: ω_world = R0 · ω_body
        dR_body = (R0.T @ T1[:3, :3] - np.eye(3)) / eps
        omega_body = np.array([
            dR_body[2, 1] - dR_body[1, 2],
            dR_body[0, 2] - dR_body[2, 0],
            dR_body[1, 0] - dR_body[0, 1],
        ]) * 0.5
        J[3:, i] = R0 @ omega_body
    return J


def _pose_error(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """Compute 6-vector error [position, axis-angle rotation].

    Returns:
        err: shape (6,). err[:3] = target - current position (mm).
            err[3:] = rotation vector (axis * angle, rad) đưa current → target.
    """
    pos_err = T_target[:3, 3] - T_current[:3, 3]

    # Rotation: R_err = R_target · R_current^T (in world frame)
    R_err = T_target[:3, :3] @ T_current[:3, :3].T
    # Convert R_err to axis-angle (logarithm map)
    cos_theta = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if abs(theta) < 1e-9:
        rot_err = np.zeros(3)
    elif abs(theta - np.pi) < 1e-6:
        # Special case: 180° rotation, axis from R_err diagonal
        # R_err = 2·n·n^T - I → diagonal = 2·n_i^2 - 1
        diag = np.diag(R_err)
        idx = np.argmax(diag)
        axis = np.zeros(3)
        axis[idx] = np.sqrt((diag[idx] + 1) * 0.5)
        for j in range(3):
            if j != idx:
                axis[j] = R_err[idx, j] / (2 * axis[idx])
        rot_err = axis * theta
    else:
        # Standard case
        axis = np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]) / (2.0 * np.sin(theta))
        rot_err = axis * theta

    return np.concatenate([pos_err, rot_err])


def inverse_kinematics(
    model,
    target_pose_world: np.ndarray,
    q_init_rad: list[float] | tuple[float, ...] | np.ndarray,
    max_iter: int = 200,
    tol_mm: float = 0.1,
    tol_rad: float = 1e-4,
    damping: float = 0.05,
    max_step_deg: float = 20.0,
) -> list[float] | None:
    """Giải IK numerical: pose 4x4 → joints (radian).

    Args:
        model: GP7 (hoặc generic 6R) DH model.
        target_pose_world: 4x4 homogeneous transform trong WORLD frame (mm).
        q_init_rad: Initial guess (radian) — thường là current joints để converge nhanh.
        max_iter: Max iterations (default 200). Mỗi iter ~50-100µs cho 6-DOF.
        tol_mm: Position tolerance (mm). Default 0.1mm — đủ cao cho mọi industrial task.
        tol_rad: Orientation tolerance (radian). 1e-4 ~ 0.006°.
        damping: DLS damping factor λ. Lớn → ổn định, chậm. Nhỏ → nhanh, kém ổn định
            ở singularity. 0.05 là default tốt (nhỏ hơn → converge nhanh khi xa solution).
        max_step_deg: Cap step size per iter để tránh overshoot. 20° là an toàn.

    Returns:
        Joints (radian) list 6 phần tử nếu converge, None nếu fail.

    Notes:
        - Joint limits được clip mỗi iter — nếu solution đúng nằm ngoài limit,
          IK sẽ converge tới điểm trên biên.
        - Multi-solution: chỉ trả 1 solution gần q_init. Để pick branch khác,
          retry với q_init khác.
    """
    q = np.asarray(q_init_rad, dtype=float).flatten()
    if len(q) != model.num_joints():
        raise ValueError(
            f"q_init phải có {model.num_joints()} phần tử, got {len(q)}"
        )
    target = np.asarray(target_pose_world, dtype=float)
    if target.shape != (4, 4):
        raise ValueError(f"target_pose phải 4x4, got {target.shape}")

    max_step_rad = np.deg2rad(max_step_deg)
    damping_sq = damping ** 2
    I6 = np.eye(6)

    # Joint limits
    # Polymorphic: URDFRobot dùng `.joints`, RobotDHModel dùng `.links`
    link_attr = getattr(model, "joints", None) or getattr(model, "links", None)
    q_min = np.array([j.joint_min for j in link_attr])
    q_max = np.array([j.joint_max for j in link_attr])

    for _ in range(max_iter):
        T_cur = _fk(model, q)
        err = _pose_error(T_cur, target)

        # Convergence check
        pos_err_norm = np.linalg.norm(err[:3])
        rot_err_norm = np.linalg.norm(err[3:])
        if pos_err_norm < tol_mm and rot_err_norm < tol_rad:
            return q.tolist()

        # Tái dùng T_cur (= FK(q)) cho jacobian thay vì tính FK(q) lần nữa.
        J = _jacobian_numerical(model, q, T0=T_cur)
        # DLS update: dq = J^T (J J^T + λ^2 I)^{-1} err
        JJt = J @ J.T
        damped = JJt + damping_sq * I6
        try:
            dq = J.T @ np.linalg.solve(damped, err)
        except np.linalg.LinAlgError:
            return None                              # Should not happen with damping > 0

        # Step size limit
        step_norm = np.linalg.norm(dq)
        if step_norm > max_step_rad:
            dq = dq * (max_step_rad / step_norm)

        q = q + dq
        # Clip to joint limits
        q = np.clip(q, q_min, q_max)

    return None                                       # Did not converge


def inverse_kinematics_seeded(
    model,
    target_pose_world: np.ndarray,
    q_init_rad: list[float] | tuple[float, ...] | np.ndarray,
    *,
    n_random_seeds: int = 8,
    seed: int = 0,
    **ik_kwargs,
) -> list[float] | None:
    """IK bền vững: thử từ `q_init` trước, fail thì retry từ nhiều seed đa dạng.

    DLS chỉ hội tụ tới 1 nghiệm gần `q_init`; với pose xa hoặc gần singularity,
    1 lần thử có thể fail (~8% khi q_init lệch >40°). Hàm này thay thế "fallback
    RoboDK SolveIK" cũ: nếu lần đầu fail, thử lại từ các seed (nhiễu quanh q_init
    + random trong joint range) → kéo tỉ lệ hội tụ về ~100% mà KHÔNG cần RoboDK.

    Ưu tiên nghiệm gần `q_init` nhất: q_init được thử ĐẦU TIÊN nên nếu nó hội tụ,
    đó là nghiệm "ít nhảy joint" nhất (tốt cho chuyển động mượt). Chỉ khi q_init
    fail mới dùng seed khác.

    Args:
        n_random_seeds: Số seed random thêm (sau các seed nhiễu xác định). 0 = tắt.
        seed: Seed RNG → kết quả tái lập được (deterministic cho test/thesis).
        **ik_kwargs: Forward tới `inverse_kinematics` (tol_mm, damping, max_iter…).

    Returns:
        Joints (radian) list 6 phần tử, hoặc None nếu mọi seed đều fail.
    """
    sol = inverse_kinematics(model, target_pose_world, q_init_rad, **ik_kwargs)
    if sol is not None:
        return sol

    q0 = np.asarray(q_init_rad, dtype=float).flatten()
    link_attr = getattr(model, "joints", None) or getattr(model, "links", None)
    q_min = np.array([j.joint_min for j in link_attr])
    q_max = np.array([j.joint_max for j in link_attr])
    rng = np.random.RandomState(seed)

    seeds: list[np.ndarray] = []
    # (a) Nhiễu xác định quanh q_init — nghiệm gần, thử trước.
    for d_deg in (15.0, 30.0, 60.0, 90.0):
        d = np.deg2rad(d_deg)
        seeds.append(np.clip(q0 + rng.uniform(-d, d, len(q0)), q_min, q_max))
    # (b) Random phủ toàn joint range — cứu pose xa / branch khác.
    for _ in range(max(0, n_random_seeds)):
        seeds.append(rng.uniform(q_min, q_max))

    for s in seeds:
        sol = inverse_kinematics(model, target_pose_world, s.tolist(), **ik_kwargs)
        if sol is not None:
            return sol
    return None
