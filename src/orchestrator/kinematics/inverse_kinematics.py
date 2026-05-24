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
from .urdf_chain import (
    URDFRobot,
    fk_with_joint_frames_batch_urdf,
    fk_with_joint_frames_urdf,
    forward_kinematics_urdf,
)


def _fk(model, q):
    """Polymorphic FK: URDFRobot hoặc RobotDHModel."""
    if isinstance(model, URDFRobot):
        return forward_kinematics_urdf(model, q)
    return forward_kinematics(model, q)


def _jacobian_analytical_urdf(
    model: URDFRobot, q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical Jacobian cho URDFRobot — không cần finite-difference.

    Cho mỗi revolute joint i:
      J[:3, i] = z_i × (p_tcp - p_i)   (linear velocity từ axis crossing lever arm)
      J[3:, i] = z_i                    (angular velocity = axis direction)

    Returns:
        T0: (4,4) FK pose tại q (tránh caller tính FK lần nữa cho convergence check)
        J:  (6, n) Jacobian.

    Performance: 1 FK pass thay vì 7 (1 + 6 finite-diff perturbations) → ~6× faster
    Jacobian computation, giảm IK total từ ~3ms xuống ~1ms.
    """
    T0, p_joints, z_joints = fk_with_joint_frames_urdf(model, q)
    p_tcp = T0[:3, 3]
    n = len(q)
    J = np.zeros((6, n))
    # Vectorized: lever_arm = p_tcp - p_joints; J[:3] = z × lever_arm
    lever = p_tcp[None, :] - p_joints       # (n, 3)
    # cross product row-wise via np.cross
    J[:3, :] = np.cross(z_joints, lever).T
    J[3:, :] = z_joints.T
    return T0, J


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
        # Special case: 180° rotation. Standard formula sin(θ)≈0 → div-by-zero.
        # Robust algorithm: R + I = 2·n·n^T at θ=π → cột nào có norm lớn nhất
        # chứa axis (numerical stable kể cả khi 2 diag entries gần bằng nhau).
        M = R_err + np.eye(3)
        col_norms = np.linalg.norm(M, axis=0)
        idx = int(np.argmax(col_norms))
        if col_norms[idx] < 1e-9:
            # Pathological R_err = -I (rotation by π around any axis) — pick X.
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = M[:, idx] / col_norms[idx]
        # Sign disambiguation: axis-angle in {+axis, -axis} đều biểu diễn 180°
        # rotation; DLS chỉ cần direction, sign không matter cho convergence.
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

    # URDFRobot → analytical Jacobian (6× faster). DH model → giữ finite-diff
    # (legacy, ít dùng — analytical formula sẽ phải implement riêng cho DH).
    use_analytical = isinstance(model, URDFRobot)

    for _ in range(max_iter):
        if use_analytical:
            T_cur, J = _jacobian_analytical_urdf(model, q)
        else:
            T_cur = _fk(model, q)
            J = _jacobian_numerical(model, q, T0=T_cur)
        err = _pose_error(T_cur, target)

        # Convergence check
        pos_err_norm = np.linalg.norm(err[:3])
        rot_err_norm = np.linalg.norm(err[3:])
        if pos_err_norm < tol_mm and rot_err_norm < tol_rad:
            return q.tolist()
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


def _pose_error_batch(
    T_curr: np.ndarray, T_target: np.ndarray,
) -> np.ndarray:
    """Batched 6-vector pose error. T_curr (N,4,4), T_target (4,4) → (N, 6).

    err[:, :3] = pos_target - pos_curr.
    err[:, 3:] = axis-angle log map của R_target · R_curr^T.

    NOTE: 180° corner case dùng standard formula với fallback safe_sin guard.
    Trong batched, individual elements gần π là rare (random seeds → uniform
    distribution của rotation, P(180°±tol) << 1%). Trade-off: tốc độ vs
    robustness — đối với enumeration mục đích "tìm nghiệm khác", chấp nhận.
    """
    N = T_curr.shape[0]
    pos_err = T_target[None, :3, 3] - T_curr[:, :3, 3]              # (N, 3)
    # R_err = R_target @ R_curr^T
    R_err = T_target[None, :3, :3] @ T_curr[:, :3, :3].transpose(0, 2, 1)
    tr = R_err[:, 0, 0] + R_err[:, 1, 1] + R_err[:, 2, 2]
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)                                    # (N,)
    sin_theta = np.sin(theta)
    axis_raw = np.stack([
        R_err[:, 2, 1] - R_err[:, 1, 2],
        R_err[:, 0, 2] - R_err[:, 2, 0],
        R_err[:, 1, 0] - R_err[:, 0, 1],
    ], axis=-1)                                                     # (N, 3)
    safe_sin = np.where(np.abs(sin_theta) > 1e-9, 2.0 * sin_theta, 2.0)
    rot_err = axis_raw / safe_sin[:, None] * theta[:, None]
    # Zero out where theta is near 0
    rot_err = np.where(theta[:, None] < 1e-9, 0.0, rot_err)
    return np.concatenate([pos_err, rot_err], axis=-1)              # (N, 6)


def inverse_kinematics_batch(
    model: URDFRobot,
    target_pose_world: np.ndarray,
    q_init_batch: np.ndarray,
    max_iter: int = 100,
    tol_mm: float = 0.5,
    tol_rad: float = 1e-3,
    damping: float = 0.05,
    max_step_deg: float = 20.0,
) -> list[list[float] | None]:
    """Batched IK: solve N independent IK problems song song qua numpy.

    Mọi N problems chạy chung outer iter loop. Mỗi iter, tất cả N batched:
    FK, Jacobian, DLS solve. numpy.linalg.solve trên (N, 6, 6) stacked matrices
    là 1 BLAS call → tận dụng SIMD + multi-core BLAS (MKL/OpenBLAS).

    Performance: N=30 sequential ~291ms → batched ~5-15ms (20-50× speedup).

    Args:
        q_init_batch: (N, num_joints) seed configurations.

    Returns:
        List dài N. Mỗi phần tử là solution joints (radian list 6) hoặc None
        nếu seed đó không converge. Thứ tự khớp với q_init_batch.

    Limitation: chỉ support URDFRobot (cần batched FK). DH model fallback dùng
    sequential `inverse_kinematics`.
    """
    if not isinstance(model, URDFRobot):
        # Fallback: chạy sequential cho DH model (rare path).
        out: list[list[float] | None] = []
        for q0 in q_init_batch:
            out.append(inverse_kinematics(
                model, target_pose_world, q0.tolist(),
                max_iter=max_iter, tol_mm=tol_mm, tol_rad=tol_rad,
                damping=damping, max_step_deg=max_step_deg))
        return out

    q_batch = np.asarray(q_init_batch, dtype=float).copy()
    target = np.asarray(target_pose_world, dtype=float)
    if target.shape != (4, 4):
        raise ValueError(f"target_pose phải 4x4, got {target.shape}")
    N, n = q_batch.shape

    q_min = np.array([j.joint_min for j in model.joints])
    q_max = np.array([j.joint_max for j in model.joints])
    max_step_rad = np.deg2rad(max_step_deg)
    damping_sq = damping ** 2
    I6 = np.eye(6)

    converged = np.zeros(N, dtype=bool)
    final_q = q_batch.copy()

    for _ in range(max_iter):
        active = ~converged
        if not active.any():
            break
        # Batched FK + analytical Jacobian inputs
        T_curr, p_joints, z_joints = fk_with_joint_frames_batch_urdf(model, q_batch)
        err = _pose_error_batch(T_curr, target)                     # (N, 6)
        pos_err_norm = np.linalg.norm(err[:, :3], axis=1)           # (N,)
        rot_err_norm = np.linalg.norm(err[:, 3:], axis=1)
        newly = active & (pos_err_norm < tol_mm) & (rot_err_norm < tol_rad)
        if newly.any():
            final_q[newly] = q_batch[newly]
            converged = converged | newly
            if converged.all():
                break
        # Build Jacobian (N, 6, n) analytical:
        #   J[:, :3, i] = z_i × (p_tcp - p_i)
        #   J[:, 3:, i] = z_i
        p_tcp = T_curr[:, :3, 3]                                    # (N, 3)
        lever = p_tcp[:, None, :] - p_joints                        # (N, n, 3)
        J = np.zeros((N, 6, n))
        # cross product per (N, n) → (N, n, 3), then transpose to (N, 3, n)
        J[:, :3, :] = np.cross(z_joints, lever).transpose(0, 2, 1)
        J[:, 3:, :] = z_joints.transpose(0, 2, 1)
        # DLS: dq = J^T (J J^T + λ²I)^-1 err  — batched
        JJt = J @ J.transpose(0, 2, 1)                              # (N, 6, 6)
        damped = JJt + damping_sq * I6                              # broadcast (N,6,6)
        try:
            # np.linalg.solve cần RHS shape (..., m, n); err (N,6) → (N,6,1).
            x = np.linalg.solve(damped, err[..., None])             # (N, 6, 1)
        except np.linalg.LinAlgError:
            break
        dq = (J.transpose(0, 2, 1) @ x).squeeze(-1)                 # (N, n)
        # Step size cap
        step_norm = np.linalg.norm(dq, axis=1)                      # (N,)
        scale = np.minimum(1.0, max_step_rad / np.maximum(step_norm, 1e-12))
        dq = dq * scale[:, None]
        # Update only active + non-newly-converged
        update_mask = active & ~newly
        q_batch[update_mask] = q_batch[update_mask] + dq[update_mask]
        # Clip to joint limits (all batch — converged sẽ idempotent vì final_q
        # đã saved value đúng).
        q_batch = np.clip(q_batch, q_min, q_max)

    # Build result list
    results: list[list[float] | None] = []
    for i in range(N):
        results.append(final_q[i].tolist() if converged[i] else None)
    return results
