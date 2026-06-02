"""
inverse_kinematics.py
─────────────────────
Numerical inverse kinematics via Damped Least Squares (DLS) — pure numpy.

No dependency on RoboDK. Allows HSE backend real mode to run WITHOUT
RoboDK API calls for each MoveJ → bypasses RoboDK Free quota entirely.

DLS algorithm:
    q_{k+1} = q_k + J^T (J J^T + λ^2 I)^{-1} · error

where:
    - J: 6x6 spatial Jacobian (numerical finite-difference from FK)
    - error: 6-vector [position (mm), rotation (rad axis-angle)]
    - λ: damping factor (default 0.1) — prevents singularity instability

Performance: ~1-5ms/call on modern CPU for GP7 (6-DOF, ≤100 iter).
Accuracy: position tol 0.5mm, orientation 1e-3 rad ~ 0.06°.

Convergence:
    - q_init close to solution → 5-15 iter
    - q_init far from solution → may not converge → return None
    - Singular config (J rank < 6) → damping rescues, still converges but slower
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
    """Polymorphic FK: URDFRobot or RobotDHModel."""
    if isinstance(model, URDFRobot):
        return forward_kinematics_urdf(model, q)
    return forward_kinematics(model, q)


def _jacobian_analytical_urdf(
    model: URDFRobot, q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical Jacobian for URDFRobot — no finite-difference needed.

    For each revolute joint i:
      J[:3, i] = z_i × (p_tcp - p_i)   (linear velocity from axis crossing lever arm)
      J[3:, i] = z_i                    (angular velocity = axis direction)

    Returns:
        T0: (4,4) FK pose at q (avoids caller recomputing FK for convergence check)
        J:  (6, n) Jacobian.

    Performance: 1 FK pass instead of 7 (1 + 6 finite-diff perturbations) → ~6× faster
    Jacobian computation, reducing IK total from ~3ms to ~1ms.
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
    """Spatial Jacobian 6x6 via finite differences.

    Args:
        T0: FK(q) already computed (avoids recomputation — caller typically just
            computed it for convergence check). None → compute internally. Passing
            it in keeps results identical.

    Returns:
        J: shape (6, num_joints). Rows 0-2 = linear velocity (mm/rad),
           rows 3-5 = angular velocity (rad/rad). Column i = derivative w.r.t. joint i.
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
        # Orientation derivative via skew approximation:
        #   R1 = R0 · (I + eps·skew(ω)) → skew(ω) ≈ (R0^T · R1 - I) / eps
        # Body-frame angular velocity → convert to world: ω_world = R0 · ω_body
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
            err[3:] = rotation vector (axis * angle, rad) rotating current → target.
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
        # Robust algorithm: R + I = 2·n·n^T at θ=π → column with largest norm
        # contains the axis (numerically stable even when 2 diag entries are nearly equal).
        M = R_err + np.eye(3)
        col_norms = np.linalg.norm(M, axis=0)
        idx = int(np.argmax(col_norms))
        if col_norms[idx] < 1e-9:
            # Pathological R_err = -I (rotation by π around any axis) — pick X.
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = M[:, idx] / col_norms[idx]
        # Sign disambiguation: axis-angle in {+axis, -axis} both represent 180°
        # rotation; DLS only needs direction, sign does not matter for convergence.
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


def manipulability(model, q_rad, char_length_mm: float = 1000.0) -> float:
    """Yoshikawa manipulability index: w = sqrt(det(Jₙ·Jₙᵀ)).

    Measures configuration "health" — the robot's ability to generate TCP
    velocity in all directions. w → 0 at **singularity** (Jacobian rank-deficient):
    wrist (θ5≈0, axes 4∥6), boundary (arm fully extended), or shoulder.
    Industrial controllers use w to slow down / warn when Cartesian-jogging
    near a singularity.

    The translational part of J (mm/rad) is normalised by `char_length_mm` to
    match the scale of the rotational part (rad/rad) — without this, mm units
    dominate and w becomes meaningless. char_length ~ robot reach (GP7 ≈ 927mm
    → use 1000mm).

    Args:
        model: URDFRobot or RobotDHModel.
        q_rad: joints (radian), 6 elements.
        char_length_mm: characteristic length for normalising the translational part.

    Returns:
        w ≥ 0. GP7: ~0.07-0.08 when healthy, < 0.01 near singularity, = 0 at
        exact singularity.
    """
    q = np.asarray(q_rad, dtype=float).flatten()
    if isinstance(model, URDFRobot):
        _, J = _jacobian_analytical_urdf(model, q)
    else:
        J = _jacobian_numerical(model, q)
    Jn = J.copy()
    Jn[:3, :] /= char_length_mm                  # normalise linear to same scale as angular
    return float(np.sqrt(max(np.linalg.det(Jn @ Jn.T), 0.0)))


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
    """Solve IK numerically: 4x4 pose → joints (radian).

    Args:
        model: GP7 (or generic 6R) DH model.
        target_pose_world: 4x4 homogeneous transform in WORLD frame (mm).
        q_init_rad: Initial guess (radian) — typically current joints for fast convergence.
        max_iter: Max iterations (default 200). Each iter ~50-100µs for 6-DOF.
        tol_mm: Position tolerance (mm). Default 0.1mm — sufficient for any industrial task.
        tol_rad: Orientation tolerance (radian). 1e-4 ~ 0.006°.
        damping: DLS damping factor λ. Larger → more stable, slower. Smaller → faster,
            less stable near singularity. 0.05 is a good default (smaller → faster
            convergence when far from solution).
        max_step_deg: Cap step size per iter to avoid overshoot. 20° is safe.

    Returns:
        Joints (radian) list of 6 elements if converged, None if failed.

    Notes:
        - Joint limits are clipped each iter — if the true solution lies outside limits,
          IK will converge to the nearest boundary point.
        - Multi-solution: returns only 1 solution near q_init. To pick a different
          branch, retry with a different q_init.
    """
    q = np.asarray(q_init_rad, dtype=float).flatten()
    if len(q) != model.num_joints():
        raise ValueError(
            f"q_init must have {model.num_joints()} elements, got {len(q)}"
        )
    target = np.asarray(target_pose_world, dtype=float)
    if target.shape != (4, 4):
        raise ValueError(f"target_pose must be 4x4, got {target.shape}")

    max_step_rad = np.deg2rad(max_step_deg)
    damping_sq = damping ** 2
    I6 = np.eye(6)

    # Joint limits
    # Polymorphic: URDFRobot uses `.joints`, RobotDHModel uses `.links`
    link_attr = getattr(model, "joints", None) or getattr(model, "links", None)
    q_min = np.array([j.joint_min for j in link_attr])
    q_max = np.array([j.joint_max for j in link_attr])

    # URDFRobot → analytical Jacobian (6× faster). DH model → keep finite-diff
    # (legacy, rarely used — analytical formula would need to be implemented for DH).
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
    """Robust IK: tries from `q_init` first, retries from diverse seeds on failure.

    DLS only converges to 1 solution near `q_init`; for poses that are far away
    or near a singularity, a single attempt can fail (~8% when q_init deviates >40°).
    This function replaces the old "fallback RoboDK SolveIK": if the first attempt
    fails, retries from multiple seeds (perturbations around q_init + random within
    joint range) → pulls convergence rate to ~100% WITHOUT RoboDK.

    Prioritises the solution closest to `q_init`: q_init is tried FIRST so if it
    converges, that is the "minimum joint-jump" solution (best for smooth motion).
    Other seeds are only used when q_init fails.

    Args:
        n_random_seeds: Number of additional random seeds (after deterministic perturbations). 0 = disabled.
        seed: RNG seed → reproducible results (deterministic for tests/thesis).
        **ik_kwargs: Forwarded to `inverse_kinematics` (tol_mm, damping, max_iter…).

    Returns:
        Joints (radian) list of 6 elements, or None if all seeds fail.
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
    # (a) Deterministic perturbations around q_init — nearby solutions, tried first.
    for d_deg in (15.0, 30.0, 60.0, 90.0):
        d = np.deg2rad(d_deg)
        seeds.append(np.clip(q0 + rng.uniform(-d, d, len(q0)), q_min, q_max))
    # (b) Random over full joint range — rescues far poses / alternative branches.
    for _ in range(max(0, n_random_seeds)):
        seeds.append(rng.uniform(q_min, q_max))

    for s in seeds:
        sol = inverse_kinematics(model, target_pose_world, s.tolist(), **ik_kwargs)
        if sol is not None:
            return sol
    return None


# ───────────────────────────────────────────────────────────────────────
# Alternative IK algorithms for thesis comparison
# ───────────────────────────────────────────────────────────────────────


def inverse_kinematics_lm(
    model,
    target_pose_world: np.ndarray,
    q_init_rad: list[float] | tuple[float, ...] | np.ndarray,
    max_iter: int = 100,
    tol_mm: float = 0.5,
    tol_rad: float = 1e-3,
    lambda_init: float = 1e-3,
    lambda_up: float = 10.0,
    lambda_down: float = 0.5,
    max_step_deg: float = 20.0,
) -> list[float] | None:
    """**Levenberg-Marquardt** IK with **adaptive damping**.

    Unlike DLS (fixed damping): LM increases/decreases λ dynamically each iter
    based on error improvement → fast Newton-like when far from solution + stable
    Gauss-Newton when close. Superlinear convergence vs DLS linear.

    Algorithm:
        loop:
            compute J, err
            dq = solve((J^T J + λ I) dq = J^T err)
            if ||err_new|| < ||err_old||: accept, λ *= lambda_down
            else: reject, λ *= lambda_up, retry

    Args:
        lambda_init: Initial damping (Marquardt suggested 1e-3).
        lambda_up/down: Damping adaptation factors (Levenberg: 10x up, 0.5x down).

    Reference:
        Levenberg 1944 / Marquardt 1963 nonlinear least squares.
    """
    q = np.asarray(q_init_rad, dtype=float).flatten()
    if len(q) != model.num_joints():
        raise ValueError(f"q_init must have {model.num_joints()} elements")
    target = np.asarray(target_pose_world, dtype=float)
    if target.shape != (4, 4):
        raise ValueError(f"target_pose must be 4x4, got {target.shape}")

    link_attr = getattr(model, "joints", None) or getattr(model, "links", None)
    q_min = np.array([j.joint_min for j in link_attr])
    q_max = np.array([j.joint_max for j in link_attr])
    n = len(q)
    max_step_rad = np.deg2rad(max_step_deg)
    use_analytical = isinstance(model, URDFRobot)

    lam = float(lambda_init)
    if use_analytical:
        T_cur, J = _jacobian_analytical_urdf(model, q)
    else:
        T_cur = _fk(model, q); J = _jacobian_numerical(model, q, T0=T_cur)
    err = _pose_error(T_cur, target)
    err_norm = float(np.linalg.norm(err))

    for _ in range(max_iter):
        pos_err_norm = float(np.linalg.norm(err[:3]))
        rot_err_norm = float(np.linalg.norm(err[3:]))
        if pos_err_norm < tol_mm and rot_err_norm < tol_rad:
            return q.tolist()

        # LM step: (J^T J + λ I) dq = J^T err
        JTJ = J.T @ J
        JTe = J.T @ err
        try:
            dq = np.linalg.solve(JTJ + lam * np.eye(n), JTe)
        except np.linalg.LinAlgError:
            return None

        # Step size cap
        step_norm = float(np.linalg.norm(dq))
        if step_norm > max_step_rad:
            dq = dq * (max_step_rad / step_norm)

        q_new = np.clip(q + dq, q_min, q_max)
        if use_analytical:
            T_new, J_new = _jacobian_analytical_urdf(model, q_new)
        else:
            T_new = _fk(model, q_new); J_new = _jacobian_numerical(model, q_new, T0=T_new)
        err_new = _pose_error(T_new, target)
        err_new_norm = float(np.linalg.norm(err_new))

        if err_new_norm < err_norm:
            # Accept step + reduce damping
            q = q_new; J = J_new; err = err_new; err_norm = err_new_norm
            lam = max(lam * lambda_down, 1e-9)
        else:
            # Reject step + increase damping
            lam = min(lam * lambda_up, 1e9)

    return None


def inverse_kinematics_sdls(
    model,
    target_pose_world: np.ndarray,
    q_init_rad: list[float] | tuple[float, ...] | np.ndarray,
    max_iter: int = 100,
    tol_mm: float = 0.5,
    tol_rad: float = 1e-3,
    gamma_max_deg: float = 45.0,
    sigma_min: float = 0.01,
) -> list[float] | None:
    """**Selectively Damped Least Squares** IK (Buss & Kim 2005).

    Unlike DLS (fixed `λ²I` damping for all directions): SDLS uses **SVD** to
    damp **only** directions near small singular values. Directions with large σ
    (well-conditioned) use 1/σ (no damping → fast Gauss-Newton). Directions with
    small σ (near singularity) use a damped pseudoinverse → stable.

    Algorithm:
        J = U Σ V^T
        For each component i:
            λ_i² = max(0, σ_min² · (σ_min/σ_i)²)  or selective damping
            δ_i = σ_i / (σ_i² + λ_i²) · (U_i^T · err)
        dq = V · diag(δ) — pre clamp each row by gamma_max
        Step-size: clamp |dq_j| ≤ gamma_max per joint

    Reference:
        Buss & Kim (2005). "Selectively Damped Least Squares for Inverse Kinematics."
        Journal of Graphics Tools, 10(3): 37-49.
    """
    q = np.asarray(q_init_rad, dtype=float).flatten()
    target = np.asarray(target_pose_world, dtype=float)
    if target.shape != (4, 4):
        raise ValueError(f"target_pose must be 4x4, got {target.shape}")
    link_attr = getattr(model, "joints", None) or getattr(model, "links", None)
    q_min = np.array([j.joint_min for j in link_attr])
    q_max = np.array([j.joint_max for j in link_attr])
    gamma_max = np.deg2rad(gamma_max_deg)
    use_analytical = isinstance(model, URDFRobot)

    for _ in range(max_iter):
        if use_analytical:
            T_cur, J = _jacobian_analytical_urdf(model, q)
        else:
            T_cur = _fk(model, q); J = _jacobian_numerical(model, q, T0=T_cur)
        err = _pose_error(T_cur, target)
        if (np.linalg.norm(err[:3]) < tol_mm
                and np.linalg.norm(err[3:]) < tol_rad):
            return q.tolist()

        # SVD: J = U Σ V^T
        try:
            U, sigma, Vt = np.linalg.svd(J, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        # Selective damping: each singular value gets its own λ
        # Buss-Kim simplified: damp inversely to σ. σ small → λ large.
        dq = np.zeros(len(q))
        for i, s in enumerate(sigma):
            if s < sigma_min * 1e-3:
                continue                            # extremely small → skip (no contribution)
            # Smooth selective damping: λ_i large when σ_i ≪ σ_min
            if s < sigma_min:
                lam_sq = (sigma_min * sigma_min) * (1.0 - (s / sigma_min)) ** 2
            else:
                lam_sq = 0.0
            inv_s = s / (s * s + lam_sq)
            # Contribution: (U_i^T · err) · V_i · inv_s
            ut_e = float(U[:, i] @ err)
            contrib = inv_s * ut_e * Vt[i, :]
            # Component-wise clamp per Buss-Kim N_max bound
            max_abs = float(np.max(np.abs(contrib)))
            if max_abs > gamma_max:
                contrib = contrib * (gamma_max / max_abs)
            dq = dq + contrib

        # Final step-size limit for full vector
        step_norm = float(np.linalg.norm(dq))
        if step_norm > gamma_max:
            dq = dq * (gamma_max / step_norm)

        q = np.clip(q + dq, q_min, q_max)

    return None


def inverse_kinematics_bfgs(
    model,
    target_pose_world: np.ndarray,
    q_init_rad: list[float] | tuple[float, ...] | np.ndarray,
    max_iter: int = 200,
    tol_mm: float = 0.5,
    tol_rad: float = 1e-3,
    w_pos: float = 1.0,
    w_rot: float = 1000.0,
) -> list[float] | None:
    """**Newton-Raphson + BFGS** IK via quasi-Newton optimization.

    Frames IK as nonlinear least squares:
        minimize F(q) = ½ ||W · err(q)||²    where err = pose_error(FK(q), target)

    **Important**: pose_error = [pos_mm (3), rot_rad (3)] mixes units → unweighted
    cost ½||err||² is dominated by the position term. **Weighted cost** with
    W = diag(w_pos, w_pos, w_pos, w_rot, w_rot, w_rot) balances the gradient
    so L-BFGS-B steps properly. Default w_rot = 1000 (≈ tol_mm / tol_rad).

    Uses **L-BFGS-B** (limited-memory BFGS with box constraints) — a quasi-Newton
    method that maintains an approximate inverse Hessian via rank-2 updates, without
    computing the true Hessian. **Joint limits** are handled naturally via box constraints.

    Analytical gradient: ∇F = J^T · W^T W · err (Gauss-Newton, drop second-order term).

    Unlike LM (which directly uses J^T J + λI Gauss-Newton step): BFGS accumulates
    curvature information over history → second-order convergence near solution.
    Worse when far from solution because initial H ≈ I.

    Reference:
        Nocedal & Wright (2006). "Numerical Optimization", Ch. 6 (BFGS).
    """
    from scipy.optimize import minimize

    q_init = np.asarray(q_init_rad, dtype=float).flatten()
    target = np.asarray(target_pose_world, dtype=float)
    if target.shape != (4, 4):
        raise ValueError(f"target_pose must be 4x4, got {target.shape}")
    link_attr = getattr(model, "joints", None) or getattr(model, "links", None)
    bounds = [(j.joint_min, j.joint_max) for j in link_attr]
    use_analytical = isinstance(model, URDFRobot)
    # Weight diagonal W² (for cost) and W (for residual scaling)
    w_sq = np.array([w_pos] * 3 + [w_rot] * 3) ** 2

    # Last accepted solution satisfying industrial tolerance — set by callback for
    # **fair early-exit** as soon as tol is reached (does not optimize past it).
    # If None at the end → fall back to scipy's converged result + verify tol.
    early_exit_q: list[np.ndarray] = []

    def _cost_and_grad(q):
        if use_analytical:
            T, J = _jacobian_analytical_urdf(model, q)
        else:
            T = _fk(model, q); J = _jacobian_numerical(model, q, T0=T)
        err = _pose_error(T, target)
        # err = target - current → F = ½ err^T W² err
        # ∂err/∂q = -J → ∇F = -J^T (W² err)
        w_err = w_sq * err
        f = 0.5 * float(err @ w_err)
        g = -(J.T @ w_err)
        return f, g

    class _StopAtTol(Exception):
        pass

    def _callback(xk):
        # After each accepted step, check unweighted pose error vs industrial tol.
        # If reached → raise to break scipy loop (fair with DLS/LM/SDLS stop-at-tol).
        T = _fk(model, xk)
        err = _pose_error(T, target)
        if (np.linalg.norm(err[:3]) < tol_mm
                and np.linalg.norm(err[3:]) < tol_rad):
            early_exit_q.append(np.asarray(xk).copy())
            raise _StopAtTol()

    try:
        result = minimize(
            _cost_and_grad, q_init, jac=True, method="L-BFGS-B",
            bounds=bounds, callback=_callback,
            options={"maxiter": max_iter, "ftol": 1e-14, "gtol": 1e-10})
        x_final = result.x
    except _StopAtTol:
        x_final = early_exit_q[-1]
    except Exception:
        return None

    # Verify final accuracy vs industrial tol
    T_final = _fk(model, x_final)
    err_final = _pose_error(T_final, target)
    if (np.linalg.norm(err_final[:3]) < tol_mm
            and np.linalg.norm(err_final[3:]) < tol_rad):
        return [float(v) for v in x_final]
    return None


def _pose_error_batch(
    T_curr: np.ndarray, T_target: np.ndarray,
) -> np.ndarray:
    """Batched 6-vector pose error. T_curr (N,4,4), T_target (4,4) → (N, 6).

    err[:, :3] = pos_target - pos_curr.
    err[:, 3:] = axis-angle log map of R_target · R_curr^T.

    NOTE: 180° corner case uses the standard formula with a safe_sin fallback guard.
    In batched mode, individual elements near π are rare (random seeds → uniform
    distribution of rotation, P(180°±tol) << 1%). Trade-off: speed vs
    robustness — acceptable for enumeration aimed at "finding alternative solutions".
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
    """Batched IK: solve N independent IK problems in parallel via numpy.

    All N problems run in a shared outer iteration loop. Each iter, all N are batched:
    FK, Jacobian, DLS solve. numpy.linalg.solve on (N, 6, 6) stacked matrices is
    a single BLAS call → leverages SIMD + multi-core BLAS (MKL/OpenBLAS).

    Performance: N=30 sequential ~291ms → batched ~5-15ms (20-50× speedup).

    Args:
        q_init_batch: (N, num_joints) seed configurations.

    Returns:
        List of length N. Each element is a solution joints (radian list of 6) or None
        if that seed did not converge. Order matches q_init_batch.

    Limitation: only supports URDFRobot (requires batched FK). DH model falls back to
    sequential `inverse_kinematics`.
    """
    if not isinstance(model, URDFRobot):
        # Fallback: run sequentially for DH model (rare path).
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
        raise ValueError(f"target_pose must be 4x4, got {target.shape}")
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
            # np.linalg.solve needs RHS shape (..., m, n); err (N,6) → (N,6,1).
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
        # Clip to joint limits (all batch — converged entries are idempotent since
        # final_q already holds the correct saved value).
        q_batch = np.clip(q_batch, q_min, q_max)

    # Build result list
    results: list[list[float] | None] = []
    for i in range(N):
        results.append(final_q[i].tolist() if converged[i] else None)
    return results
