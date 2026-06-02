"""
pieper_gp7.py
─────────────
Analytical (closed-form) IK for Yaskawa GP7 using Pieper decoupling.

GP7 has a **spherical wrist** — 3 wrist axes (J4, J5, J6) all intersect at one
point called the Wrist Center Point (WCP). In our URDF chain, J5 and J6 have
origin (0,0,0) in the parent frame, so all 3 axes pass through the J4 origin (=WCP).

Pieper decoupling splits the 6-DOF IK into 2 independent 3-DOF sub-problems:
  1. **Position**: solve q1, q2, q3 from WCP_world (depends only on WCP position)
  2. **Orientation**: solve q4, q5, q6 from R_J3_J6 = R_world_J3⁻¹ · R_world_J6

Yields **up to 8 solutions** natively:
  - 2 for J1 (front / back, J1 ± π/2 around shoulder)
  - 2 for elbow (up / down — asin has 2 branches)
  - 2 for wrist (normal / flip — XYX Euler has 2 branches via β = ±)
  → 2 × 2 × 2 = 8 configurations

Speed: ~0.24 ms/call (~5× faster than DLS) — benchmark
`scripts/17_compare_fk_ik.py --fair --samples 200` (machine-dependent).
Accuracy: float-precision exact (~1e-13 mm, float noise floor), no DLS
tolerance gap.

Reference:
  Pieper, D. L. (1968). "The Kinematics of Manipulators Under Computer Control."
  Stanford PhD thesis. Chapter 4: spherical wrist decomposition.

────────────────────────────────────────────────────────────────────────
GP7 URDF parameters (from gp7_urdf):
  J1 (S): origin (0, 0, 0),    axis (0, 0, 1)
  J2 (L): origin (40, 0, 0),   axis (0, 1, 0)
  J3 (U): origin (0, 0, 445),  axis (0, -1, 0)
  J4 (R): origin (440, 0, 40), axis (-1, 0, 0)   ← WCP at J4 origin
  J5 (B): origin (0, 0, 0),    axis (0, -1, 0)
  J6 (T): origin (0, 0, 0),    axis (-1, 0, 0)
  flange: (80, 0, 0)
  tool0:  RPY(π, -π/2, 0) → R_J6_tool0 = [[0,0,1],[0,-1,0],[1,0,0]]

Derived constants:
  A1 = 40    (shoulder X offset)
  L2 = 445   (upper arm length, J2 → J3)
  A3_x = 440 (forearm X)
  A3_z = 40  (forearm Z offset → small wrist offset, gives non-zero PSI)
  L3 = √(A3_x² + A3_z²) = √195200 ≈ 441.8  (effective forearm length)
  PSI = atan2(A3_z, A3_x) ≈ 5.2°  (forearm tilt at q3=0)
  D6 = 80    (flange offset from WCP along tool0 Z)
"""
from __future__ import annotations

import math

import numpy as np

from .urdf_chain import URDFRobot

# ── GP7 geometric constants ────────────────────────────────────────────
_A1 = 40.0
_L2 = 445.0
_A3_X = 440.0
_A3_Z = 40.0
_L3 = math.sqrt(_A3_X * _A3_X + _A3_Z * _A3_Z)
_PSI = math.atan2(_A3_Z, _A3_X)
_D6 = 80.0

# R_J6_tool0 = Ry(-π/2) · Rx(π) — fixed URDF tool0 rotation
_R_J6_TOOL0 = np.array([
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
])


def _Rx(q: float) -> np.ndarray:
    c, s = math.cos(q), math.sin(q)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(q: float) -> np.ndarray:
    c, s = math.cos(q), math.sin(q)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _Rz(q: float) -> np.ndarray:
    c, s = math.cos(q), math.sin(q)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _wrap_to_pi(angle: float) -> float:
    """Wrap angle to (-π, π]."""
    a = math.fmod(angle + math.pi, 2 * math.pi)
    if a <= 0: a += 2 * math.pi
    return a - math.pi


def _wrap_to_limits(angle: float, lo: float, hi: float) -> float | None:
    """Find equivalent angle (angle + 2π·k) that lies in [lo, hi].

    Necessary because GP7 joint limits are asymmetric (e.g. J3 = -116°..+255°,
    J6 = ±360°). A naive wrap-to-(-π, π] may push a solution outside the limits
    even though the other branch (+2π or -2π) is perfectly valid.

    Returns None if no k fits.
    """
    # Start from canonical (-π, π] representation
    base = _wrap_to_pi(angle)
    tol = 1e-9
    # Try base, base+2π, base-2π, base+4π, base-4π (cover joint ranges ≤ ±720°)
    for k in (0, 1, -1, 2, -2):
        cand = base + 2 * math.pi * k
        if (lo - tol) <= cand <= (hi + tol):
            return cand
    return None


def _within_limits(q: float, lo: float, hi: float) -> bool:
    # Allow slight numerical overshoot at boundaries.
    return (lo - 1e-9) <= q <= (hi + 1e-9)


def _base_R_t(model: URDFRobot) -> tuple[np.ndarray, np.ndarray]:
    """Return base rotation (3x3) and translation (3,) of robot S frame in world."""
    from .urdf_chain import _base_transform
    T = _base_transform(model)
    return T[:3, :3].copy(), T[:3, 3].copy()


def _solve_q123(wcp_s: np.ndarray, joint_lim: list[tuple[float, float]]
                 ) -> list[tuple[int, int, float, float, float]]:
    """Solve q1, q2, q3 from the WCP position in the robot S frame.

    Returns a list of (front_back, elbow, q1, q2, q3) — up to 4 configs
    (2 front/back × 2 elbow). front_back: 0=front, 1=rear. elbow: 0=up, 1=down.
    """
    x, y, z = float(wcp_s[0]), float(wcp_s[1]), float(wcp_s[2])
    results: list[tuple[int, int, float, float, float]] = []
    L2_sq = _L2 * _L2
    L3_sq = _L3 * _L3

    for front_back in (0, 1):
        if front_back == 0:                      # front
            q1_raw = math.atan2(y, x)
            r_signed = math.sqrt(x * x + y * y)
        else:                                    # back (J1 flipped 180°)
            q1_raw = math.atan2(y, x) + math.pi
            r_signed = -math.sqrt(x * x + y * y)
        # Joint limit aware wrap (J1 limit ±170° symmetric, but be consistent)
        q1 = _wrap_to_limits(q1_raw, *joint_lim[0])
        if q1 is None: continue

        x_prime = r_signed - _A1
        z_prime = z
        D_sq = x_prime * x_prime + z_prime * z_prime
        zeta = (D_sq - L2_sq - L3_sq) / (2.0 * _L2)
        ratio = zeta / _L3
        if abs(ratio) > 1.0 + 1e-9:
            continue                              # pose unreachable for this q1
        ratio = max(-1.0, min(1.0, ratio))
        asin_z = math.asin(ratio)

        for elbow in (0, 1):
            if elbow == 0:                       # asin branch → "elbow up"
                q3_raw = asin_z - _PSI
            else:                                # π - asin → "elbow down"
                q3_raw = (math.pi - asin_z) - _PSI
            # J3 limit asymmetric (-116°…+255°) → try +2π unwrap if wrap-to-π fails.
            q3 = _wrap_to_limits(q3_raw, *joint_lim[2])
            if q3 is None: continue

            xi = _A3_X * math.cos(q3) - _A3_Z * math.sin(q3)
            eta = _L2 + _A3_X * math.sin(q3) + _A3_Z * math.cos(q3)
            denom = xi * x_prime + eta * z_prime
            numer = eta * x_prime - xi * z_prime
            q2_raw = math.atan2(numer, denom)
            q2 = _wrap_to_limits(q2_raw, *joint_lim[1])
            if q2 is None: continue

            results.append((front_back, elbow, q1, q2, q3))
    return results


def _solve_q456(M: np.ndarray, joint_lim: list[tuple[float, float]]
                 ) -> list[tuple[int, float, float, float]]:
    """Solve q4, q5, q6 from M = R_J3⁻¹ · R_world_J6.

    Returns (wrist_flip, q4, q5, q6) — wrist_flip: 0=non-flip, 1=flip.

    M = Rx(-q4) · Ry(-q5) · Rx(-q6)  (URDF axes: J4=-X, J5=-Y, J6=-X)
    Let α = -q4, β = -q5, γ = -q6 → M = Rx(α) Ry(β) Rx(γ)  (XYX Euler).

    Decompose:
      M[0,0] = cos(β)
      M[1,0] = sin(α) sin(β)
      M[2,0] = -cos(α) sin(β)
      M[0,1] = sin(β) sin(γ)
      M[0,2] = sin(β) cos(γ)

    2 branches: β ∈ [0, π] or β ∈ [-π, 0] (sign of sin(β) flips → α, γ flip by π).
    """
    results: list[tuple[int, float, float, float]] = []
    cb = float(np.clip(M[0, 0], -1.0, 1.0))
    # Singular wrist: β = 0 (cos = 1) — only α + γ is determined
    if abs(cb - 1.0) < 1e-9:
        alpha = math.atan2(M[2, 1], M[1, 1])
        q4 = _wrap_to_limits(-alpha, *joint_lim[3])
        if q4 is None: return results
        q5 = _wrap_to_limits(0.0, *joint_lim[4])
        if q5 is None: return results
        q6 = _wrap_to_limits(0.0, *joint_lim[5])
        if q6 is None: return results
        results.append((0, q4, q5, q6))
        return results
    if abs(cb + 1.0) < 1e-9:
        alpha = -math.atan2(M[1, 2], M[1, 1])
        q4 = _wrap_to_limits(-alpha, *joint_lim[3])
        if q4 is None: return results
        q5 = _wrap_to_limits(-math.pi, *joint_lim[4])
        if q5 is None: return results
        q6 = _wrap_to_limits(0.0, *joint_lim[5])
        if q6 is None: return results
        results.append((0, q4, q5, q6))
        return results

    # Standard 2-branch case
    for wrist_flip in (0, 1):
        if wrist_flip == 0:
            beta = math.acos(cb)                  # β ∈ [0, π], sin(β) ≥ 0
        else:
            beta = -math.acos(cb)                 # β ∈ [-π, 0], sin(β) ≤ 0
        sb = math.sin(beta)
        alpha = math.atan2(M[1, 0] / sb, -M[2, 0] / sb)
        gamma = math.atan2(M[0, 1] / sb, M[0, 2] / sb)
        q4 = _wrap_to_limits(-alpha, *joint_lim[3])
        if q4 is None: continue
        q5 = _wrap_to_limits(-beta, *joint_lim[4])
        if q5 is None: continue
        q6 = _wrap_to_limits(-gamma, *joint_lim[5])
        if q6 is None: continue
        results.append((wrist_flip, q4, q5, q6))
    return results


def inverse_kinematics_pieper_gp7(
    model: URDFRobot,
    target_pose_world: np.ndarray,
) -> list[list[float]]:
    """Analytical IK for GP7 — returns all solutions (up to 8).

    Returns: list of joint vectors (radians, length 6). Empty list if unreachable.

    Performance: ~0.24 ms/call (see scripts/17_compare_fk_ik.py --fair).
    Accuracy: float-precision exact (FK(sol) matches target to ~1e-13 mm).
    """
    target = np.asarray(target_pose_world, dtype=float)
    if target.shape != (4, 4):
        raise ValueError(f"target_pose must be 4x4, got {target.shape}")
    # Convert target → robot S frame
    base_R, base_t = _base_R_t(model)
    p_world = target[:3, 3]
    R_world = target[:3, :3]
    p_S = base_R.T @ (p_world - base_t)
    R_S = base_R.T @ R_world

    # WCP in S frame: TCP - 80mm * tool0_Z_world (then expressed in S)
    # Note: tool0_Z in world = R_world[:, 2] = R_S[:, 2] after transform.
    wcp_s = p_S - _D6 * R_S[:, 2]

    # Joint limits
    jl = [(j.joint_min, j.joint_max) for j in model.joints]

    # R_world_J6 = R_world_tool0 · R_J6_tool0^T
    # R_J6_tool0 is symmetric (its own transpose) → R_world_J6 = R_S · R_J6_tool0
    R_J6_in_S = R_S @ _R_J6_TOOL0.T

    solutions: list[list[float]] = []
    for _fb, _el, q1, q2, q3 in _solve_q123(wcp_s, jl):
        # R_J3 in S frame = Rz(q1) · Ry(q2) · Ry(-q3) = Rz(q1) · Ry(q2 - q3)
        # Note: J3 axis = -Y, rotation by q3 about -Y = rotation by -q3 about Y.
        # The frame after J3 has accumulated rotation Rz(q1)·Ry(q2-q3).
        R_J3 = _Rz(q1) @ _Ry(q2 - q3)
        M = R_J3.T @ R_J6_in_S
        for _wf, q4, q5, q6 in _solve_q456(M, jl):
            solutions.append([q1, q2, q3, q4, q5, q6])
    return solutions


def _joint_turn_variants(
    joints_rad: tuple[float, ...],
    joint_lim: list[tuple[float, float]],
) -> list[list[float]]:
    """Enumerate valid "joint turn" variants (±360°) for one solution.

    Joints with range > 360° (e.g. J6 = ±360° → 720° range) allow the same
    end orientation to be reached at multiple turns (q, q±360°). RoboDK lists
    these as separate configurations. Returns the Cartesian product of all valid
    wrapped values within joint limits (deduped). Posture flags are unchanged
    (turns only add/subtract full revolutions).
    """
    import itertools
    per_joint: list[list[float]] = []
    for q, (lo, hi) in zip(joints_rad, joint_lim):
        vals = []
        for k in (-2, -1, 0, 1, 2):
            cand = q + 2.0 * math.pi * k
            if (lo - 1e-9) <= cand <= (hi + 1e-9):
                vals.append(cand)
        per_joint.append(vals or [q])
    return [list(combo) for combo in itertools.product(*per_joint)]


def inverse_kinematics_pieper_gp7_tagged(
    model: URDFRobot,
    target_pose_world: np.ndarray,
    include_turns: bool = True,
) -> list[dict]:
    """Like inverse_kinematics_pieper_gp7 but TAGS each solution with config flags.

    Each config is distinguished by 3 binary flags (standard industrial posture flags):
      • front:    arm reaching forward (True) or base rotated rearward (False=Rear)
      • elbow_up: elbow up (True) or down (False)
      • no_flip:  wrist not flipped (True) or flipped (False)
    → config_id = front_back*4 + elbow*2 + wrist_flip  (0..7)

    Flags are taken DIRECTLY from the algorithm branch that produced the solution
    (not post-classified), so they are absolutely accurate.

    include_turns=True: also enumerates joint-turn variants (±360°) like RoboDK —
    multiple rows with the same config_id but different turn counts (e.g. J6).
    False = base 8 solutions only.

    Returns: list[dict] — keys: id, front, elbow_up, no_flip, joints_deg(6).
             Sorted by (id, joints). Empty if unreachable.
    """
    target = np.asarray(target_pose_world, dtype=float)
    if target.shape != (4, 4):
        raise ValueError(f"target_pose must be 4x4, got {target.shape}")
    base_R, base_t = _base_R_t(model)
    p_S = base_R.T @ (target[:3, 3] - base_t)
    R_S = base_R.T @ target[:3, :3]
    wcp_s = p_S - _D6 * R_S[:, 2]
    jl = [(j.joint_min, j.joint_max) for j in model.joints]
    R_J6_in_S = R_S @ _R_J6_TOOL0.T

    out: list[dict] = []
    seen: set = set()
    for fb, el, q1, q2, q3 in _solve_q123(wcp_s, jl):
        R_J3 = _Rz(q1) @ _Ry(q2 - q3)
        M = R_J3.T @ R_J6_in_S
        for wf, q4, q5, q6 in _solve_q456(M, jl):
            base = (q1, q2, q3, q4, q5, q6)
            variants = (_joint_turn_variants(base, jl) if include_turns
                        else [list(base)])
            for var in variants:
                key = tuple(round(v, 6) for v in var)
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "id": fb * 4 + el * 2 + wf,
                    "front": fb == 0,
                    "elbow_up": el == 0,
                    "no_flip": wf == 0,
                    "joints_deg": [math.degrees(a) for a in var],
                })
    out.sort(key=lambda c: (c["id"], c["joints_deg"]))
    return out


def inverse_kinematics_pieper_gp7_nearest(
    model: URDFRobot,
    target_pose_world: np.ndarray,
    q_init_rad: list[float] | tuple[float, ...] | np.ndarray,
) -> list[float] | None:
    """Pieper IK + pick solution nearest q_init (for continuous motion).

    Returns: joint vector (radians) or None if unreachable.
    """
    sols = inverse_kinematics_pieper_gp7(model, target_pose_world)
    if not sols:
        return None
    q_init = np.asarray(q_init_rad, dtype=float)
    best_dist = float("inf")
    best = sols[0]
    for sol in sols:
        d = float(np.max(np.abs(np.asarray(sol) - q_init)))
        if d < best_dist:
            best_dist = d; best = sol
    return best
