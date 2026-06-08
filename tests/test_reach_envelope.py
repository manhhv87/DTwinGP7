"""
test_reach_envelope.py
──────────────────────
Verify sphere reachability model client-side.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.orchestrator.backends.reach_envelope import ReachEnvelope


class TestGP7Default:
    def test_default_uses_927_reach(self):
        env = ReachEnvelope.gp7_default()
        assert env.reach_max_mm == 927.0
        assert env.reach_min_mm == 150.0

    def test_custom_base(self):
        env = ReachEnvelope.gp7_default(base_xyz_mm=(0, 0, 630))
        assert env.base_xyz_mm == (0, 0, 630)


class TestCanReach:
    @pytest.fixture
    def env(self):
        return ReachEnvelope(
            base_xyz_mm=(0, 0, 630),
            reach_max_mm=927.0,
            reach_min_mm=150.0,
        )

    def test_within_envelope_ok(self, env):
        # 700mm horizontal từ base = √(700² + 0²) khoảng cách
        target = np.array([700, 0, 500])
        # dist from base = √(700² + 0² + 130²) ≈ 712 → OK (150..927)
        assert env.can_reach(target) is True

    def test_outside_envelope_far_rejected(self, env):
        target = np.array([2000, 0, 0])
        assert env.can_reach(target) is False

    def test_inside_min_dead_zone_rejected(self, env):
        target = np.array([50, 0, 630])           # 50mm từ base — dead zone
        assert env.can_reach(target) is False

    def test_at_max_boundary_ok(self, env):
        # Tại exactly 927mm — inclusive
        target = np.array([927, 0, 630])
        assert env.can_reach(target) is True

    def test_4x4_pose_extracted(self, env):
        T = np.eye(4)
        T[:3, 3] = [700, 0, 500]
        assert env.can_reach(T) is True

    def test_list_extracted(self, env):
        assert env.can_reach([700, 0, 500]) is True

    def test_invalid_target_permissive(self, env):
        # Không extract được → coi như reachable (an toàn cho test code)
        assert env.can_reach("nonsense") is True


class TestDistanceFromBase:
    def test_distance_correct(self):
        env = ReachEnvelope(base_xyz_mm=(0, 0, 0), reach_max_mm=1000, reach_min_mm=0)
        d = env.distance_from_base([3, 4, 0])         # pythagoras
        assert d == pytest.approx(5.0)

    def test_distance_none_for_invalid(self):
        env = ReachEnvelope(base_xyz_mm=(0, 0, 0), reach_max_mm=1000, reach_min_mm=0)
        assert env.distance_from_base("xxx") is None


class TestIntegrationWithHSEBackend:
    """ReachEnvelope wire vào MotomanHSEBackend.MoveJ_Test."""

    def test_movej_test_uses_envelope_when_provided(self):
        from src.orchestrator.backends.motoman_hse import MotomanHSEBackend

        env = ReachEnvelope(base_xyz_mm=(0, 0, 0), reach_max_mm=500, reach_min_mm=100)
        backend = MotomanHSEBackend(ip="x", reach_envelope=env)

        T_in = np.eye(4)
        T_in[:3, 3] = [300, 0, 0]
        assert backend.MoveJ_Test(None, T_in) == 0

        T_out = np.eye(4)
        T_out[:3, 3] = [1000, 0, 0]
        assert backend.MoveJ_Test(None, T_out) == -1

    def test_movej_test_permissive_without_envelope(self):
        from src.orchestrator.backends.motoman_hse import MotomanHSEBackend

        backend = MotomanHSEBackend(ip="x")    # no envelope
        T = np.eye(4)
        T[:3, 3] = [99999, 0, 0]
        assert backend.MoveJ_Test(None, T) == 0    # permissive


class TestWorkspaceReachEnvelopeMesh:
    """The §3 WorkSpace mesh = brute-force FK reach envelope (RoboDK-style).

    Guards two things the old hand-derived/fixed-offset version got wrong:
      • accuracy vs the GP7 datasheet reach (~927 mm at the wrist centre);
      • the tool/flange envelope must grow by the real offset length (wrist swept),
        so tool > flange > wrist.
    Mesh generation is headless (no OpenGL needed)."""

    def _fn(self):
        import types
        from src.orchestrator.viewports.gp7_app_qt import GP7AppQt
        from src.orchestrator.kinematics.urdf_chain import gp7_urdf
        m = gp7_urdf(base_xyz_mm=(0.0, 0.0, 0.0))
        T_tool = np.eye(4); T_tool[2, 3] = 100.0          # 100 mm TCP offset
        stub = types.SimpleNamespace(
            _model=m, _base_xyz=(0.0, 0.0, 0.0),
            _tool_frames=[("flange", np.eye(4)), ("tcp", T_tool)], _tool_idx=1,
            _reach_envelope_cache={})
        return GP7AppQt._compute_reach_envelope_mesh.__get__(stub)

    @staticmethod
    def _radius_mm(mesh):
        b = mesh.bounds
        return max(abs(b[0]), abs(b[1]), abs(b[2]), abs(b[3])) * 1000.0

    def test_wrist_matches_datasheet_reach(self):
        fn = self._fn()
        mesh, inner = fn("wrist")
        assert mesh is not None and mesh.n_points > 0
        r = self._radius_mm(mesh)
        assert 880 < r < 960, f"wrist reach {r:.0f}mm should be ~927mm (datasheet)"

    def test_tool_extends_reach_beyond_flange_beyond_wrist(self):
        fn = self._fn()
        rw = self._radius_mm(fn("wrist")[0])
        rf = self._radius_mm(fn("flange")[0])
        rt = self._radius_mm(fn("tool")[0])
        assert rt > rf > rw, f"expected tool>flange>wrist, got {rt:.0f}/{rf:.0f}/{rw:.0f}"
        assert 120 < (rt - rw) < 240    # ~180mm (80 flange + 100 tool)

    def test_none_model_returns_none(self):
        import types
        from src.orchestrator.viewports.gp7_app_qt import GP7AppQt
        stub = types.SimpleNamespace(_model=None)
        assert GP7AppQt._compute_reach_envelope_mesh.__get__(stub)("wrist") is None


class TestWorkspaceMatchesFK:
    """Lock the dented WorkSpace envelope to an INDEPENDENT FK ground truth: the mesh
    radius per azimuth (app uses analytic J1) must match the true max reach found by
    explicitly sweeping J1 (different method). Also checks the opposite dents."""

    def _meshes(self):
        import types
        from src.orchestrator.viewports.gp7_app_qt import GP7AppQt
        from src.orchestrator.kinematics.urdf_chain import gp7_urdf
        m = gp7_urdf(base_xyz_mm=(0.0, 0.0, 0.0))
        stub = types.SimpleNamespace(
            _model=m, _base_xyz=(0.0, 0.0, 0.0),
            _tool_frames=[("flange", np.eye(4))], _tool_idx=0,
            _reach_envelope_cache={})
        outer, inner = GP7AppQt._compute_reach_envelope_mesh.__get__(stub)("wrist")
        return m, outer, inner

    @staticmethod
    def _fk_maxr(model, az, j2_filter):
        import math
        from src.orchestrator.kinematics.urdf_chain import link_frames_urdf
        jl = [(j.joint_min, j.joint_max) for j in model.joints]
        best = 0.0
        for q1 in np.linspace(jl[0][0], jl[0][1], 90):
            for qL in np.linspace(jl[1][0], jl[1][1], 16):
                if not j2_filter(qL):
                    continue
                for qU in np.linspace(jl[2][0], jl[2][1], 16):
                    p = dict(link_frames_urdf(model, [q1, qL, qU, 0, 0, 0]))["link_B"][:3, 3]
                    r = math.hypot(p[0], p[1])
                    a = math.degrees(math.atan2(p[1], p[0]))
                    if abs(((a - az + 180) % 360) - 180) < 6 and r > best:
                        best = r
        return best

    @staticmethod
    def _mesh_maxr(mesh, az):
        P = mesh.points
        a = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
        r = np.hypot(P[:, 0], P[:, 1]) * 1000.0
        sel = r[np.abs(((a - az + 180) % 360) - 180) < 6]
        return float(sel.max()) if len(sel) else 0.0

    def test_outer_matches_fk_all_configs(self):
        m, outer, _ = self._meshes()
        for az in (0, 90, 180, 270):
            mesh = self._mesh_maxr(outer, az)
            fk = self._fk_maxr(m, az, lambda q: True)
            assert abs(mesh - fk) < 90, f"outer az={az}: mesh {mesh:.0f} vs FK {fk:.0f}"

    @staticmethod
    def _mesh_minr_band(mesh, zlo_mm, zhi_mm):
        P = mesh.points
        z = P[:, 2] * 1000.0
        r = np.hypot(P[:, 0], P[:, 1]) * 1000.0
        sel = r[(z >= zlo_mm) & (z <= zhi_mm)]
        return float(sel.min()) if len(sel) else 0.0

    @staticmethod
    def _fk_minr_band(model, zlo_mm, zhi_mm):
        import math
        from src.orchestrator.kinematics.urdf_chain import link_frames_urdf
        jl = [(j.joint_min, j.joint_max) for j in model.joints]
        best = 1e9
        for q1 in np.linspace(jl[0][0], jl[0][1], 24):
            for qL in np.linspace(jl[1][0], jl[1][1], 30):
                for qU in np.linspace(jl[2][0], jl[2][1], 36):
                    p = dict(link_frames_urdf(model, [q1, qL, qU, 0, 0, 0]))["link_B"][:3, 3]
                    if zlo_mm <= p[2] <= zhi_mm:
                        r = math.hypot(p[0], p[1])
                        if 1.0 <= r < best:
                            best = r
        return best if best < 1e9 else 0.0

    def test_single_envelope_no_separate_inner(self):
        # The void is now carved into ONE kidney mesh — no second inner surface.
        _, outer, inner = self._meshes()
        assert outer is not None and inner is None

    def test_inner_void_r233(self):
        # The single kidney has the datasheet inner concave dent: in the mid-height
        # band the closest approach to the J1 axis (~R233) sits far inside the outer
        # reach and matches the FK min-reach there.
        m, kidney, _ = self._meshes()
        band = (-40.0, 40.0)                 # raw FK z band straddling the R233 bulge
        mesh = self._mesh_minr_band(kidney, *band)
        fk = self._fk_minr_band(m, *band)
        assert abs(mesh - fk) < 60, f"void wall {mesh:.0f} vs FK {fk:.0f}"
        assert 160 < mesh < 300, f"inner dent {mesh:.0f} not near datasheet R233"
        assert self._mesh_maxr(kidney, 0) - mesh > 400, "dent must be inside outer shell"

    def test_envelope_vertical_extent(self):
        # Kidney top/bottom match datasheet (wrist P-point: +887 / -806 mm rel L-axis;
        # = 1217 / -476 rel floor). Tolerance covers the nz height-binning.
        _, kidney, _ = self._meshes()
        z = kidney.points[:, 2] * 1000.0
        assert abs(z.max() - 887) < 45, f"top {z.max():.0f} vs 887"
        assert abs(z.min() - (-806)) < 45, f"bottom {z.min():.0f} vs -806"

    def test_outer_dent(self):
        # outer reaches farther in front than behind (dent at rear) — physics.
        m, outer, _ = self._meshes()
        assert self._mesh_maxr(outer, 0) > self._mesh_maxr(outer, 180) + 50
