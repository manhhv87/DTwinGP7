"""
test_uncertainty.py
───────────────────
Unit tests for the bootstrap hand-eye uncertainty (paper §3.4 → Eq. (1) widths).

Synthetic data follows the same eye-to-hand relation as test_hand_eye_solver:
    T_target2cam = inv(T_BC) @ T_gripper2base @ T_board_in_gripper
with small measurement noise injected so the bootstrap spread is non-zero.

Run:
    pytest tests/test_uncertainty.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="bootstrap re-solves via cv2.calibrateHandEye")
from scipy.spatial.transform import Rotation  # noqa: E402

from src.calibration.hand_eye_solver import invert_transform  # noqa: E402
from src.calibration.uncertainty import (  # noqa: E402
    bootstrap_hand_eye,
    save_sigma_json,
)
from src.synthgen import load_sigma_json  # noqa: E402


def make_T(rpy_deg, xyz_mm) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T[:3, 3] = xyz_mm
    return T


def noisy_dataset(T_BC, T_TG, n=18, noise_mm=0.5, noise_deg=0.1, seed=42):
    """Pose pairs consistent with T_BC + small PnP-like noise on target2cam."""
    rng = np.random.default_rng(seed)
    g2b, t2c = [], []
    for _ in range(n):
        T_GB = make_T(rng.uniform(-40, 40, 3), rng.uniform(-300, 300, 3))
        T = invert_transform(T_BC) @ T_GB @ T_TG
        # measurement noise: small rotation + translation on the board pose
        dR = Rotation.from_rotvec(
            np.deg2rad(rng.normal(0, noise_deg, 3))).as_matrix()
        T[:3, :3] = T[:3, :3] @ dR
        T[:3, 3] += rng.normal(0, noise_mm, 3)
        g2b.append(T_GB)
        t2c.append(T)
    return g2b, t2c


T_BC_TRUE = make_T([180, 0, 30], [400.0, 0.0, 850.0])   # overhead camera, mm
T_TG = make_T([5, -10, 15], [0.0, 0.0, 120.0])          # board on gripper


class TestBootstrapHandEye:
    def test_recovers_reference_and_positive_sigma(self):
        g2b, t2c = noisy_dataset(T_BC_TRUE, T_TG)
        result = bootstrap_hand_eye(g2b, t2c, n_boot=60, seed=0)

        T_ref = np.asarray(result["T_ref_mm"])
        assert np.linalg.norm(T_ref[:3, 3] - T_BC_TRUE[:3, 3]) < 3.0  # mm

        s_t = np.asarray(result["sigma_trans_mm"])
        s_r = np.asarray(result["sigma_rot_deg"])
        assert np.all(s_t > 0.0) and np.all(s_t < 10.0)
        assert np.all(s_r > 0.0) and np.all(s_r < 2.0)
        assert result["n_valid"] >= 50
        assert result["source"] == "bootstrap"

    def test_deterministic_per_seed(self):
        g2b, t2c = noisy_dataset(T_BC_TRUE, T_TG)
        r1 = bootstrap_hand_eye(g2b, t2c, n_boot=40, seed=3)
        r2 = bootstrap_hand_eye(g2b, t2c, n_boot=40, seed=3)
        assert r1["sigma_trans_mm"] == r2["sigma_trans_mm"]
        assert r1["sigma_rot_deg"] == r2["sigma_rot_deg"]

    def test_noisier_data_wider_sigma(self):
        g2b_a, t2c_a = noisy_dataset(T_BC_TRUE, T_TG, noise_mm=0.2, noise_deg=0.05)
        g2b_b, t2c_b = noisy_dataset(T_BC_TRUE, T_TG, noise_mm=2.0, noise_deg=0.5)
        quiet = bootstrap_hand_eye(g2b_a, t2c_a, n_boot=40, seed=1)
        loud = bootstrap_hand_eye(g2b_b, t2c_b, n_boot=40, seed=1)
        assert (np.mean(loud["sigma_trans_mm"])
                > 2.0 * np.mean(quiet["sigma_trans_mm"]))

    def test_length_mismatch_raises(self):
        g2b, t2c = noisy_dataset(T_BC_TRUE, T_TG, n=12)
        with pytest.raises(ValueError, match="mismatch"):
            bootstrap_hand_eye(g2b, t2c[:-1], n_boot=20)


class TestSigmaJson:
    def test_roundtrip_into_sampler_loader(self, tmp_path):
        g2b, t2c = noisy_dataset(T_BC_TRUE, T_TG)
        result = bootstrap_hand_eye(g2b, t2c, n_boot=40, seed=0)
        path = save_sigma_json(result, tmp_path / "sigma.json")

        loaded = load_sigma_json(path)             # the synthgen-side loader
        assert loaded is not None
        assert loaded["sigma_trans_mm"] == result["sigma_trans_mm"]
        assert loaded["source"] == "bootstrap"

    def test_missing_file_returns_none(self, tmp_path):
        assert load_sigma_json(tmp_path / "absent.json") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
