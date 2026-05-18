"""
test_cell_loader.py
───────────────────
Unit tests cho module cell.

Strategy:
  - Pure unit tests: pose_utils (không cần RoboDK)
  - Validation tests: cell_models (không cần RoboDK)
  - Integration tests: cell_loader với MOCK RoboDK API

Run:
    pytest tests/ -v
    pytest tests/test_cell_loader.py::test_load_valid_config -v
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.cell.cell_loader import CellLoader
from src.cell.cell_models import CellConfig, PoseConfig, RobotConfig
from src.cell.pose_utils import (
    make_homogeneous,
    matrix_to_rpy_xyz,
)


# ════════════════════════════════════════════════════════════════
# pose_utils tests
# ════════════════════════════════════════════════════════════════


class TestPoseUtils:
    """Tests cho pose math helpers — không cần RoboDK."""

    def test_make_identity(self):
        T = make_homogeneous([0, 0, 0], [0, 0, 0])
        np.testing.assert_array_almost_equal(T, np.eye(4))

    def test_make_translation_only(self):
        T = make_homogeneous([100, 200, 300])
        np.testing.assert_array_almost_equal(T[:3, 3], [100, 200, 300])
        np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))

    def test_make_yaw_90(self):
        T = make_homogeneous([0, 0, 0], [0, 0, 90])
        expected_R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])
        np.testing.assert_array_almost_equal(T[:3, :3], expected_R, decimal=5)

    def test_roundtrip_xyz_rpy(self):
        """Forward + inverse phải trả lại giá trị gốc."""
        xyz_orig = [123.4, -56.7, 890.1]
        rpy_orig = [12.0, -34.0, 56.0]

        T = make_homogeneous(xyz_orig, rpy_orig)
        xyz_back, rpy_back = matrix_to_rpy_xyz(T)

        np.testing.assert_array_almost_equal(xyz_back, xyz_orig)
        np.testing.assert_array_almost_equal(rpy_back, rpy_orig)

    def test_invalid_xyz_size_raises(self):
        with pytest.raises(ValueError, match="xyz_mm must have 3"):
            make_homogeneous([100, 200])

    def test_invalid_rpy_size_raises(self):
        with pytest.raises(ValueError, match="rpy_deg must have 3"):
            make_homogeneous([0, 0, 0], [0])


# ════════════════════════════════════════════════════════════════
# Pydantic validation tests
# ════════════════════════════════════════════════════════════════


class TestPoseConfig:
    def test_valid_pose(self):
        p = PoseConfig(xyz_mm=(100, 200, 300), rpy_deg=(0, 0, 90))
        assert p.xyz_mm == (100, 200, 300)
        assert p.rpy_deg == (0, 0, 90)

    def test_default_rpy(self):
        p = PoseConfig(xyz_mm=(0, 0, 0))
        assert p.rpy_deg == (0.0, 0.0, 0.0)

    def test_xyz_too_large_raises(self):
        with pytest.raises(ValueError, match="vượt giới hạn"):
            PoseConfig(xyz_mm=(10000, 0, 0))

    def test_rpy_out_of_range_raises(self):
        with pytest.raises(ValueError, match="ngoài range"):
            PoseConfig(xyz_mm=(0, 0, 0), rpy_deg=(0, 0, 500))


class TestRobotConfig:
    def test_valid_library_source(self):
        r = RobotConfig(
            name="GP7",
            source="library",
            library_name="Yaskawa-GP7.robot",
            pose=PoseConfig(xyz_mm=(0, 0, 0)),
            home_joints_deg=[0, 0, 0, 0, 0, 0],
        )
        assert r.name == "GP7"

    def test_missing_library_name_raises(self):
        with pytest.raises(ValueError, match="library_name"):
            RobotConfig(
                name="GP7",
                source="library",
                pose=PoseConfig(xyz_mm=(0, 0, 0)),
                home_joints_deg=[0, 0, 0, 0, 0, 0],
            )

    def test_wrong_num_joints_raises(self):
        with pytest.raises(ValueError):
            RobotConfig(
                name="GP7",
                source="library",
                library_name="GP7.robot",
                pose=PoseConfig(xyz_mm=(0, 0, 0)),
                home_joints_deg=[0, 0, 0],  # only 3, not 6
            )


class TestCellConfig:
    """Test loading từ YAML."""

    @pytest.fixture
    def valid_config_dict(self) -> dict:
        return {
            "robot": {
                "name": "Yaskawa GP7",
                "source": "library",
                "library_name": "Yaskawa-GP7.robot",
                "pose": {"xyz_mm": [0, 0, 0], "rpy_deg": [0, 0, 0]},
                "home_joints_deg": [0, 0, 0, 0, -90, 0],
            },
            "worktable": {
                "mesh": "models/worktable.stl",
                "pose": {"xyz_mm": [400, 0, -10]},
            },
            "camera": {
                "type": "virtual",
                "pose": {"xyz_mm": [400, 0, 850], "rpy_deg": [180, 0, 0]},
            },
            "gripper": {
                "name": "Gripper",
                "tcp_offset_xyz_mm": [0, 0, 130],
            },
        }

    def test_valid_minimal_config(self, valid_config_dict):
        config = CellConfig.model_validate(valid_config_dict)
        assert config.robot.name == "Yaskawa GP7"
        assert config.worktable.color_rgb == (0.6, 0.6, 0.7)  # default

    def test_frame_reference_validation(self, valid_config_dict):
        valid_config_dict["frames"] = [
            {"name": "Frame1", "pose": {"xyz_mm": [0, 0, 0]}, "parent": "NonExistent"}
        ]
        with pytest.raises(ValueError, match="parent 'NonExistent'"):
            CellConfig.model_validate(valid_config_dict)

    def test_object_parent_frame_validation(self, valid_config_dict):
        valid_config_dict["objects"] = [
            {"name": "obj1", "mesh": "x.stl", "parent_frame": "MissingFrame"}
        ]
        with pytest.raises(ValueError, match="parent_frame 'MissingFrame'"):
            CellConfig.model_validate(valid_config_dict)

    def test_real_robot_requires_ip(self, valid_config_dict):
        valid_config_dict["robot_connection"] = {"enabled": True}
        with pytest.raises(ValueError, match="enabled=true requires 'ip'"):
            CellConfig.model_validate(valid_config_dict)


# ════════════════════════════════════════════════════════════════
# CellLoader integration tests (with mocked RoboDK)
# ════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_rdk(monkeypatch):
    """Mock RoboDK Robolink class."""
    mock_robolink = MagicMock()
    mock_rdk_instance = MagicMock()
    mock_robolink.return_value = mock_rdk_instance
    mock_rdk_instance.ItemList.return_value = []

    # Mock các method tạo items
    fake_robot = MagicMock(name="MockRobot")
    fake_table = MagicMock(name="MockTable")
    fake_camera = MagicMock(name="MockCamera")
    fake_tool = MagicMock(name="MockTool")
    fake_frame = MagicMock(name="MockFrame")

    mock_rdk_instance.AddFile.side_effect = lambda *a, **kw: fake_robot if "Yaskawa" in str(a) else fake_table
    mock_rdk_instance.Cam2D_Add.return_value = fake_camera
    mock_rdk_instance.AddFrame.return_value = fake_frame
    fake_robot.AddTool.return_value = fake_tool

    # Patch import in cell_loader
    import src.cell.cell_loader as cl_module
    monkeypatch.setattr(
        cl_module,
        "_resolve_path",
        lambda self, p: Path(p),
        raising=False,
    )

    return mock_rdk_instance, {
        "robot": fake_robot,
        "table": fake_table,
        "camera": fake_camera,
        "tool": fake_tool,
        "frame": fake_frame,
    }


class TestCellLoader:
    """Integration tests cho CellLoader với mocked RoboDK."""

    @pytest.fixture
    def minimal_config(self) -> CellConfig:
        return CellConfig.model_validate({
            "robot": {
                "name": "Yaskawa GP7",
                "source": "library",
                "library_name": "Yaskawa-GP7.robot",
                "pose": {"xyz_mm": [0, 0, 0]},
                "home_joints_deg": [0, 0, 0, 0, -90, 0],
            },
            "worktable": {
                "mesh": "models/worktable.stl",
                "pose": {"xyz_mm": [400, 0, -10]},
            },
            "camera": {
                "type": "virtual",
                "pose": {"xyz_mm": [400, 0, 850]},
                "intrinsics": {
                    "fov_deg": 87,
                    "focal_length_mm": 1.93,
                    "size_px": [1280, 720],
                },
            },
            "gripper": {
                "name": "Gripper",
                "tcp_offset_xyz_mm": [0, 0, 130],
            },
        })

    def test_loader_can_instantiate(self, minimal_config):
        loader = CellLoader(minimal_config)
        assert loader.config == minimal_config
        assert loader._rdk is None  # lazy connect

    def test_loader_uses_default_library_path(self, minimal_config):
        loader = CellLoader(minimal_config)
        assert loader.library_path == Path("C:/RoboDK/Library")

    def test_loader_uses_custom_library_path(self, minimal_config, tmp_path):
        loader = CellLoader(minimal_config, library_path=tmp_path)
        assert loader.library_path == tmp_path


class TestCellConfigFromYAML:
    """Test load từ file YAML thực."""

    def test_load_from_valid_yaml_file(self, tmp_path):
        yaml_content = """
robot:
  name: Test
  source: library
  library_name: Yaskawa-GP7.robot
  pose:
    xyz_mm: [0, 0, 0]
  home_joints_deg: [0, 0, 0, 0, -90, 0]
worktable:
  mesh: models/table.stl
  pose:
    xyz_mm: [400, 0, 0]
camera:
  type: virtual
  pose:
    xyz_mm: [400, 0, 850]
gripper:
  name: Gripper
  tcp_offset_xyz_mm: [0, 0, 130]
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        config = CellConfig.from_yaml(config_file)
        assert config.robot.name == "Test"

    def test_load_from_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            CellConfig.from_yaml("does_not_exist.yaml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
