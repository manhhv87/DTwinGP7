"""
anchored_config.py
──────────────────
Typed schema for the synthetic-data generator (config/synthgen.yaml).

Encodes the paper's factor table: which rendering factors are ANCHORED
(distributions centred on the twin's calibrated/measured values, widths
kappa*sigma) and which are FREE task variables (fully randomized under both
the anchored and the blind treatment). The blind baseline uses the blind_*
ranges and NO calibration information (Tobin-2017 / Zhu-2025 style).

Pure pydantic + yaml — no heavy imports; testable everywhere.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

# Multiplier applied to the anchored light intensity when a failure-mode
# condition requests a specific lighting regime (C3 loop). Labels must match
# the --lighting labels used when running trials (03_run_experiment.py).
LIGHTING_CONDITION_SCALE: dict[str, float] = {
    "bright": 1.0,
    "medium": 0.6,
    "dim": 0.3,
}


class CameraFactors(BaseModel):
    """Camera anchor + sampling widths. sigma_* are FALLBACKS — the bootstrap
    file (T_base_camera_sigma.json) takes precedence when present."""

    intrinsics: dict[str, float]
    sigma_trans_mm: list[float] = Field(default=[2.0, 2.0, 2.0])
    sigma_rot_deg: list[float] = Field(default=[0.5, 0.5, 0.5])
    sigma_intrinsics_px: float = 2.0
    blind_trans_mm: float = 100.0
    blind_rot_deg: float = 10.0
    blind_intrinsics_px: float = 30.0

    @model_validator(mode="after")
    def _check(self) -> "CameraFactors":
        for key in ("fx", "fy", "ppx", "ppy", "width", "height"):
            if key not in self.intrinsics:
                raise ValueError(f"camera.intrinsics missing key '{key}'")
        if len(self.sigma_trans_mm) != 3 or len(self.sigma_rot_deg) != 3:
            raise ValueError("sigma_trans_mm / sigma_rot_deg must have 3 components")
        return self


class LightingFactors(BaseModel):
    """Light sources: anchored around the measured cell lighting proxy."""

    anchor_positions_mm: list[list[float]]
    anchor_intensity_w: float = 60.0
    anchor_color_temp_k: float = 5000.0
    sigma_position_mm: float = 150.0
    sigma_intensity_frac: float = 0.20
    sigma_color_temp_k: float = 300.0
    blind_count_range: list[int] = Field(default=[1, 4])
    blind_position_box_mm: list[list[float]] = Field(
        default=[[200.0, 1200.0], [-600.0, 600.0], [800.0, 2000.0]]
    )
    blind_intensity_range_w: list[float] = Field(default=[10.0, 200.0])
    blind_color_temp_range_k: list[float] = Field(default=[2500.0, 9000.0])

    @model_validator(mode="after")
    def _check(self) -> "LightingFactors":
        if not self.anchor_positions_mm:
            raise ValueError("lighting.anchor_positions_mm must not be empty")
        for p in self.anchor_positions_mm:
            if len(p) != 3:
                raise ValueError("each anchor light position must be [x, y, z] mm")
        return self


class BackgroundFactors(BaseModel):
    """Worktable / background appearance."""

    anchor_albedo_rgb: list[float] = Field(default=[0.50, 0.52, 0.55])
    sigma_albedo: float = 0.05
    anchor_roughness: float = 0.8
    blind_albedo_range: list[float] = Field(default=[0.05, 0.95])


class ObjectFactors(BaseModel):
    """Task variables — fully randomized under BOTH treatments."""

    classes: dict[str, str]
    class_ids: dict[str, int]
    count_range: list[int] = Field(default=[1, 3])
    region_x_mm: list[float] = Field(default=[400.0, 1000.0])
    region_y_mm: list[float] = Field(default=[-200.0, 200.0])
    table_z_mm: float = 500.0
    yaw_range_deg: list[float] = Field(default=[-90.0, 90.0])
    min_separation_mm: float = 120.0

    @model_validator(mode="after")
    def _check(self) -> "ObjectFactors":
        missing = set(self.classes) ^ set(self.class_ids)
        if missing:
            raise ValueError(f"objects.classes / class_ids mismatch on: {sorted(missing)}")
        if self.region_x_mm[0] >= self.region_x_mm[1]:
            raise ValueError("region_x_mm must be [min, max]")
        if self.region_y_mm[0] >= self.region_y_mm[1]:
            raise ValueError("region_y_mm must be [min, max]")
        return self


class DistractorFactors(BaseModel):
    """Nuisance objects that are not workpieces (never labelled)."""

    count_range: list[int] = Field(default=[0, 3])
    blind_count_range: list[int] = Field(default=[0, 5])
    size_range_mm: list[float] = Field(default=[20.0, 80.0])


class SynthgenConfig(BaseModel):
    """Root schema of config/synthgen.yaml."""

    camera: CameraFactors
    lighting: LightingFactors
    background: BackgroundFactors
    objects: ObjectFactors
    distractors: DistractorFactors

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SynthgenConfig":
        """Load + validate a synthgen YAML config."""
        import yaml  # lazy import

        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
