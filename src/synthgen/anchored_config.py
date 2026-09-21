"""
anchored_config.py
──────────────────
Typed schema for the synthetic-data generator (config/synthgen.yaml).

Encodes the paper's factor table: which rendering factors are ANCHORED
(distributions centred on the twin's calibrated/measured values, widths
kappa*sigma) and which are FREE task variables (fully randomized under both
the anchored and the blind treatment). The blind baseline uses the blind_*
ranges around the same camera reference, with a different appearance recipe;
it is not independent of calibration. E5 ablations fix/remove one factor after
the full anchored draw; their intervention is recorded in scene manifests.

Pure pydantic + yaml — no heavy imports; testable everywhere.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "LIGHTING_CONDITION_SCALE",
    "BackgroundFactors",
    "CameraFactors",
    "DistractorFactors",
    "LightingFactors",
    "MaterialFactors",
    "ObjectFactors",
    "SynthgenConfig",
]

# Multiplier applied to the anchored light intensity when a failure-mode
# condition requests a specific lighting regime (C3 loop). Labels must match
# the --lighting labels used when running trials (03_run_experiment.py).
#
# CALIBRATED 12/09/2026, not guessed. The dataset audit measured the mean image
# brightness of the three real regimes as 1.00 / 0.83 / 0.46 of the bright one
# over 3524 photographs. Rendering one scene at a ladder of light energies and
# matching the same statistic gives the energy factors below: the renderer's
# response is heavily compressive (roughly the fourth root of the energy, because
# Blender 4.2 tone-maps with AgX), so a factor is nowhere near the brightness
# ratio it produces. The previous values, 0.6 and 0.3, rendered a "dim" scene
# about four times brighter than the real dim condition.
LIGHTING_CONDITION_SCALE: dict[str, float] = {
    "bright": 1.0,
    "medium": 0.48,
    "dim": 0.08,
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


class MaterialFactors(BaseModel):
    """Appearance of ONE product class.

    albedo_rgb and sigma_albedo_rel are MEASURED, not invented: the median colour of
    that class's human-labelled pixels over the standard-condition images, and the
    spread of that median across images. roughness and metallic are physical
    ESTIMATES — they set how the surface reflects, which is what makes the
    mirror-finish class lose depth on the real sensor.

    sigma_albedo_rel is RELATIVE (a fraction of the colour), because the measured
    spread is proportional to each channel: the carton's per-channel spread is
    9/101, 8/81, 6/56, all near a tenth. That is brightness varying between images,
    not hue, so the sampler scales all three channels together. Perturbing the
    channels independently at that width would turn a grey steel box green.
    """

    albedo_rgb: list[float]
    sigma_albedo_rel: float = 0.10
    roughness: float = 0.6
    metallic: float = 0.0

    @model_validator(mode="after")
    def _check(self) -> "MaterialFactors":
        if len(self.albedo_rgb) != 3:
            raise ValueError("albedo_rgb must be [r, g, b]")
        if not all(0.0 <= c <= 1.0 for c in self.albedo_rgb):
            raise ValueError("albedo_rgb components must lie in [0, 1]")
        if self.sigma_albedo_rel < 0.0:
            raise ValueError("sigma_albedo_rel is a fraction and cannot be negative")
        if not 0.0 <= self.roughness <= 1.0 or not 0.0 <= self.metallic <= 1.0:
            raise ValueError("roughness and metallic must lie in [0, 1]")
        return self


class ObjectFactors(BaseModel):
    """Task variables — fully randomized under BOTH treatments."""

    # Mot lop co the co nhieu vat that khac co (carton: hai co) -> danh sach mesh.
    classes: dict[str, str | list[str]]
    class_ids: dict[str, int]
    count_range: list[int] = Field(default=[1, 3])
    region_x_mm: list[float] = Field(default=[400.0, 1000.0])
    region_y_mm: list[float] = Field(default=[-200.0, 200.0])
    table_z_mm: float = 500.0
    yaw_range_deg: list[float] = Field(default=[-90.0, 90.0])
    min_separation_mm: float = 120.0
    # Appearance per class. A class with no entry renders in neutral grey, which
    # teaches the detector nothing about that product's colour, so the generator
    # script warns about any class missing here.
    materials: dict[str, MaterialFactors] = Field(default_factory=dict)
    sigma_roughness: float = 0.08
    blind_albedo_range: list[float] = Field(default=[0.05, 0.95])
    blind_roughness_range: list[float] = Field(default=[0.05, 1.0])

    @model_validator(mode="after")
    def _check(self) -> "ObjectFactors":
        missing = set(self.classes) ^ set(self.class_ids)
        if missing:
            raise ValueError(f"objects.classes / class_ids mismatch on: {sorted(missing)}")
        unknown = set(self.materials) - set(self.classes)
        if unknown:
            raise ValueError(f"objects.materials names unknown classes: {sorted(unknown)}")
        empty = sorted(k for k, v in self.classes.items() if isinstance(v, list) and not v)
        if empty:
            raise ValueError(f"objects.classes has an empty mesh list for: {empty}")
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
