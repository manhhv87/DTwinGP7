"""
anchored_config.py
──────────────────
Typed schema for the synthetic-data generator (config/synthgen.yaml).

Encodes the paper's factor table: which rendering factors are ANCHORED
(distributions centred on the twin's calibrated/measured values, widths
kappa*sigma) and which are FREE (drawn alike under both the anchored and the
blind treatment and in every E5 arm). Free are the task variables (product
poses, counts) and the factors that neither calibration nor measurement pins
down: the session exposure gain, the luminaire layout and the overhead share of
the light. The two recipes differ only in the measured factors: camera
extrinsics, light level and colour around the fitted profile, table and product
appearance, the blind_* ranges being wider than anchored at every kappa. The
blind baseline keeps the same camera reference, so it is not independent of
calibration. E5 ablations fix/remove one factor after the full anchored draw;
their intervention is recorded in scene manifests.

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
    "ExposureFactors",
    "FirstObjectMix",
    "LightingFactors",
    "MaterialFactors",
    "ObjectFactors",
    "SynthgenConfig",
]

# Factor on the profile's light level for the lighting label of a failure-mode
# condition (C3 loop), relative to lighting.anchor_condition, the regime the level
# was fitted in: a label renders at SCALE[label] / SCALE[anchor_condition]. Labels
# are the --lighting labels the trials log (03_run_experiment.py).
#
# Only "dim": the hard profile renders the condition that every E3-E5 trial runs in
# and logs as "dim", so the label maps to the profile itself, and any other label is
# refused rather than rendered at a guess. The former bright/medium/dim factors
# 1 / 0.48 / 0.08 were fitted in standard light through Filmic; applied to the hard
# profile, 0.08 would render the guided arm's upper sheet at a median luma of about 2.
LIGHTING_CONDITION_SCALE: dict[str, float] = {
    "dim": 1.0,
}


def _range(value: list[float], name: str, lo: float | None = None) -> None:
    if len(value) != 2 or not value[0] <= value[1] or (lo is not None and value[0] < lo):
        raise ValueError(f"{name} must be [low, high] with low <= high"
                         + ("" if lo is None else f" and low >= {lo}"))


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
    """Light of the hard condition: 600 mm area luminaires above the cell and a
    uniform, tinted overhead (hard_profile.luminaire_lights).

    The level is the irradiance at the centre of the card block, anchor_irradiance,
    fitted on the empty-table frames of the condition check in anchor_condition.
    MEASURED, so anchored around the profile and wide in the blind recipe: the level
    (anchored: times exp(eta), eta ~ N(0, (kappa*sigma_intensity_frac)^2); blind: a
    log-uniform multiple from blind_intensity_rel_range), each luminaire's colour
    temperature and the overhead's tint. FREE, drawn alike in both recipes and every
    E5 arm, because no capture fixes them: where the luminaires hang (one beyond the
    far edge of the table, a second anywhere with extra_luminaire_probability), how
    the luminaires share their light (Dirichlet) and the overhead share of the
    irradiance at the cards (log-uniform in overhead_share_range).
    """

    anchor_condition: str = "dim"
    anchor_irradiance: float
    reference_point_mm: list[float] = Field(default=[572.7, 20.0])
    anchor_color_temp_k: float = 5000.0
    sigma_intensity_frac: float = 0.20
    sigma_color_temp_k: float = 300.0
    luminaire_size_mm: list[float] = Field(default=[600.0, 600.0])
    luminaire_height_mm: list[float] = Field(default=[1700.0, 2000.0])
    far_luminaire_x_mm: list[float] = Field(default=[1100.0, 2800.0])
    far_luminaire_y_mm: list[float] = Field(default=[-500.0, 400.0])
    extra_luminaire_probability: float = 0.5
    extra_luminaire_x_mm: list[float] = Field(default=[0.0, 2800.0])
    extra_luminaire_y_mm: list[float] = Field(default=[-1500.0, 1500.0])
    overhead_share_range: list[float] = Field(default=[0.25, 0.95])
    overhead_tint_rgb: list[float] = Field(default=[0.886, 1.0, 0.852])
    sigma_overhead_tint: float = 0.04
    blind_intensity_rel_range: list[float] = Field(default=[0.111, 9.0])
    blind_color_temp_range_k: list[float] = Field(default=[2500.0, 9000.0])
    blind_overhead_tint_log: float = 0.5

    @model_validator(mode="after")
    def _check(self) -> "LightingFactors":
        if self.anchor_condition not in LIGHTING_CONDITION_SCALE:
            raise ValueError(f"lighting.anchor_condition must be one of "
                             f"{sorted(LIGHTING_CONDITION_SCALE)}")
        if self.anchor_irradiance <= 0.0:
            raise ValueError("lighting.anchor_irradiance must be positive")
        if len(self.reference_point_mm) != 2:
            raise ValueError("lighting.reference_point_mm must be [x, y] mm")
        if len(self.luminaire_size_mm) != 2 or min(self.luminaire_size_mm) <= 0.0:
            raise ValueError("lighting.luminaire_size_mm must be two positive sides")
        _range(self.luminaire_height_mm, "lighting.luminaire_height_mm", lo=1.0)
        for key in ("far_luminaire_x_mm", "far_luminaire_y_mm", "extra_luminaire_x_mm",
                    "extra_luminaire_y_mm", "blind_color_temp_range_k"):
            _range(getattr(self, key), f"lighting.{key}")
        if not 0.0 <= self.extra_luminaire_probability <= 1.0:
            raise ValueError("lighting.extra_luminaire_probability must lie in [0, 1]")
        lo, hi = self.overhead_share_range
        if len(self.overhead_share_range) != 2 or not 0.0 < lo <= hi <= 1.0:
            raise ValueError("lighting.overhead_share_range must be [low, high] in (0, 1]")
        if len(self.overhead_tint_rgb) != 3 or min(self.overhead_tint_rgb) <= 0.0:
            raise ValueError("lighting.overhead_tint_rgb must be three positive values")
        if len(self.blind_intensity_rel_range) != 2:
            raise ValueError("lighting.blind_intensity_rel_range must be [low, high]")
        lo, hi = self.blind_intensity_rel_range
        if not 0.0 < lo < 1.0 < hi:
            raise ValueError("lighting.blind_intensity_rel_range must bracket the anchor: "
                             "0 < low < 1 < high, in multiples of anchor_irradiance")
        return self


class BackgroundFactors(BaseModel):
    """Worktable / background appearance."""

    anchor_albedo_rgb: list[float] = Field(default=[0.50, 0.52, 0.55])
    sigma_albedo: float = 0.05
    anchor_roughness: float = 0.8
    blind_albedo_range: list[float] = Field(default=[0.05, 0.95])


class MaterialFactors(BaseModel):
    """Appearance of ONE product class.

    A dielectric class (carton, wood, plastic) has albedo_rgb and sigma_albedo_rel,
    MEASURED, not invented: the median colour of that class's human-labelled pixels
    over the standard-condition images, and the spread of that median across images.
    A metal class (metallic 1) has instead f0_rgb, its PHYSICAL normal reflectance
    (linear), which the renderer multiplies by render.reflectance_scale like every
    specular reflectance; sigma_f0_rel is its relative spread at kappa 1,
    roughness_range the roughness spread at kappa 1, and coat_ior/coat_roughness a
    clear coat over the metal (the steel box: rough metal under a glossy coat).
    roughness and metallic are physical ESTIMATES or fits; they set how the surface
    reflects, which is what makes the mirror-finish class lose depth on the real
    sensor.

    sigma_albedo_rel is RELATIVE (a fraction of the colour), because the measured
    spread is proportional to each channel: the carton's per-channel spread is
    9/101, 8/81, 6/56, all near a tenth. That is brightness varying between images,
    not hue, so the sampler scales all three channels together. Perturbing the
    channels independently at that width would turn a grey steel box green.
    The sampler applies it as a log scale, exp(eta) with eta ~ N(0, (kappa*s)^2):
    the old factor max(0, 1 + eta) rendered 18% of inox boxes black at kappa 2.
    """

    albedo_rgb: list[float] | None = None
    sigma_albedo_rel: float = 0.10
    roughness: float = 0.6
    metallic: float = 0.0
    f0_rgb: list[float] | None = None
    sigma_f0_rel: float = 0.0
    roughness_range: list[float] | None = None
    coat_ior: float | None = None
    coat_roughness: float = 0.30
    sigma_coat_roughness: float = 0.0

    @model_validator(mode="after")
    def _check(self) -> "MaterialFactors":
        if (self.albedo_rgb is None) == (self.f0_rgb is None):
            raise ValueError("a material has either albedo_rgb (dielectric) or f0_rgb (metal)")
        colour = self.albedo_rgb if self.f0_rgb is None else self.f0_rgb
        if len(colour) != 3:
            raise ValueError("albedo_rgb / f0_rgb must be [r, g, b]")
        if not all(0.0 <= c <= 1.0 for c in colour):
            raise ValueError("albedo_rgb / f0_rgb components must lie in [0, 1]")
        if self.f0_rgb is not None and self.metallic != 1.0:
            raise ValueError("f0_rgb is the reflectance of a metal: metallic must be 1")
        if self.sigma_albedo_rel < 0.0 or self.sigma_f0_rel < 0.0 or self.sigma_coat_roughness < 0.0:
            raise ValueError("spreads cannot be negative")
        if not 0.0 <= self.roughness <= 1.0 or not 0.0 <= self.metallic <= 1.0:
            raise ValueError("roughness and metallic must lie in [0, 1]")
        if self.roughness_range is not None:
            _range(self.roughness_range, "roughness_range")
            if not self.roughness_range[0] <= self.roughness <= self.roughness_range[1]:
                raise ValueError("roughness_range must contain roughness")
        if self.coat_ior is not None and self.coat_ior <= 1.0:
            raise ValueError("coat_ior must exceed 1")
        if not 0.0 <= self.coat_roughness <= 1.0:
            raise ValueError("coat_roughness must lie in [0, 1]")
        return self


class FirstObjectMix(BaseModel):
    """How often one class is the FIRST product of a scene, alone or stacked.

    Counts, not probabilities: they are rows of the physical pose lists, so the
    configuration can be checked against those files. single counts the rows with
    no support; stacked_on counts the rows per support class (the support stands
    on the table, this class rests on top of it).
    """

    single: int = 0
    stacked_on: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "FirstObjectMix":
        if self.single < 0 or any(v < 0 for v in self.stacked_on.values()):
            raise ValueError("first_object_mix counts cannot be negative")
        return self


class ObjectFactors(BaseModel):
    """Task variables — fully randomized under BOTH treatments.

    count_range counts PLACEMENTS on the table: a single product, or a support
    with a product resting on it. Product footprints never overlap; the
    footprint is the mesh's length x width (mesh_sizes_mm), turned by its yaw.
    """

    # Mot lop co the co nhieu vat that khac co (carton: hai co) -> danh sach mesh.
    classes: dict[str, str | list[str]]
    class_ids: dict[str, int]
    count_range: list[int] = Field(default=[1, 2])
    region_x_mm: list[float] = Field(default=[400.0, 1000.0])
    region_y_mm: list[float] = Field(default=[-200.0, 200.0])
    table_z_mm: float = 500.0
    yaw_range_deg: list[float] = Field(default=[-90.0, 90.0])
    # [length, width, height] mm of every mesh named in classes: length along the
    # mesh X axis, width along Y, the base at z = 0 (scripts/gen_product_meshes.py).
    mesh_sizes_mm: dict[str, list[float]] = Field(default_factory=dict)
    # Class of the first product and its support, from the physical pose lists.
    # Empty: uniform class, never stacked.
    first_object_mix: dict[str, FirstObjectMix] = Field(default_factory=dict)
    # A stacked product's centre lies within this distance of its support's centre,
    # per axis.
    stack_offset_mm: float = 15.0
    # Appearance per class. A class with no entry renders in neutral grey, which
    # teaches the detector nothing about that product's colour, so the generator
    # script warns about any class missing here.
    materials: dict[str, MaterialFactors] = Field(default_factory=dict)
    sigma_roughness: float = 0.08
    blind_albedo_range: list[float] = Field(default=[0.0, 0.95])
    blind_roughness_range: list[float] = Field(default=[0.02, 1.0])
    # Blind: share of products drawn with a clear coat (index 1.5, roughness from
    # blind_roughness_range), so the wide range holds the coated steel box too.
    blind_coat_probability: float = 0.5

    def meshes_of(self, cls: str) -> list[str]:
        meshes = self.classes[cls]
        return list(meshes) if isinstance(meshes, list) else [meshes]

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
        if len(self.count_range) != 2 or not 1 <= self.count_range[0] <= self.count_range[1]:
            raise ValueError("objects.count_range must be [min, max] with 1 <= min <= max")
        meshes = {m for cls in self.classes for m in self.meshes_of(cls)}
        no_size = sorted(meshes - set(self.mesh_sizes_mm))
        if no_size:
            raise ValueError(f"objects.mesh_sizes_mm has no size for: {no_size}")
        for mesh, size in self.mesh_sizes_mm.items():
            if len(size) != 3 or min(size) <= 0.0:
                raise ValueError(f"objects.mesh_sizes_mm[{mesh}] must be [length, width, "
                                 f"height] mm, all positive")
        for cls, mix in self.first_object_mix.items():
            names = {cls, *mix.stacked_on}
            if names - set(self.classes):
                raise ValueError(f"objects.first_object_mix names unknown classes: "
                                 f"{sorted(names - set(self.classes))}")
            if cls in mix.stacked_on:
                raise ValueError(f"objects.first_object_mix stacks {cls} on itself")
        if self.first_object_mix and not sum(
                m.single + sum(m.stacked_on.values()) for m in self.first_object_mix.values()):
            raise ValueError("objects.first_object_mix holds no rows")
        if self.stack_offset_mm < 0.0:
            raise ValueError("objects.stack_offset_mm cannot be negative")
        # The top's centre must stay over its support: the offset reaches
        # stack_offset_mm * sqrt(2) from the support's centre in the worst direction.
        supports = {s for m in self.first_object_mix.values() for s, k in m.stacked_on.items() if k}
        for sup in sorted(supports):
            half = min(min(self.mesh_sizes_mm[m][:2]) for m in self.meshes_of(sup)) / 2.0
            if self.stack_offset_mm * 2 ** 0.5 >= half:
                raise ValueError(f"objects.stack_offset_mm lets a product overhang the centre "
                                 f"of its {sup} support (half width {half:g} mm)")
        return self


class DistractorFactors(BaseModel):
    """Nuisance objects that are not workpieces (never labelled)."""

    count_range: list[int] = Field(default=[0, 3])
    blind_count_range: list[int] = Field(default=[0, 5])
    size_range_mm: list[float] = Field(default=[20.0, 80.0])


class ExposureFactors(BaseModel):
    """Session exposure: a sensor gain per scene, log-uniform in gain_range, times
    the gain at which the profile was fitted (sensor.gain). FREE factor: drawn the
    same way in both recipes, at every kappa and in every E5 arm.

    gain_range is where both table parts of the profile meet the condition check's
    luminance bounds, check_luminance (hard_profile.admitted_gain_range, from the
    profile colours reference_upper_rgb/reference_lower_rgb at gain 1)."""

    gain_range: list[float]
    check_luminance: list[float] = Field(default=[10.0, 45.0])
    reference_upper_rgb: list[float]
    reference_lower_rgb: list[float]

    @model_validator(mode="after")
    def _check(self) -> "ExposureFactors":
        if len(self.gain_range) != 2 or not 0.0 < self.gain_range[0] < self.gain_range[1]:
            raise ValueError("exposure.gain_range must be [lo, hi] with 0 < lo < hi")
        return self


class SynthgenConfig(BaseModel):
    """Root schema of config/synthgen.yaml."""

    camera: CameraFactors
    lighting: LightingFactors
    background: BackgroundFactors
    objects: ObjectFactors
    distractors: DistractorFactors
    exposure: ExposureFactors

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SynthgenConfig":
        """Load + validate a synthgen YAML config."""
        import yaml  # lazy import

        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
