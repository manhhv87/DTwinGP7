"""
scene_sampler.py
────────────────
Sample renderer scene specifications for calibration-anchored (or blind)
domain randomization — the implementation of Eq. (1) of the paper:

    theta_i ~ D_i(theta_i*, kappa * sigma_i)

ANCHORED mode: the measured nuisance parameters (camera extrinsics/intrinsics,
light level and colour, table and product appearance) are drawn around the
twin's calibrated or fitted state theta* with widths kappa*sigma; the camera's
sigma comes from the bootstrap of the hand-eye calibration
(config/calibration/T_base_camera_sigma.json) or, as a fallback, from
config/synthgen.yaml. Quantities that must stay positive (light level, albedo,
metal reflectance) are perturbed as theta* * exp(eta), so no kappa drives them
to zero or onto a floor value.

BLIND mode: wide hand-set ranges for the same measured factors, around the same
camera reference, with a different appearance recipe. It is not independent of
calibration. Its light level is a log-uniform multiple of the anchored level.

FREE factors are drawn alike in both modes and in every E5 arm: the task
variables (one or more placements on the table whose footprints never overlap;
the first product's class, and whether it rests on a support, follow the
physical pose lists; distractors keep off the products) and the factors no
capture pins down, which are drawn over physical or admitted bounds: where the
luminaires hang and how they share the light, the overhead share of the light,
and the session exposure gain.

The sampler runs in the repo venv and writes plain-JSON scene specs; rendering
happens separately under `blenderproc run` (render_blenderproc.py), which only
consumes these JSON files. This split keeps the sampling logic unit-testable
without Blender.
"""
from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from . import top_texture
from .anchored_config import LIGHTING_CONDITION_SCALE, SynthgenConfig
from .hard_profile import luminaire_lights
from .seed_ranges import WORD

# Repo root: mesh and texture paths in the config are relative to it.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Schema version of the scene-spec JSON contract with render_blenderproc.py.
# 2: an object's xyz_mm z is the height of its base, which is the top face of its
# support for a stacked product (stack_on_index names the support); scene i draws
# from SeedSequence([seed, i]), where version 1 drew from default_rng(seed + i).
# 3: the lights are area luminaires plus a uniform overhead (spec "environment"),
# a metal carries its physical f0_rgb, and every scene has a session exposure gain
# (spec "exposure"), drawn from SeedSequence([seed, i, 2]); the renderer refuses a
# spec without one.
# 4: an anchored product of a mesh with measured top-face textures carries one of them
# (material "top_texture": file, SHA-256, scale), drawn from SeedSequence([seed, i, 3]).
# A change of the RNG scheme needs a bump and its rule in seed_ranges.STREAM_RULE.
SPEC_VERSION = 4
ABLATION_FACTORS = ("none", "illumination", "background", "distractors", "camera", "pose")
# Draws of position and yaw for one product or distractor before it is left out.
PLACEMENT_TRIES = 200
# Third SeedSequence word of the stream that moves distractors in the pose arm, so
# that the full scene's own draws stay untouched.
_RELOCATION_STREAM = 1
# Third word of the session exposure's stream: the gain depends on (seed, index)
# only, so the kappa arms, the E5 arms and both recipes of a seed share it.
_EXPOSURE_STREAM = 2
# Third word of the stream of the anchored products' top-face textures: one draw per
# product in spec order, textured or not, so a product's texture depends on (seed,
# index, its place in the list) only, and every kappa and E5 arm of a seed shares it.
_TEXTURE_STREAM = 3
# Index of refraction of the clear coat the blind recipe puts on some products.
BLIND_COAT_IOR = 1.5


def lighting_scale(label: str | None, anchor_condition: str) -> float:
    """Factor on the profile's light level for a condition's lighting label.

    The level was fitted in anchor_condition; a label renders at
    SCALE[label] / SCALE[anchor_condition], None at the anchor's own level. A label
    the table does not hold raises: it used to render at 1.0 unnoticed.
    """
    for name in sorted({label, anchor_condition} - {None}):
        if name not in LIGHTING_CONDITION_SCALE:
            raise ValueError(f"lighting '{name}' is not rendered by the hard-condition profile "
                             f"({sorted(LIGHTING_CONDITION_SCALE)}): mine only hard-condition "
                             "sessions")
    if label is None:
        return 1.0
    return LIGHTING_CONDITION_SCALE[label] / LIGHTING_CONDITION_SCALE[anchor_condition]


def _rotvec_deg_to_matrix(rv_deg: np.ndarray) -> np.ndarray:
    """Rotation vector (degrees, axis*angle) → 3x3 rotation matrix (Rodrigues)."""
    rv = np.deg2rad(np.asarray(rv_deg, dtype=float))
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.eye(3)
    axis = rv / angle
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def footprint(center_xy: Any, length: float, width: float, yaw_deg: float) -> np.ndarray:
    """Corners (4x2, mm) of a length x width rectangle turned by yaw about +Z.

    The length runs along the mesh X axis, which is how the renderer turns a mesh
    by the spec's yaw.
    """
    c, s = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    half = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]) * [
        length / 2.0, width / 2.0]
    return half @ np.array([[c, s], [-s, c]]) + np.asarray(center_xy, dtype=float)


def footprints_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex quadrilaterals.

    True when they share area; rectangles that only touch count as apart.
    """
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        for normal in np.column_stack([-edges[:, 1], edges[:, 0]]):
            pa, pb = a @ normal, b @ normal
            if pa.max() <= pb.min() or pb.max() <= pa.min():
                return False
    return True


def load_sigma_json(path: str | Path) -> dict[str, Any] | None:
    """Load the bootstrap sigma file written by calibration.uncertainty.

    Returns None if the file does not exist (caller falls back to YAML sigmas).
    """
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


class SceneSampler:
    """Draw scene specs under the anchored or blind treatment.

    Args:
        config: Validated synthgen config (factor table).
        T_BC_mm: Calibrated camera pose in base frame, 4x4, mm (eye-to-hand).
        mode: "anchored" | "blind".
        kappa: Width multiplier for anchored sampling (Eq. 1). Ignored in blind.
        sigma: Optional dict from T_base_camera_sigma.json with keys
            sigma_trans_mm [3] and sigma_rot_deg [3]; overrides YAML fallbacks.
        seed: Run seed, an integer in [0, 2**32). Scene i draws from
            default_rng(SeedSequence([seed, i])): any subset of a run regenerates
            exactly, and runs with different seeds share no scenes, whatever their
            index ranges. Within one seed, an index range must not be reused.
        ablation: One E5 factor to fix/remove, or "none" for the full recipe.
            Available only for unconditional anchored sampling. All factors are
            drawn before applying the intervention, preserving paired RNG draws.
    """

    def __init__(
        self,
        config: SynthgenConfig,
        T_BC_mm: np.ndarray,
        mode: str = "anchored",
        kappa: float = 2.0,
        sigma: dict[str, Any] | None = None,
        seed: int = 0,
        ablation: str = "none",
    ) -> None:
        if mode not in ("anchored", "blind"):
            raise ValueError(f"mode must be 'anchored' or 'blind', got '{mode}'")
        if ablation not in ABLATION_FACTORS:
            raise ValueError(f"ablation must be one of {ABLATION_FACTORS}, got '{ablation}'")
        if ablation != "none" and mode != "anchored":
            raise ValueError("E5 ablation requires mode='anchored'")
        if not 0 <= int(seed) < WORD:
            # numpy splits a larger seed into two words: SeedSequence([2**32 + 5, 0])
            # is SeedSequence([5, 1]), so its scenes could be another seed's.
            raise ValueError(f"seed must be an integer in [0, 2**32), got {seed}")
        T = np.asarray(T_BC_mm, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"T_BC_mm must be 4x4, got {T.shape}")
        self.config = config
        self.T_BC = T
        self.mode = mode
        self.kappa = float(kappa)
        self.seed = int(seed)
        self.ablation = ablation

        cam = config.camera
        if sigma is not None:
            self.sigma_trans = np.asarray(sigma["sigma_trans_mm"], dtype=float)
            self.sigma_rot = np.asarray(sigma["sigma_rot_deg"], dtype=float)
            self.sigma_source = str(sigma.get("source", "bootstrap"))
        else:
            self.sigma_trans = np.asarray(cam.sigma_trans_mm, dtype=float)
            self.sigma_rot = np.asarray(cam.sigma_rot_deg, dtype=float)
            self.sigma_source = "yaml_fallback"

        # SHA-256 of every product mesh file: the renderer imports these STLs, and their
        # shape (the steel box's rounded corners) is not in the config, so the generator
        # profile, hence generator_sha256, carries them. A missing file stays None here and
        # fails at render.
        self.mesh_sha256: dict[str, str | None] = {}
        for mesh in sorted({m for cls in config.objects.classes for m in config.objects.meshes_of(cls)}):
            path = REPO_ROOT / mesh
            self.mesh_sha256[mesh] = (hashlib.sha256(path.read_bytes()).hexdigest()
                                      if path.is_file() else None)

        # Measured top-face textures per mesh (anchored only); every file is checked
        # against its SHA-256 here, and the list enters the generator profile.
        self.top_textures: dict[str, list[dict[str, Any]]] = {}
        if config.objects.top_textures:
            self.top_textures = top_texture.load_manifest(REPO_ROOT, config.objects.top_textures)
            meshes = {m for cls in config.objects.classes for m in config.objects.meshes_of(cls)}
            for mesh in self.top_textures:
                if mesh not in meshes:
                    raise ValueError(f"{config.objects.top_textures}: textures for {mesh}, "
                                     "which no class uses")
                cls = next(c for c in config.objects.classes if mesh in config.objects.meshes_of(c))
                mat = config.objects.materials.get(cls)
                if mat is not None and mat.albedo_rgb is None:
                    # (a class without a material renders grey, untextured)
                    raise ValueError(f"{config.objects.top_textures}: textures for {mesh}, but "
                                     f"{cls} is a metal, whose f0_rgb they cannot multiply")

        self.ablation_protocol = {
            "protocol_version": 3,
            "factor": ablation,
            "reference": self._ablation_reference(),
        }

    def _ablation_reference(self) -> dict[str, Any] | list[Any] | None:
        """Resolve fixed settings once, from the supplied configuration only."""
        if self.ablation == "none":
            return None
        if self.ablation == "camera":
            return {"T_BC_mm": self.T_BC.tolist(),
                    "intrinsics": dict(self.config.camera.intrinsics)}
        if self.ablation == "illumination":
            # The measured light factors at the profile; the free ones (luminaire
            # layout, overhead share, exposure) stay as drawn.
            lit = self.config.lighting
            return {"rule": "profile_level_and_colour",
                    "irradiance": float(lit.anchor_irradiance),
                    "color_temp_k": float(lit.anchor_color_temp_k),
                    "overhead_tint_rgb": [float(v) for v in lit.overhead_tint_rgb],
                    "kept": ["luminaire_layout", "overhead_share", "exposure"]}
        if self.ablation == "background":
            bg = self.config.background
            return {"kind": "plane", "z_mm": self.config.objects.table_z_mm,
                    "albedo_rgb": list(bg.anchor_albedo_rgb),
                    "roughness": float(bg.anchor_roughness)}
        if self.ablation == "distractors":
            return []

        # Pose is fixed conditional on the number of placements; class, size,
        # material and stacking still vary. Slots are cell centres along the longer
        # region axis (X breaks a tie); a stacked product sits square on the centre
        # of its support's slot.
        obj = self.config.objects
        ranges = np.asarray([obj.region_x_mm, obj.region_y_mm], dtype=float)
        widths = ranges[:, 1] - ranges[:, 0]
        axis = int(np.argmax(widths))
        centre = ranges.mean(axis=1)
        counts = obj.count_range
        yaw = float(np.mean(obj.yaw_range_deg))
        # Every mesh, at the fixed yaw, must fit its slot, or the reference itself
        # would put products inside one another.
        reach = max(float(np.ptp(footprint((0.0, 0.0), *size[:2], yaw)[:, axis]))
                    for size in obj.mesh_sizes_mm.values())
        if counts[1] > 1 and widths[axis] / counts[1] < reach:
            raise ValueError(
                f"pose reference slots overlap: {counts[1]} slots of "
                f"{widths[axis] / counts[1]:.1f} mm cannot hold a {reach:.1f} mm footprint; "
                "revise and document the reference region/count protocol")
        poses_by_count = {}
        for n in range(counts[0], counts[1] + 1):
            poses = []
            for i in range(n):
                xy = centre.copy()
                xy[axis] = ranges[axis, 0] + (i + 0.5) * widths[axis] / n
                poses.append({"xyz_mm": [float(xy[0]), float(xy[1]), obj.table_z_mm],
                              "yaw_deg": yaw})
            poses_by_count[str(n)] = poses
        return {"rule": "count_conditional_long_axis_cell_centres",
                "axis": "xy"[axis], "poses_by_count": poses_by_count,
                "stacked": "support_slot_centre",
                "distractors": "moved_off_products"}

    def _generator_profile(self) -> dict[str, Any]:
        """Persist remaining distributions and effective camera widths for audit."""
        return {"config": self.config.model_dump(mode="json"),
                "T_BC_mm": self.T_BC.tolist(),
                "sigma_trans_mm": self.sigma_trans.tolist(),
                "sigma_rot_deg": self.sigma_rot.tolist(),
                "top_textures": self.top_textures,
                "mesh_sha256": self.mesh_sha256}

    def _check_condition(self, condition: dict[str, Any] | None) -> None:
        if self.ablation != "none" and condition:
            raise ValueError("E5 ablation cannot be combined with failure conditioning")
        if condition:
            lighting_scale(condition.get("lighting"), self.config.lighting.anchor_condition)

    def _apply_ablation(self, spec: dict[str, Any], index: int) -> None:
        reference = deepcopy(self.ablation_protocol["reference"])
        if self.ablation == "pose":
            objects = spec["objects"]
            on_table = [o for o in objects if "stack_on_index" not in o]
            for obj, pose in zip(on_table, reference["poses_by_count"][str(len(on_table))]):
                obj.update(pose)
            for obj in objects:
                if "stack_on_index" in obj:
                    support = objects[obj["stack_on_index"]]
                    x, y, z = support["xyz_mm"]
                    obj["xyz_mm"] = [x, y, z + self._size(support["mesh"])[2]]
                    obj["yaw_deg"] = support["yaw_deg"]
            self._relocate_distractors(spec, index)
        elif self.ablation == "illumination":
            lit = self.config.lighting
            spec["lights"], spec["environment"] = self._light(
                [li["position_mm"] for li in spec["lights"]],
                [li["irradiance_share"] for li in spec["lights"]],
                spec["environment"]["overhead_share"], lit.anchor_irradiance,
                [lit.anchor_color_temp_k] * len(spec["lights"]), lit.overhead_tint_rgb)
        elif self.ablation != "none":
            spec[self.ablation] = reference
        spec["ablation"] = deepcopy(self.ablation_protocol)

    def _relocate_distractors(self, spec: dict[str, Any], index: int) -> None:
        """Pose arm: move each distractor that the fixed poses would cut through.

        The full scene placed every distractor clear of the products where they
        were; at the slot poses some would sit inside a product. Only those move,
        keeping their shape, size, yaw and colour, to a free position drawn from a
        stream of their own; one that finds no free position is left out.
        """
        products = [self._footprint(o) for o in spec["objects"]]
        squares = [footprint(d["xyz_mm"][:2], d["size_mm"], d["size_mm"], d["yaw_deg"])
                   for d in spec["distractors"]]
        blocked = [any(footprints_overlap(sq, p) for p in products) for sq in squares]
        if not any(blocked):
            return
        taken = products + [sq for sq, hit in zip(squares, blocked) if not hit]
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, int(index), _RELOCATION_STREAM]))
        kept = []
        for d, hit in zip(spec["distractors"], blocked):
            if hit:
                pose = self._place(rng, d["size_mm"], d["size_mm"], self._region(), taken,
                                   yaw_deg=d["yaw_deg"])
                if pose is None:
                    continue
                d["xyz_mm"] = [pose[0], pose[1], d["xyz_mm"][2]]
            kept.append(d)
        spec["distractors"] = kept

    # ────────────────────────────────────────────────────────────
    # Per-factor sampling
    # ────────────────────────────────────────────────────────────

    def _sample_camera(self, rng: np.random.Generator) -> dict[str, Any]:
        cam = self.config.camera
        T = self.T_BC.copy()
        if self.mode == "anchored":
            d_t = rng.normal(0.0, self.kappa * self.sigma_trans, size=3)
            d_r = rng.normal(0.0, self.kappa * self.sigma_rot, size=3)
            d_px = rng.normal(0.0, self.kappa * cam.sigma_intrinsics_px, size=4)
        else:
            d_t = rng.uniform(-cam.blind_trans_mm, cam.blind_trans_mm, size=3)
            d_r = rng.uniform(-cam.blind_rot_deg, cam.blind_rot_deg, size=3)
            d_px = rng.uniform(-cam.blind_intrinsics_px, cam.blind_intrinsics_px, size=4)

        T[:3, 3] += d_t
        T[:3, :3] = T[:3, :3] @ _rotvec_deg_to_matrix(d_r)

        intr = dict(cam.intrinsics)
        for key, dv in zip(("fx", "fy", "ppx", "ppy"), d_px):
            intr[key] = float(intr[key] + dv)
        return {"T_BC_mm": T.tolist(), "intrinsics": intr}

    def _sample_lights(
        self, rng: np.random.Generator, lighting_condition: str | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Luminaires and overhead of one scene: the spec's lights and environment.

        The free factors come first and are drawn alike in both recipes: one
        luminaire beyond the far edge of the table, a second anywhere with
        extra_luminaire_probability, their shares of the luminaire light, and the
        overhead share. Then the measured ones: the level (the irradiance at the
        cards), each luminaire's colour temperature and the overhead's tint, around
        the profile (anchored) or over the wide ranges (blind).
        """
        lit = self.config.lighting
        z_table = self.config.objects.table_z_mm
        scale = lighting_scale(lighting_condition, lit.anchor_condition)

        def centre(x_range: list[float], y_range: list[float]) -> list[float]:
            x, y = float(rng.uniform(*x_range)), float(rng.uniform(*y_range))
            return [x, y, z_table + float(rng.uniform(*lit.luminaire_height_mm))]

        centres = [centre(lit.far_luminaire_x_mm, lit.far_luminaire_y_mm)]
        if rng.random() < lit.extra_luminaire_probability:
            centres.append(centre(lit.extra_luminaire_x_mm, lit.extra_luminaire_y_mm))
        weights = rng.dirichlet(np.ones(len(centres)))
        log_lo, log_hi = np.log(lit.overhead_share_range)
        share = float(np.exp(rng.uniform(log_lo, log_hi)))
        if self.mode == "anchored":
            # The level is the fitted one times exp(eta), never at a floor.
            level = np.exp(rng.normal(0.0, self.kappa * lit.sigma_intensity_frac))
            temps = lit.anchor_color_temp_k + rng.normal(
                0.0, self.kappa * lit.sigma_color_temp_k, size=len(centres))
            tint = np.asarray(lit.overhead_tint_rgb, dtype=float) * np.exp(
                rng.normal(0.0, self.kappa * lit.sigma_overhead_tint, size=3))
        else:
            # A multiple of the fitted level, so a refit moves the wide range with it.
            level = np.exp(rng.uniform(*np.log(lit.blind_intensity_rel_range)))
            temps = rng.uniform(*lit.blind_color_temp_range_k, size=len(centres))
            tint = np.exp(rng.uniform(-lit.blind_overhead_tint_log, lit.blind_overhead_tint_log,
                                      size=3))
        return self._light(centres, weights, share, lit.anchor_irradiance * scale * level,
                           np.clip(temps, 1500.0, 12000.0), tint)

    def _light(self, centres: list, weights: Any, share: float, irradiance: float,
               temps: Any, tint: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        lit = self.config.lighting
        reference = [*lit.reference_point_mm, self.config.objects.table_z_mm]
        return luminaire_lights(centres, weights, share, irradiance, [float(t) for t in temps],
                                tint, lit.luminaire_size_mm, reference)

    def _sample_exposure(self, index: int) -> dict[str, float]:
        """Session exposure gain of scene index, log-uniform over the admitted range,
        from a stream of its own (SeedSequence([seed, index, 2])): it depends on
        (seed, index) only, so every kappa, E5 arm and recipe of a seed shares it."""
        lo, hi = self.config.exposure.gain_range
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, int(index), _EXPOSURE_STREAM]))
        return {"gain": float(np.exp(rng.uniform(np.log(lo), np.log(hi))))}

    def _sample_background(self, rng: np.random.Generator) -> dict[str, Any]:
        bg = self.config.background
        if self.mode == "anchored":
            # Same story as the workpieces: the measured spread of the table's colour
            # is equal across channels (12, 13, 12 on 150, 151, 150), so it is the
            # brightness that differs between images, not the hue. Scale the table as
            # a whole and jitter the channels only slightly; independent draws at this
            # width hand the anchored arm pink and green tables, which is precisely
            # what anchoring is supposed to rule out. Both factors are exp(.), so the
            # table never goes black.
            base = np.asarray(bg.anchor_albedo_rgb, dtype=float)
            rel = bg.sigma_albedo / max(1e-6, float(base.mean()))
            scale = np.exp(rng.normal(0.0, self.kappa * rel))
            hue = np.exp(rng.normal(0.0, self.kappa * rel * 0.2, size=3))
            albedo = base * scale * hue
        else:
            lo, hi = bg.blind_albedo_range
            albedo = rng.uniform(lo, hi, size=3)
        return {
            "kind": "plane",
            "z_mm": self.config.objects.table_z_mm,
            "albedo_rgb": np.clip(albedo, 0.0, 1.0).tolist(),
            "roughness": float(bg.anchor_roughness),
        }

    def _sample_material(self, rng: np.random.Generator, cls: str) -> dict[str, Any]:
        """Appearance of one workpiece.

        Anchored: a dielectric class's measured colour, widened by kappa*sigma, and
        its estimated roughness; a metal class's physical reflectance F0 times
        exp(eta), eta ~ N(0, (kappa*sigma_f0_rel)^2), its roughness spread over
        kappa times roughness_range about the fitted value, and its clear coat's
        roughness N(coat_roughness, (kappa*sigma_coat_roughness)^2), and its top-face
        pattern as fitted (face_pattern, no spread). Blind: the wide
        hand-set ranges, using none of those measurements — including metallic, so a
        cardboard box may come out chrome, which is exactly what a blind treatment
        does — and a clear coat on blind_coat_probability of the products. Empty dict
        when the class has no material configured; the renderer then leaves the mesh
        in default grey.
        """
        obj = self.config.objects
        mat = obj.materials.get(cls)
        if self.mode == "blind":
            albedo = rng.uniform(*obj.blind_albedo_range, size=3)
            out = {"albedo_rgb": np.clip(albedo, 0.0, 1.0).tolist(),
                   "roughness": float(np.clip(rng.uniform(*obj.blind_roughness_range), 0.02, 1.0)),
                   "metallic": float(rng.uniform(0.0, 1.0))}
            if rng.random() < obj.blind_coat_probability:
                out["coat_ior"] = BLIND_COAT_IOR
                out["coat_roughness"] = float(np.clip(rng.uniform(*obj.blind_roughness_range),
                                                      0.02, 1.0))
            return out
        if mat is None:
            return {}
        if mat.f0_rgb is not None:
            f0 = np.asarray(mat.f0_rgb, dtype=float) * np.exp(
                rng.normal(0.0, self.kappa * mat.sigma_f0_rel))
            lo, hi = mat.roughness_range or (mat.roughness, mat.roughness)
            rough = mat.roughness + self.kappa * rng.uniform(lo - mat.roughness, hi - mat.roughness)
            out = {"f0_rgb": np.clip(f0, 0.0, 1.0).tolist(),
                   "roughness": float(np.clip(rough, 0.02, 1.0)),
                   "metallic": float(mat.metallic)}
            if mat.coat_ior is not None:
                coat = mat.coat_roughness + rng.normal(0.0, self.kappa * mat.sigma_coat_roughness)
                out["coat_ior"] = float(mat.coat_ior)
                out["coat_roughness"] = float(np.clip(coat, 0.02, 1.0))
            if mat.face_pattern is not None:
                out["face_pattern"] = {"floor": float(mat.face_pattern.floor),
                                       "sigma_mm": [float(v) for v in mat.face_pattern.sigma_mm]}
            return out
        else:
            # The measured spread is proportional to each channel (brightness between
            # images, not hue), so all three channels scale together. A small
            # independent term keeps some genuine colour variation without turning a
            # grey steel box green, which independent draws at the full width do.
            # Both are exp(.): the factor max(0, 1 + eta) rendered 18% of inox boxes
            # black at kappa 2 and 32% at kappa 4.
            assert mat.albedo_rgb is not None
            base = np.asarray(mat.albedo_rgb, dtype=float)
            scale = np.exp(rng.normal(0.0, self.kappa * mat.sigma_albedo_rel))
            hue = np.exp(rng.normal(0.0, self.kappa * mat.sigma_albedo_rel * 0.2, size=3))
            albedo = base * scale * hue
            rough = float(mat.roughness + rng.normal(0.0, self.kappa * obj.sigma_roughness))
            metal = float(mat.metallic)
        return {
            "albedo_rgb": np.clip(albedo, 0.0, 1.0).tolist(),
            "roughness": float(np.clip(rough, 0.02, 1.0)),
            "metallic": float(np.clip(metal, 0.0, 1.0)),
        }

    def _assign_textures(self, objects: list[dict[str, Any]], index: int) -> None:
        """Anchored: give each product of a textured mesh one of its measured top-face
        textures, uniformly, from the stream SeedSequence([seed, index, 3]). One draw
        per product in spec order whether textured or not, so no product's texture
        depends on the others' classes; the main stream is not touched, so every
        other draw of the scene is that of a sampler without textures."""
        if self.mode != "anchored" or not self.top_textures:
            return
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, int(index), _TEXTURE_STREAM]))
        for obj in objects:
            u = float(rng.random())
            entries = self.top_textures.get(obj["mesh"])
            if entries and obj.get("material"):
                obj["material"]["top_texture"] = top_texture.pick(entries, u)

    def _region(self) -> tuple[float, float, float, float]:
        obj = self.config.objects
        return (obj.region_x_mm[0], obj.region_x_mm[1],
                obj.region_y_mm[0], obj.region_y_mm[1])

    def _sample_xy(
        self, rng: np.random.Generator, region: tuple[float, float, float, float]
    ) -> tuple[float, float]:
        x0, x1, y0, y1 = region
        return float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))

    def _sample_yaw(self, rng: np.random.Generator) -> float:
        lo, hi = self.config.objects.yaw_range_deg
        return float(rng.uniform(lo, hi))

    def _size(self, mesh: str) -> list[float]:
        return self.config.objects.mesh_sizes_mm[mesh]

    def _footprint(self, obj: dict[str, Any]) -> np.ndarray:
        length, width, _ = self._size(obj["mesh"])
        return footprint(obj["xyz_mm"][:2], length, width, obj["yaw_deg"])

    def _place(
        self,
        rng: np.random.Generator,
        length: float,
        width: float,
        region: tuple[float, float, float, float],
        taken: list[np.ndarray],
        yaw_deg: float | None = None,
    ) -> tuple[float, float, float] | None:
        """(x, y, yaw) uniform over the poses whose footprint clears every one in
        taken, which then holds it too; None after PLACEMENT_TRIES draws. A given
        yaw_deg is kept; otherwise every draw draws the yaw as well."""
        for _ in range(PLACEMENT_TRIES):
            x, y = self._sample_xy(rng, region)
            yaw = self._sample_yaw(rng) if yaw_deg is None else float(yaw_deg)
            fp = footprint((x, y), length, width, yaw)
            if not any(footprints_overlap(fp, t) for t in taken):
                taken.append(fp)
                return x, y, yaw
        return None

    def _draw_first(
        self, rng: np.random.Generator, forced_class: str | None
    ) -> tuple[str, str | None]:
        """Class of the first product and the class of its support (None: it stands
        on the table), drawn with the frequencies of the pose-list rows. A forced
        class keeps only that class's rows."""
        obj = self.config.objects
        if forced_class is not None and forced_class not in obj.classes:
            raise ValueError(f"condition class '{forced_class}' not in objects.classes")
        if obj.first_object_mix:
            rows = []
            for cls in sorted(obj.first_object_mix):
                mix = obj.first_object_mix[cls]
                rows.append((cls, None, mix.single))
                rows += [(cls, sup, k) for sup, k in sorted(mix.stacked_on.items())]
        else:
            rows = [(cls, None, 1) for cls in sorted(obj.classes)]
        rows = [r for r in rows if r[2] > 0]
        if forced_class is not None:
            rows = [r for r in rows if r[0] == forced_class] or [(forced_class, None, 1)]
        cdf = np.cumsum([r[2] for r in rows], dtype=float)
        k = int(np.searchsorted(cdf / cdf[-1], rng.random(), side="right"))
        cls, support, _ = rows[min(k, len(rows) - 1)]
        return cls, support

    def _draw_look(self, rng: np.random.Generator, cls: str) -> tuple[str, dict[str, Any]]:
        """Mesh and material of one product of class cls."""
        # A class may map to several physical sizes (carton: two); pick one per
        # placement, from the same seeded generator so scenes stay reproducible.
        meshes = self.config.objects.classes[cls]
        if isinstance(meshes, list):
            mesh = meshes[int(rng.integers(0, len(meshes)))]
        else:
            mesh = meshes
        return mesh, self._sample_material(rng, cls)

    def _product(
        self,
        cls: str,
        mesh: str,
        xyz: list[float],
        yaw: float,
        material: dict[str, Any],
        stack_on_index: int | None = None,
    ) -> dict[str, Any]:
        spec_obj: dict[str, Any] = {
            "class_name": cls,
            "class_id": self.config.objects.class_ids[cls],
            "mesh": mesh,
            "xyz_mm": [float(v) for v in xyz],
            "yaw_deg": float(yaw),
        }
        if stack_on_index is not None:
            spec_obj["stack_on_index"] = stack_on_index
        if material:
            spec_obj["material"] = material
        return spec_obj

    def _sample_objects(
        self, rng: np.random.Generator, condition: dict[str, Any] | None
    ) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
        """Products, and the footprints they occupy on the table.

        The first placement is drawn before any other, so it is uniform over its
        region: the whole envelope, or the cell a failure condition names. Each
        further placement is uniform over the poses that clear the footprints
        already there, and is left out when PLACEMENT_TRIES draws find none.
        """
        obj = self.config.objects
        names = sorted(obj.classes)
        n = int(rng.integers(obj.count_range[0], obj.count_range[1] + 1))
        region = self._region()
        forced_class = None
        first_region = region
        if condition:
            forced_class = condition.get("class_name") or None
            if condition.get("region"):
                r = condition["region"]
                first_region = (float(r[0]), float(r[1]), float(r[2]), float(r[3]))

        cls, support_cls = self._draw_first(rng, forced_class)
        mesh, material = self._draw_look(rng, cls)
        x, y = self._sample_xy(rng, first_region)
        yaw = self._sample_yaw(rng)
        z = obj.table_z_mm
        if support_cls is None:
            objects = [self._product(cls, mesh, [x, y, z], yaw, material)]
        else:
            # Staged as in the trials: the support stands at the placement and the
            # product rests on its top face, its centre within stack_offset_mm of the
            # support's centre. The trials do not call the support's yaw, so it is
            # drawn on its own.
            sup_mesh, sup_material = self._draw_look(rng, support_cls)
            sup_yaw = self._sample_yaw(rng)
            dx, dy = rng.uniform(-obj.stack_offset_mm, obj.stack_offset_mm, size=2)
            objects = [
                self._product(cls, mesh, [x + dx, y + dy, z + self._size(sup_mesh)[2]],
                              yaw, material, stack_on_index=1),
                self._product(support_cls, sup_mesh, [x, y, z], sup_yaw, sup_material),
            ]
        taken = [self._footprint(o) for o in objects]
        for _ in range(1, n):
            c = names[int(rng.integers(0, len(names)))]
            mesh, material = self._draw_look(rng, c)
            pose = self._place(rng, *self._size(mesh)[:2], region, taken)
            if pose is not None:
                objects.append(self._product(c, mesh, [pose[0], pose[1], z], pose[2], material))
        return objects, taken

    def _sample_distractors(
        self, rng: np.random.Generator, taken: list[np.ndarray]
    ) -> list[dict[str, Any]]:
        """Distractors in the placement envelope, clear of the products and of one
        another. Each occupies its size-wide square at its yaw, which holds the
        sphere and the cylinder as well. One that finds no free position in
        PLACEMENT_TRIES draws is left out."""
        dis = self.config.distractors
        obj = self.config.objects
        rng_range = dis.count_range if self.mode == "anchored" else dis.blind_count_range
        n = int(rng.integers(rng_range[0], rng_range[1] + 1))
        taken = list(taken)
        out: list[dict[str, Any]] = []
        for _ in range(n):
            shape = ["cube", "sphere", "cylinder"][int(rng.integers(0, 3))]
            size = float(rng.uniform(*dis.size_range_mm))
            yaw = float(rng.uniform(-180.0, 180.0))
            albedo = rng.uniform(0.05, 0.95, size=3).tolist()
            pose = self._place(rng, size, size, self._region(), taken, yaw_deg=yaw)
            if pose is None:
                continue
            out.append({
                "shape": shape,
                "size_mm": size,
                "xyz_mm": [pose[0], pose[1], obj.table_z_mm],
                "yaw_deg": yaw,
                "albedo_rgb": albedo,
            })
        return out

    # ────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────

    def sample(self, index: int, condition: dict[str, Any] | None = None) -> dict[str, Any]:
        """Draw one scene spec. Deterministic in (seed, index, condition).

        Args:
            index: Scene index within the run, an integer in [0, 2**32). With the run
                seed it seeds the scene's generator, SeedSequence([seed, index]).
            condition: Optional C3 failure-mode constraint:
                {"class_name": str|None, "region": [x0,x1,y0,y1]|None,
                 "lighting": str|None} — forces the first product's class (its
                support, if any, follows that class's pose-list rows) and the
                region of its placement, and scales the light level to the named
                regime (lighting_scale; an unknown label raises).
        """
        self._check_condition(condition)
        if not 0 <= int(index) < WORD:
            raise ValueError(f"index must be an integer in [0, 2**32), got {index}")
        # SeedSequence mixes both words, so the stream of (seed, index) is not the
        # stream of (seed + 1, index - 1), as it was when seeded with seed + index.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        lighting_condition = (condition or {}).get("lighting")
        camera = self._sample_camera(rng)
        lights, environment = self._sample_lights(rng, lighting_condition)
        background = self._sample_background(rng)
        objects, taken = self._sample_objects(rng, condition)
        self._assign_textures(objects, index)
        spec = {
            "spec_version": SPEC_VERSION,
            "index": int(index),
            "seed": self.seed,
            "mode": self.mode,
            "kappa": self.kappa if self.mode == "anchored" else None,
            "sigma_source": self.sigma_source,
            "camera": camera,
            "lights": lights,
            "environment": environment,
            "exposure": self._sample_exposure(index),
            "background": background,
            "objects": objects,
            "distractors": self._sample_distractors(rng, taken),
            "condition": condition or None,
        }
        self._apply_ablation(spec, index)
        return spec

    def write_specs(
        self,
        out_dir: str | Path,
        n: int,
        condition: dict[str, Any] | None = None,
        start_index: int = 0,
    ) -> Path:
        """Write n scene specs + manifest.json to out_dir. Returns manifest path."""
        self._check_condition(condition)
        if n < 1 or start_index < 0:
            raise ValueError("n must be positive and start_index non-negative")
        out = Path(out_dir)
        files = [f"scene_{i:06d}.json" for i in range(start_index, start_index + n)]
        manifest_path = out / "manifest.json"
        manifest = {
            "spec_version": SPEC_VERSION,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n": n,
            "start_index": start_index,
            "mode": self.mode,
            "kappa": self.kappa if self.mode == "anchored" else None,
            "seed": self.seed,
            "sigma_source": self.sigma_source,
            "ablation": deepcopy(self.ablation_protocol),
            "generator_profile": self._generator_profile(),
            "condition": condition or None,
            "files": files,
        }
        # Merge with an existing manifest (multi-batch runs, e.g. C3 allocations).
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                old = json.load(f)
            # Check BEFORE writing scenes: never mix arms or silently overwrite a
            # run. Legacy manifests lack the profile needed to prove equivalence.
            for key in ("spec_version", "mode", "kappa", "seed", "sigma_source",
                        "ablation", "generator_profile"):
                if old.get(key) != manifest[key]:
                    raise ValueError(f"Existing manifest differs in '{key}'; use a new output directory")
            if set(old.get("files", [])) & set(files):
                raise ValueError("Scene indices already exist; use a new output directory or start_index")
            manifest["files"] = old.get("files", []) + files
            manifest["n"] = len(manifest["files"])
            manifest["start_index"] = min(old["start_index"], start_index)
            manifest["batches"] = old.get("batches", [{
                "start_index": old["start_index"], "n": old["n"],
                "condition": old.get("condition"),
            }]) + [
                {"start_index": start_index, "n": n, "condition": condition or None}
            ]
        if any((out / name).exists() for name in files):
            raise ValueError("Scene file already exists; use a new output directory or start_index")
        out.mkdir(parents=True, exist_ok=True)
        for i, name in zip(range(start_index, start_index + n), files):
            spec = self.sample(i, condition=condition)
            with (out / name).open("w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest_path
