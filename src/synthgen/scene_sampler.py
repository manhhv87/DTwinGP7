"""
scene_sampler.py
────────────────
Sample renderer scene specifications for calibration-anchored (or blind)
domain randomization — the implementation of Eq. (1) of the paper:

    theta_i ~ D_i(theta_i*, kappa * sigma_i)

ANCHORED mode: nuisance parameters (camera extrinsics/intrinsics, lighting,
background) are drawn around the twin's calibrated ground state theta* with
widths kappa*sigma; sigma comes from the bootstrap of the hand-eye calibration
(config/calibration/T_base_camera_sigma.json) or, as a fallback, from
config/synthgen.yaml. Task variables (object pose, count) are fully random.

BLIND mode: wide hand-set perturbation ranges around the same camera reference,
with a different appearance recipe. It is not independent of calibration.

The sampler runs in the repo venv and writes plain-JSON scene specs; rendering
happens separately under `blenderproc run` (render_blenderproc.py), which only
consumes these JSON files. This split keeps the sampling logic unit-testable
without Blender.
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .anchored_config import LIGHTING_CONDITION_SCALE, SynthgenConfig

# Schema version of the scene-spec JSON contract with render_blenderproc.py.
SPEC_VERSION = 1
ABLATION_FACTORS = ("none", "illumination", "background", "distractors", "camera", "pose")


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
        seed: Base RNG seed. Scene i uses default_rng(seed + i) → any subset of
            a run can be regenerated deterministically.
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

        self.ablation_protocol = {
            "protocol_version": 1,
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
            lit = self.config.lighting
            return [{"type": "point", "position_mm": list(pos),
                     "energy_w": float(lit.anchor_intensity_w),
                     "color_temp_k": float(lit.anchor_color_temp_k)}
                    for pos in lit.anchor_positions_mm]
        if self.ablation == "background":
            bg = self.config.background
            return {"kind": "plane", "z_mm": self.config.objects.table_z_mm,
                    "albedo_rgb": list(bg.anchor_albedo_rgb),
                    "roughness": float(bg.anchor_roughness)}
        if self.ablation == "distractors":
            return []

        # Pose is fixed conditional on count; class, size and material still vary.
        # Slots are cell centres along the longer region axis (X breaks a tie).
        obj = self.config.objects
        ranges = np.asarray([obj.region_x_mm, obj.region_y_mm], dtype=float)
        widths = ranges[:, 1] - ranges[:, 0]
        axis = int(np.argmax(widths))
        centre = ranges.mean(axis=1)
        counts = obj.count_range
        if len(counts) != 2 or counts[0] < 1 or counts[1] < counts[0]:
            raise ValueError("pose ablation requires a positive ordered objects.count_range")
        if counts[1] > 1 and widths[axis] / counts[1] < obj.min_separation_mm:
            raise ValueError("pose reference slots cannot satisfy min_separation_mm; "
                             "revise and document the reference region/count protocol")
        poses_by_count = {}
        yaw = float(np.mean(obj.yaw_range_deg))
        for n in range(counts[0], counts[1] + 1):
            poses = []
            for i in range(n):
                xy = centre.copy()
                xy[axis] = ranges[axis, 0] + (i + 0.5) * widths[axis] / n
                poses.append({"xyz_mm": [float(xy[0]), float(xy[1]), obj.table_z_mm],
                              "yaw_deg": yaw})
            poses_by_count[str(n)] = poses
        return {"rule": "count_conditional_long_axis_cell_centres",
                "axis": "xy"[axis], "poses_by_count": poses_by_count}

    def _generator_profile(self) -> dict[str, Any]:
        """Persist remaining distributions and effective camera widths for audit."""
        return {"config": self.config.model_dump(mode="json"),
                "T_BC_mm": self.T_BC.tolist(),
                "sigma_trans_mm": self.sigma_trans.tolist(),
                "sigma_rot_deg": self.sigma_rot.tolist()}

    def _check_condition(self, condition: dict[str, Any] | None) -> None:
        if self.ablation != "none" and condition:
            raise ValueError("E5 ablation cannot be combined with failure conditioning")

    def _apply_ablation(self, spec: dict[str, Any]) -> None:
        reference = deepcopy(self.ablation_protocol["reference"])
        if self.ablation == "pose":
            poses = reference["poses_by_count"][str(len(spec["objects"]))]
            for obj, pose in zip(spec["objects"], poses):
                obj.update(pose)
        elif self.ablation != "none":
            key = "lights" if self.ablation == "illumination" else self.ablation
            spec[key] = reference
        spec["ablation"] = deepcopy(self.ablation_protocol)

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
    ) -> list[dict[str, Any]]:
        lit = self.config.lighting
        scale = LIGHTING_CONDITION_SCALE.get(lighting_condition or "", 1.0)
        lights: list[dict[str, Any]] = []
        if self.mode == "anchored":
            for pos in lit.anchor_positions_mm:
                p = np.asarray(pos, dtype=float) + rng.normal(
                    0.0, self.kappa * lit.sigma_position_mm, size=3
                )
                energy = lit.anchor_intensity_w * scale * (
                    1.0 + rng.normal(0.0, self.kappa * lit.sigma_intensity_frac)
                )
                temp = lit.anchor_color_temp_k + rng.normal(
                    0.0, self.kappa * lit.sigma_color_temp_k
                )
                lights.append({
                    "type": "point",
                    "position_mm": p.tolist(),
                    "energy_w": float(max(1.0, energy)),
                    "color_temp_k": float(np.clip(temp, 1500.0, 12000.0)),
                })
        else:
            n = int(rng.integers(lit.blind_count_range[0], lit.blind_count_range[1] + 1))
            (x0, x1), (y0, y1), (z0, z1) = lit.blind_position_box_mm
            for _ in range(max(1, n)):
                lights.append({
                    "type": "point",
                    "position_mm": [
                        float(rng.uniform(x0, x1)),
                        float(rng.uniform(y0, y1)),
                        float(rng.uniform(z0, z1)),
                    ],
                    "energy_w": float(rng.uniform(*lit.blind_intensity_range_w) * scale),
                    "color_temp_k": float(rng.uniform(*lit.blind_color_temp_range_k)),
                })
        return lights

    def _sample_background(self, rng: np.random.Generator) -> dict[str, Any]:
        bg = self.config.background
        if self.mode == "anchored":
            # Same story as the workpieces: the measured spread of the table's colour
            # is equal across channels (12, 13, 12 on 150, 151, 150), so it is the
            # brightness that differs between images, not the hue. Scale the table as
            # a whole and jitter the channels only slightly; independent draws at this
            # width hand the anchored arm pink and green tables, which is precisely
            # what anchoring is supposed to rule out.
            base = np.asarray(bg.anchor_albedo_rgb, dtype=float)
            rel = bg.sigma_albedo / max(1e-6, float(base.mean()))
            scale = max(0.0, 1.0 + rng.normal(0.0, self.kappa * rel))
            hue = rng.normal(0.0, self.kappa * rel * 0.2, size=3)
            albedo = base * scale * (1.0 + hue)
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

        Anchored: the class's measured colour, widened by kappa*sigma, and its
        estimated reflectance. Blind: the wide hand-set ranges, using none of those
        measurements — including metallic, so a cardboard box may come out chrome,
        which is exactly what a blind treatment does. Empty dict when the class has
        no material configured; the renderer then leaves the mesh in default grey.
        """
        obj = self.config.objects
        mat = obj.materials.get(cls)
        if self.mode == "blind":
            albedo = rng.uniform(*obj.blind_albedo_range, size=3)
            rough = float(rng.uniform(*obj.blind_roughness_range))
            metal = float(rng.uniform(0.0, 1.0))
        elif mat is None:
            return {}
        else:
            # The measured spread is proportional to each channel (brightness between
            # images, not hue), so all three channels scale together. A small
            # independent term keeps some genuine colour variation without turning a
            # grey steel box green, which independent draws at the full width do.
            base = np.asarray(mat.albedo_rgb, dtype=float)
            scale = 1.0 + rng.normal(0.0, self.kappa * mat.sigma_albedo_rel)
            hue = rng.normal(0.0, self.kappa * mat.sigma_albedo_rel * 0.2, size=3)
            albedo = base * max(0.0, scale) * (1.0 + hue)
            rough = float(mat.roughness + rng.normal(0.0, self.kappa * obj.sigma_roughness))
            metal = float(mat.metallic)
        return {
            "albedo_rgb": np.clip(albedo, 0.0, 1.0).tolist(),
            "roughness": float(np.clip(rough, 0.02, 1.0)),
            "metallic": float(np.clip(metal, 0.0, 1.0)),
        }

    def _sample_xy(
        self, rng: np.random.Generator, region: tuple[float, float, float, float]
    ) -> tuple[float, float]:
        x0, x1, y0, y1 = region
        return float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))

    def _place_non_overlapping(
        self,
        rng: np.random.Generator,
        n: int,
        region: tuple[float, float, float, float],
        min_sep: float,
        max_tries: int = 100,
    ) -> list[tuple[float, float]]:
        """Rejection-sample n positions with pairwise min separation.

        Falls back to progressively halving min_sep instead of failing — a
        cluttered scene is still a valid (and useful) training scene.
        """
        placed: list[tuple[float, float]] = []
        sep = float(min_sep)
        while len(placed) < n:
            for _ in range(max_tries):
                x, y = self._sample_xy(rng, region)
                if all((x - px) ** 2 + (y - py) ** 2 >= sep**2 for px, py in placed):
                    placed.append((x, y))
                    break
            else:
                sep *= 0.5  # relax and keep going
                if sep < 1.0:
                    placed.append(self._sample_xy(rng, region))
        return placed

    def _sample_objects(
        self, rng: np.random.Generator, condition: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        obj = self.config.objects
        names = sorted(obj.classes)
        n = int(rng.integers(obj.count_range[0], obj.count_range[1] + 1))
        n = max(1, n)

        region = (
            obj.region_x_mm[0], obj.region_x_mm[1],
            obj.region_y_mm[0], obj.region_y_mm[1],
        )
        forced_class = None
        forced_region = None
        if condition:
            forced_class = condition.get("class_name") or None
            if condition.get("region"):
                r = condition["region"]
                forced_region = (float(r[0]), float(r[1]), float(r[2]), float(r[3]))

        positions = self._place_non_overlapping(rng, n, region, obj.min_separation_mm)
        objects: list[dict[str, Any]] = []
        for i, (x, y) in enumerate(positions):
            if i == 0 and forced_class is not None:
                cls = forced_class
                if cls not in obj.classes:
                    raise ValueError(f"condition class '{cls}' not in objects.classes")
            else:
                cls = names[int(rng.integers(0, len(names)))]
            if i == 0 and forced_region is not None:
                x, y = self._sample_xy(rng, forced_region)
            yaw = float(rng.uniform(obj.yaw_range_deg[0], obj.yaw_range_deg[1]))
            # A class may map to several physical sizes (carton: two); pick one per
            # placement, from the same seeded generator so scenes stay reproducible.
            meshes = obj.classes[cls]
            if isinstance(meshes, list):
                mesh = meshes[int(rng.integers(0, len(meshes)))]
            else:
                mesh = meshes
            spec_obj = {
                "class_name": cls,
                "class_id": obj.class_ids[cls],
                "mesh": mesh,
                "xyz_mm": [x, y, obj.table_z_mm],
                "yaw_deg": yaw,
            }
            material = self._sample_material(rng, cls)
            if material:
                spec_obj["material"] = material
            objects.append(spec_obj)
        return objects

    def _sample_distractors(self, rng: np.random.Generator) -> list[dict[str, Any]]:
        dis = self.config.distractors
        obj = self.config.objects
        rng_range = dis.count_range if self.mode == "anchored" else dis.blind_count_range
        n = int(rng.integers(rng_range[0], rng_range[1] + 1))
        region = (
            obj.region_x_mm[0], obj.region_x_mm[1],
            obj.region_y_mm[0], obj.region_y_mm[1],
        )
        out: list[dict[str, Any]] = []
        for _ in range(n):
            x, y = self._sample_xy(rng, region)
            out.append({
                "shape": ["cube", "sphere", "cylinder"][int(rng.integers(0, 3))],
                "size_mm": float(rng.uniform(*dis.size_range_mm)),
                "xyz_mm": [x, y, obj.table_z_mm],
                "yaw_deg": float(rng.uniform(-180.0, 180.0)),
                "albedo_rgb": rng.uniform(0.05, 0.95, size=3).tolist(),
            })
        return out

    # ────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────

    def sample(self, index: int, condition: dict[str, Any] | None = None) -> dict[str, Any]:
        """Draw one scene spec. Deterministic in (seed, index, condition).

        Args:
            index: Scene index within the run (also drives the RNG stream).
            condition: Optional C3 failure-mode constraint:
                {"class_name": str|None, "region": [x0,x1,y0,y1]|None,
                 "lighting": str|None} — forces the first object's class/region
                and scales lighting to the named regime.
        """
        self._check_condition(condition)
        rng = np.random.default_rng(self.seed + index)
        lighting_condition = (condition or {}).get("lighting")
        spec = {
            "spec_version": SPEC_VERSION,
            "index": int(index),
            "seed": self.seed,
            "mode": self.mode,
            "kappa": self.kappa if self.mode == "anchored" else None,
            "sigma_source": self.sigma_source,
            "camera": self._sample_camera(rng),
            "lights": self._sample_lights(rng, lighting_condition),
            "background": self._sample_background(rng),
            "objects": self._sample_objects(rng, condition),
            "distractors": self._sample_distractors(rng),
            "condition": condition or None,
        }
        self._apply_ablation(spec)
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
