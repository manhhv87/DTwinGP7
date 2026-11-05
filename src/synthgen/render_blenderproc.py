# `blenderproc run` REFUSES a script whose first line that is neither blank nor a
# comment is anything other than the blenderproc import: it strips comments and blank
# lines and checks the very first survivor (SetupUtility.check_if_setup_utilities_are_
# at_the_top). A module docstring counts as that first line and fails the check, so the
# import leads the file, the description is assigned to __doc__ below, and there is no
# `from __future__ import annotations` (it may only follow a docstring, never an import).
# Blender 4.2 ships Python 3.11, where the builtin generics below need no future import.
import blenderproc as bproc

__doc__ = """
render_blenderproc.py
─────────────────────
BlenderProc2 rendering backend for the anchored/blind synthetic generator.

⚠ This file does NOT run in the repo venv. BlenderProc bundles its own Blender
Python; run it from the repo root as:

    blenderproc run src/synthgen/render_blenderproc.py -- \
        --scenes data/synth/<run>/specs --out data/synth/<run>/render

(`pip install blenderproc` in any Python ≥3.10 env provides the `blenderproc`
launcher; first run downloads Blender automatically.)

Input:  scene_*.json specs written by scene_sampler.SceneSampler (mm units,
        OpenCV camera convention: +Z looks forward/down), and the render, sensor and
        hard_profile sections of config/synthgen.yaml (--config).
Output per frame (contract consumed by auto_label.convert_render_output):
    frame_<idx>_rgb.png     rendered RGB, 8-bit sRGB after the sensor model
    frame_<idx>_inst.png    instance map (uint16 PNG; 0 = background)
    frame_<idx>_meta.json   {"instances": {inst_id: class_id}, "scene": <spec file>}
and once per output directory render_manifest.json: samples, device, Blender and
BlenderProc versions, the profile and its hash (hard_profile.build_manifest).

Design notes:
  - mm → m conversion happens HERE (Blender works in meters).
  - OpenCV cam → Blender cam: right-multiply by diag(1,-1,-1) (flip Y,Z axes).
  - Every colour in the config and the specs is sRGB-encoded (a measured camera
    value / 255); Base Color is scene-linear, so each one is linearised first
    (sensor_model.srgb_to_linear).
  - The scene is the hard condition of the trials (hard_profile.py): green sheet in
    two parts split by the black strip, the 21 cards, the cell around the table's
    edges. The spec's background albedo is the upper sheet's colour.
  - Light: the spec's area luminaires (600 mm panels above the cell) and its uniform
    overhead (spec environment.radiance_rgb, the world), as the sampler drew them
    (hard_profile.luminaire_lights).
  - Reflectances: a dielectric's Base Color is its albedo in the units of the config
    (the bare standard table is 0.305 linear), about 0.46 of physical. Every PHYSICAL
    specular reflectance is multiplied by render.reflectance_scale to match: a
    metal's f0_rgb, the dielectric specular (Specular IOR Level 0.5, F0 0.04) and a
    clear coat's weight.
  - Cycles renders scene-linear EXR; sensor_model.apply_sensor then applies white
    balance, the gain (sensor.gain times the scene's exposure gain), Poisson-Gaussian
    noise and the sRGB curve (Blender's "Standard" view, no tone mapping) in numpy.
    A spec without an exposure draw (spec_version < 3) is refused.
  - Meshes are placed so their bounding-box bottom rests at the spec height
    xyz_mm[2]: the table top, or the top of the part underneath for a stack (STL
    origins are arbitrary).
  - SMOKE TEST before a full run:
        blenderproc run src/synthgen/render_blenderproc.py -- \
            --scenes <dir-with-3-specs> --out /tmp/smoke --samples 16
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MM = 1e-3  # mm → m
# Per-frame noise stream: drawn from (spec seed, scene index, this tag), so the
# sensor noise of a frame is reproducible yet independent of the sampler's draws.
NOISE_STREAM_TAG = 0x5E45
# Specular IOR Level of a dielectric: 0.5 leaves the F0 of its IOR (1.5: 0.04),
# before render.reflectance_scale.
DIELECTRIC_SPECULAR_LEVEL = 0.5


def _cv_to_blender_cam(T_cv_m: np.ndarray) -> np.ndarray:
    """OpenCV camera pose (+Z forward, +Y down) → Blender (-Z forward, +Y up)."""
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return T_cv_m @ flip


def _pose_from_spec(xyz_mm: list[float], yaw_deg: float) -> np.ndarray:
    """(x, y, z) mm + yaw → 4x4 pose in meters (rotation about Z only)."""
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = np.asarray(xyz_mm, dtype=float) * MM
    return T


def main() -> int:
    # bpy exists only inside Blender; blenderproc itself is imported at the top of the
    # file, because the launcher refuses a script that does not open with that import.
    try:
        import bpy
    except ImportError:
        print(
            "bpy not available — this script only runs under Blender. Run via:\n"
            "  blenderproc run src/synthgen/render_blenderproc.py -- --scenes ... --out ...",
            file=sys.stderr,
        )
        return 2
    from PIL import Image  # bundled with the blenderproc environment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", required=True, help="Dir with scene_*.json specs")
    parser.add_argument("--out", required=True, help="Output dir for rendered frames")
    parser.add_argument("--config", default="config/synthgen.yaml",
                        help="Config with the render, sensor and hard_profile sections "
                             "(relative paths are taken from the repo root)")
    parser.add_argument("--samples", type=int, default=0,
                        help="Cycles samples/px; 0 = render.samples of the config. "
                             "Lower only for a smoke test: the value is written to the "
                             "manifest, and packaging refuses arms that differ in it")
    parser.add_argument("--limit", type=int, default=0,
                        help="Render only the first N specs (0 = all)")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto",
                        help="Render device. 'gpu' REQUIRES a usable GPU and stops if "
                             "there is none, so a batch meant for the GPU machine "
                             "cannot quietly spend hours on its CPU instead. "
                             "'auto' leaves the choice to BlenderProc.")
    parser.add_argument("--save-linear", action="store_true",
                        help="Also write frame_<idx>_linear.npy, the scene-linear render "
                             "before the sensor model (scripts/fit_hard_profile.py)")
    args = parser.parse_args()

    scenes_dir = Path(args.scenes)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent.parent
    # The two helpers are plain modules next to this file. Imported as top-level
    # modules, because src.synthgen's __init__ pulls in pydantic, which Blender's
    # Python does not have.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hard_profile as hp
    import sensor_model as sm

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    cfg = hp.load_render_config(config_path)
    sm.check_sensor(cfg["sensor"])
    rcfg = cfg["render"]
    samples = int(args.samples) if args.samples > 0 else int(rcfg["samples"])

    spec_files = sorted(scenes_dir.glob("scene_*.json"))
    if args.limit > 0:
        spec_files = spec_files[: args.limit]
    if not spec_files:
        print(f"No scene_*.json in {scenes_dir}", file=sys.stderr)
        return 1

    # Frames already in the directory must come from a batch rendered the same way,
    # which only a manifest can show. Frames without one predate the manifest (the
    # Filmic, uniform-plane renders) and must not be mixed with these.
    manifest_path = out_dir / hp.MANIFEST_NAME
    old_manifest = None
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            old_manifest = json.load(f)
    elif any(out_dir.glob("frame_*_rgb.png")):
        print(f"{out_dir} holds frames but no {hp.MANIFEST_NAME}: they were rendered "
              "before the hard profile. Use a new output directory.", file=sys.stderr)
        return 1

    bproc.init()
    if args.device == "cpu":
        bproc.renderer.set_render_devices(use_only_cpu=True)
    elif args.device == "gpu":
        bproc.renderer.set_render_devices(desired_gpu_device_type=["OPTIX", "CUDA", "HIP"])
        # BlenderProc falls back to the CPU when it finds no usable GPU, and says so
        # only in passing. On a 3000-image batch that fallback costs hours, so check.
        if bpy.context.scene.cycles.device != "GPU":
            print("--device gpu was asked for, but Cycles ended up on the CPU: no "
                  "usable OPTIX/CUDA/HIP device. Check the driver, or run with "
                  "--device cpu deliberately.", file=sys.stderr)
            return 3
    cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
    device = str(bpy.context.scene.cycles.device)
    if device == "GPU":
        device = f"GPU:{cycles_prefs.compute_device_type}"
    print(f"Render device: {device} (--device {args.device})")

    # Render settings come from the config so that every batch of every arm uses the
    # same ones, and they are written to the manifest. BlenderProc's defaults
    # (glossy_bounces 0, max_bounces 3) left a mirror nothing to reflect, so the inox
    # box, the class the hard condition is about, rendered black.
    bproc.renderer.set_max_amount_of_samples(samples)
    bproc.renderer.set_noise_threshold(float(rcfg["noise_threshold"]))
    bproc.renderer.set_denoiser(rcfg["denoiser"])
    bproc.renderer.set_light_bounces(
        diffuse_bounces=int(rcfg["diffuse_bounces"]),
        glossy_bounces=int(rcfg["glossy_bounces"]),
        max_bounces=int(rcfg["max_bounces"]),
        transmission_bounces=int(rcfg["transmission_bounces"]),
        transparent_max_bounces=int(rcfg["transparent_max_bounces"]))
    # Scene-linear float output; the sensor model does the camera's encoding. The
    # view transform is set to Standard as well so that no tone curve hides anywhere.
    bproc.renderer.set_output_format(file_format="OPEN_EXR", color_depth=32,
                                     view_transform=rcfg["view_transform"])
    # Physical specular reflectances in the albedo units of the config (see __doc__).
    reflectance_scale = float(rcfg["reflectance_scale"])
    render_settings = {
        "samples": samples,
        "denoiser": rcfg["denoiser"],
        "noise_threshold": float(rcfg["noise_threshold"]),
        "diffuse_bounces": int(rcfg["diffuse_bounces"]),
        "glossy_bounces": int(rcfg["glossy_bounces"]),
        "max_bounces": int(rcfg["max_bounces"]),
        "transmission_bounces": int(rcfg["transmission_bounces"]),
        "transparent_max_bounces": int(rcfg["transparent_max_bounces"]),
        "view_transform": rcfg["view_transform"],
        "reflectance_scale": reflectance_scale,
        "output": "OPEN_EXR 32-bit scene-linear, then sensor_model.apply_sensor",
    }
    blender_version = str(bpy.app.version_string)
    bproc_version = str(getattr(bproc, "__version__", "unknown"))

    def write_manifest(frames: list[str]) -> None:
        new = hp.build_manifest(cfg, render_settings, device=device,
                                blender_version=blender_version,
                                blenderproc_version=bproc_version,
                                config_path=str(config_path), scenes_dir=str(scenes_dir),
                                frames=frames)
        merged = hp.merge_manifest(old_manifest, new) if old_manifest else new
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)

    # Refuse a mismatching batch before spending hours on it, not after.
    write_manifest([])

    def import_mesh(path: Path):
        """Load STL/OBJ/PLY → bproc MeshObject (STL via bpy operator fallback)."""
        suffix = path.suffix.lower()
        if suffix in (".obj", ".ply"):
            return bproc.loader.load_obj(str(path))[0]
        before = set(bpy.data.objects)
        try:
            bpy.ops.wm.stl_import(filepath=str(path))          # Blender ≥ 4.0
        except AttributeError:
            bpy.ops.import_mesh.stl(filepath=str(path))        # Blender 3.x
        new = [o for o in bpy.data.objects if o not in before]
        if not new:
            raise RuntimeError(f"STL import produced no object: {path}")
        # Blender 4.2's STL importer shades every face smooth. A box of 12 triangles
        # shares 8 vertices, so the interpolated normals bend each face like a pillow:
        # the mirror top of the stainless case became a convex mirror that shrank a
        # 600 mm luminaire to a dot. The parts are flat-faced boxes.
        for poly in new[0].data.polygons:
            poly.use_smooth = False
        return bproc.python.types.MeshObjectUtility.MeshObject(new[0])

    def place_on_surface(obj, z_support_m: float) -> None:
        """Shift the object so its bounding-box bottom sits at the support height."""
        bb = np.array(obj.get_bound_box())          # 8x3 world coords
        dz = z_support_m - float(bb[:, 2].min())
        loc = np.array(obj.get_location())
        obj.set_location(loc + np.array([0.0, 0.0, dz]))

    def flat_polygon(name: str, polygon_mm, z_mm: float):
        """One flat face, normal up, from a CCW polygon in mm."""
        mesh = bpy.data.meshes.new(name)
        verts = [(float(x) * MM, float(y) * MM, float(z_mm) * MM) for x, y in polygon_mm]
        mesh.from_pydata(verts, [], [list(range(len(verts)))])
        mesh.update()
        return bproc.object.create_from_blender_mesh(mesh, name)

    def matte(name: str, albedo_srgb):
        """Lambertian material: the static elements were fitted as pure diffuse
        surfaces, so their brightness is exactly albedo times irradiance."""
        mat = bproc.material.create(name)
        mat.set_principled_shader_value(
            "Base Color", [float(v) for v in sm.srgb_to_linear(albedo_srgb)] + [1.0])
        mat.set_principled_shader_value("Roughness", 1.0)
        mat.set_principled_shader_value("Specular IOR Level", 0.0)
        return mat

    def product_material(name: str, ms: dict):
        """Principled material of a workpiece: a metal's physical F0 (f0_rgb, linear)
        or a dielectric's albedo (albedo_rgb, sRGB-encoded) as Base Color, the
        dielectric specular and a clear coat, every physical reflectance times
        render.reflectance_scale."""
        mat = bproc.material.create(name)
        if "f0_rgb" in ms:
            base = [float(v) * reflectance_scale for v in ms["f0_rgb"]]
        else:
            base = [float(v) for v in sm.srgb_to_linear(ms["albedo_rgb"])]
        mat.set_principled_shader_value("Base Color", base + [1.0])
        mat.set_principled_shader_value("Roughness", float(ms["roughness"]))
        mat.set_principled_shader_value("Metallic", float(ms["metallic"]))
        mat.set_principled_shader_value("Specular IOR Level",
                                        DIELECTRIC_SPECULAR_LEVEL * reflectance_scale)
        if "coat_ior" in ms:
            mat.set_principled_shader_value("Coat Weight", reflectance_scale)
            mat.set_principled_shader_value("Coat Roughness", float(ms["coat_roughness"]))
            mat.set_principled_shader_value("Coat IOR", float(ms["coat_ior"]))
        return mat

    rendered: list[str] = []
    for spec_path in spec_files:
        with spec_path.open("r", encoding="utf-8") as f:
            spec = json.load(f)
        idx = int(spec["index"])
        # Only the profile fit's reference render (mode "reference") goes without a
        # session exposure and an overhead; a sampled spec without them predates
        # spec_version 3 and would render at a fixed exposure in the old light.
        exposure, environment = spec.get("exposure"), spec.get("environment")
        if spec.get("mode") != "reference" and (exposure is None or environment is None):
            raise ValueError(f"{spec_path.name}: no exposure draw or environment "
                             f"(spec_version {spec.get('spec_version')}); regenerate the specs")
        exposure_gain = 1.0 if exposure is None else float(exposure["gain"])
        bproc.utility.reset_keyframes()
        # Full scene rebuild per spec keeps specs independent (slower but simple;
        # optimise later by caching meshes if render throughput becomes the bottleneck).
        for obj in bproc.object.get_all_mesh_objects():
            obj.delete()
        # Lights are NOT mesh objects, so the loop above leaves them standing. Without
        # this, every scene keeps the lights of all scenes before it: frame 50 of a
        # batch was lit by 100 lamps, each frame brighter than the last, which silently
        # destroys the lighting factor the anchored generator exists to control.
        for light_obj in [o for o in bpy.data.objects if o.type == "LIGHT"]:
            bpy.data.objects.remove(light_obj, do_unlink=True)
        # Materials and meshes are created per scene too and would otherwise pile up
        # as bg_mat.001, bg_mat.002 ... over a 3000-image run.
        for mat in [m for m in bpy.data.materials if m.users == 0]:
            bpy.data.materials.remove(mat)
        for mesh_data in [m for m in bpy.data.meshes if m.users == 0]:
            bpy.data.meshes.remove(mesh_data)

        # ── Static scene of the hard condition (never labelled: class id -1) ──
        bg = spec["background"]
        static = hp.static_elements(cfg, bg["albedo_rgb"], float(bg["z_mm"]))
        for el in static:
            plane = flat_polygon(f"{el['name']}_{idx}", el["polygon_mm"], el["z_mm"])
            plane.replace_materials(matte(f"{el['name']}_mat_{idx}", el["albedo_rgb"]))
            plane.set_cp("custom_class_id", -1)

        # ── Workpieces ──
        instances: dict[int, int] = {}
        for obj_spec in spec["objects"]:
            mesh = import_mesh(repo_root / obj_spec["mesh"])
            # The STL is modelled in mm, so the mm→m factor is baked into the pose
            # matrix. A separate set_scale() call is silently undone here:
            # set_local2world_mat writes the whole matrix, whose linear part carries
            # scale 1, and the mesh stays 1000x too large — large enough to swallow
            # the camera, which renders as a black frame with no object in it.
            T = _pose_from_spec(obj_spec["xyz_mm"], obj_spec["yaw_deg"])
            T[:3, :3] = T[:3, :3] * MM
            mesh.set_local2world_mat(T)
            place_on_surface(mesh, float(obj_spec["xyz_mm"][2]) * MM)
            mesh.set_cp("custom_class_id", int(obj_spec["class_id"]))
            # Without this the workpiece renders in default grey, whatever the
            # sampler drew: an STL carries no material of its own.
            ms = obj_spec.get("material")
            if ms:
                mesh.replace_materials(product_material(f"mat_{idx}_{obj_spec['class_name']}", ms))

        # ── Distractors (never labelled: class id stays -1) ──
        for d in spec.get("distractors", []):
            shape = {"cube": "CUBE", "sphere": "SPHERE", "cylinder": "CYLINDER"}[d["shape"]]
            size_m = float(d["size_mm"]) * MM
            prim = bproc.object.create_primitive(shape, scale=[size_m / 2.0] * 3)
            T = _pose_from_spec(d["xyz_mm"], d.get("yaw_deg", 0.0))
            prim.set_local2world_mat(T)
            place_on_surface(prim, float(d["xyz_mm"][2]) * MM)
            dmat = bproc.material.create(f"distractor_{idx}")
            dmat.set_principled_shader_value(
                "Base Color", [float(v) for v in sm.srgb_to_linear(d["albedo_rgb"])] + [1.0])
            dmat.set_principled_shader_value("Specular IOR Level",
                                             DIELECTRIC_SPECULAR_LEVEL * reflectance_scale)
            prim.replace_materials(dmat)
            prim.set_cp("custom_class_id", -1)

        # ── Light: area luminaires (facing down, Blender's default) and the overhead ──
        for li in spec["lights"]:
            if li.get("type") != "area":
                raise ValueError(f"{spec_path.name}: light type {li.get('type')!r}; the hard "
                                 "profile renders area luminaires only")
            light = bproc.types.Light()
            light.set_type("AREA")
            light.blender_obj.data.shape = "RECTANGLE"
            light.blender_obj.data.size = float(li["size_mm"][0]) * MM
            light.blender_obj.data.size_y = float(li["size_mm"][1]) * MM
            light.set_location(np.asarray(li["position_mm"], dtype=float) * MM)
            light.set_energy(float(li["energy_w"]))
            light.set_color([float(v) for v in sm.light_rgb(li["color_temp_k"], cfg["white_k"])])
        world = [0.0, 0.0, 0.0] if environment is None else environment["radiance_rgb"]
        bproc.renderer.set_world_background([float(v) for v in world], strength=1.0)

        # ── Camera (OpenCV convention in spec → Blender) ──
        cam = spec["camera"]
        intr = cam["intrinsics"]
        K = np.array([
            [intr["fx"], 0.0, intr["ppx"]],
            [0.0, intr["fy"], intr["ppy"]],
            [0.0, 0.0, 1.0],
        ])
        W, H = int(intr["width"]), int(intr["height"])
        bproc.camera.set_intrinsics_from_K_matrix(K, W, H)
        T_cv_m = np.asarray(cam["T_BC_mm"], dtype=float)
        T_cv_m[:3, 3] *= MM
        bproc.camera.add_camera_pose(_cv_to_blender_cam(T_cv_m))

        # ── Render + write ──
        # Segmentation is enabled HERE, once the scene exists. BlenderProc wires the
        # segmentation AOV into the materials that are present at the moment of this
        # call, so a mesh or a material created afterwards still renders in colour but
        # is absent from BOTH segmentation maps — every frame then yields an empty
        # label file, with nothing in the logs to say so.
        bproc.renderer.enable_segmentation_output(map_by=["instance", "custom_class_id"],
                                                  default_values={"custom_class_id": -1})
        data = bproc.renderer.render()
        linear = np.asarray(data["colors"][0], dtype=np.float32)[..., :3]
        inst = np.asarray(data["instance_segmaps"][0])
        for attr in data["instance_attribute_maps"][0]:
            cls = int(attr.get("custom_class_id", -1))
            if cls >= 0:
                instances[int(attr["idx"])] = cls
        noise_rng = np.random.default_rng([int(spec.get("seed", 0)), idx, NOISE_STREAM_TAG])
        rgb = sm.apply_sensor(linear, cfg["sensor"], noise_rng, exposure_gain=exposure_gain)

        stem = f"frame_{idx:06d}"
        Image.fromarray(rgb).save(out_dir / f"{stem}_rgb.png")
        Image.fromarray(inst.astype(np.uint16)).save(out_dir / f"{stem}_inst.png")
        if args.save_linear:
            np.save(out_dir / f"{stem}_linear.npy", linear)
        with (out_dir / f"{stem}_meta.json").open("w", encoding="utf-8") as f:
            json.dump({"instances": {str(k): v for k, v in instances.items()},
                       "scene": spec_path.name, "exposure_gain": exposure_gain}, f, indent=2)
        # A scene holding more lights or meshes than its spec asked for means the
        # per-scene reset leaked, and every later frame is lit by the frames before
        # it. That went unnoticed once already, so it stops the run now.
        n_lights = len([o for o in bpy.data.objects if o.type == "LIGHT"])
        n_meshes = len(bproc.object.get_all_mesh_objects())
        want_meshes = len(static) + len(spec["objects"]) + len(spec.get("distractors", []))
        if n_lights != len(spec["lights"]) or n_meshes != want_meshes:
            raise RuntimeError(
                f"{stem}: scene has {n_lights} lights and {n_meshes} meshes, but the "
                f"spec asks for {len(spec['lights'])} and {want_meshes} — the "
                f"per-scene reset leaked objects from an earlier frame.")
        rendered.append(stem)
        write_manifest(rendered)
        print(f"[{stem}] objects={len(spec['objects'])} "
              f"distractors={len(spec.get('distractors', []))} lights={n_lights}")

    print(f"Rendered {len(spec_files)} frames → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
