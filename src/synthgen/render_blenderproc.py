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
        OpenCV camera convention: +Z looks forward/down).
Output per frame (contract consumed by auto_label.convert_render_output):
    frame_<idx>_rgb.png     rendered RGB
    frame_<idx>_inst.png    instance map (uint16 PNG; 0 = background)
    frame_<idx>_meta.json   {"instances": {inst_id: class_id}, "scene": <spec file>}

Design notes:
  - mm → m conversion happens HERE (Blender works in meters).
  - OpenCV cam → Blender cam: right-multiply by diag(1,-1,-1) (flip Y,Z axes).
  - Meshes are placed so their bounding-box bottom rests on the table plane
    (the sampler provides table-top z; STL origins are arbitrary).
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


def _kelvin_to_rgb(temp_k: float) -> tuple[float, float, float]:
    """Approximate blackbody colour (Tanner Helland fit), normalized 0-1."""
    t = max(1000.0, min(12000.0, float(temp_k))) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * np.log(t) - 161.1195681661
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
    b = 255.0 if t >= 66 else (
        0.0 if t <= 19 else 138.5177312231 * np.log(t - 10.0) - 305.0447927307
    )
    rgb = np.clip([r, g, b], 0.0, 255.0) / 255.0
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


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
    parser.add_argument("--samples", type=int, default=64,
                        help="Cycles samples/px (16 for smoke test, 64+ for real)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Render only the first N specs (0 = all)")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto",
                        help="Render device. 'gpu' REQUIRES a usable GPU and stops if "
                             "there is none, so a batch meant for the GPU machine "
                             "cannot quietly spend hours on its CPU instead. "
                             "'auto' leaves the choice to BlenderProc.")
    args = parser.parse_args()

    scenes_dir = Path(args.scenes)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent.parent

    spec_files = sorted(scenes_dir.glob("scene_*.json"))
    if args.limit > 0:
        spec_files = spec_files[: args.limit]
    if not spec_files:
        print(f"No scene_*.json in {scenes_dir}", file=sys.stderr)
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
    print(f"Render device: {bpy.context.scene.cycles.device} (--device {args.device})")
    bproc.renderer.set_max_amount_of_samples(args.samples)

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
        return bproc.python.types.MeshObjectUtility.MeshObject(new[0])

    def place_on_surface(obj, z_table_m: float) -> None:
        """Shift the object so its bounding-box bottom sits on the table plane."""
        bb = np.array(obj.get_bound_box())          # 8x3 world coords
        dz = z_table_m - float(bb[:, 2].min())
        loc = np.array(obj.get_location())
        obj.set_location(loc + np.array([0.0, 0.0, dz]))

    for spec_path in spec_files:
        with spec_path.open("r", encoding="utf-8") as f:
            spec = json.load(f)
        idx = int(spec["index"])
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
        # Materials are created per scene too and would otherwise pile up as
        # bg_mat.001, bg_mat.002 ... over a 3000-image run.
        for mat in [m for m in bpy.data.materials if m.users == 0]:
            bpy.data.materials.remove(mat)

        # ── Background plane ──
        bg = spec["background"]
        z_table_m = float(bg["z_mm"]) * MM
        plane = bproc.object.create_primitive("PLANE", scale=[2.0, 2.0, 1.0])
        plane.set_location([0.7, 0.0, z_table_m])   # centred on the worktable
        mat = bproc.material.create("bg_mat")
        mat.set_principled_shader_value("Base Color", list(bg["albedo_rgb"]) + [1.0])
        mat.set_principled_shader_value("Roughness", float(bg["roughness"]))
        plane.replace_materials(mat)
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
            place_on_surface(mesh, z_table_m)
            mesh.set_cp("custom_class_id", int(obj_spec["class_id"]))
            # Without this the workpiece renders in default grey, whatever the
            # sampler drew: an STL carries no material of its own.
            ms = obj_spec.get("material")
            if ms:
                omat = bproc.material.create(f"mat_{idx}_{obj_spec['class_name']}")
                omat.set_principled_shader_value(
                    "Base Color", list(ms["albedo_rgb"]) + [1.0])
                omat.set_principled_shader_value("Roughness", float(ms["roughness"]))
                omat.set_principled_shader_value("Metallic", float(ms["metallic"]))
                mesh.replace_materials(omat)

        # ── Distractors (never labelled: class id stays -1) ──
        for d in spec.get("distractors", []):
            shape = {"cube": "CUBE", "sphere": "SPHERE", "cylinder": "CYLINDER"}[d["shape"]]
            size_m = float(d["size_mm"]) * MM
            prim = bproc.object.create_primitive(shape, scale=[size_m / 2.0] * 3)
            T = _pose_from_spec(d["xyz_mm"], d.get("yaw_deg", 0.0))
            prim.set_local2world_mat(T)
            place_on_surface(prim, z_table_m)
            dmat = bproc.material.create(f"distractor_{idx}")
            dmat.set_principled_shader_value(
                "Base Color", list(d["albedo_rgb"]) + [1.0])
            prim.replace_materials(dmat)
            prim.set_cp("custom_class_id", -1)

        # ── Lights ──
        for li in spec["lights"]:
            light = bproc.types.Light()
            light.set_type("POINT")
            light.set_location(np.asarray(li["position_mm"], dtype=float) * MM)
            light.set_energy(float(li["energy_w"]))
            light.set_color(_kelvin_to_rgb(li["color_temp_k"]))

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
        rgb = np.asarray(data["colors"][0], dtype=np.uint8)
        inst = np.asarray(data["instance_segmaps"][0])
        for attr in data["instance_attribute_maps"][0]:
            cls = int(attr.get("custom_class_id", -1))
            if cls >= 0:
                instances[int(attr["idx"])] = cls

        stem = f"frame_{idx:06d}"
        Image.fromarray(rgb).save(out_dir / f"{stem}_rgb.png")
        Image.fromarray(inst.astype(np.uint16)).save(out_dir / f"{stem}_inst.png")
        with (out_dir / f"{stem}_meta.json").open("w", encoding="utf-8") as f:
            json.dump({"instances": {str(k): v for k, v in instances.items()},
                       "scene": spec_path.name}, f, indent=2)
        # A scene holding more lights or meshes than its spec asked for means the
        # per-scene reset leaked, and every later frame is lit by the frames before
        # it. That went unnoticed once already, so it stops the run now.
        n_lights = len([o for o in bpy.data.objects if o.type == "LIGHT"])
        n_meshes = len(bproc.object.get_all_mesh_objects())
        want_meshes = 1 + len(spec["objects"]) + len(spec.get("distractors", []))
        if n_lights != len(spec["lights"]) or n_meshes != want_meshes:
            raise RuntimeError(
                f"{stem}: scene has {n_lights} lights and {n_meshes} meshes, but the "
                f"spec asks for {len(spec['lights'])} and {want_meshes} — the "
                f"per-scene reset leaked objects from an earlier frame.")
        print(f"[{stem}] objects={len(spec['objects'])} "
              f"distractors={len(spec.get('distractors', []))} lights={n_lights}")

    print(f"Rendered {len(spec_files)} frames → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
