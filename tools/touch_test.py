#!/usr/bin/env python
"""touch_test.py — hand-eye touch test with the printed ChArUco board (READ-ONLY).

The calibration says where the camera is; this checks it against the robot, which is
the only thing in the cell that knows exactly where it is. Lay the ChArUco board flat
on the empty table. The tool grabs one frame, detects the chessboard corners, and
turns eight of them into robot base-frame X, Y through the calibrated camera pose and
the measured table plane. It draws those eight corners, numbered, on a snapshot. Then,
one by one, jog the tip of a pointer clamped upright between the jaws (TOOL01 active,
translation only, so the pendant X, Y is the tip's) onto each drawn corner and type the X and Y the
pendant shows (COORD = Robot). Camera-predicted minus pendant-measured is the touch-test
error the paper reports as mean / RMS / max.

Sends nothing to the robot. Needs config/calibration/T_base_camera.npy and
table_plane.json, and TOOL01 registered on the pendant. The board must be OFF the
gripper and lying on the table.

Usage:
    python tools/touch_test.py --square-mm <measured square> --marker-mm <measured marker>
    python tools/touch_test.py --square-mm 45 --marker-mm 34 --image results/board.png --no-prompt

The board thickness matters: its top face is above the table plane by that much, and
the ray meets the lifted plane. Default 4 mm, the board of this cell with calipers; pass
your own measurement with --board-thickness-mm.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.depth_modes import TablePlane, _ray, ray_plane_depth  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--square-mm", type=float, required=True,
                   help="Measured side of one chessboard square of the PRINTED board.")
    p.add_argument("--marker-mm", type=float, required=True,
                   help="Measured side of one ArUco marker of the printed board.")
    p.add_argument("--squares", type=int, nargs=2, default=(7, 5), metavar=("COLS", "ROWS"))
    p.add_argument("--dict", default="DICT_4X4_50")
    p.add_argument("--board-thickness-mm", type=float, default=4.0,
                   help="Top face of the board above the table. Default 4, the board of "
                        "this cell measured with calipers; measure your own.")
    p.add_argument("--points", type=int, default=8,
                   help="How many corners to touch (spread over the board). Default 8.")
    p.add_argument("--calib-dir", default="config/calibration")
    p.add_argument("--image", default=None,
                   help="Use this image instead of the camera (needs --intrinsics).")
    p.add_argument("--intrinsics", default=None,
                   help="JSON with fx, fy, ppx, ppy (e.g. T_base_camera_meta.json) when "
                        "--image is given. Default: the meta file next to the calibration.")
    p.add_argument("--no-prompt", action="store_true",
                   help="Only predict and draw; do not ask for pendant readings.")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def resolve(path: str) -> Path:
    q = Path(path)
    return q if q.is_absolute() else PROJECT_ROOT / q


def pixel_to_base(intr: dict, T_BC: np.ndarray, plane: TablePlane, u: float, v: float,
                  height_mm: float) -> np.ndarray | None:
    """Base-frame point where the ray through (u, v) meets the table plane lifted by height_mm."""
    depth_m = ray_plane_depth(intr, u, v, T_BC, plane, height_mm=height_mm)
    if depth_m is None:
        return None
    t, r = _ray(intr, u, v, T_BC)
    return t + r * (depth_m * 1000.0)


def pick_spread(ids: np.ndarray, pts: np.ndarray, k: int) -> list[int]:
    """Indices of k corners spread over the detected set.

    The four extreme corners come first (they have the longest lever arm, so a
    rotation error in the calibration shows there), then farthest-point sampling
    fills the rest.
    """
    n = len(ids)
    if n <= k:
        return list(range(n))
    s, dfx = pts[:, 0] + pts[:, 1], pts[:, 0] - pts[:, 1]
    chosen: list[int] = []
    for i in (np.argmin(s), np.argmax(s), np.argmin(dfx), np.argmax(dfx)):
        if int(i) not in chosen and len(chosen) < k:
            chosen.append(int(i))
    d = np.min([np.linalg.norm(pts - pts[c], axis=1) for c in chosen], axis=0)
    while len(chosen) < k:
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(pts - pts[nxt], axis=1))
    return chosen


def main() -> int:
    args = parse_args()
    calib_dir = resolve(args.calib_dir)
    T_BC = np.load(calib_dir / "T_base_camera.npy")
    plane = TablePlane.from_dict(json.loads((calib_dir / "table_plane.json").read_text(encoding="utf-8")))

    if args.image:
        img = cv2.imread(str(resolve(args.image)))
        if img is None:
            print(f"cannot read {args.image}", file=sys.stderr)
            return 2
        meta = resolve(args.intrinsics) if args.intrinsics else calib_dir / "T_base_camera_meta.json"
        intr = json.loads(meta.read_text(encoding="utf-8"))
        intr = intr.get("intrinsics", intr)
    else:
        from src.perception.camera import D455Camera
        cam = D455Camera()
        try:
            rgb, _ = cam.get_frame()
        finally:
            cam.stop()
        if rgb is None:
            print("no frame from the camera", file=sys.stderr)
            return 2
        img = np.ascontiguousarray(rgb)
        intr = cam.intrinsics

    aruco = cv2.aruco
    board = aruco.CharucoBoard(tuple(args.squares), args.square_mm / 1000.0,
                               args.marker_mm / 1000.0,
                               aruco.getPredefinedDictionary(getattr(aruco, args.dict)))
    detector = aruco.CharucoDetector(board)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < args.points:
        print(f"only {0 if ids is None else len(ids)} chessboard corners found; need at least "
              f"{args.points}. Board fully in view, flat, and lit?", file=sys.stderr)
        return 3
    pts = corners.reshape(-1, 2)
    ids = ids.reshape(-1)
    sel = pick_spread(ids, pts, args.points)

    rows = []
    for n, i in enumerate(sel, start=1):
        u, v = float(pts[i, 0]), float(pts[i, 1])
        p = pixel_to_base(intr, T_BC, plane, u, v, args.board_thickness_mm)
        if p is None:
            continue
        rows.append({"point": n, "corner_id": int(ids[i]), "u_px": round(u, 1), "v_px": round(v, 1),
                     "x_cam_mm": round(float(p[0]), 1), "y_cam_mm": round(float(p[1]), 1)})

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = out_dir / f"touch_test_{ts}.png"
    vis = img.copy()
    for r in rows:
        c = (int(r["u_px"]), int(r["v_px"]))
        cv2.circle(vis, c, 14, (0, 140, 255), 2)
        cv2.putText(vis, str(r["point"]), (c[0] + 16, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 140, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(snap), vis)

    print(f"\n{len(ids)} corners detected; {len(rows)} chosen. Numbered snapshot: {snap}")
    print("Camera prediction (robot base frame, mm):")
    print("  point  corner   u      v      X_cam    Y_cam")
    for r in rows:
        print("  %5d  %6d  %6.1f %6.1f  %7.1f  %7.1f" % (
            r["point"], r["corner_id"], r["u_px"], r["v_px"], r["x_cam_mm"], r["y_cam_mm"]))

    if args.no_prompt:
        return 0

    print("\nOpen the snapshot. For each numbered corner: jog the pointer tip onto that corner,")
    print("read X and Y on the pendant (COORD = Robot), type them as two numbers, ENTER.")
    print("Type s to skip a corner you cannot reach.\n")
    errs = []
    for r in rows:
        while True:
            try:
                ans = input(f"point {r['point']} (camera says X={r['x_cam_mm']:.1f} Y={r['y_cam_mm']:.1f}): ").strip()
            except EOFError:
                ans = "s"
            if ans.lower() == "s":
                r.update(x_robot_mm="", y_robot_mm="", dx_mm="", dy_mm="", err_mm="")
                break
            try:
                x, y = (float(t) for t in ans.replace(",", " ").split())
            except ValueError:
                print("   two numbers, e.g.  575.3 80.1")
                continue
            dx, dy = r["x_cam_mm"] - x, r["y_cam_mm"] - y
            e = float(np.hypot(dx, dy))
            r.update(x_robot_mm=x, y_robot_mm=y, dx_mm=round(dx, 1), dy_mm=round(dy, 1),
                     err_mm=round(e, 1))
            errs.append(e)
            print(f"   error {e:.1f} mm (dx {dx:+.1f}, dy {dy:+.1f})")
            break

    csv_path = out_dir / f"touch_test_{ts}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    if errs:
        a = np.asarray(errs)
        summary = {"n": int(a.size), "mean_mm": round(float(a.mean()), 2),
                   "rms_mm": round(float(np.sqrt((a ** 2).mean())), 2),
                   "max_mm": round(float(a.max()), 2),
                   "bias_dx_mm": round(float(np.mean([r["dx_mm"] for r in rows if r["dx_mm"] != ""])), 2),
                   "bias_dy_mm": round(float(np.mean([r["dy_mm"] for r in rows if r["dy_mm"] != ""])), 2)}
        print(f"\nTouch test over {a.size} points: mean {summary['mean_mm']} mm, "
              f"RMS {summary['rms_mm']} mm, max {summary['max_mm']} mm; "
              f"mean offset dx {summary['bias_dx_mm']:+.2f}, dy {summary['bias_dy_mm']:+.2f} mm")
        print("A mean offset well above the spread means a systematic shift: a tilted or off-centre "
              "pointer, or the "
              "calibration, not noise.")
    (out_dir / f"touch_test_{ts}.json").write_text(json.dumps(
        {"calibration_dir": str(calib_dir), "board": {"squares": list(args.squares),
         "square_mm": args.square_mm, "marker_mm": args.marker_mm, "dict": args.dict,
         "thickness_mm": args.board_thickness_mm}, "points": rows, "summary": summary},
        indent=2), encoding="utf-8")
    print(f"Saved {csv_path.name} and the JSON summary next to it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
