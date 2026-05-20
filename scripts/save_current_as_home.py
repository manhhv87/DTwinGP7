#!/usr/bin/env python
"""
save_current_as_home.py
───────────────────────
Lưu joints hiện tại của robot trong RoboDK GUI làm `home_joints_deg`
trong cell_layout.yaml + cell_layout_real.yaml.

Workflow:
  1. Mở RoboDK + chạy `build_station.py` để có robot trong station
  2. Trong RoboDK GUI: kéo thanh trượt joint hoặc nhập số trực tiếp
     (Robot panel → Joint angles) để move robot tới tư thế bạn muốn làm home
  3. Chạy script này — `robot.Joints()` được đọc và ghi vào YAML:
       python scripts/save_current_as_home.py
  4. Chạy lại `build_station.py` để áp dụng (mỗi lần build robot move về
     home mới này).

Tốn ~2 API call → an toàn RoboDK Free quota.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIGS = ["config/cell_layout.yaml", "config/cell_layout_real.yaml"]


def _joints_to_list(joints):
    """Convert RoboDK Mat → flat list of 6 floats. None nếu không parse được."""
    if joints is None:
        return None
    try:
        data = joints.tolist() if hasattr(joints, "tolist") else list(joints)
    except Exception:
        return None
    if not data:
        return None
    # Flat: [j1, j2, ..., j6]
    if isinstance(data[0], (int, float)) and len(data) >= 6:
        return [float(j) for j in data[:6]]
    # Nested 1xN: [[j1, j2, ..., j6]]
    if isinstance(data[0], list) and len(data) == 1 and len(data[0]) >= 6:
        return [float(j) for j in data[0][:6]]
    # Nested Nx1: [[j1], [j2], ..., [j6]]
    if isinstance(data[0], list) and len(data) >= 6 and all(
        isinstance(row, list) and len(row) >= 1 for row in data[:6]
    ):
        return [float(row[0]) for row in data[:6]]
    return None


def _write_home(path: Path, joints: list[float]) -> bool:
    """Thay dòng `home_joints_deg:` trong file YAML (preserve other lines)."""
    if not path.exists():
        print(f"  ⚠ Bỏ qua (không có file): {path.name}")
        return False
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    val = "[" + ", ".join(f"{j:g}" for j in joints) + "]"
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        if stripped.startswith("home_joints_deg:"):
            indent = ln[: len(ln) - len(stripped)]
            eol = "\n" if ln.endswith("\n") else ""
            lines[i] = (
                f"{indent}home_joints_deg: {val}"
                f"   # capture từ RoboDK GUI (save_current_as_home.py){eol}"
            )
            path.write_text("".join(lines), encoding="utf-8")
            print(f"  ✓ {path.name}")
            return True
    print(f"  ⚠ Không tìm thấy 'home_joints_deg:' trong {path.name}")
    return False


def main() -> int:
    # Console Windows cp1252 → ép UTF-8 để in ✓ + tiếng Việt.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    try:
        from robodk.robolink import ITEM_TYPE_ROBOT, Robolink
    except ImportError:
        print("✗ Chưa cài robodk: pip install robodk")
        return 1

    rdk = Robolink()
    robots = rdk.ItemList(ITEM_TYPE_ROBOT)
    if not robots:
        print("✗ Không tìm thấy robot trong RoboDK.")
        print("  → Chạy scripts/build_station.py trước.")
        return 1

    robot = robots[0]
    joints = _joints_to_list(robot.Joints())
    if joints is None:
        print("✗ Không đọc được joints từ robot.")
        return 1

    jrounded = [round(j, 2) for j in joints]
    print("=" * 60)
    print(f" Joints hiện tại của robot:")
    print(f"   {jrounded}")
    print("=" * 60)
    print("Ghi vào config:")
    for cfg in CONFIGS:
        _write_home(PROJECT_ROOT / cfg, jrounded)
    print("\nXong. Chạy lại 'python scripts/build_station.py' để áp dụng.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
