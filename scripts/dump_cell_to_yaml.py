#!/usr/bin/env python
"""
dump_cell_to_yaml.py
────────────────────
Helper: đọc trạng thái cell hiện tại trong RoboDK GUI, xuất ra YAML.

Dùng khi prototyping:
    1. Kéo thả tay trong RoboDK GUI tạo cell ưng ý
    2. Chạy script này → ra file `current_cell_dump.yaml`
    3. Copy các tọa độ relevant vào config/cell_layout.yaml

Lưu ý: Output là YAML "thô" liệt kê mọi item — không phải file config chuẩn.
Bạn cần biên tập lại để match schema CellConfig.

Usage:
    python scripts/dump_cell_to_yaml.py
    python scripts/dump_cell_to_yaml.py --output my_cell.yaml
    python scripts/dump_cell_to_yaml.py --filter-types Frame Tool
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from src.cell.pose_utils import matrix_to_rpy_xyz, robodk_pose_to_matrix  # noqa: E402

logger = logging.getLogger("dump_cell")


# RoboDK item type constants (from robolink)
ITEM_TYPE_MAP = {
    1: "Station",
    2: "Robot",
    3: "Frame",
    4: "Tool",
    5: "Object",
    6: "Target",
    7: "Program",
    8: "Instruction",
    9: "Python",
    10: "Machining",
    11: "BallbarValidation",
    12: "Calibration",
    13: "ValidIsoCube",
    14: "Camera",
    15: "GenericItem",
}


def get_item_type_name(item) -> str:
    """Lấy tên loại item (Robot, Frame, ...)."""
    try:
        t = item.Type()
        return ITEM_TYPE_MAP.get(t, f"Unknown({t})")
    except Exception:
        return "Unknown"


def dump_item(item) -> Dict:
    """Dump 1 item → dict."""
    item_type = get_item_type_name(item)
    data = {
        "name": item.Name(),
        "type": item_type,
    }

    # Pose (chỉ với items có pose)
    if item_type in ("Robot", "Frame", "Tool", "Object", "Target", "Camera"):
        try:
            pose_mat = robodk_pose_to_matrix(item.Pose())
            xyz, rpy = matrix_to_rpy_xyz(pose_mat)
            data["pose"] = {
                "xyz_mm": [round(v, 3) for v in xyz],
                "rpy_deg": [round(v, 3) for v in rpy],
            }
        except Exception as e:
            data["pose_error"] = str(e)

    # Joints (robot)
    if item_type == "Robot":
        try:
            joints = item.Joints().tolist()
            if isinstance(joints[0], list):
                joints = joints[0]
            data["current_joints_deg"] = [round(j, 3) for j in joints]
        except Exception:
            pass

    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump RoboDK station state → YAML for prototyping."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="current_cell_dump.yaml",
        help="Output YAML file (default: current_cell_dump.yaml)",
    )
    parser.add_argument(
        "--filter-types",
        nargs="*",
        default=None,
        help="Chỉ dump items có type này (vd: --filter-types Robot Frame)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    try:
        from robodk.robolink import Robolink
        RDK = Robolink()
    except Exception as e:
        logger.error("Không kết nối được tới RoboDK: %s", e)
        logger.error("Đảm bảo RoboDK GUI đang mở!")
        return 1

    items_raw = RDK.ItemList()
    logger.info("Tìm thấy %d items trong station", len(items_raw))

    if not items_raw:
        logger.warning("Station rỗng — không có gì để dump")
        return 0

    # Dump
    dumped: List[Dict] = []
    for item in items_raw:
        try:
            data = dump_item(item)
            if args.filter_types and data["type"] not in args.filter_types:
                continue
            dumped.append(data)
            logger.info("  [%s] %s", data["type"], data["name"])
        except Exception as e:
            logger.warning("Skip item: %s", e)

    # Write
    output_data = {
        "metadata": {
            "dumped_at": datetime.now().isoformat(),
            "source": "RoboDK Software GUI",
            "total_items": len(dumped),
            "note": (
                "Đây là dump thô. Để dùng làm config CellLoader, "
                "biên tập theo schema CellConfig (xem cell_layout.yaml example)."
            ),
        },
        "items": dumped,
    }

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info("✓ Dumped %d items → %s", len(dumped), output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
