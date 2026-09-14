#!/usr/bin/env python
"""
gen_product_meshes.py
─────────────────────
Sinh mesh hộp cho các vật của bộ dữ liệu paper, theo KÍCH THƯỚC ĐO BẰNG THƯỚC KẸP.

NGUỒN SỐ ĐO
-----------
Người dùng đo bằng thước kẹp ngày 2026-09-11 (dài × rộng × cao, mm):

    carton (2 cỡ)  180×120×120   và   180×100× 80
    plastic_box    180×140× 30
    wood_box       190×190×150
    metal_box      190×135× 85
    inox_box       180× 85× 40

Lớp carton gồm HAI hộp thật khác cỡ nên có hai mesh, carton_large và
carton_small, cùng nhãn lớp carton (xem config/synthgen.yaml, mục objects).

VÌ SAO BỎ SỐ ĐO TỪ DEPTH
------------------------
Bản trước đo từ ảnh RGB-D và quy đổi cạnh ngang bằng tiêu cự suy từ FOV 69°. Đó
là FOV màu của D435, không phải D455, nên mọi cạnh ngang bị co khoảng 1,4 lần
(ví dụ metal 139×99 thay vì 190×135). Chiều cao lấy từ độ sâu nên vẫn đúng, lệch
thước kẹp 1 đến 5 mm, và đó là phép kiểm chéo cho dữ liệu depth. Hai đỉnh chiều
cao 82 và 125 mm của carton từng bị hiểu là một hộp đặt trên hai mặt; thực ra là
hai hộp khác cỡ.

VÌ SAO HÌNH HỌC PHẢI ĐÚNG
-------------------------
Chế độ plane của C4 (postprocess.z_from_ground_plane) đặt mặt trên vật ở
Z_bàn + h, với h lấy từ mesh; sai h bao nhiêu thì điểm kẹp sai bấy nhiêu theo
phương đứng. Hàm kẹp mở từ 180 mm (đóng hết) tới 216 mm (mở hết), nên mỗi vật chỉ
kẹp được theo cạnh 180 đến 190 mm của nó, và khe hở mỗi bên khi hạ xuống chỉ
13 đến 18 mm.

Chạy:
    python scripts/gen_product_meshes.py            # ghi models/objects/*.stl
    python scripts/gen_product_meshes.py --dry-run
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "models" / "objects"

# tên → (X dài, Y ngắn, Z cao) mm.  Trục khớp mesh cũ: giữa theo X/Y, đáy ở Z = 0.
PRODUCTS: dict[str, tuple[float, float, float]] = {
    "carton_large": (180.0, 120.0, 120.0),
    "carton_small": (180.0, 100.0,  80.0),
    "plastic_box":  (180.0, 140.0,  30.0),
    "wood_box":     (190.0, 190.0, 150.0),
    "metal_box":    (190.0, 135.0,  85.0),
    "inox_box":     (180.0,  85.0,  40.0),
}

# Mesh cũ không còn đúng với vật nào: carton.stl một cỡ, kích thước ngang sai.
STALE = ("carton",)


def box_triangles(sx: float, sy: float, sz: float):
    """12 tam giác của hình hộp: giữa theo X/Y, đáy ở z=0. Pháp tuyến hướng ra ngoài."""
    x, y = sx / 2.0, sy / 2.0
    v = [(-x, -y, 0.0), (x, -y, 0.0), (x, y, 0.0), (-x, y, 0.0),
         (-x, -y, sz), (x, -y, sz), (x, y, sz), (-x, y, sz)]
    quads = [((0, 3, 2, 1), (0, 0, -1)),      # đáy
             ((4, 5, 6, 7), (0, 0, 1)),       # đỉnh
             ((0, 1, 5, 4), (0, -1, 0)),
             ((1, 2, 6, 5), (1, 0, 0)),
             ((2, 3, 7, 6), (0, 1, 0)),
             ((3, 0, 4, 7), (-1, 0, 0))]
    for (a, b, c, d), n in quads:
        yield n, (v[a], v[b], v[c])
        yield n, (v[a], v[c], v[d])


def write_stl(path: Path, sx: float, sy: float, sz: float) -> int:
    tris = list(box_triangles(sx, sy, sz))
    buf = bytearray(b"\x00" * 80)                 # header rỗng, khớp mesh cũ
    buf += struct.pack("<I", len(tris))
    for n, (p, q, r) in tris:
        buf += struct.pack("<3f", *n)
        for pt in (p, q, r):
            buf += struct.pack("<3f", *pt)
        buf += struct.pack("<H", 0)
    path.write_bytes(bytes(buf))
    return len(buf)


def main(dry: bool) -> None:
    print("%-14s %-22s %-22s %s" % ("lớp", "mesh cũ", "mesh mới", "trạng thái"))
    for name, (sx, sy, sz) in PRODUCTS.items():
        p = OUT_DIR / f"{name}.stl"
        old = "(chưa có)"
        if p.exists():
            b = p.read_bytes()
            nt = struct.unpack("<I", b[80:84])[0]
            import numpy as np
            vv = np.concatenate([np.frombuffer(b[84 + i * 50 + 12:84 + i * 50 + 48],
                                               "<f4").reshape(3, 3) for i in range(nt)])
            e = vv.max(0) - vv.min(0)
            old = "%.0f×%.0f×%.0f" % (e[0], e[1], e[2])
        new = "%.0f×%.0f×%.0f" % (sx, sy, sz)
        if dry:
            print("%-14s %-22s %-22s %s" % (name, old, new, "(dry-run)"))
            continue
        n = write_stl(p, sx, sy, sz)
        print("%-14s %-22s %-22s ghi %d byte" % (name, old, new, n))
    for name in STALE:
        p = OUT_DIR / f"{name}.stl"
        if p.exists():
            if dry:
                print("%-14s sẽ bị xoá (đã thay bằng carton_large và carton_small)" % name)
            else:
                p.unlink()
                print("%-14s đã xoá (đã thay bằng carton_large và carton_small)" % name)
    if dry:
        print("\n(dry-run — chưa ghi gì)")
    else:
        print("\n-> %s" % OUT_DIR)
        print("Nhớ đồng bộ: config/synthgen.yaml (classes + class_ids),")
        print("             src/synthgen/auto_label.py DEFAULT_CLASS_NAMES,")
        print("             src/utils/dataset_check.py EXPECTED_NAMES.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().dry_run)
