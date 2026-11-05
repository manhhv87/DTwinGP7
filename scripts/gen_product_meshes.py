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

GÓC BO CỦA HỘP KIM LOẠI
-----------------------
metal_box là hộp thiếc có bốn góc đứng bo tròn (nhìn từ trên). Bán kính 28,8 mm
ĐO trên nhãn YOLO-seg của tập train (06/11/2026): mọi ảnh std/metal_box của
_work/splits/train.txt có đúng một đa giác metal_box nằm trọn trong khung (68
ảnh); mỗi đa giác khớp một hình chữ nhật bo góc trong mặt phẳng ảnh (tâm, góc,
dài l, rộng w, bán kính rho, tính bằng px; khoảng cách từ đỉnh nhãn tới đường
bao, soft-L1), rồi r = rho x 135 / w và rho x 190 / l, lấy trung bình hai cách.
Cách này không cần chiều cao camera của các buổi chụp tháng 8 (trước hiệu chuẩn
18/09). Giữ 45 ảnh có l / w lệch 190 / 135 dưới 4 %: trung vị 28,8 mm, bootstrap
95 % 27,1-29,5, sai lệch đỉnh trung vị 0,62 px (script và số: scratchpad
synth_fix/geom, an/corner_fit2d.py). Nắp phẳng: ảnh độ sâu của lượt gắp E2 khó
cho mặt trên cao 83,1 mm ở giữa và 83,7-84,5 mm sát mép, không có độ phồng hay
mép bo đo được, nên mặt trên và mặt đáy phẳng, cạnh trên vuông.

Mỗi góc bo chia 16 đoạn (5,6 độ; mặt cong lệch cung tròn 0,04 mm). Renderer tô
mượt các mặt đứng có pháp tuyến lệch trục X, Y của mesh (dải góc bo) và tô phẳng
mọi mặt khác (render_blenderproc.import_mesh), nên mặt trên vẫn phẳng.

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

# Bán kính góc đứng bo tròn (mm, nhìn từ trên); vật không có tên ở đây là hộp góc vuông.
CORNER_RADIUS_MM: dict[str, float] = {
    "metal_box": 28.8,      # đo trên nhãn train, xem đầu file
}
CORNER_SEGMENTS = 16        # số đoạn của mỗi góc bo 90 độ

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


def rounded_outline(sx: float, sy: float, r: float, n_arc: int = CORNER_SEGMENTS):
    """Đường bao nhìn từ trên của hình chữ nhật sx x sy bo bốn góc bán kính r, ngược chiều kim
    đồng hồ, mỗi góc n_arc đoạn (n_arc + 1 điểm, kể cả hai đầu nối với cạnh thẳng)."""
    import math
    if not 0.0 < r < min(sx, sy) / 2.0:
        raise ValueError(f"bán kính góc {r} phải nằm trong (0, {min(sx, sy) / 2.0})")
    x, y = sx / 2.0 - r, sy / 2.0 - r
    pts = []
    for cx, cy, a0 in ((x, y, 0.0), (-x, y, 90.0), (-x, -y, 180.0), (x, -y, 270.0)):
        for k in range(n_arc + 1):
            a = math.radians(a0 + 90.0 * k / n_arc)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def rounded_box_triangles(sx: float, sy: float, sz: float, r: float):
    """Tam giác của hộp có bốn góc đứng bo bán kính r: giữa theo X/Y, đáy ở z = 0, pháp tuyến
    hướng ra ngoài. Mặt trên, mặt đáy là quạt tam giác quanh tâm; mỗi đoạn của đường bao cho
    một mặt đứng (hai tam giác)."""
    import math
    ring = rounded_outline(sx, sy, r)
    n = len(ring)
    top_c, bot_c = (0.0, 0.0, sz), (0.0, 0.0, 0.0)
    for i in range(n):
        (ax, ay), (bx, by) = ring[i], ring[(i + 1) % n]
        yield (0.0, 0.0, 1.0), (top_c, (ax, ay, sz), (bx, by, sz))
        yield (0.0, 0.0, -1.0), (bot_c, (bx, by, 0.0), (ax, ay, 0.0))
        ex, ey = bx - ax, by - ay
        ln = math.hypot(ex, ey)
        nrm = (ey / ln, -ex / ln, 0.0)            # ngoài, vì đường bao ngược chiều kim đồng hồ
        a0, b0, b1, a1 = (ax, ay, 0.0), (bx, by, 0.0), (bx, by, sz), (ax, ay, sz)
        yield nrm, (a0, b0, b1)
        yield nrm, (a0, b1, a1)


def write_stl(path: Path, sx: float, sy: float, sz: float, r: float = 0.0) -> int:
    tris = list(rounded_box_triangles(sx, sy, sz, r) if r > 0.0 else box_triangles(sx, sy, sz))
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
        r = CORNER_RADIUS_MM.get(name, 0.0)
        new = "%.0f×%.0f×%.0f" % (sx, sy, sz) + (" góc r %.1f" % r if r else "")
        if dry:
            print("%-14s %-22s %-22s %s" % (name, old, new, "(dry-run)"))
            continue
        n = write_stl(p, sx, sy, sz, r)
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
