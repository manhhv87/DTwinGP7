# models/ — Trọng số model + mesh

Model YOLOv8-seg được **train trên máy Linux GPU** (việc train nằm ngoài
repo này). Thư mục này chứa trọng số đã train để inference + các file mesh.

## ⭐ File trọng số cần có (cho `--mode real`)

| File | Mô tả | Nguồn |
|---|---|---|
| `yolov8s-seg_best.pt` | Trọng số production (variant `s`) | Copy từ máy train |

`config/experiment.yaml :: model_path` trỏ tới `models/yolov8s-seg_best.pt`.
Đổi giá trị này nếu dùng tên/đường dẫn khác.

> **Sim mode (`--headless` hoặc default `--mode sim`)** dùng `MockDetector` →
> KHÔNG cần file `.pt`. Chỉ cần khi `--mode real` (D455 + GP7 thật).

## ⭐ Đưa trọng số từ máy train về

Sau khi train xong trên máy Linux, copy file trọng số tốt nhất về máy chạy
và đặt đúng tên mà `experiment.yaml` trỏ tới:

```bash
copy best.pt  models\yolov8s-seg_best.pt
```

## ⭐ Định dạng hỗ trợ

`ObjectDetector` nạp được cả `.pt` và `.onnx`. Nếu muốn chạy inference không
cần PyTorch GPU, có thể export ONNX trên máy train (`yolo export model=best.pt
format=onnx`) rồi trỏ `model_path` tới file `.onnx`.

## ⭐ Mesh STL

| Thư mục | Nội dung |
|---|---|
| `models/` | `worktable.stl`, `pedestal.stl`, `gripper.stl`, `floor.stl` |
| `models/gp7_links/` | 7 STL link Yaskawa GP7 (`gp7_base_link.stl` … `gp7_link_6_t.stl`) cho URDF chain |
| `models/objects/` | `tray.stl` (Galaxy S23 use case), `bottle.stl`, `cup.stl`, `bolt.stl` |

> **Lớp vật (class):** `tray/bottle/cup/bolt` là **mặc định**. Danh sách lớp của
> bài toán nay **định nghĩa được** (`CellConfig.object_classes`, sửa qua dock
> Camera → Quản lý… trong app); detection thật lấy tên lớp từ chính model YOLO.
> Xem [`../docs/GIOI_THIEU_PHAN_MEM.md`](../docs/GIOI_THIEU_PHAN_MEM.md) §3.1.

`config/cell_layout.yaml` tham chiếu các đường dẫn mesh này; `O3DGuiSimRobot`
load chúng vào Open3D Filament viewport khi mở (sim non-headless hoặc real
mode mirror). Cùng mesh dùng cho:
- Sim mode (`--mode sim`) — SimRobot animate qua URDF, `tray` là target gắp chính
- Real mode (`--mode real --backend hse`) — Open3D viewport mirror robot thật
  từ HSE joints @2Hz; mesh tĩnh (bàn/pedestal) cộng arm động + gripper rendered

## ⭐ Sinh STL primitive nếu thiếu

Repo có sẵn các STL được commit. Nếu thiếu / muốn sinh lại:

```bash
pip install trimesh
python scripts/gen_primitive_meshes.py                  # tất cả primitives
python scripts/gen_primitive_meshes.py --only gripper   # chỉ 1 file
```

## ⭐ Gripper subsystem (CC-Link) — xem tài liệu khác

Gripper khí nén double-acting điều khiển bằng PLC Mitsubishi; PC giao tiếp qua
**YRC1000 làm CC-Link bridge** (HSE ↔ CC-Link). Đây không phải asset nên không
mô tả chi tiết ở đây — tài liệu đầy đủ (tránh trùng lặp) ở:

- **Thiết kế** (kiến trúc 3 thiết bị / 2 giao thức, sequence diagram, memory map
  5-bit, latency budget): [`../docs/phat_bieu_bai_toan_v3_2_HD.md` §7.9](../docs/phat_bieu_bai_toan_v3_2_HD.md).
- **Setup production** (cấu hình CC-Link master trên YRC TP, PLC ladder bridge,
  wiring, verify end-to-end): [`../docs/HUONG_DAN_CAI_DAT.md` §2.9](../docs/HUONG_DAN_CAI_DAT.md).

## ⭐ Ghi chú policy commit

- `.stl` (cell asset, ~5MB tổng) **được commit** vào repo để clone là dùng được ngay.
- `.pt` / `.onnx` (YOLO weights, vài chục–vài trăm MB) **KHÔNG commit** — copy về từ máy train.
- `T_base_camera.npy` (calibration sim) **được commit** — sinh tự động từ `calibration_from_layout.py`.
- `.rdk` (RoboDK station save) **KHÔNG commit** — repo không còn dùng RoboDK
  cho viewport/motion. RoboDK chỉ cần khi chạy `scripts/13_verify_vs_robodk.py`
  hoặc `scripts/17_compare_fk_ik.py` (verify/benchmark FK/IK).
