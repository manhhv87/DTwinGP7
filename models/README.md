# models/ — Trọng số model + mesh

Model YOLOv8-seg được **train trên máy Linux GPU** (việc train nằm ngoài
repo này). Thư mục này chứa trọng số đã train để inference + các file mesh.

## File trọng số cần có (cho `--mode real`)

| File | Mô tả | Nguồn |
|---|---|---|
| `yolov8s-seg_best.pt` | Trọng số production (variant `s`) | Copy từ máy train |

`config/experiment.yaml :: model_path` trỏ tới `models/yolov8s-seg_best.pt`.
Đổi giá trị này nếu dùng tên/đường dẫn khác.

> **Sim mode (`--headless` hoặc default `--mode sim`)** dùng `MockDetector` →
> KHÔNG cần file `.pt`. Chỉ cần khi `--mode real` (D455 + GP7 thật).

## Đưa trọng số từ máy train về

Sau khi train xong trên máy Linux, copy file trọng số tốt nhất về máy chạy
và đặt đúng tên mà `experiment.yaml` trỏ tới:

```bash
copy best.pt  models\yolov8s-seg_best.pt
```

## Định dạng hỗ trợ

`ObjectDetector` nạp được cả `.pt` và `.onnx`. Nếu muốn chạy inference không
cần PyTorch GPU, có thể export ONNX trên máy train (`yolo export model=best.pt
format=onnx`) rồi trỏ `model_path` tới file `.onnx`.

## Mesh STL

| Thư mục | Nội dung |
|---|---|
| `models/` | `worktable.stl`, `pedestal.stl`, `gripper.stl`, `floor.stl` |
| `models/objects/` | `tray.stl` (Galaxy S23 use case), `bottle.stl`, `cup.stl`, `bolt.stl` |

`config/cell_layout.yaml` tham chiếu các đường dẫn mesh này; `CellLoader`
nạp chúng vào RoboDK khi dựng cell. Cùng mesh dùng cho:
- RoboDK viewport visualization (cell builder)
- Cell sim mode (--mode sim) — tray là target gắp chính
- Real mode (--mode real --backend hse) — visualization khi mirror robot thật

## Sinh STL primitive nếu thiếu

Repo có sẵn các STL được commit. Nếu thiếu / muốn sinh lại:

```bash
pip install trimesh
python scripts/gen_primitive_meshes.py            # tất cả primitives
python scripts/gen_primitive_meshes.py --only gripper   # chỉ 1 file
```

## CIO ladder cho gripper IO (HSE backend, real mode)

Khi dùng `--backend hse` để điều khiển gripper khí nén qua `setDO`, cần setup
1 lần qua INFORM Ladder Editor trên YRC1000:

| Network I/O bit | Mapping | Mô tả |
|---|---|---|
| `27010` | → Y-output #1 (ví dụ) | Solenoid van khí nén gripper |

Logic ladder: nếu bit 27010 ON → solenoid ON → gripper đóng (kẹp).
Bit 27010 = `NETWORK_IO_BASE` trong `src/orchestrator/backends/motoman_hse.py`.

Sửa `NETWORK_IO_BASE` nếu setup CIO khác trong project của bạn.

## Ghi chú policy commit

- `.stl` (cell asset, ~5MB tổng) **được commit** vào repo để clone là dùng được ngay.
- `.pt` / `.onnx` (YOLO weights, vài chục–vài trăm MB) **KHÔNG commit** — copy về từ máy train.
- `T_base_camera.npy` (calibration sim) **được commit** — sinh tự động từ `calibration_from_layout.py`.
- `.rdk` (RoboDK station save) **KHÔNG commit** — dựng lại bằng `build_station.py` mỗi lần.
