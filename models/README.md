# models/ — Trọng số model + mesh

Model YOLOv8-seg được **train trên máy Linux GPU** (việc train nằm ngoài
repo này). Thư mục này chứa trọng số đã train để inference + các file mesh.

## File trọng số cần có

| File | Mô tả | Nguồn |
|---|---|---|
| `yolov8s-seg_best.pt` | Trọng số production (variant `s`) | Copy từ máy train |

`config/experiment.yaml :: model_path` trỏ tới `models/yolov8s-seg_best.pt`.
Đổi giá trị này nếu dùng tên/đường dẫn khác.

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
| `models/` | `worktable.stl`, `camera_mount.stl`, `gripper.stl` |
| `models/objects/` | `bottle.stl`, `box.stl`, `bolt.stl` — mesh CAD của 3 loại vật |

`config/cell_layout.yaml` tham chiếu các đường dẫn mesh này; `CellLoader`
nạp chúng vào RoboDK khi dựng cell.

## Ghi chú

File `.pt` / `.onnx` / `.stl` KHÔNG commit vào Git (xem `.gitignore`) — dung
lượng lớn.
