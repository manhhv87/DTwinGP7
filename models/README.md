# models/ — Trọng số model + mesh

Model YOLOv8-seg được **train trên máy Linux GPU**. Repo cung cấp quy trình
[đóng gói và huấn luyện](../docs/TRAINING_WORKFLOW.md); thư mục này chứa trọng số
được chọn để inference và các file mesh. Việc có trọng số chưa xác nhận cell đã hiệu chuẩn
hoặc đã hoàn tất thí nghiệm vật lý.

## ⭐ File trọng số cần có (cho `--mode real`)

| File | Mô tả | Nguồn |
|---|---|---|
| `e2_seed1_best.pt` | Ứng viên baseline E2, YOLOv8s-seg, năm lớp | Được commit riêng tại `d736b75`; có trên nhánh `feat/synthgen-c2-c3-dataset` |
| Các file E3/E4/E5 | Checkpoint riêng của từng cấu hình, seed/vòng | Chuyển từ máy train cùng manifest, log và SHA-256 |

`config/experiment.yaml :: model_path` hiện trỏ tới `models/e2_seed1_best.pt`.
Link artifact cố định theo commit:
[e2_seed1_best.pt trên GitHub](https://github.com/manhhv87/DTwinGP7/blob/d736b75fd2089348cbf4d11b0d41fbcf68110f83/models/e2_seed1_best.pt).
Khi clone nhánh có commit này, Git lấy file cùng mã nguồn. Nếu checkout khác, kiểm tra
commit chứa file trước khi kết luận rằng trọng số đã có.

Seed 1 này được chọn lịch sử theo **validation mask mAP@0.5:0.95** cao nhất giữa các seed.
Checkpoint trong mỗi run dùng tổng box+mask validation mAP@0.5:0.95. Không gọi lựa chọn
seed 1 lịch sử là quy tắc composite giữa các seed của chiến dịch sắp tới. Quy tắc mới,
seeds và cách khóa model nằm trong [campaign manifest](../docs/CAMPAIGN_MANIFEST.md).

> **Sim mode (`--mode sim`)** dùng `MockDetector` →
> KHÔNG cần file `.pt`. Chỉ cần khi `--mode real` (D455 + GP7 thật).
> `--headless` chỉ tắt giao diện; không tự chuyển một lệnh `--mode real` thành mô phỏng.

## ⭐ Đưa trọng số từ máy train về

Sau khi train và chọn bằng validation trên Linux, chuyển `best.pt` được chọn về máy chạy
với tên riêng cho cấu hình, seed và vòng. Ví dụ sau khi đã chuyển file tới Windows:

```powershell
Copy-Item -LiteralPath '<file_da_chuyen_ve>\best.pt' -Destination 'models\e3_anchored_k2_seed0_best.pt'
Get-FileHash -Algorithm SHA256 -LiteralPath 'models\e3_anchored_k2_seed0_best.pt'
```

Tên trên chỉ là ví dụ, không xác nhận κ=2/seed=0 đã được chọn. Trỏ `model_path` của cấu hình
thí nghiệm tới đúng file; ghi đường dẫn gốc, SHA-256, seed, recipe, package manifest và điểm
validation vào hồ sơ model. So khớp SHA-256 ở Linux và Windows sau chuyển. Không ghi đè
baseline để thay cấu hình, không chọn lại checkpoint sau khi xem final-test outcomes.

## ⭐ Định dạng hỗ trợ

`ObjectDetector` nạp được cả `.pt` và `.onnx`. Nếu muốn chạy inference không
cần PyTorch GPU, có thể export ONNX trên máy train (`yolo export model=best.pt
format=onnx`) rồi trỏ `model_path` tới file `.onnx`.

## ⭐ Mesh STL

| Thư mục | Nội dung |
|---|---|
| `models/` | `worktable.stl`, `pedestal.stl`, `gripper.stl`, `floor.stl` |
| `models/gp7_links/` | 7 STL link Yaskawa GP7 (`gp7_base_link.stl` … `gp7_link_6_t.stl`) cho URDF chain |
| `models/objects/` | **Bộ vật của paper (5 lớp, 6 mesh)**: `carton_large.stl`, `carton_small.stl`, `plastic_box.stl`, `wood_box.stl`, `metal_box.stl`, `inox_box.stl` — dựng theo số đo thước kẹp bằng `scripts/gen_product_meshes.py`. Bộ cũ cho sim/demo: `tray.stl`, `bottle.stl`, `cup.stl`, `bolt.stl` |

> **Lớp vật (class):** bài toán thật dùng 5 lớp `carton / plastic_box / wood_box /
> metal_box / inox_box`, trong đó `carton` gồm HAI cỡ hộp (180×120×120 và
> 180×100×80 mm) nên nhận danh sách mesh — xem `config/synthgen.yaml`.
> `tray/bottle/cup/bolt` nay chỉ là mặc định của kịch bản sim/demo. Danh sách lớp
> **định nghĩa được** (`CellConfig.object_classes`, sửa qua dock
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
python scripts/gen_primitive_meshes.py                  # tất cả primitives (bàn, bệ, gripper…)
python scripts/gen_primitive_meshes.py --only gripper   # chỉ 1 file
python scripts/gen_product_meshes.py                    # 6 mesh vật của paper, theo số thước kẹp
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
- `.gitignore` loại file mới khớp `*.pt`. Ngoại lệ đã theo dõi trong Git là
  `models/e2_seed1_best.pt` ở commit `d736b75`; ignore không xóa một file đã tracked.
  Các trọng số mới chuyển riêng từ Linux, không mặc nhiên có trên GitHub.
- `.onnx` là artifact export; không mặc nhiên tương đương pipeline `.pt` đã đánh giá.
  Lưu phiên bản export, tham số và kiểm tra đầu ra trước khi dùng trong một chiến dịch.
- `T_base_camera.npy` (calibration sim) **được commit** — sinh tự động từ `calibration_from_layout.py`.
- `.rdk` (RoboDK station save) **KHÔNG commit** — repo không còn dùng RoboDK
  cho viewport/motion. RoboDK chỉ cần khi chạy `scripts/13_verify_vs_robodk.py`
  hoặc `scripts/17_compare_fk_ik.py` (verify/benchmark FK/IK).
