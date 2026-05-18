# pickplace_gp7 — Vision-guided Pick-and-Place cho Yaskawa GP7

> Tích hợp YOLOv8-seg vào hệ thống Digital Twin (RoboDK Free) cho bài toán
> gắp–thả sản phẩm ở vị trí ngẫu nhiên. Stack tối giản, license 0đ.
>
> Tài liệu đề tài đặt ở thư mục `../tai_lieu/`.

## Quickstart

```bash
# 1. Cài dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Verify config + chạy tests (không cần RoboDK/D455)
python -m src.cell.cell_models validate config/cell_layout.yaml
pytest tests/ -q

# 3. Mở RoboDK GUI (empty) → dựng cell
python scripts/build_station.py

# 4. Chạy thí nghiệm pick-and-place ở chế độ sim
python scripts/03_run_experiment.py --mode sim --trials 50
```

## Vị trí trong thư mục tổng

```
files/
├── pickplace_gp7/     ← BỘ CODE (thư mục này)
└── tai_lieu/          ← Tài liệu: HUONG_DAN_CAI_DAT.md + đề tài + phụ lục
```

## Cấu trúc bộ code

```
pickplace_gp7/
├── config/                       # YAML configs (validate bằng Pydantic)
│   ├── cell_layout.yaml           # cell sim
│   ├── cell_layout_real.yaml      # cell cho robot thật
│   ├── experiment.yaml            # tham số Orchestrator + thí nghiệm
│   └── calibration/               # T_base_camera.npy (output hand-eye)
├── src/
│   ├── cell/                      # Phụ lục A — "Cell là code"
│   │   ├── cell_models.py          # Pydantic schemas
│   │   ├── cell_loader.py          # dựng station từ config
│   │   ├── exceptions.py · pose_utils.py
│   ├── perception/                # Thị giác
│   │   ├── camera.py               # D455Camera + MockCamera
│   │   ├── detector.py             # ObjectDetector (YOLO) + MockDetector
│   │   ├── postprocess.py          # mask → centroid + PCA + depth → pose 3D
│   │   └── perception_node.py      # vòng lặp perception (threaded)
│   ├── orchestrator/              # Điều phối
│   │   ├── coord_conv.py           # transforms (pure numpy)
│   │   ├── state_machine.py        # state machine pick-and-place
│   │   └── orchestrator.py         # chu trình + lớp an toàn digital-twin
│   ├── calibration/               # Hand-eye eye-to-hand
│   │   ├── hand_eye_solver.py       # solver (mặc định Park, KHÔNG Tsai)
│   │   └── capture_calibration.py   # phát hiện ChArUco + thu pose
│   ├── logging/                   # TrialLogger → CSV
│   └── utils/                     # helpers (logging, YAML)
├── scripts/                       # CLI entry points
│   ├── build_station.py · dump_cell_to_yaml.py
│   ├── 01_collect_dataset.py       # chụp dataset bằng D455
│   ├── 02_run_calibration.py       # hand-eye calibration
│   ├── 03_run_experiment.py        # chạy thí nghiệm pick-and-place
│   └── 04_analyze_results.py       # phân tích thống kê + figures
├── tests/                         # 67 unit/integration tests
├── data/raw/ · models/ · results/ · figures/ · logs/   # output (gitignored)
├── clean.bat                       # tiện ích xóa __pycache__
└── requirements.txt · pyproject.toml
```

> **Train model:** việc huấn luyện YOLOv8 làm trên máy Linux GPU bằng công cụ
> riêng, **không nằm trong repo này**. Repo chỉ nhận file trọng số `.pt`/`.onnx`
> (đặt vào `models/`) để inference. Xem `models/README.md`.

## Luồng dữ liệu

```
config/*.yaml → cell_loader → RoboDK station (Digital Twin)
D455 → detector (YOLOv8-seg) → postprocess → pose 3D (camera frame)
                                                   ↓ hand-eye T_BC
                              orchestrator → pose 3D (base frame)
                                                   ↓ kiểm tra reachability
                              RoboDK → GP7 (sim hoặc thật) → TrialLogger → CSV
```

## Thiết kế để test được

Mọi thư viện phần cứng (`pyrealsense2`, `robodk`, `ultralytics`) đều được
**lazy-import** — module logic thuần import được và test được trên máy không
có phần cứng. Có `MockCamera` / `MockDetector` cho pipeline sim đầy đủ.

```bash
pytest tests/ -q                              # toàn bộ — không cần phần cứng
pytest tests/test_hand_eye_solver.py -v       # solver hand-eye
pytest tests/ --cov=src --cov-report=term     # với coverage
```

| Tầng test | Phần cứng cần | Cách chạy |
|---|---|---|
| Unit + integration (67 tests) | Không | `pytest tests/` |
| System SIM | RoboDK GUI | `03_run_experiment.py --mode sim` |
| System REAL | RoboDK + D455 + GP7 | `03_run_experiment.py --mode real` |

## Hai lưu ý kỹ thuật quan trọng

1. **Hand-eye eye-to-hand**: `cv2.calibrateHandEye` mặc định cho eye-in-hand;
   solver ở đây tự nghịch đảo `gripper2base → base2gripper` để ra đúng `T_BC`.
2. **KHÔNG dùng Tsai-Lenz** cho setup này: camera nhìn xuống nên `T_BC` xoay
   ~180° — đúng điểm kỳ dị của Tsai. Mặc định dùng **Park**.

## Liên kết

- Hướng dẫn cài đặt: `../tai_lieu/HUONG_DAN_CAI_DAT.md`
- Tài liệu đề tài: `../tai_lieu/phat_bieu_bai_toan_v3_2_HD.md`
- Phụ lục A (chi tiết cell module): `../tai_lieu/Phu_luc_A_README_HD.md`
- RoboDK API: https://robodk.com/doc/en/PythonAPI/

---
*pickplace_gp7 — Version 1.0*
