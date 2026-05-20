# pickplace_gp7 — Vision-guided Pick-and-Place cho Yaskawa GP7

> Tích hợp YOLOv8-seg vào hệ thống Digital Twin (RoboDK Free) cho bài toán
> gắp–thả sản phẩm ở vị trí ngẫu nhiên. Stack tối giản, license 0đ.
>
> **Use case thực tế**: gắp khay (tray) đựng điện thoại Galaxy S23 trên dây
> chuyền assembly, dùng pneumatic parallel-jaw gripper custom.
> Demo vision multi-class với 3 vật khác (bottle/cup/bolt) tùy chọn.
>
> Tài liệu: `docs/` (HUONG_DAN_CAI_DAT, phat_bieu_bai_toan, Phu_luc_A).
> Repo GitHub: https://github.com/manhhv87/DTwinGP7

## Quickstart

```bash
# 1. Cài dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Verify config + chạy tests (không cần RoboDK/D455)
python -m src.cell.cell_models validate config/cell_layout.yaml
pytest tests/ -q                              # kỳ vọng: 79 passed

# 3. Mở RoboDK GUI (empty) → dựng cell (minimal = chỉ tray, ~20 API calls)
python scripts/build_station.py --minimal

# 4. Chạy thí nghiệm pick-and-place ở chế độ sim
python scripts/03_run_experiment.py --mode sim --trials 5 --minimal-build

# Headless mode (0 API call, scale ~500 trials cho thống kê thesis)
python scripts/03_run_experiment.py --mode sim --trials 500 --headless
```

## Cấu trúc repo

```
pickplace_gp7/             ← root repo (DTwinGP7 trên GitHub)
├── docs/                  ← HUONG_DAN_CAI_DAT, phat_bieu_bai_toan, Phu_luc_A
├── config/                ← YAML configs
├── models/                ← STL meshes + YOLOv8 weights
├── src/                   ← Python source
├── scripts/               ← CLI entry points
├── tests/                 ← 79 tests
└── results/ figures/ logs/  ← output (gitignored)
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
├── scripts/                       # CLI entry points (10 scripts)
│   ├── build_station.py             # dựng cell (--minimal cho RoboDK Free)
│   ├── 01_collect_dataset.py        # chụp dataset bằng D455
│   ├── 02_run_calibration.py        # hand-eye calibration (ChArUco)
│   ├── 03_run_experiment.py         # thí nghiệm — sim/real/--headless/--minimal-build
│   ├── 04_analyze_results.py        # phân tích thống kê + figures
│   ├── calibration_from_layout.py   # sinh T_BC từ cell config (sim/headless)
│   ├── convert_glb_to_stl.py        # GLB → STL utility
│   ├── diagnose_layout.py           # in toạ độ world thật của items trong RoboDK
│   ├── gen_primitive_meshes.py      # sinh STL primitives (gripper, tray, ...)
│   └── save_current_as_home.py      # ghi joints hiện tại làm home (sau khi jog tay)
├── tests/                         # 79 unit/integration tests
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
| Unit + integration (79 tests) | Không | `pytest tests/` |
| System SIM | RoboDK GUI | `03_run_experiment.py --mode sim` |
| System SIM headless | Không (SimRobot mock) | `03_run_experiment.py --headless --trials 500` |
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
# DTwinGP7
