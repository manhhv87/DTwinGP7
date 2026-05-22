# PickPlaceGP7 — Vision-Guided Pick-and-Place for Yaskawa GP7

> Tích hợp YOLOv8-seg vào hệ thống **Level-4 Bidirectional Digital Twin** cho
> bài toán gắp–thả sản phẩm ở vị trí ngẫu nhiên. Stack tối giản — HSE là motion
> path cho robot thật, Open3D làm viewport cho sim (RoboDK chỉ dùng để
> verify FK/IK trong `scripts/13_verify_vs_robodk.py`).
>
> **Use case thực tế**: gắp khay (tray) đựng điện thoại Galaxy S23 trên dây
> chuyền assembly, dùng pneumatic parallel-jaw gripper custom.
> Demo vision multi-class với 3 vật khác (bottle/cup/bolt) tùy chọn.
>
> Repo GitHub: https://github.com/manhhv87/DTwinGP7

## ⭐ Stack tổng quan

```mermaid
graph LR
    A[Open3D viewport<br/>render URDF] -->|sim mode| B[Python<br/>YOLO OpenCV<br/>Orchestrator<br/>Digital Twin L4<br/>+ DLS IK]
    B ==>|HSE<br/>UDP plus FTP| GP7[Yaskawa GP7<br/>YRC1000<br/>HSE Server ON]

    style A fill:#1565C0,stroke:#fff,stroke-width:2px,color:#fff
    style B fill:#2E7D32,stroke:#fff,stroke-width:2px,color:#fff
    style GP7 fill:#E65100,stroke:#fff,stroke-width:2px,color:#fff
```

**HSE là motion path cho robot thật** — project implement protocol public
Yaskawa HW1485553 để nói chuyện thẳng với YRC1000, không phụ thuộc RoboDK
driver license. Sim mode render qua **Open3D** từ URDF chain (verified match 
RoboDK FK 0.00mm), real mode dùng HSE backend + telemetry CSV @10Hz (replay offline qua
`07_replay_telemetry.py`). RoboDK chỉ cần khi chạy `scripts/13_verify_vs_robodk.py`
để verify FK/IK client-side so với RoboDK reference.

## ⭐ Đọc tài liệu nào trước?

| Bạn cần | Đọc file |
|---|---|
| **Chạy thử trên PC** (không có robot) | [`docs/HUONG_DAN_SU_DUNG.md`](docs/HUONG_DAN_SU_DUNG.md) — workflow + commands theo kịch bản |
| **Cài đặt từ đầu** | [`docs/HUONG_DAN_CAI_DAT.md`](docs/HUONG_DAN_CAI_DAT.md) — Python + Open3D + D455 + YRC1000 HSE |
| **Hiểu kiến trúc tổng thể** | [`docs/phat_bieu_bai_toan_v3_2_HD.md`](docs/phat_bieu_bai_toan_v3_2_HD.md) — sơ đồ + thiết kế hệ thống |
| **Setup STL mesh + YOLO weights + gripper IO** | [`models/README.md`](models/README.md) — assets + CIO ladder |

## ⭐ Quickstart 30 giây

```bash
pip install -r requirements.txt
pytest tests/ -q                                              # → 293 passed
python scripts/03_run_experiment.py --mode sim --headless --trials 500
python scripts/03_run_experiment.py --mode sim --trials 500 --minimal-build
```

4 lệnh trên chạy được trên **mọi PC**, không cần phần cứng. Đầu ra: CSV
trong `results/`. Cho workflow chi tiết hơn (Open3D GUI, real robot, ultra-fast,
phân tích figure), xem [`HUONG_DAN_SU_DUNG.md`](docs/HUONG_DAN_SU_DUNG.md).

## ⭐ Kiến trúc Level-4 Bidirectional Digital Twin

```mermaid
flowchart TB
    CAM[D455 Camera] --> PER[Perception<br/>YOLO postprocess]
    PER --> ORC[Orchestrator<br/>state machine predictive safety]
    ORC --> DT[DigitalTwinMirror<br/>L4 facade]
    DT -->|Command path| BE[Motion Backend<br/>HSE real or SimRobot dev]
    BE -->|UDP HSE plus FTP INFORM<br/>real mode only| GP7[YRC1000 GP7]
    GP7 -.->|State sync<br/>Joints 10Hz| DT
    DT -.->|viewport_callback 2Hz<br/>sim only| O3D[Open3D viewport<br/>URDF render]
    DT --> TEL[Telemetry CSV<br/>drift detection<br/>alarm auto-Stop]

    style ORC fill:#2E7D32,stroke:#fff,stroke-width:2px,color:#fff
    style DT fill:#9C27B0,stroke:#fff,stroke-width:3px,color:#fff
    style BE fill:#1565C0,stroke:#fff,stroke-width:2px,color:#fff
    style GP7 fill:#E65100,stroke:#fff,stroke-width:3px,color:#fff
    style O3D fill:#1565C0,stroke:#fff,stroke-width:2px,color:#fff
```

**Bidirectional**: PC → robot (motion command) + robot → PC (joint state @10Hz).
Trong sim non-headless, **O3DGuiSimRobot** vừa làm motion backend vừa render
viewport (Filament GUI, navigate mượt). Real mode: HSE backend ghi telemetry
CSV @10Hz + **live Open3D mirror** render trạng thái THẬT của robot @2Hz
(O3DGuiSimRobot làm viewport-only, joints từ HSE poll; GUI main thread,
experiment worker thread). Tắt mirror bằng `--no-viewport-mirror` cho batch
500+ trial; replay offline bằng `07_replay_telemetry.py`.

Chi tiết kiến trúc + so sánh backend đầy đủ: [phat_bieu mục 2](docs/phat_bieu_bai_toan_v3_2_HD.md#2-sơ-đồ-kết-nối-hệ-thống--level-4-bidirectional-digital-twin).

## ⭐ Chế độ chạy

| Chế độ | Phần cứng | Use case |
|---|---|---|
| **Sim headless** | 0 | Thống kê 500+ trial offline, CI/CD |
| **Sim Open3D GUI** | Open3D (pip) | Demo trực quan (SimRobot + viewport render từ URDF) |
| **Real (HSE)** | YRC1000 + GP7 + D455 | Production |
| **Real ultra-fast** | Như trên | ~50ms/trial overhead, scale 500+ trial trên robot thật |

Workflow + commands chi tiết: [`HUONG_DAN_SU_DUNG.md`](docs/HUONG_DAN_SU_DUNG.md).

## ⭐ Tóm tắt cấu trúc repo

```
DTwinGP7/                        ← root repo (DTwinGP7 trên GitHub)
├── README.md                    ← file này (entry point)
├── docs/                        ← 4 file hướng dẫn (xem bảng trên)
├── config/                      ← YAML configs (KHÔNG sửa code)
├── models/                      ← STL meshes + YOLOv8 weights (xem models/README.md)
├── src/                         ← Python source (logic, không chạy trực tiếp)
│   ├── cell/                      CellConfig Pydantic schema (YAML → base pose + camera + mesh paths)
│   ├── perception/                YOLO + D455 + postprocess
│   ├── orchestrator/              trial pick-place + digital twin L4 + kinematics + backends
│   ├── calibration/               hand-eye ChArUco
│   ├── logging/ · utils/
├── scripts/                     ← CLI entry points (01–07, 11, 13 + helpers, BẠN CHẠY)
├── tests/                       ← 293 unit/integration tests
└── results/ · figures/ · logs/  ← output (gitignored)
```

Chi tiết module tree: xem [phat_bieu mục 3](docs/phat_bieu_bai_toan_v3_2_HD.md#3-cấu-trúc-thư-mục-code).

## ⭐ Đóng góp kỹ thuật chính

1. **Level-4 Bidirectional Digital Twin** — Open3D Filament viewport render URDF
   live cho cả sim VÀ real mode (real mirror @2Hz từ HSE poll); telemetry CSV
   @10Hz + drift detection + alarm auto-Stop
2. **HSE motion architecture** — Python nói chuyện thẳng YRC1000 qua public spec
   Yaskawa HW1485553
3. **3-tier motion optimization** — single-shot → batch M3 → ultra-fast M3++
   (~30× speedup so với approach single-shot)
4. **2-source IK architecture** — `--ik-source {yrc, client}`:
   - **YRC**: gửi pose Cartesian thẳng tới YRC1000, controller tự IK (0 PC overhead)
   - **Client**: DLS pure Python với URDF chain (verified match RoboDK SolveFK 0.00mm)
5. **C2 safety pipeline** — 2 tầng kiểm an toàn pure-Python TRƯỚC khi gửi MoveJ:
   - **Reach envelope** (~µs/pose): sphere 150–927 mm tính từ J1, reject pose
     ngoài envelope tại PLAN state
   - **Predictive trajectory check** (~2ms/trial sau optimization, real mode
     auto-bật): solve client DLS IK cho 6 waypoint, interpolate, verify joint
     limit + self-collision sphere TOÀN BỘ trajectory bằng pure-Python FK. Trial
     unsafe bị reject với failure_reason `predicted_joint_limit` hoặc
     `predicted_self_collision` — KHÔNG gửi command lên controller thật
6. **Kinematics performance** — 12-32× speedup vs naive impl bằng pure-Python/
   numpy: LRU-cache per-link/per-model constants, batched FK qua numpy matmul
   (N,4,4), `einsum` squared-distance pre-filter cho self-collision check,
   redundant-FK elimination trong DLS IK. Bit-identical với reference (max_err
   = 0.0 trên 3000+ random configs). Chi tiết: [phat_bieu §7.5.2](docs/phat_bieu_bai_toan_v3_2_HD.md#752-kinematics-performance-optimization)

Tài liệu chi tiết: [phat_bieu_bai_toan_v3_2_HD.md](docs/phat_bieu_bai_toan_v3_2_HD.md).

## ⭐ Train YOLO model

Việc huấn luyện YOLOv8 làm trên máy Linux GPU bằng công cụ riêng,
**không nằm trong repo này**. Repo chỉ nhận file trọng số `.pt`/`.onnx`
(đặt vào `models/`) để inference. Xem [`models/README.md`](models/README.md)
+ [phat_bieu Phần C](docs/phat_bieu_bai_toan_v3_2_HD.md#phần-c--xây-dựng--huấn-luyện-model).

---

*PickPlaceGP7 — Level-4 Bidirectional Digital Twin + Yaskawa HSE motion*
