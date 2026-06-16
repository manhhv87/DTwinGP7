# HƯỚNG DẪN LẬP TRÌNH — DTwinGP7

> Tài liệu **học lập trình** với hệ thống DTwinGP7: từ khái niệm nền → lập trình
> robot bằng GUI → ngôn ngữ INFORM → script Python trong app → lập trình bằng
> Python SDK (động học, tọa độ, thị giác) → vision-guided → bài tập thực hành.
>
> **Đối tượng:** người mới, muốn tự lập trình robot GP7 và mở rộng hệ thống.
> Mỗi mục có **ví dụ chạy được**. Đọc tuần tự từ §0.
>
> | Liên quan | File |
> |---|---|
> | **Thao tác bằng GUI** (không cần code) | [`HUONG_DAN_GUI.md`](HUONG_DAN_GUI.md) |
> | Giới thiệu tổng quan + chức năng các phần | [`GIOI_THIEU_PHAN_MEM.md`](GIOI_THIEU_PHAN_MEM.md) |
> | Cài đặt | [`HUONG_DAN_CAI_DAT.md`](HUONG_DAN_CAI_DAT.md) |
> | Workflow + CLI flags | [`HUONG_DAN_SU_DUNG.md`](HUONG_DAN_SU_DUNG.md) |

---

## Mục lục

- [§0. Chuẩn bị](#0-chuẩn-bị)
- [§1. Kiến thức nền tảng](#1-kiến-thức-nền-tảng-bắt-buộc-hiểu-trước)
- [§2. Cách 1 — Lập trình robot bằng GUI](#2-cách-1--lập-trình-robot-bằng-gui-16_app_qtpy)
- [§3. Ngôn ngữ INFORM (.JBI)](#3-ngôn-ngữ-inform-jbi--robot-thực-thi-cái-gì)
- [§4. Cách 2 — Script Python trong app](#4-cách-2--lập-trình-bằng-python-script-trong-app)
- [§5. Cách 3 — Lập trình bằng Python SDK](#5-cách-3--lập-trình-bằng-python-sdk-script-độc-lập)
- [§6. Lập trình vision-guided](#6-lập-trình-vision-guided-camera--gắp)
- [§7. Bài tập thực hành](#7-bài-tập-thực-hành-tăng-dần)
- [§8. Lỗi thường gặp](#8-lỗi-thường-gặp-khi-lập-trình)
- [§9. Tham chiếu API chi tiết](#9-tham-chiếu-api-chi-tiết)
- [§10. Mini-project: vision → IK → .JBI](#10-mini-project-vision--ik--jbi)
- [§11. API nâng cao: HSE backend + Orchestrator](#11-api-nâng-cao--backend-hse--orchestrator)
- [§12. Định dạng project `.json` (kiến trúc chương trình)](#12-định-dạng-project-json-kiến-trúc-chương-trình)

---

## §0. Chuẩn bị

```powershell
# 1. Kích hoạt môi trường ảo (đã cài theo HUONG_DAN_CAI_DAT.md)
.venv\Scripts\Activate.ps1
# 2. Kiểm tra
pytest tests/ -q          # → 452 passed
```

**Quy tắc vàng về đơn vị** (nhớ kỹ — sai đơn vị là lỗi phổ biến nhất):

| Đại lượng | Đơn vị | Ghi chú |
|---|---|---|
| Vị trí (X, Y, Z) | **mm** | Toàn dự án |
| Góc xoay pose (Rx, Ry, Rz) | **độ** | Trong pose 6 số + GUI |
| **Góc khớp khi gọi hàm FK/IK** | **radian** | ⚠ Hàm `forward_kinematics*`/`inverse_kinematics*` nhận/trả **radian** |
| Góc khớp trong GUI / target / `MoveJ` | **độ** | Phải `math.radians()`/`math.degrees()` khi chuyển qua lại |
| Ma trận pose | 4×4 homogeneous | Translation ở `T[:3,3]` (mm) |
| Scene 3D (viewport) | mét | mesh ×0.001 — chỉ liên quan render, không liên quan logic |

**Chạy script Python tự viết:** đặt file ở **thư mục gốc repo** rồi `python ten_file.py`,
hoặc thêm 2 dòng đầu để `import src...` luôn chạy được:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # nếu để ở gốc repo
```

---

## §1. Kiến thức nền tảng (bắt buộc hiểu trước)

### 1.1. Hệ tọa độ (frames)
- **World / Base frame**: gốc tọa độ thế giới (≈ chân đế robot). Mọi pose mặc định
  tính trong frame này.
- **Tool / TCP frame**: điểm tâm dụng cụ (đầu gripper). FK trả về pose của TCP.
- **Reference frame (UF)**: hệ quy chiếu phụ do người dùng đặt (vd góc bàn).
- **Camera frame**: gốc ở camera; cần `T_base_camera` để đổi về base.

### 1.2. Pose là gì?
Một **pose** = vị trí + hướng của một frame, biểu diễn 2 cách tương đương:
- **6 số**: `[X, Y, Z, Rx, Ry, Rz]` (mm, mm, mm, độ, độ, độ) — quy ước ZYX Yaskawa.
- **Ma trận 4×4**: `[[R(3×3), t(3×1)], [0,0,0,1]]` — R là xoay, t là tịnh tiến (mm).

Chuyển đổi (có sẵn trong `viewports/control_panel.py`):
```python
from src.orchestrator.viewports.control_panel import (
    _matrix_to_xyz_rpy_deg,   # 4x4 → (X,Y,Z mm, Rx,Ry,Rz deg)
    _xyz_rpy_to_matrix,       # (X,Y,Z, Rx,Ry,Rz) → 4x4
)
T = _xyz_rpy_to_matrix(500, 0, 400, 180, 0, 0)     # pose nhìn xuống tại (500,0,400)
x, y, z, rx, ry, rz = _matrix_to_xyz_rpy_deg(T)
```

### 1.3. FK và IK
- **FK (Forward Kinematics)**: biết 6 góc khớp → tính pose TCP. *Luôn có 1 đáp án.*
- **IK (Inverse Kinematics)**: biết pose TCP → tính 6 góc khớp. *Có thể 0, 1, hoặc
  nhiều (tới 8) đáp án (các "cấu hình" tay khác nhau), hoặc không với tới được.*
  GP7 có cổ tay cầu (spherical wrist) nên IK giải được **giải tích (closed-form)
  Pieper** — đây là đường chính: exact (sai số ~1e-13 mm), trả tất cả nghiệm rồi
  chọn nhánh gần tư thế hiện tại, xử lý tool offset. IK số (DLS, lặp) chỉ làm
  **fallback** khi cần.

### 1.4. Hand-eye calibration
Ma trận `T_base_camera` (4×4, file `.npy`) cho biết camera nằm ở đâu so với base
robot, để đổi tọa độ vật từ ảnh sang robot:
`p_base = T_base_camera @ p_camera`.

### 1.5. Grasp pose (gắp top-down)
Gripper chỉ thẳng xuống (-Z), xoay quanh trục đứng theo góc `yaw` của vật. Hàm
`make_grasp_pose(xyz_base_mm, yaw_rad)` dựng sẵn pose này.

### 1.6. GP7 — thông số cần nhớ
- 6 khớp quay (S, L, U, R, B, T). Mỗi khớp có giới hạn (joint_min/max).
- Tầm với: bán kính ~150–927 mm tính từ khớp J1 (`ReachEnvelope`).

---

## §2. Cách 1 — Lập trình robot bằng GUI (`16_app_qt.py`)

Cách **trực quan nhất** (giống RoboDK / teach pendant), không cần code: jog robot →
teach pose đặt tên → ghép lệnh MoveJ/MoveL + gripper → **▶ Run Sim** → **Export .JBI**
→ **⚙ Run on Robot**.

```powershell
python scripts/16_app_qt.py                                    # app trống
python scripts/16_app_qt.py --config config/cell_layout.yaml   # nạp sẵn 1 cell
python scripts/16_app_qt.py --program examples/sample_program.json  # mở chương trình mẫu
```

> 📖 **Thao tác chi tiết click-by-click** (bản đồ menu đầy đủ, phím tắt, panel
> Program/Controls/Camera, teach, camera vision-guided, chạy robot): xem sổ tay
> [`HUONG_DAN_GUI.md`](HUONG_DAN_GUI.md). Mục này chỉ tóm tắt để nối sang phần code.

Mọi chương trình dựng bằng GUI đều được **dịch sang INFORM `.JBI`** khi Export — hiểu
INFORM (§3) giúp đọc/sửa file robot sinh ra. Muốn sinh lệnh **tự động/hàng loạt** thì
chuyển sang Python: script trong app (§4) hoặc SDK độc lập (§5).

---

## §3. Ngôn ngữ INFORM (.JBI) — robot thực thi cái gì?

Mọi chương trình bạn dựng đều dịch sang **INFORM III** (ngôn ngữ job của Yaskawa)
khi Export. Hiểu nó giúp đọc/sửa file `.JBI` và biết app đang sinh gì.

| Lệnh app | INFORM | Ý nghĩa |
|---|---|---|
| MoveJ | `MOVJ <pos> VJ=10.00` | Di chuyển khớp (joint interpolation) tới pose, tốc độ % |
| MoveL | `MOVL <pos> V=100.0` | Di chuyển thẳng (linear) tới pose, tốc độ mm/s |
| MoveC | `MOVC <mid> / <end>` | Cung tròn qua điểm giữa + điểm cuối |
| SetGripper close | `DOUT OT#(1) ON` | Bật ngõ ra số 1 (đóng kẹp) |
| SetDO | `DOUT OT#(n) ON/OFF` | Bật/tắt ngõ ra số n |
| Wait | `TIMER T=0.30` | Chờ (giây) |
| WaitIO | `WAIT IN#(n)=ON T=...` | Chờ ngõ vào n = ON (timeout tùy chọn) |
| SetSpeed | (modal) `VJ=`, `V=` | Đặt tốc độ cho các MOV kế tiếp |
| SetRounding | (modal) `PL=0..8` | Bo góc quỹ đạo |
| SetTool | (modal) `TL=n` | Chọn tool coordinate |
| SetRefFrame | (modal) `UF#(n)` | Chọn user frame |
| ShowMessage | `MSG "..."` | Hiện thông báo trên pendant |
| CallJob | `CALL JOB:NAME` | Gọi job con |

> "Modal" = lệnh đặt trạng thái, áp cho mọi MOV phía sau cho tới khi đổi. Khi
> Export, app gấp tốc độ/tool/frame vào từng dòng MOV.

Ví dụ một job `.JBI` đơn giản (gắp 1 vật rồi về home):
```
/JOB
//NAME PICK_DEMO
NOP
SET P000 (HOME joints)
SET P001 (PICK joints)
MOVJ P000 VJ=10.00
MOVJ P001 VJ=10.00
DOUT OT#(1) ON
TIMER T=0.30
MOVJ P000 VJ=10.00
END
```

---

## §4. Cách 2 — Lập trình bằng Python Script trong app

Trong app: **Program → Generate from Python script…** → cửa sổ editor có biến `p`
(đối tượng `ScriptProgramAPI`). Bấm **▶ Run** để sinh hàng loạt lệnh vào job hiện tại.

### 4.1. API đầy đủ của `p`
| Hàm | Tác dụng |
|---|---|
| `p.add_movej(joints)` | MoveJ với 6 góc khớp (**độ**) |
| `p.add_movel(tcp_pose)` | MoveL với pose `[X,Y,Z mm, Rx,Ry,Rz độ]` (world) |
| `p.add_movej_to(name)` | MoveJ tới target đặt tên (phải tồn tại) |
| `p.add_movel_to(name)` | MoveL tới target đặt tên |
| `p.add_grip(close)` | Đóng kẹp (`True`) / mở (`False`) |
| `p.add_wait(seconds)` | Chờ |
| `p.add_setspeed(vj_pct, v_mm_s)` | Đặt tốc độ khớp % + thẳng mm/s |
| `p.add_msg(text)` | Thông báo (≤32 ký tự) |
| `p.add_call(job_name)` | Gọi job con |
| `p.targets` | Dict target hiện có (chỉ đọc) |
| `p.active_job` | Tên job đang chỉnh |

Helper có sẵn trong editor: `math`, `np` (numpy).

### 4.2. Ví dụ — 8 điểm trên vòng tròn (cần có target HOME)
```python
import math
p.add_setspeed(15, 120)            # VJ=15%, V=120 mm/s
p.add_movej_to('HOME')
cx, cy, z = 500.0, 0.0, 300.0      # tâm vòng tròn (mm)
r = 150.0
for i in range(8):
    a = i * 2 * math.pi / 8
    x = cx + r * math.cos(a)
    y = cy + r * math.sin(a)
    p.add_movel([x, y, z, 180.0, 0.0, 0.0])   # gripper nhìn xuống
p.add_movej_to('HOME')
```

### 4.3. Ví dụ — palletizing lưới 3×3
```python
p.add_movej_to('HOME')
x0, y0, z = 400.0, -100.0, 250.0
step = 60.0
for r in range(3):
    for c in range(3):
        x = x0 + r * step
        y = y0 + c * step
        p.add_movel([x, y, z + 50, 180, 0, 0])   # tiếp cận
        p.add_movel([x, y, z,      180, 0, 0])   # hạ xuống
        p.add_grip(False)                          # nhả vật
        p.add_wait(0.2)
        p.add_movel([x, y, z + 50, 180, 0, 0])   # nhấc lên
p.add_movej_to('HOME')
```

### 4.4. Ví dụ — lặp qua mọi target đã teach
```python
for name in p.targets:             # p.targets là dict {name: {...}}
    p.add_movej_to(name)
    p.add_wait(0.3)
```

---

## §5. Cách 3 — Lập trình bằng Python SDK (script độc lập)

Viết script `.py` riêng dùng trực tiếp các module trong `src/`. Mọi ví dụ dưới đây
chạy độc lập (đặt file ở gốc repo, `python ten_file.py`).

### 5.1. Forward Kinematics — góc khớp → pose TCP
```python
import math
from src.orchestrator.kinematics.urdf_chain import gp7_urdf, forward_kinematics_urdf

model = gp7_urdf(base_xyz_mm=(0.0, 0.0, 330.0))     # GP7 trên đế cao 330mm
joints_deg = [0, 0, 0, 0, 0, 0]
T = forward_kinematics_urdf(model, [math.radians(q) for q in joints_deg])  # 4x4 (mm)
print("TCP tại (mm):", T[:3, 3].round(1))
```
> `gp7_urdf` dùng mô hình URDF đã **verify khớp RoboDK 0.00 mm**.

### 5.2. Inverse Kinematics — pose → góc khớp

**Đường chính — IK giải tích Pieper** (closed-form, exact, trả mọi cấu hình tay):
```python
import math
from src.orchestrator.kinematics.urdf_chain import gp7_urdf
from src.orchestrator.kinematics.pieper_gp7 import (
    inverse_kinematics_pieper_gp7,          # trả tất cả nghiệm (≤8)
    inverse_kinematics_pieper_gp7_nearest,  # trả nghiệm gần q_init nhất
)
from src.orchestrator.viewports.control_panel import _xyz_rpy_to_matrix

model = gp7_urdf(base_xyz_mm=(0.0, 0.0, 330.0))
T_target = _xyz_rpy_to_matrix(450, 0, 400, 180, 0, 0)   # pose mong muốn (mm, độ)
q_init = [math.radians(q) for q in [0, 0, 0, 0, 0, 0]]  # seed = tư thế hiện tại

sols = inverse_kinematics_pieper_gp7(model, T_target)     # list (có thể 0..8 nghiệm)
print(f"Số cấu hình tay với tới được: {len(sols)}")
best = inverse_kinematics_pieper_gp7_nearest(model, T_target, q_init)  # cho chuyển động liên tục
if best is None:
    print("Không với tới được pose này")
else:
    print("Góc khớp (độ):", [round(math.degrees(q), 1) for q in best])
```
Pieper exact (~1e-13 mm), nhanh, deterministic — dùng làm IK mặc định trong app
(Find branches / Change Config) và Orchestrator client-IK.

**Fallback — IK số DLS** (lặp, generic 6R, khi cần robot khác / cấu hình đặc biệt):
```python
from src.orchestrator.kinematics import inverse_kinematics_seeded
sol_rad = inverse_kinematics_seeded(model, T_target, q_init)   # bền: thử nhiều seed
```

### 5.3. Tọa độ & grasp pose
```python
import numpy as np
from src.orchestrator.coord_conv import (
    transl, rotz, transform_point, camera_to_base, make_grasp_pose, load_calibration)

# Đổi 1 điểm từ camera frame sang base frame
T_BC = load_calibration("config/calibration/T_base_camera.npy")   # 4x4 (mm)
xyz_cam = np.array([25.0, 30.0, 800.0])         # vật trong camera frame (mm)
xyz_base = camera_to_base(xyz_cam, T_BC)
print("Vật trong base frame (mm):", xyz_base.round(1))

# Dựng pose gắp top-down tại vật, xoay theo yaw của vật
T_grasp = make_grasp_pose(xyz_base, yaw_rad=0.5)   # 4x4 → đưa vào IK ở §5.2
```

### 5.4. Đọc / kiểm tra cấu hình Cell (YAML)
```python
from src.cell import CellConfig

cfg = CellConfig.from_yaml("config/cell_layout.yaml")
print("Robot base (mm):", cfg.robot.pose.xyz_mm)
print("Các lớp vật của bài toán:", cfg.object_classes)
if cfg.camera is not None:
    print("Camera pose:", cfg.camera.pose.xyz_mm, "| intrinsics:", cfg.camera.intrinsics)

# Validate file YAML từ dòng lệnh:
#   python -m src.cell.cell_models validate config/cell_layout.yaml
```

### 5.5. Kiểm tra quỹ đạo an toàn (joint limit + tự va chạm)
```python
import math
from src.orchestrator.kinematics import gp7_default, interpolate_joints, \
    check_joint_limits, check_self_collision_spheres

model = gp7_default(base_xyz_mm=(0.0, 0.0, 330.0))
waypoints = [                                  # mỗi waypoint = 6 góc (radian)
    [math.radians(q) for q in [0, 0, 0, 0, 0, 0]],
    [math.radians(q) for q in [45, 20, -10, 0, 30, 0]],
]
samples = interpolate_joints(waypoints, dt=0.05, max_joint_speed_deg_s=60.0)
limit_viol = check_joint_limits(model, samples)             # [] = OK
collisions = check_self_collision_spheres(model, samples)   # [] = OK
print(f"{len(samples)} điểm | vượt giới hạn: {len(limit_viol)} | va chạm: {len(collisions)}")
```

### 5.6. Robot mô phỏng + chạy thí nghiệm
`SimRobot` (`src.orchestrator.sim_robot`) là robot ảo thuần Python (Joints/MoveJ/
SolveIK/setDO). Để chạy **trọn chu trình pick-place + thống kê**, dùng entry chính
thay vì tự ghép tay:
```powershell
python scripts/03_run_experiment.py --mode sim --trials 500 --headless
python scripts/04_analyze_results.py --csv "results/*.csv"
```

### 5.7. Thị giác — phát hiện vật → pose 3D (không cần phần cứng)
```python
from src.perception import MockCamera, MockDetector, PoseExtractor
from src.perception.detector import field_dict

cam = MockCamera()                              # camera ảo (D455Camera nếu có thật)
det = MockDetector(scripted=[[
    MockDetector.make_detection("tray", mask_box=(560, 320, 760, 440))]])
extractor = PoseExtractor(cam.intrinsics)

rgb, depth = cam.get_frame()                    # (ảnh BGR, depth mét)
for d in det.detect(rgb):
    enriched = extractor.extract(field_dict(d), depth)
    if enriched and enriched.get("pose_camera"):
        x, y, z, yaw = enriched["pose_camera"]  # mm, mm, mm, radian (camera frame)
        print(f"{enriched['class_name']}: pose_camera = ({x:.0f},{y:.0f},{z:.0f}) yaw={yaw:.2f}")
```
> Camera thật: thay `MockCamera()`→`D455Camera()`; `MockDetector`→`ObjectDetector(model_path="models/best.pt")`.

---

## §6. Lập trình vision-guided (camera → gắp)

Ghép §5.7 + §5.3 + §5.2 thành luồng "thấy → gắp" hoàn chỉnh:

```python
import math, numpy as np
from src.perception import MockCamera, MockDetector, PoseExtractor
from src.perception.detector import field_dict
from src.orchestrator.coord_conv import camera_to_base, make_grasp_pose, load_calibration
from src.orchestrator.kinematics.urdf_chain import gp7_urdf
from src.orchestrator.kinematics import inverse_kinematics_seeded

# 1) Thị giác → pose vật trong camera frame
cam = MockCamera()
det = MockDetector(scripted=[[MockDetector.make_detection("tray", mask_box=(560,320,760,440))]])
extractor = PoseExtractor(cam.intrinsics)
rgb, depth = cam.get_frame()
obj = next(extractor.extract(field_dict(d), depth) for d in det.detect(rgb))
pose_cam = obj["pose_camera"]                       # (x,y,z mm, yaw rad)

# 2) Đổi sang base frame (hand-eye) → dựng grasp pose
T_BC = load_calibration("config/calibration/T_base_camera.npy")
xyz_base = camera_to_base(np.array(pose_cam[:3]), T_BC)
T_grasp = make_grasp_pose(xyz_base, yaw_rad=float(pose_cam[3]))

# 3) IK → góc khớp để gắp
model = gp7_urdf(base_xyz_mm=(0.0, 0.0, 330.0))
q_init = [0.0]*6
sol = inverse_kinematics_seeded(model, T_grasp, q_init)
print("Gắp được" if sol else "Ngoài tầm với",
      "→ joints(độ):", [round(math.degrees(q),1) for q in sol] if sol else None)
```

**Trong GUI** (không cần code): dock **Camera (D455)** → bật Detector → Start →
**Detect → Teach grasp** (chính là 3 bước trên) → **Pick → Program** → **Run on Robot**.
Xem [`HUONG_DAN_SU_DUNG.md` §4 Kịch bản F](HUONG_DAN_SU_DUNG.md).

---

## §7. Bài tập thực hành (tăng dần)

1. **GUI cơ bản**: teach 3 target (HOME, A, B), tạo chương trình HOME→A→grip→B→
   nhả→HOME, chạy Sim, Export .JBI. *(Kỹ năng: §2, §3)*
2. **Script vòng lặp**: dùng editor Python sinh quỹ đạo zig-zag 5×4 quét mặt bàn. *(§4)*
3. **FK/IK tay**: viết script in pose TCP tại 5 tư thế khớp khác nhau; rồi IK ngược
   lại, kiểm tra `FK(IK(pose)) ≈ pose`. *(§5.1, §5.2)*
4. **An toàn**: tạo 2 waypoint cố tình vượt giới hạn khớp, dùng `check_joint_limits`
   phát hiện. *(§5.5)*
5. **Vision-guided**: đổi `mask_box` trong §6 thành 3 vị trí khác nhau, in xyz_base
   tương ứng; quan sát thay đổi. *(§6)*
6. **Tùy biến bài toán**: thêm lớp vật mới (vd `gear`) qua dock Camera → Quản lý
   class, lưu cell, kiểm tra `cfg.object_classes`. *(§5.4)*

---

## §8. Lỗi thường gặp khi lập trình

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| `ModuleNotFoundError: src` | Chạy script ngoài gốc repo | Đặt file ở gốc repo, hoặc thêm `sys.path` (§0) |
| IK trả `None` | Pose ngoài tầm với / gần singularity | Đổi pose gần robot hơn; dùng `inverse_kinematics_seeded` (đã thử nhiều seed) |
| FK ra pose lạ, lệch xa | Truyền góc khớp bằng **độ** thay vì radian | Bọc `math.radians(q)` |
| `add_movej` báo "joints phải 6 phần tử" | Thiếu/thừa khớp | Đảm bảo list 6 số |
| `KeyError: Target '...' không tồn tại` | `add_movej_to` tên chưa teach | Teach target trước, hoặc kiểm tra `p.targets` |
| `FileNotFoundError: T_base_camera.npy` | Chưa có calibration | `python scripts/calibration_from_layout.py` (sim) |
| Teach grasp trong app báo "chưa load robot" | IK cần robot model | File → Load Robot GP7 trước |
| Lỗi đơn vị (mm↔m) | Nhầm scene-mét với logic-mm | Logic luôn mm; chỉ render mới ×0.001 |

---

## §9. Tham chiếu API chi tiết

> **Quy ước (đọc 1 lần):** góc khớp = **radian** trong mọi hàm FK/IK; pose 6 số =
> **mm + độ**; ma trận 4×4 homogeneous (translation ở `T[:3,3]`, mm). `model` là
> `URDFRobot` (từ `gp7_urdf`) hoặc `RobotDHModel` (từ `gp7_default`) — các hàm
> FK/IK/quỹ đạo nhận **cả hai** (polymorphic). Mỗi mục ghi rõ **module import**.

### §9.1. Pose ↔ ma trận
Import: `from src.orchestrator.viewports.control_panel import _xyz_rpy_to_matrix, _matrix_to_xyz_rpy_deg`

| Hàm | Tham số | Trả về |
|---|---|---|
| `_xyz_rpy_to_matrix(x, y, z, rx_deg, ry_deg, rz_deg)` | x/y/z (mm); rx/ry/rz (**độ**) | `np.ndarray` 4×4 (mm) |
| `_matrix_to_xyz_rpy_deg(T)` | `T` 4×4 | tuple `(X, Y, Z mm, Rx, Ry, Rz độ)` |

Góc theo **ZYX intrinsic** (Fanuc/Motoman); hai hàm nghịch đảo của nhau.

```python
from src.orchestrator.viewports.control_panel import (
    _xyz_rpy_to_matrix, _matrix_to_xyz_rpy_deg)
T = _xyz_rpy_to_matrix(500, 0, 400, 180, 0, 0)        # pose nhìn xuống tại (500,0,400)
x, y, z, rx, ry, rz = _matrix_to_xyz_rpy_deg(T)
```

### §9.2. Forward Kinematics (góc khớp → pose)
Import: `from src.orchestrator.kinematics.urdf_chain import gp7_urdf, forward_kinematics_urdf, link_frames_urdf`
· `from src.orchestrator.kinematics import gp7_default, forward_kinematics, joint_positions`

| Hàm | Tham số (mặc định) | Trả về |
|---|---|---|
| `gp7_urdf(base_xyz_mm=(0,0,0), base_rpy_rad=(0,0,0), tool_offset_mm=0.0)` | base_xyz (mm), base_rpy (rad), tool_offset (mm) | `URDFRobot` (verify RoboDK 0.00mm) |
| `gp7_default(base_xyz_mm=(0,0,0), base_rpy_rad=(0,0,0), tool_offset_mm=0.0)` | như trên | `RobotDHModel` (Modified DH, legacy) |
| `forward_kinematics_urdf(model, joints_rad)` | joints_rad: 6 góc (**radian**) | `np.ndarray` 4×4 — pose tool0 (mm) |
| `forward_kinematics(model, joints_rad)` | (dùng được cả URDF lẫn DH) | 4×4 (mm) |
| `joint_positions(model, joints_rad)` | | `list[np.ndarray]` — origin (x,y,z) từng joint |
| `link_frames_urdf(model, joints_rad)` | | `list[(tên_link, T 4×4)]` — cho viewport vẽ |

> `base_xyz_mm` = vị trí gốc **J1** trong world. GP7 đứng sàn ⇒ `(0,0,330)`
> (330mm = base_link→J1); robot trên pedestal cao H ⇒ `(0,0,330+H)`.

```python
import math
from src.orchestrator.kinematics.urdf_chain import gp7_urdf, forward_kinematics_urdf
model = gp7_urdf(base_xyz_mm=(0.0, 0.0, 330.0))
T = forward_kinematics_urdf(model, [math.radians(q) for q in [0, 0, 0, 0, 0, 0]])
print("TCP (mm):", T[:3, 3].round(1))
```

### §9.3. Inverse Kinematics (pose → góc khớp)
Import phân tích: `from src.orchestrator.kinematics.pieper_gp7 import inverse_kinematics_pieper_gp7, inverse_kinematics_pieper_gp7_nearest, inverse_kinematics_pieper_gp7_tagged`
Import số: `from src.orchestrator.kinematics import inverse_kinematics, inverse_kinematics_seeded`
· `from src.orchestrator.kinematics.inverse_kinematics import inverse_kinematics_lm, inverse_kinematics_sdls, inverse_kinematics_bfgs, inverse_kinematics_batch`

Tham số chung: `target_pose_world` = pose mong muốn của tool0 (4×4 world, mm);
`q_init_rad` = tư thế hiện tại (6 góc radian, dùng làm seed).

**Phân tích — closed-form, khuyến nghị cho GP7:**

| Hàm | Trả về | Khi nào dùng |
|---|---|---|
| `inverse_kinematics_pieper_gp7(model, T)` | `list[list[float]]` (≤8 nghiệm radian; `[]` nếu unreachable) | Liệt kê **mọi cấu hình tay** |
| `inverse_kinematics_pieper_gp7_nearest(model, T, q_init_rad)` | `list[float]` \| `None` | **Mặc định** — nghiệm gần `q_init` (chuyển động mượt) |
| `inverse_kinematics_pieper_gp7_tagged(model, T, include_turns=True)` | `list[dict]` (nghiệm + nhãn front/back, elbow, wrist) | Dialog đổi cấu hình |

**Số — iterative, generic 6R (fallback / robot khác):**

| Hàm (tham số mặc định) | Thuật toán | Trả về |
|---|---|---|
| `inverse_kinematics(model, T, q_init_rad, max_iter=200, tol_mm=0.1, tol_rad=1e-4)` | DLS, λ cố định | `list[float]` \| `None` |
| `inverse_kinematics_seeded(model, T, q_init_rad, *, n_random_seeds=8, seed=0)` | DLS + nhiều seed — **bền nhất** | `list[float]` \| `None` |
| `inverse_kinematics_lm(model, T, q_init_rad, max_iter=100, tol_mm=0.5, tol_rad=1e-3)` | Levenberg-Marquardt | `list[float]` \| `None` |
| `inverse_kinematics_sdls(model, T, q_init_rad, …)` | Selectively Damped LS (SVD) | `list[float]` \| `None` |
| `inverse_kinematics_bfgs(model, T, q_init_rad, …)` | Quasi-Newton (scipy) | `list[float]` \| `None` |
| `inverse_kinematics_batch(model, T, q_init_batch, …)` | N seed song song | `list` các nghiệm |

> **Chọn hàm nào?** GP7 → `inverse_kinematics_pieper_gp7_nearest` (nhanh, exact).
> Cần mọi cấu hình → `inverse_kinematics_pieper_gp7`. Fallback/robot khác →
> `inverse_kinematics_seeded`. `None` = ngoài tầm với / gần singularity.

```python
import math
from src.orchestrator.kinematics.pieper_gp7 import inverse_kinematics_pieper_gp7_nearest
from src.orchestrator.viewports.control_panel import _xyz_rpy_to_matrix
T = _xyz_rpy_to_matrix(450, 0, 400, 180, 0, 0)
q = inverse_kinematics_pieper_gp7_nearest(model, T, [0.0] * 6)
print("Ngoài tầm với" if q is None else [round(math.degrees(a), 1) for a in q])
```

### §9.4. Quỹ đạo & kiểm tra an toàn
Import: `from src.orchestrator.kinematics import interpolate_joints, check_joint_limits, check_self_collision_spheres`

| Hàm (tham số mặc định) | Trả về (`[]` = OK) |
|---|---|
| `interpolate_joints(waypoints, dt=0.05, max_joint_speed_deg_s=30.0)` | `list[TrajectorySample]` |
| `check_joint_limits(model, samples)` | `list[(sample_idx, joint_idx, góc_rad)]` |
| `check_self_collision_spheres(model, samples, radii_mm=…, min_non_adjacent_gap=3)` | `list[(sample_idx, i, j, dist_mm)]` |

`waypoints` = list các `[6 góc radian]`. `TrajectorySample`: `.t` (giây), `.joints_rad` (list 6 góc radian).

```python
import math
from src.orchestrator.kinematics import (
    gp7_default, interpolate_joints, check_joint_limits, check_self_collision_spheres)
model = gp7_default(base_xyz_mm=(0, 0, 330))
wps = [[math.radians(q) for q in row] for row in ([0]*6, [45, 20, -10, 0, 30, 0])]
samples = interpolate_joints(wps, dt=0.05, max_joint_speed_deg_s=60)
safe = not check_joint_limits(model, samples) and not check_self_collision_spheres(model, samples)
print(len(samples), "điểm — an toàn:", safe)
```

### §9.5. Tọa độ & grasp pose
Import: `from src.orchestrator.coord_conv import transl, rotx, rotz, transform_point, camera_to_base, make_grasp_pose, load_calibration, save_calibration`

| Hàm | Tham số | Trả về |
|---|---|---|
| `transl(x, y, z)` | mm | 4×4 tịnh tiến |
| `rotx(angle_rad)` / `rotz(angle_rad)` | radian | 4×4 xoay |
| `transform_point(p_xyz, T)` | điểm (3,), T 4×4 | điểm (3,) đã biến đổi |
| `camera_to_base(xyz_cam, T_BC)` | điểm camera (mm), T_BC 4×4 | điểm base (mm) |
| `make_grasp_pose(xyz_base_mm, yaw_rad=0.0, yaw_offset_deg=0.0)` | vị trí (mm), yaw (radian), bù yaw (độ) | 4×4 — gripper nhìn xuống, xoay yaw |
| `load_calibration(path)` | file `.npy` | 4×4 (raise `FileNotFoundError`/`ValueError`) |
| `save_calibration(path, T_BC)` | file, 4×4 | `None` |

```python
import numpy as np
from src.orchestrator.coord_conv import load_calibration, camera_to_base, make_grasp_pose
T_BC = load_calibration("config/calibration/T_base_camera.npy")
xyz_base = camera_to_base(np.array([25.0, 30.0, 800.0]), T_BC)   # camera→base (mm)
T_grasp = make_grasp_pose(xyz_base, yaw_rad=0.5)                 # → đưa vào IK §9.3
```

### §9.6. Cell config
Import: `from src.cell import CellConfig`

| API | Mô tả |
|---|---|
| `CellConfig.from_yaml(path) -> CellConfig` | Load + validate (raise nếu schema sai) |
| `cfg.to_yaml(path) -> None` | Ghi lại YAML (giữ mọi field) |
| `cfg.robot.pose.xyz_mm` / `.rpy_deg` | Base pose robot (chỉ `robot` bắt buộc) |
| `cfg.camera` | `CameraConfig` \| `None`: `.type`(virtual/real), `.mount`(eye_to_hand/eye_in_hand), `.pose`, `.intrinsics` |
| `cfg.camera.intrinsics.hfov_vfov_deg()` | `(hfov, vfov)` độ — dựng frustum |
| `cfg.gripper.tcp_offset_xyz_mm` | TCP offset gripper (mm) |
| `cfg.objects` / `cfg.object_classes` | list `ObjectConfig` / list tên lớp (mặc định `[tray, bottle, cup, bolt]`) |

```python
from src.cell import CellConfig
cfg = CellConfig.from_yaml("config/cell_layout.yaml")
print("Base (mm):", cfg.robot.pose.xyz_mm, "| classes:", cfg.object_classes)
# CLI validate:  python -m src.cell.cell_models validate config/cell_layout.yaml
```

### §9.7. Thị giác
Import: `from src.perception import D455Camera, MockCamera, ObjectDetector, MockDetector, Detection, PoseExtractor`
· `from src.perception.detector import field_dict`

| API | Trả về / thuộc tính |
|---|---|
| `D455Camera()` / `MockCamera(intrinsics=None, rgb_frames=None, depth_frames=None)` | `.intrinsics` (dict fx,fy,ppx,ppy,width,height); `.get_frame() -> (rgb_bgr, depth_m)`; `.stop()` |
| `ObjectDetector(model_path="models/yolov8s-seg_best.pt", conf=0.5, iou=0.45, class_names=None)` | `.detect(rgb) -> list[Detection]`; raise `FileNotFoundError` nếu thiếu weights |
| `MockDetector(scripted=None)` | `.detect(rgb)` trả kịch bản kế tiếp (lặp vòng) |
| `MockDetector.make_detection(class_name="bottle", confidence=0.95, mask_hw=(720,1280), mask_box=(600,320,680,420))` | `Detection` (tạo nhanh 1 vật giả) |
| `field_dict(det) -> dict` | Detection → dict cho `PoseExtractor` |
| `PoseExtractor(intrinsics).extract(detection_dict, depth)` | dict điền `"pose_camera": (x, y, z mm, yaw rad)` |

`Detection`: `class_id, class_name, confidence, mask(H,W), bbox(x1,y1,x2,y2)` + (sau `extract`) `pose_camera, pixel_uv, mask_area, height_mm`.
Postprocess lẻ (`src.perception`): `mask_centroid(mask)`, `masked_depth(depth, mask)`, `mask_pca_yaw(mask) -> rad`, `deproject_pixel(intrinsics, u, v, …)`.

```python
from src.perception import MockCamera, MockDetector, PoseExtractor
from src.perception.detector import field_dict
cam = MockCamera()
det = MockDetector(scripted=[[MockDetector.make_detection("tray", mask_box=(560, 320, 760, 440))]])
ext = PoseExtractor(cam.intrinsics)
rgb, depth = cam.get_frame()
for d in det.detect(rgb):
    obj = ext.extract(field_dict(d), depth)
    if obj and obj.get("pose_camera"):
        print(d.class_name, "→ pose_camera:", obj["pose_camera"])
```

### §9.8. Script API trong app — biến `p` (`ScriptProgramAPI`)
Có sẵn trong **Program → Generate from Python script…** (kèm sẵn `math`, `np`).

| Phương thức | Tham số | Tác dụng |
|---|---|---|
| `p.add_movej(joints)` | list 6 góc (**độ**) | MoveJ; raise `ValueError` nếu ≠ 6 |
| `p.add_movel(tcp_pose)` | `[X,Y,Z mm, Rx,Ry,Rz độ]` (world) | MoveL |
| `p.add_movej_to(name)` / `p.add_movel_to(name)` | tên target | Move tới target; raise `KeyError` nếu chưa teach |
| `p.add_grip(close)` | `bool` | Đóng (True) / mở (False) kẹp |
| `p.add_wait(seconds)` | giây | TIMER |
| `p.add_setspeed(vj_pct, v_mm_s)` | % khớp, mm/s | Đặt tốc độ cho MOV kế tiếp |
| `p.add_msg(text)` | str (cắt còn 32 ký tự) | MSG trên pendant |
| `p.add_call(job_name)` | str (chữ/số/`_`) | Gọi job con |
| `p.targets` | property | dict target hiện có (read-only) |
| `p.active_job` | property | tên job đang chỉnh |

```python
import math
p.add_setspeed(15, 120)                # VJ=15%, V=120 mm/s
p.add_movej_to('HOME')
for i in range(8):                     # 8 điểm trên vòng tròn (cần target HOME)
    a = i * 2 * math.pi / 8
    p.add_movel([500 + 150*math.cos(a), 150*math.sin(a), 300, 180, 0, 0])
p.add_movej_to('HOME')
```

---

## §10. Mini-project: vision → IK → .JBI (giải thích từng dòng)

> Capstone ghép §5–§9 thành **1 script chạy được headless** (Mock, không cần phần
> cứng): thấy vật → đổi sang base → grasp pose → IK → dựng joints approach/grasp/
> place → sinh file INFORM `.JBI`. Đặt ở **gốc repo**, chạy `python mini_pick.py`.
> (Đã kiểm chứng: ra `MINIPICK.JBI` 31 dòng.)

```python
# mini_pick.py — vision-guided pick → sinh INFORM .JBI (chạy headless, Mock)
import math
import numpy as np
from src.perception import MockCamera, MockDetector, PoseExtractor
from src.perception.detector import field_dict
from src.orchestrator.coord_conv import camera_to_base, make_grasp_pose, load_calibration
from src.orchestrator.kinematics.urdf_chain import gp7_urdf
from src.orchestrator.kinematics.pieper_gp7 import inverse_kinematics_pieper_gp7_nearest
from src.orchestrator.backends.inform_codegen import gen_pick_place_job

# [1] Model robot GP7 (J1 ở 330mm trên sàn) + tư thế home
model = gp7_urdf(base_xyz_mm=(0.0, 0.0, 330.0))
HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]                          # độ
home_rad = [math.radians(q) for q in HOME]

# [2] Thị giác → pose vật trong CAMERA frame (đổi Mock→D455/ObjectDetector khi có thật)
cam = MockCamera()
det = MockDetector(scripted=[[
    MockDetector.make_detection("tray", mask_box=(560, 320, 760, 440))]])
extractor = PoseExtractor(cam.intrinsics)
rgb, depth = cam.get_frame()
obj = extractor.extract(field_dict(det.detect(rgb)[0]), depth)
x, y, z, yaw = obj["pose_camera"]                              # mm,mm,mm,rad (camera)

# [3] Camera→base (hand-eye) + grasp pose top-down + điểm approach (trên 80mm)
T_BC = load_calibration("config/calibration/T_base_camera.npy")
xyz_base = camera_to_base(np.array([x, y, z]), T_BC)           # vd [725,-25,400] mm
T_grasp = make_grasp_pose(xyz_base, yaw_rad=float(yaw))        # 4x4 (gripper nhìn xuống)
T_approach = T_grasp.copy(); T_approach[2, 3] += 80.0          # nâng Z (world) 80mm

# [4] IK: pose → góc khớp (radian). q_init = home để chọn nghiệm hợp lý.
to_deg = lambda q: [round(math.degrees(a), 3) for a in q]
grasp_rad = inverse_kinematics_pieper_gp7_nearest(model, T_grasp, home_rad)
approach_rad = inverse_kinematics_pieper_gp7_nearest(model, T_approach, home_rad)
if grasp_rad is None or approach_rad is None:
    raise SystemExit("Grasp/approach ngoài tầm với — đổi vị trí vật hoặc base robot.")

# [5] Điểm thả (place): dịch +200mm theo Y; transfer = trên place 80mm
T_place = T_grasp.copy(); T_place[1, 3] += 200.0
T_transfer = T_place.copy(); T_transfer[2, 3] += 80.0
place_rad = inverse_kinematics_pieper_gp7_nearest(model, T_place, grasp_rad)
transfer_rad = inverse_kinematics_pieper_gp7_nearest(model, T_transfer, grasp_rad)
if place_rad is None or transfer_rad is None:
    raise SystemExit("Place ngoài tầm với — đổi offset place.")

# [6] Sinh INFORM .JBI: home→approach→grasp→[close]→lift→transfer→place→[open]→home
jbi = gen_pick_place_job(
    name="MINIPICK", home_deg=HOME,
    approach_deg=to_deg(approach_rad), grasp_deg=to_deg(grasp_rad),
    transfer_deg=to_deg(transfer_rad), place_deg=to_deg(place_rad),
    gripper_do_index=1, gripper_delay_s=0.3, speed_pct=10.0)
with open("MINIPICK.JBI", "w", encoding="utf-8", newline="") as f:  # utf-8 (comment có tiếng Việt), newline="" giữ CRLF
    f.write(jbi)
print(jbi)

# [7] (TÙY CHỌN — CẦN ROBOT THẬT) nạp + chạy qua HSE. Robot REMOTE, servo ON, speed thấp.
# import time
# from src.orchestrator.backends.motoman_hse import MotomanHSEBackend
# bk = MotomanHSEBackend(ip="192.168.1.100", tool_no=1); bk.connect()
# assert bk.Valid(), "YRC1000 không phản hồi"
# bk.upload_job(jbi, "MINIPICK"); bk.job_select("MINIPICK"); bk.job_start()
# while bk.read_status_running(): time.sleep(0.2)
# bk.disconnect()
```

**Giải thích từng khối:**

| Khối | Làm gì | API chính (mục) |
|---|---|---|
| [1] | Dựng model GP7 (J1 ở 330mm) + tư thế home | `gp7_urdf` (§9.2) |
| [2] | Mock camera + 1 detection → `pose_camera` (camera frame) | `MockCamera`/`MockDetector`/`PoseExtractor` (§9.7) |
| [3] | Đổi điểm sang base + grasp pose nhìn xuống + điểm approach | `camera_to_base`, `make_grasp_pose` (§9.5) |
| [4] | IK pose→joints (None = ngoài tầm với → dừng có thông báo) | `inverse_kinematics_pieper_gp7_nearest` (§9.3) |
| [5] | Tính điểm place + transfer rồi IK | như [4] |
| [6] | Ghép 5 waypoint thành INFORM `.JBI` rồi lưu file | `gen_pick_place_job` (§11.2) |
| [7] | (tùy chọn) upload + chạy thật qua HSE | `MotomanHSEBackend` (§11.1) |

> ⚠ Khối [7] điều khiển robot thật — chỉ bỏ comment khi YRC1000 ở **REMOTE mode**,
> servo ON, speed thấp, tay sẵn sàng E-stop (xem `HUONG_DAN_CAI_DAT.md` §2.9–§2.10).

---

## §11. API nâng cao — backend HSE + Orchestrator

> Phần cho ai muốn **can thiệp sâu** (tự điều khiển robot / sinh job / chạy thí
> nghiệm bằng code). Cách dễ nhất vẫn là CLI `scripts/03_run_experiment.py`
> ([`HUONG_DAN_SU_DUNG.md`](HUONG_DAN_SU_DUNG.md)); phần này dành khi cần tùy biến.

### §11.1. Backend HSE — `MotomanHSEBackend`
Import: `from src.orchestrator.backends.motoman_hse import MotomanHSEBackend`
Constructor: `MotomanHSEBackend(ip, port=10040, timeout_s=2.0, ftp_user="", ftp_pass="", ftp_job_dir="/MPRAM1/JBI", max_speed_pct=20.0, tool_no=1, reach_envelope=None)`

| Nhóm | Phương thức | Tác dụng |
|---|---|---|
| Lifecycle | `connect()` · `disconnect()` · `Valid() -> bool` | Mở/đóng UDP socket; heartbeat |
| Trạng thái | `Joints() -> [6 độ]` · `read_status_running() -> bool` · `read_alarm() -> (code, sub)` | Đọc khớp thật / robot đang chạy / alarm |
| Job | `upload_job(job_text, job_name)` (FTP) · `job_select(name)` · `job_start()` | Nạp + chọn + chạy 1 `.JBI` |
| Motion | `MoveJ(target)` · `MoveL(target)` · `timer(s)` | `target` = list 6 góc (độ) **hoặc** 4×4 pose (YRC tự IK) |
| I/O | `setDO(index, value)` · `set_io(bit_addr, value)` · `read_io(bit_addr) -> int` | Gripper / network I/O |
| Tối ưu | `batch(job_name=None)` (context) · `enable_ultra_fast(True)` | Gom 1 trial = 1 job / template P-var |
| An toàn | `Stop()` | Emergency: tắt servo |

> `MoveJ`/`MoveL` **polymorphic**: truyền 4×4 pose (Cartesian — YRC1000 tự IK, cần
> TOOL01 ở `CAI_DAT §2.10`) hoặc list 6 góc (độ). Cờ `supports_cartesian_pose=True`.

```python
# Gom chuỗi move + gripper vào 1 job (batch). CẦN ROBOT THẬT.
bk = MotomanHSEBackend(ip="192.168.1.100", tool_no=1); bk.connect()
with bk.batch("DEMO"):                 # mọi lệnh dưới → gom vào 1 INFORM job
    bk.MoveJ([0, 0, 0, 0, 0, 0])       # joints (độ)
    bk.MoveJ([10, -5, 20, 0, 30, -15])
    bk.setDO(1, 1)                     # đóng gripper
    bk.timer(0.3)
    bk.MoveJ([0, 0, 0, 0, 0, 0])
# Thoát `with` → tự upload + select + start + chờ xong
bk.disconnect()
```

### §11.2. Sinh INFORM `.JBI` — `inform_codegen`
Import: `from src.orchestrator.backends.inform_codegen import InformJobBuilder, gen_pick_place_job, gen_pvar_template_job`

| API | Mô tả |
|---|---|
| `InformJobBuilder(name, max_speed_pct=30.0)` | Builder dạng chain (mỗi method `return self`) |
| `.add_position(name, joints_deg, pos_token=None)` | Khai vị trí trong `//POS` (joints **độ**). Mặc định sinh **P-var job-local** (`P00000`…) đúng format YRC1000 nhận; truyền `pos_token='P5'`/`'C3'` để giữ token gốc khi round-trip .JBI import |
| `.movj(name, speed_pct=None, tool_no, pl, user_frame)` · `.movl(name, speed_mm_s=100)` · `.movc(...)` | Lệnh chuyển động |
| `.dout(idx, on)` · `.timer(s)` · `.wait_in(idx, on, timeout_s)` · `.msg(text)` · `.call_job(name)` · `.comment(text)` | I/O / timing / phụ trợ |
| `.render() -> str` | Sinh full text `.JBI` (CRLF, Yaskawa convention) |
| `gen_pick_place_job(name, home_deg, approach_deg, grasp_deg, transfer_deg, place_deg, …) -> str` | Helper 1 chu trình pick-place hoàn chỉnh |
| `gen_pvar_template_job(name, num_positions, …) -> str` | Template P-var cho ultra-fast mode |

```python
from src.orchestrator.backends.inform_codegen import InformJobBuilder
jbi = (InformJobBuilder("HELLO", max_speed_pct=10)
       .add_position("home", [0, 0, 0, 0, 0, 0])
       .add_position("p1",   [10, -5, 20, 0, 30, -15])
       .movj("home").movj("p1").dout(1, True).timer(0.3).movj("home")
       .render())
```

### §11.3. Đổi frame cho HSE Cartesian — `frame_convert`
Import: `from src.orchestrator.frame_convert import world_to_robot_base, matrix_to_xyzrpy_yaskawa`

| Hàm | Trả về |
|---|---|
| `world_to_robot_base(T_world, robot_base_xyz_mm, robot_base_rpy_deg=(0,0,0))` | 4×4 pose trong BASE frame |
| `matrix_to_xyzrpy_yaskawa(T)` | `(x, y, z mm, Rx, Ry, Rz độ)` — XYZ-fixed (encoding HSE BASE) |

### §11.4. Kiểm tầm với — `ReachEnvelope`
Import: `from src.orchestrator.backends.reach_envelope import ReachEnvelope`

```python
env = ReachEnvelope.gp7_default(base_xyz_mm=(0, 0, 330))
env.can_reach([725, -25, 400])           # True/False (sphere 150–927mm từ J1)
env.distance_from_base([725, -25, 400])  # khoảng cách (mm) hoặc None
```

### §11.5. Chạy chu trình bằng code — `Orchestrator`
Import: `from src.orchestrator.orchestrator import Orchestrator`
Constructor: `Orchestrator(perception_queue, config=None, robot=<backend>, logger_obj=None)` — `robot` **bắt buộc** (SimRobot / DigitalTwinMirror / MotomanHSEBackend), raise `ValueError` nếu thiếu.
Methods: `run_one_cycle(trial_id=-1) -> bool` · `run_n_trials(n) -> dict` (thống kê).

> Để chạy đủ pipeline (load cell, perception thread, telemetry, viewport), dùng
> `scripts/03_run_experiment.py` thay vì tự ghép `Orchestrator` — script đã wire
> sẵn mọi thứ. Tự dựng `Orchestrator` chỉ khi cần nhúng vào app khác.

---

## §12. Định dạng project `.json` (kiến trúc chương trình)

`.json` là **định dạng project nội bộ** của Program Editor — nơi chương trình được lập
trình được lưu lại. Khác với `.JBI` (một job, cú pháp INFORM để nạp lên YRC1000), `.json`
giữ **nhiều job + thư viện target + cấu hình post-processor + nguồn gốc `.JBI`** trong một
file. Sinh bởi **File → Save program (.json)** (`mixin_program_io.py`), nạp bởi **Open**
/ **Load**. Định dạng có **versioning** (v1 → v3); v3 là hiện hành, v1/v2 vẫn nạp được.

### 12.1. Cấu trúc tổng thể (v3)

```jsonc
{
  "version": 3,                         // số phiên bản schema (hiện tại = 3)
  "active_job": "MAIN",                 // job đang xem (chỉ là view-state)
  "targets": {                          // thư viện pose dùng chung cho mọi job
    "P6": {
      "joints":   [-38, 16, 13, 1, -66, 167],   // 6 góc khớp (độ)
      "tcp_pose": [x, y, z, Rx, Ry, Rz],         // pose TCP (mm + độ, ZYX)
      "jbi_token": "P00001"                       // (tùy chọn) tên P-var .JBI gốc
    }
  },
  "jobs": {                             // map: tên job → danh sách lệnh (theo thứ tự)
    "MAIN": [ { "type": "MoveJ", "target_name": "P6" }, ... ],
    "PP1":  [ ... ]
  },
  "post_processor": {                   // cấu hình post-processor (mục 12.3)
    "max_speed_pct": 30.0,             // trần an toàn VJ cho mọi move
    "default_vj":    10.0,             // VJ% mặc định trước lệnh SET SPEED đầu tiên
    "default_v_mms": 100.0            // V mm/s mặc định trước lệnh SET SPEED đầu tiên
  },

  // ── Các khối CHỈ xuất khi có dữ liệu ──
  "job_folders":   { "PP1": "PPTUNG" },          // ///FOLDERNAME mỗi job
  "jbi_raw":       { "PP1": "<text .JBI gốc>" }, // nếu project import từ .JBI
  "jbi_positions": { "P10": [j1..j6] }           // bảng P-var gốc (re-export byte-exact)
}
```

| Khóa | Bắt buộc | Ý nghĩa |
|---|---|---|
| `version` | ✓ | Schema version (3). Bộ nạp tự nhận v1/v2 cũ. |
| `active_job` | ✓ | Job đang hiển thị khi lưu (không tính vào dirty-check). |
| `targets` | ✓ | Pose có tên, dùng chung cho mọi job. `joints` (độ) là nguồn chuẩn; `tcp_pose` để hiển thị; `jbi_token` giữ tên P-var để round-trip `.JBI` chính xác. |
| `jobs` | ✓ | Mỗi job = list **Instruction object** (mục 12.2), giữ nguyên thứ tự thực thi. |
| `post_processor` | ✓ (v3) | Tốc độ mặc định + trần VJ; xem mục 12.3. |
| `job_folders` | — | `///FOLDERNAME` của từng job (cấu trúc thư mục pendant). |
| `jbi_raw`, `jbi_positions` | — | Văn bản `.JBI` gốc + bảng P-var, chỉ có khi project được **import từ `.JBI`**, để Export lại **byte-exact**. |

### 12.2. Instruction object — kiến trúc một lệnh

Mỗi phần tử trong `jobs[*]` là một object với khóa bắt buộc `type` cùng các trường riêng
theo loại. Tốc độ/PL/Tool/Frame là **modal** (lưu thành lệnh `Set*` riêng, áp cho các
move kế tiếp), nên move chỉ giữ vị trí (+ `speed_var` nếu dùng biến tốc độ).

| `type` | Trường JSON | Ghi chú |
|---|---|---|
| `MoveJ` | `target_name` \| `joints`[6] \| `pos_index_var`; `speed_var?` | Move khớp; tham chiếu 1 target hoặc joints trực tiếp hoặc P[Bxxx]. |
| `MoveL` | `target_name` \| `tcp_pose`[6] \| `pos_index_var`; `speed_var?` | Move thẳng. |
| `MoveC` | `tcp_pose_mid`[6], `tcp_pose`[6] | Cung tròn (điểm giữa + điểm cuối). |
| `SetSpeed` | `speed_joint_pct`, `speed_linear_mm_s` | VJ% (khớp) + V mm/s (thẳng). V chỉ hiệu lực với MOVL/MOVC (xem §3). |
| `SetRounding` | `rounding_pl` | PL 0–8. |
| `SetTool` | `tool_no` | TL# 0–15. |
| `SetRefFrame` | `ref_frame_no` | UF# 0–15. |
| `SetGripper` | `close` (bool) | Kẹp đóng/mở (→ DOUT). |
| `SetDO` | `do_index`, `do_state` | Ghi 1 output (OT#). |
| `PulseDO` | `do_index` | Xung output tức thời. |
| `Wait` | `seconds` | TIMER. |
| `WaitIO` | `io_index`, `io_state`, `io_timeout_s` | Chờ IN#=ON/OFF (+timeout). |
| `ShowMessage` | `message` | MSG. |
| `CallJob` | `job_name` | CALL JOB. |
| `SimEvent` | `event_name`, `event_payload` | Checkpoint mô phỏng (comment trong .JBI). |
| `Label` | `label_name` | Nhãn `*LABEL`. |
| `Jump` | `label_name` + *điều kiện* | Nhảy có/không điều kiện. |
| `SetVar` | `var_name`, `var_op`, `var_arg?` \| `var_expr?` | Gán/biến đổi biến B/I (INC/DEC không cần arg). |
| `ClearVar` | `var_name`, `clear_count` | Xóa n thanh ghi. |
| `ReadGroupIn`, `WriteGroupOut` | `var_name`, `io_group`, `io_group_kind` | DIN/DOUT nhóm. |
| `IfThen`, `ElseIf`, `While` | *điều kiện* | Mở khối điều kiện. Biến thể **EXP** (`IFTHENEXP`/`ELSEIFEXP`/`WHILEEXP`) = chính các type này với `cond_exp: true`, **không phải type riêng**. |
| `Else`, `EndIf`, `EndWhile`, `ClearStack` | (chỉ `type`) | Lệnh không tham số. |

**Trường điều kiện** (dùng bởi `Jump`/`IfThen`/`ElseIf`/`While`):
- Điều kiện đơn: `cond_lhs`, `cond_op`, `cond_rhs` (vd `"B000"`, `">"`, `"5"`).
- Điều kiện ghép: `cond_terms` = `[[lhs, op, rhs], …]` + `cond_join` = `"AND"`\|`"OR"`.
- `cond_exp: true` → render họ từ khóa **EXP**: `IFTHENEXP` / `ELSEIFEXP` / `WHILEEXP`
  (và `JUMP` có điều kiện EXP). Đây **không phải type riêng** — chỉ là cờ trên cùng các
  type `IfThen`/`ElseIf`/`While`/`Jump`. Điều kiện ghép luôn dùng `ANDEXP`/`OREXP`.

### 12.3. Ghi chú ngữ nghĩa

- **Tốc độ modal & V:** `SetSpeed` lưu cả VJ lẫn V. Khi Export `.JBI`, V chỉ ghi cho
  `MOVL`/`MOVC` (`MOVJ` chỉ mang `VJ=`) — vì vậy `.json` là cách **giữ trọn V** kể cả
  khi move kế là MOVJ. `post_processor.default_vj/v` là tốc độ áp cho move **trước** lệnh
  `SetSpeed` đầu tiên.
- **Tương thích ngược:** v1 = list lệnh trần (→ 1 job `MAIN`); v2 = `{"targets", "program"}`
  (→ 1 job `MAIN`); v3 = đa job như trên. Khóa thiếu khi nạp được điền mặc định an toàn.
- **Round-trip `.JBI`:** project import từ `.JBI` mang theo `jbi_raw` + `jbi_positions`,
  nên sau Save/Load `.json` vẫn Export lại `.JBI` **giống hệt từng byte** (nếu chưa sửa).
- **Dirty-check:** `targets` + `jobs` + `post_processor` cấu thành chữ ký project; đổi bất
  kỳ phần nào → app hỏi lưu khi đóng. `active_job` là view-state, không tính.

> Cấu hình **cell** (robot/bàn/camera/frame) **không** nằm trong `.json` này — lưu riêng
> qua **File → Save Cell to YAML** (schema Pydantic, xem [§9.6](#96-cell-config)).

---

## Liên kết
- Giới thiệu + chức năng các phần: [`GIOI_THIEU_PHAN_MEM.md`](GIOI_THIEU_PHAN_MEM.md)
- Thao tác giao diện (click-by-click): [`HUONG_DAN_GUI.md`](HUONG_DAN_GUI.md)
- Workflow + CLI: [`HUONG_DAN_SU_DUNG.md`](HUONG_DAN_SU_DUNG.md)
- Digital Twin + robot thật: [`HUONG_DAN_DIGITAL_TWIN.md`](HUONG_DAN_DIGITAL_TWIN.md)
- Cài đặt: [`HUONG_DAN_CAI_DAT.md`](HUONG_DAN_CAI_DAT.md)
- Thiết kế hệ thống: [`phat_bieu_bai_toan_v3_2_HD.md`](phat_bieu_bai_toan_v3_2_HD.md)
- Mã nguồn kinematics: [`../src/orchestrator/kinematics/`](../src/orchestrator/kinematics/)

---

*Hướng dẫn lập trình — DTwinGP7.*
