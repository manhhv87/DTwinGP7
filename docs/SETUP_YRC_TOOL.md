# Setup TOOL01 trên YRC1000

Tài liệu này hướng dẫn config TOOL coordinate cho gripper trên Yaskawa YRC1000
teach pendant. **Bắt buộc làm 1 lần** trước khi chạy real mode với HSE Cartesian
(`--ik-source yrc`).

## Tại sao cần TOOL01

YRC1000 cần biết TCP offset (Tool Center Point) của gripper để compute IK đúng.
- Robot có sẵn TOOL00 = flange (origin tại mặt cuối arm)
- Gripper đặt thêm offset Z = ~100mm (chiều dài gripper)
- TOOL01 lưu offset này → khi PC gửi pose Cartesian, YRC tự bù offset → robot
  di chuyển sao cho fingertip (không phải flange) ở pose target

Không setup TOOL01 → robot sẽ đi với offset 100mm sai → gripper đâm bàn hoặc
gắp hụt vật.

## Cần chuẩn bị

- YRC1000 ở **MAINTENANCE mode** hoặc **MANAGEMENT mode** (key switch trên TP)
- Biết TCP offset gripper theo `cell_layout.yaml`:
  ```yaml
  tool:
    tcp_offset_mm: [0.0, 0.0, 100.0]   # ← lấy giá trị này
  ```
- TP (teach pendant) trong tay, robot **servo OFF** (an toàn)

## Bước 1: Vào menu Tool

1. Press **MAIN MENU** trên TP
2. Chọn **ROBOT → TOOL**
3. Màn hình hiện list TOOL00-TOOL63 (mặc định tất cả zero)

## Bước 2: Chọn TOOL01

1. Cursor xuống dòng **TOOL: 1**
2. Press **SELECT**
3. Màn hình hiện 6 fields:
   ```
   X    0.000 mm
   Y    0.000 mm
   Z    0.000 mm
   Rx   0.0000 deg
   Ry   0.0000 deg
   Rz   0.0000 deg
   ```

## Bước 3: Nhập TCP offset

1. Cursor lên field **Z**
2. Press **EDIT** hoặc **MODIFY**
3. Gõ `100.000` (theo `tcp_offset_mm[2]` trong cell config)
4. Press **ENTER**
5. X, Y giữ `0.000` (gripper centered)
6. Rx, Ry, Rz giữ `0.0000` (gripper aligned với flange)

> Nếu gripper của bạn có offset X/Y (vd lệch sang phải 20mm), nhập tương ứng.
> Nếu gripper xoay (vd kẹp ngang), nhập Rz.

## Bước 4: Lưu

1. Press **COMPLETE** hoặc **REGISTER**
2. Màn hình hiện confirmation "TOOL DATA REGISTERED"
3. Press **TOP MENU** để thoát

## Verify

Trên TP, vẫn ở menu **ROBOT → TOOL**:

1. Cursor lên TOOL: 1
2. Press **DISP** (display)
3. Confirm: Z = 100.000, others = 0

## Test với robot (servo ON, REDUCED SPEED)

⚠ Bước này robot SẼ di chuyển. Đảm bảo:
- Key switch ở **REMOTE mode**
- Speed limit **10%** (override slider trên TP)
- Tay sẵn sàng E-stop

```powershell
# Trên PC: chạy diagnostic test
python scripts/11_test_yrc_cartesian.py --tool-no 1 --speed-pct 10
```

Script này gửi 3 Cartesian pose đơn giản (home → offset Z+50mm → home), verify:
- Robot di chuyển smoothly
- Fingertip đi tới đúng pose (không lệch 100mm)
- Không alarm

## Optional: USER01 frame

Nếu workspace có origin custom (vd góc worktable thay vì robot base):

1. **MAIN MENU → ROBOT → USER COORD**
2. Chọn USER: 1
3. Define qua 3-point teaching (Origin, X-axis point, Y-axis point):
   - Robot mode JOG về 3 vị trí lần lượt
   - Press **REGISTER** ở mỗi point
4. Save

Sau đó trong code:
```yaml
robot_connection:
  user_frame_no: 1   # dùng UF01 thay vì BASE
```

Project default dùng BASE (UF=0) — đủ cho hầu hết pick-place. UF01 chỉ cần khi
muốn config tương đối với workpiece.

## Troubleshooting

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| TOOL không edit được | Key switch ở PLAY mode | Đổi sang MAINTENANCE/MANAGEMENT |
| "TOOL CONST ERROR" sau khi save | Giá trị nhập vượt limit | Z < 600mm, các trục < ±360° |
| Robot di chuyển lệch 100mm vẫn còn | TP cache stale | Restart YRC1000 (power cycle) |
| HSE READ_POSITION trả tool_no=0 | INFORM job không có TL=1 | Verify `motoman_hse.py:tool_no=1` constructor |

## Tham chiếu

- Yaskawa Operator's Manual cho YRC1000 §"Tool File Setting"
- INFORM Language Manual (RE-CKI-A464) §"TL Coordinate"
