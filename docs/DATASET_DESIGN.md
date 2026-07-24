# DATASET DESIGN (repo) — v2.0, 21/07/2026

> ⚠️ **Bản đầy đủ (cẩm nang thao tác từng bước) là nguồn chân lý:**
> `D:\Scientific\2026\DigitalTwin\DATASET_DESIGN.md` (v2.0). File repo này chỉ giữ **bản tóm
> tắt + lệnh** để làm việc cạnh code. Bản v1.0 cũ (tray/bottle/cup/bolt, VERIFY-1 khẩu độ, cửa
> sổ [26,106], mesh photogrammetry) **đã bị thay thế hoàn toàn** — đừng dùng lại.

## Chốt phần cứng (Hướng A)

Gripper 2 ngón khí nén; **thay 2 con lăn đầu ngón bằng má phẳng bọc cao su** → kẹp ma sát 2 mặt
phẳng. Dải kẹp thật ~185–200 mm (đo lại sau khi lắp má). Vật đặt trên **băng tải stop-and-go**;
camera **D455 eye-to-hand** nhìn xuống. Feedback CC-Link (X505 = carrier-detect).

## Bộ vật (4 hộp, cùng chiều kẹp ~190 mm)

| id | class (code) | vật | KT thô (mm) | vai trò |
|---|---|---|---|---|
| 0 | `carton` | hộp giấy sản phẩm | 190×110×70 | use-case; giấy; PCA-yaw rõ |
| 1 | `plastic_box` | hộp nhựa có nắp | 185×140×85 | nhựa; tỷ lệ vuông hơn |
| 2 | `wood_box` | hộp/khối gỗ | 190×90×60 | gỗ vân; dài–hẹp |
| 3 | `metal_box` | hộp kim loại phủ mờ | 190×150×55 | lớp khó (phản quang); gần vuông (test guard yaw) |

Đa dạng qua **vật liệu/màu/tỷ lệ**, KHÔNG qua hình (gripper một-cỡ). **Không vật trụ tròn**
(má phẳng ép mặt cong → trượt). Kích thước là **placeholder** — đo caliper vật thật rồi cập nhật
`config/synthgen.yaml` + regenerate STL. Thứ tự class khớp: `synthgen.yaml` · `auto_label.DEFAULT_CLASS_NAMES`
· `dataset_check.EXPECTED_NAMES` · `23_launch_retrain.CLASS_NAMES` = `[carton, plastic_box, wood_box, metal_box]`.
(Sim/demo vẫn dùng bộ `bottle/cup/bolt/tray` riêng trong `detector.DEFAULT_CLASS_NAMES` — không liên quan dataset paper.)

## Ngân sách + verify

400 train + 100 val + 30 nền (CHUẨN) + 300 test-std + 300 test-hard (dim/side_light/novel_bg).
Split **theo session**, không offline augmentation. Sau khi ráp từ Roboflow:

```powershell
python scripts/24_verify_dataset.py --data data/real_dataset --min-per-class 180
python scripts/24_verify_dataset.py --data data/real_dataset --fix   # nếu lệch class-id alphabet
```

## Depth (C4)

D455 RGB-D. Hai đường lấy z: **RGB-D** (`masked_depth` median) và **plane** (`z_from_ground_plane`
— giao tia với `Z_băng_tải + h_class`, không đọc depth → cứu `metal_box` thủng depth). So sánh
= đóng góp C4, thí nghiệm E2.

Chi tiết đầy đủ (kịch bản session, Roboflow từng click, mesh, lỗi thường gặp, checklist):
**`D:\Scientific\2026\DigitalTwin\DATASET_DESIGN.md`** và định vị publish:
`D:\Scientific\2026\DigitalTwin\PUBLICATION_DESIGN.md`.
