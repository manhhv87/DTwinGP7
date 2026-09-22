# HƯỚNG DẪN CHẠY THÍ NGHIỆM TRÊN CELL GP7

Bản 22/09/2026, cho người chạy cell. Phần sinh ảnh, huấn luyện và phân tích ở
`HUONG_DAN_HUAN_LUYEN.md`, không cần đọc.

**Ba điều giữ suốt:**

- Robot chỉ tự chạy từ Pha 2. Mọi lệnh chạy thật phải có `--confirm-each-trial` (dừng chờ ENTER
  trước mỗi lượt) và `--no-viewport-mirror` (thiếu nó, Ctrl+C không dừng được robot).
- **DỪNG** = dừng ngay, ghi lại, gửi số về, không tự sửa rồi chạy tiếp. **GHI SỐ** = ghi vào nhật ký
  rồi đi tiếp.
- Nút dừng khẩn luôn trong tầm tay. Không đưa tay vào cell khi servo bật. Không đổi tốc độ.

IP tủ điều khiển: `192.168.1.100`.

---

## 1. Lấy mã và kiểm máy

Máy cần Python 3.10 trở lên và Git. Mở PowerShell. Lần đầu:

```
git clone -b feat/synthgen-c2-c3-dataset https://github.com/manhhv87/DTwinGP7.git
cd DTwinGP7
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Những lần sau:

```
cd DTwinGP7
git pull
.venv\Scripts\activate
```

`git pull` báo `config/synthgen.yaml` hoặc `config/calibration/` bị sửa ở máy thì chạy
`git checkout -- config/synthgen.yaml config/calibration/` rồi pull lại.

Kiểm máy, chưa cần robot:

```
pytest tests/ -q
python scripts/03_run_experiment.py --mode sim --headless --trials 5
```

**DỪNG nếu** có chữ `failed`, hoặc lệnh thứ hai không tạo file mới trong `results\`. Tỷ lệ `0.0%`
ở lệnh mô phỏng là bình thường. Một bài test đo nhịp thời gian thỉnh thoảng trượt khi máy bận:
hỏng đúng một bài thì chạy lại.

Mô hình nhận dạng `models/e2_seed1_best.pt` có sẵn trên git; không đổi `model_path`.

---

## 2. Ngày đầu tiên: tám việc, theo thứ tự, đạt hết mới sang Pha 2

Robot không tự chạy ở việc nào; việc 5 đến 7 chỉ jog tay ở chế độ TEACH.

| # | Việc | Đạt khi | Ghi lại |
|---|---|---|---|
| 1 | Mục 1 ở trên | `git log -1` là `335e365` hoặc mới hơn; không `failed` | số `passed` |
| 2 | Cắm D455, chạy `python -c "from src.perception.camera import D455Camera; print(D455Camera().intrinsics)"` | `fx` 645,0 · `fy` 644,2 · `ppx` 647,3 · `ppy` 368,8 (lệch dưới 0,1) | bốn số |
| 3 | Bàn trống, chạy `python scripts/02_run_calibration.py --table-only` | dòng `Table plane:` cho `top` ≈ 593,9 mm, `tilt` ≈ 0,32° (lệch dưới 2 mm và 0,2°) | `top`, `rms`, `tilt`; khớp thì `git checkout -- config/calibration/table_plane.json` |
| 4 | Thước kẹp đo cạnh ô và cạnh dấu của tấm bàn cờ đã in, ba ô, lấy trung bình | 45,0 và 34,0 ± 0,2 mm | hai số |
| 5 | Làm cây chỉ, khai TOOL02 và TOOL01 (mục 3) | xoay quanh điểm mốc mà mũi cây chỉ không rời | Z của TOOL02 và TOOL01 |
| 6 | Hệ Robot, jog TCP tới (525, −160), (625, −160), (625, 200), (525, 200) | tới cả bốn, không báo giới hạn khớp | có/không |
| 7 | Chạm thử 8 điểm (mục 4) | có file `results\touch_test_*.json` | mean / RMS / max |
| 8 | Đo điểm thả và độ cao băng tải (mục 3) | băng tải không cao hơn mặt bàn | `place_position` |

Việc 2 và 3: đúng camera đã hiệu chuẩn, chưa xê dịch. Việc 4: hiệu chuẩn 18/09 đúng tỷ lệ. Chỉ
việc 3 hoặc 4 hỏng mới phải làm lại hiệu chuẩn (mục 4). Việc 7 lệch mà 3, 4 đạt thì xem lại TOOL02
trước: sai tool cho độ lệch cùng chiều ở mọi điểm, công cụ in riêng độ lệch trung bình theo x, y.

Xong tám việc, gửi tám dòng số về, rồi mới sang Pha 2.

---

## 3. Ba số phải đo tay

| # | Việc | Ghi vào đâu | Hiện tại |
|---|---|---|---|
| 1 | Cạnh ô, cạnh dấu tấm bàn cờ đã in | gõ vào lệnh hiệu chuẩn, nếu phải làm lại | in danh nghĩa 45 / 34 |
| 2 | TCP má kẹp (TOOL01) và mũi cây chỉ (TOOL02) | TOOL01 vào `config/cell_layout_real.yaml` → `gripper.tcp_offset_xyz_mm` | `[0, 0, 100]`, số giữ chỗ |
| 3 | Điểm thả trên băng tải | `config/experiment.yaml` → `place_position` | `[700, 120, 700]`, số giữ chỗ |

Mặt băng tải không được cao hơn mặt bàn; thấp hơn thì vật rơi đúng phần chênh, chỉ chấp nhận vài cm.
Thông số camera đã điền sẵn. **Không sửa `robot.pose.xyz_mm`** (`[0, 0, 630]`); preflight từ chối
chạy nếu khác lúc hiệu chuẩn.

**Hai tool, vì sao cần cả hai.** TOOL01 là điểm gắp: điểm giữa hai thanh má kẹp, ngang đầu
thanh. Nó nằm trong không khí, không chạm được gì. Mọi việc phải *chạm* (kiểm tool, lấy dấu thẻ,
chạm thử) dùng TOOL02 = mũi một **cây chỉ** kẹp trong má.

**Làm cây chỉ:** một thanh gỗ hoặc nhựa cứng, dài khoảng 190 mm để hai má kẹp được (má đóng hết
còn 180 mm, mở hết 216 mm), đóng một đinh hoặc vít nhọn xuống ở giữa, mũi thò ra chừng 30 đến
50 mm. Kẹp thanh vào giữa hai má, đẩy sát lên càng ngang cho khỏi xoay, mũi hướng xuống.

**Khai TOOL02 bằng chức năng hiệu chuẩn tool trên pendant**, không đo tay: chìa MAINTENANCE hoặc
MANAGEMENT, chọn TOOL: 2, vào UTILITY → CALIBRATION, đưa mũi cây chỉ chạm **cùng một điểm mốc**
(đầu một đinh dựng đứng trên bàn) từ **5 hướng cổ tay khác nhau**, mỗi lần bấm ghi điểm, rồi
COMPLETE; pendant tự tính X, Y, Z của mũi. Kiểm: chọn TOOL02, hệ Robot, mũi chạm mốc, bấm phím
xoay; mũi không rời mốc là đạt. Tên nút có thể khác đôi chút theo phiên bản, xem mục Tool
calibration trong sổ tay pendant.

**Suy TOOL01 từ TOOL02:** thước kẹp đo khoảng cách theo phương thẳng đứng từ **mũi cây chỉ** tới
**đầu dưới thanh má kẹp** khi đang kẹp: gọi là d. Khai TOOL01 với Z = Z(TOOL02) − d, X = Y = 0
(cây chỉ tự vào giữa vì hai má đóng đối xứng). Ghi Z của TOOL01 vào `cell_layout_real.yaml`.
Lệnh gắp dùng TOOL01 (`--tool-no 1`); mọi việc chạm dùng TOOL02. Đổi cây chỉ là khai lại TOOL02.

---

## 4. Pha 1: hiệu chuẩn tay–mắt. Đã làm ngày 18/09/2026

Bốn file kết quả đã có trên git (`config\calibration\`): 25 tư thế, camera cách bàn 713 mm, mặt
bàn nghiêng 0,32°. **Không đụng vào camera.** Còn hai việc: đo tấm bàn cờ (việc 4 ngày đầu) và
chạm thử.

**Chạm thử 8 điểm** (cần TOOL02 đã khai): **tháo tấm bàn cờ khỏi má kẹp** (mở má là tấm rời ra), đặt
nó nằm phẳng trên bàn trống trong tầm nhìn camera, lắp cây chỉ vào má, chọn TOOL02, rồi chạy với
hai số vừa đo:

```
python tools/touch_test.py --square-mm <cạnh ô> --marker-mm <cạnh dấu> --board-thickness-mm 3
```

Công cụ chụp một khung, chọn 8 góc ô rải khắp tấm, lưu ảnh đánh số và in X, Y của từng góc theo
gốc robot. Với từng góc: jog mũi cây chỉ chạm đúng góc đó, đọc X, Y trên pendant (COORD = Robot), gõ
hai số vào. Kết quả ra `results\touch_test_*.csv` và `.json`. Mục tiêu 3 mm là mục tiêu, không
phải cửa chặn: 3,4 mm vẫn chạy, báo cáo đúng 3,4.

**Chỉ làm lại hiệu chuẩn khi** cạnh ô đo khác 45,0 quá 0,2 mm, hoặc camera đã bị đụng. Khi đó:

1. In `charuco_a3_o45mm.pdf` ở **100%**, đo vạch 100 mm in sẵn, giấy mờ, dán lên alu 3 mm. Gá lên
   má kẹp bằng hai thanh nhôm hộp 20 × 40 dán ở lưng tấm, hai mặt ngoài cách nhau 200 mm; mặt in
   ngửa lên; tấm cao 250 đến 470 mm trên bàn (xem hai bản vẽ).
2. Robot ở TEACH, bàn trống, chạy:

   ```
   python scripts/02_run_calibration.py --hse-ip 192.168.1.100 --squares 7 5 --square-mm <cạnh ô> --marker-mm <cạnh dấu> --dict DICT_4X4_50 --method park --bootstrap 200
   ```

3. Mỗi tư thế: jog cho camera thấy rõ cả tấm, bấm ENTER, đợi `Captured pose #n`. Đủ **25 đến 30
   tư thế, xoay mặt bích nhiều hướng, rải khắp bàn**. Gõ `s` để giải. Rồi dọn bàn, đưa robot ra
   khỏi tầm nhìn, ENTER để đo mặt bàn.

**DỪNG nếu** không đủ bốn file, dưới 25 tư thế, hoặc các tư thế na ná nhau. **GHI SỐ:** `tilt_deg`,
`rms_mm` trong `table_plane.json`. Chỉ xê dịch bàn thì chạy lại với `--table-only`.

![Bố trí chung và kích thước tấm](ban_ve_ga_ban_co.png)

![Chi tiết gá và trình tự lắp](ban_ve_ga_3d.png)

---

## 5. Pha 2: chạy thử, một lượt rồi năm lượt

**Chuyển TEACH sang REMOTE là lúc nguy hiểm nhất.** Trước khi xoay chìa: dọn vật lạ, đếm người,
không còn tay ai trong cell, nút dừng khẩn trong tầm tay, nói to cho cả phòng. Xoay chìa xong mới
bật servo. Đặt sẵn một vật lên bàn.

```
python scripts/03_run_experiment.py --mode real --trials 1 --depth-mode rgbd --ik-source yrc --tool-no 1 --confirm-each-trial --no-viewport-mirror
```

Phần mềm tự kiểm tra (preflight) trước khi robot nhúc nhích; thiếu gì nó từ chối chạy và in cách
sửa. Chạy được 1 lượt thì chạy `--trials 5`.

**DỪNG nếu:** preflight báo lỗi; đầu má kẹp xuống thấp hơn mặt bàn cộng khoảng an toàn; robot đi
sai hướng, hoặc `motion_error` lặp lại.

**GHI SỐ:** vật rơi lệch bao nhiêu so với điểm dạy (thước); số lượt hỏng trên 5 và lý do.

**Xong Pha 2:** dán hai vạch băng dính ở điểm thả, cách nhau 30 mm (±15 mm quanh tâm). Từ đây chấm
đạt hay hỏng là chấm theo vạch này. Ghi vào nhật ký so vạch với tâm vật hay mép vật, giữ nguyên
suốt chiến dịch.

---

## 6. Pha 3: E1, đo nền tảng

```
pytest tests/ -q > results/e1_tests.txt
python scripts/17_compare_fk_ik.py --samples 500 --fair
python tools/hse_rtt.py 192.168.1.100 --n 1000 --csv results/e1_hse_rtt.csv | tee results/e1_rtt_tomtat.txt
python scripts/03_run_experiment.py --mode real --trials 50 --depth-mode rgbd --pose-list config/pose_lists/std_v2.csv --telemetry-hz 10 --confirm-each-trial --no-viewport-mirror
python scripts/05_analyze_telemetry.py latest | tee results/e1_telemetry_tomtat.txt
```

Ba lệnh đầu không di chuyển robot. Lệnh thứ tư gắp 50 lượt theo thẻ (cần thẻ đã dán, mục 7).
`tee` giữ lại màn hình thành file; thiếu nó thì số mất.

**DỪNG nếu** robot tự dừng ngoài ý muốn hoặc mất kết nối. **GHI SỐ:** `p50`/`p95`/`p99` của RTT
và tần số telemetry đạt được (có trong hai file `_tomtat.txt`). `p95` trên 100 ms là cảnh báo,
không phải lỗi.

---

## 7. Pha 4: E2, chiến dịch chính

### Thẻ vị trí

21 thẻ, 3 hàng × 7 cột, chỉ phủ phần bàn mà camera nhìn trọn cả vật cao và chồng hai lớp, robot
với tới. Đổi camera hoặc đổi dụng cụ thì lưới phải tính lại.

1. In `position_cards.pdf` (2 trang A4) ở **100%**, đo vạch 50 mm in sẵn, cắt theo viền.
2. **Lấy dấu bằng robot:** lắp cây chỉ, chọn TOOL02, hệ Robot, jog tới đúng X, Y in trên thẻ
   (x 525 / 575 / 625; y từ −160 đến 200), hạ mũi chạm bàn, đánh dấu. 21 lần. Xong tháo cây chỉ.
3. **Dán thẻ vào dấu**, mũi tên hướng +x (ra xa robot), băng dính trong phủ kín. Dán xong jog lại về
   thẻ 1: TCP phải rơi đúng tâm.

![Toạ độ thẻ đo từ gốc robot](ban_ve_toa_do_the.png)

### Mỗi lượt

Chương trình gọi thẻ và góc, ví dụ `card=14  x=575.0 mm  y=80.0 mm  yaw=85.0 deg  class=metal_box`,
rồi dừng chờ.

1. Đặt đúng loại vật, tâm vật lên chữ thập giữa thẻ, cạnh dài trùng vạch góc (bội số 5°). Dung
   sai ±15 mm. Có dòng `STACKED` thì đặt vật đế trước, vật gọi lên trên.
2. **Rút tay ra**, bấm ENTER. Không nhìn màn hình nhận dạng lúc đặt.
3. Gắp xong, màn hình hỏi `part in the right place? [y]=yes [n]=no`. **y** khi vật nằm gọn trong vạch
   30 mm ở điểm thả; **n** khi kẹp trượt, rơi giữa đường, hoặc ngoài vạch. Trả lời theo mắt thấy.
4. Lấy vật về khỏi điểm thả trong lúc chờ ENTER lượt kế. Không với ra băng tải lúc robot đang về.

### Chạy mù, một lệnh

Đổi `2026-09-20-sang` thành tên buổi thật, dùng đúng tên đó cho mọi lệnh trong buổi; `AN` là tên
viết tắt người đặt vật. Chạy `--dry-run` trước để xem lịch.

```
python tools/run_blinded_campaign.py --arm rgbd "--depth-mode rgbd" --arm plane "--depth-mode plane" --arm fusion "--depth-mode fusion" --pose-list config/pose_lists/std_v2.csv --trials 200 --block 25 --session 2026-09-20-sang --operator AN --seed 7 --common "--mode real --ik-source yrc --tool-no 1 --confirm-each-trial --no-viewport-mirror --save-frames --frames-dir D:/Scientific/Dataset/DigitalTwin/frames"
```

Màn hình chỉ hiện `[A] Trial 7/25 — PLACE: card=14 ...`; mã A, B, C bốc ngẫu nhiên, kết quả không
hiện. File khoá `results\blinding_keys\key_<buổi>.json` nói mã nào là cấu hình nào: **không mở**.
Một lỗi lộ tên chế độ thì ghi vào nhật ký khối đó đã lộ.

**Khối đối chứng**, đầu buổi và cuối buổi, cùng 20 tư thế:

```
python scripts/03_run_experiment.py --mode real --depth-mode rgbd --pose-list config/pose_lists/std_v2.csv --pose-slice 200:220 --session-id 2026-09-20-sang --block-id doichung-dau --operator-id AN --confirm-each-trial --no-viewport-mirror
```

Cuối buổi chạy lại, đổi `--block-id doichung-cuoi`.

**Bộ khó**, một buổi riêng: dựng đúng như buổi chụp `novelbg_dim` ngày 27/08/2026 (tấm nền xanh lá
phủ kín bàn, đèn mức "dim"; mở vài ảnh trong thư mục đó ra so). Tấm nền che thẻ, nên dán một bộ thẻ
thứ hai lên nền, lấy dấu lại bằng robot, đúng 21 toạ độ. Rồi:

```
python tools/run_blinded_campaign.py --arm real_only "--depth-mode rgbd" --pose-list config/pose_lists/hard_v2.csv --trials 200 --block 25 --session 2026-09-21-sang --operator AN --seed 8 --common "--mode real --ik-source yrc --tool-no 1 --confirm-each-trial --no-viewport-mirror --save-frames --frames-dir D:/Scientific/Dataset/DigitalTwin/frames --lighting dim"
```

### Cuối buổi

Mỗi khối 25 lượt là một file CSV. Dồn vào thư mục của buổi, tách hai file đối chứng (nhận ra bằng
giờ chạy) sang thư mục riêng:

```
mkdir results\2026-09-20-sang
mkdir results\2026-09-20-sang-doichung
move results\experiment_real_*.csv results\2026-09-20-sang\
move results\telemetry_*.csv results\2026-09-20-sang\
```

**Gửi về:** thư mục của buổi, `logs\experiment.log`, thư mục khung ảnh trên ổ D, file khoá **chưa
mở**, và trang nhật ký. Sao lưu `results\` và `logs\` ra ổ ngoài.

---

## 8. Pha 5: đo lại với mô hình mới

Người huấn luyện gửi về file `.pt` kèm mã SHA-256, tên danh sách thẻ và số lượt. Chép file vào
`models\`, kiểm SHA-256 trùng, sửa `model_path` trong `config\experiment.yaml`, rồi chạy đúng lệnh
chiến dịch mù ở mục 7 với danh sách được chỉ định. Mọi buổi Pha 5 đều trên **bộ khó**.

Riêng buổi adaptation của E4 (80 lượt, một nhánh, không mù):

```
python scripts/03_run_experiment.py --mode real --depth-mode rgbd --pose-list config/pose_lists/hard_v2.csv --pose-slice 220:300 --session-id <buổi> --block-id adapt-vong1 --operator-id AN --confirm-each-trial --no-viewport-mirror --save-frames --frames-dir D:/Scientific/Dataset/DigitalTwin/frames --lighting dim
```

| Lúc nào | Chạy gì | Số lượt |
|---|---|---:|
| Pha 2 | chạy thử | 1 rồi 5 |
| Pha 3 | E1 | 50 |
| Pha 4 | 3 chế độ độ sâu × 200, bộ chuẩn | 600 |
| Pha 4 | real-only, bộ khó | 200 |
| mỗi buổi Pha 4, 5 | đối chứng đầu và cuối buổi | 20 + 20 |
| Pha 5, E3 | anchored và wide-range | 200 + 200 |
| Pha 5, E4 | adaptation 2 vòng; đánh giá guided-2 và control-2 | 80 + 80; 200 + 200 |
| Pha 5, E5 | ba ablation | 3 × 200 |

Mỗi lượt đặt vật bằng tay: bấm giờ 10 lượt đầu rồi nhân lên để biết một buổi mất bao lâu.

---

## 9. Nhật ký

Máy tự ghi từng lượt vào CSV (thành công, `human_ok` người chấm, lý do hỏng, thời gian, toạ độ,
`pose_id`, buổi, khối, người), telemetry, và toàn bộ màn hình vào `logs\experiment.log`. Người chỉ
ghi thứ máy không biết. Sau mỗi buổi, điền:

| Mục | Ghi gì |
|---|---|
| Ngày, giờ bắt đầu và kết thúc | |
| Pha nào, cấu hình gì | ví dụ: Pha 4, bộ chuẩn |
| Số lượt chạy, số lượt hỏng | |
| Tên các file trong `results\` | |
| Điều kiện phòng | đèn gì, có che nắng không, ai đi qua chắn sáng |
| Bất thường | robot dừng, vật rơi, thẻ xê dịch, camera bị chạm, lỗi lộ tên chế độ |
