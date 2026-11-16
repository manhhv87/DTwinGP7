# CHẠY E3: TỜ NGẮN

Lệnh chép từ file này (`docs\E3_CHAY_NHANH.md`) mở trên máy cell. Mọi lệnh gõ trong PowerShell, ở
thư mục `DTwinGP7`, sau `.venv\Scripts\activate` (đầu dòng nhắc có `(.venv)`).

**DỪNG** nghĩa là: robot đang chạy thì bấm dừng khẩn; chụp màn hình; gửi thầy; chờ trả lời.
Từ phần 1 tới hết buổi `e4-2`: không `git pull` lần nữa; không sửa TOOL01, job `DTWINRUN`, file
trong `config\`; không chạm, không hiệu chuẩn lại camera.

## 1. LẤY VỀ (một lần)

```
cd DTwinGP7
.venv\Scripts\activate
git pull
git log -1 --format=%h -- models/e3_anchored.pt
```

**Đạt khi:** dòng cuối in `58df5b5`. `git pull` báo lỗi, hoặc dòng cuối khác: DỪNG.

## 2. KIỂM TRA TRƯỚC BUỔI `e3-1` (một lần)

**1. Ba file mô hình.** Lệnh dưới tự tính mã SHA-256 của ba file trong `models\` và so với mã đúng
viết sẵn trong lệnh. Mỗi file in một dòng: tên file và `True` (đúng file) hoặc `False` (sai file).

```
"e3_anchored.pt 6d5e2d7bf969f05163a325d2c10d582302850ca0f893f737fc8dce45902a9f95", "e3_wide.pt 311af74bdcc3a27e1b398058492a81acbf2c00fbe9d27c90439ac9f6025bd2cd", "e2_seed1_best.pt 52e25b0349d1586b45f7a0077e4b38479d845621435e85722555ea65cf5e702a" | ForEach-Object { $ten, $ma = $_ -split ' '; "{0}  {1}" -f $ten, ((Get-FileHash "models\$ten").Hash -eq $ma) }
```

**Đạt khi:** ba dòng đều `True`. Có `False`: DỪNG.

**2. Mã chạy không đổi.**

```
git diff --stat 3d06434 HEAD -- scripts src config tools
git status --porcelain --untracked-files=no
```

**Đạt khi:** cả hai lệnh không in dòng nào. In ra dòng nào: DỪNG.

**3. Điểm công cụ (lệnh 6 của B3).**

- Trên tay dạy vào MAIN MENU → ROBOT → TOOL → chọn tool số 1, chụp một ảnh màn hình này (ảnh thấy
  đủ X, Y, Z, Rx, Ry, Rz). Không sửa số nào.
- Chìa ở TEACH, chọn TOOL01, hệ Robot, rồi chạy:

  ```
  python scripts/30_measure_tcp_twin_vs_controller.py --hse-ip 192.168.125.100
  ```

  Dòng đầu phải có `controller reports tool 1; expected 1`. Số tool khác 1: chọn TOOL01 trên tay dạy.
- Jog tay, tốc độ thấp, lần lượt tới 20 tư thế dưới. Ở mỗi tư thế: buông phím jog, robot đứng yên
  thì bấm ENTER trên máy tính. Độ cao tính từ mặt bàn tới đầu dưới má kẹp, ước chừng. Cổ tay chúc
  thẳng xuống, rồi xoay hay nghiêng thêm nếu dòng đó ghi.

  | Tư thế | Chỗ | Độ cao | Xoay, nghiêng |
  |---|---|---|---|
  | 1 đến 5 | thẻ 1, 3, 11, 19, 21 | 50 mm | không |
  | 6 đến 10 | thẻ 1, 3, 11, 19, 21 | 200 mm | Rz 45 độ |
  | 11 đến 14 | thẻ 2, 10, 12, 20 | 150 mm | 15 độ: quanh Rx ở thẻ 2, 20; quanh Ry ở thẻ 10, 12 |
  | 15 đến 17 | dấu điểm thả | 50, 150, 250 mm | không |
  | 18 đến 20 | dấu điểm thả | 150 mm | lần lượt Rz -45 độ, Rx 15 độ, Ry 15 độ |

  Máy in `NOT STILL`: chờ robot đứng yên, bấm ENTER lại. `TOOL MISMATCH`: chọn TOOL01, bấm ENTER
  lại.
- **Đạt khi:** in đủ 20 tư thế rồi mục `== Summary ==`, và trong `results\` có hai file `e1_tcp_*`.
  Máy in `No position reply from the controller`: DỪNG. Xong: đưa robot về HOME như các buổi E2.

**4. Chạy thử lệnh của buổi `e3-1`** (có `--dry-run`, robot không chạy).

```
$B = "e3-1"; $F = 0
python tools/run_blinded_campaign.py --arm anchored "--depth-mode rgbd --model-path models/e3_anchored.pt" --arm wide "--depth-mode rgbd --model-path models/e3_wide.pt" --arm real_only "--depth-mode rgbd --model-path models/e2_seed1_best.pt" --pose-list config/pose_lists/hard_v2.csv --first-pose $F --trials 100 --block 25 --session $B --operator PQM --common "--mode real --ik-source yrc --tool-no 1 --confirm-each-trial --no-viewport-mirror --save-frames --frames-dir D:/Scientific/Dataset/DigitalTwin/frames --lighting dim" --dry-run
```

**Đạt khi:** in `Placement check: off for every arm (the arms differ in more than --depth-mode).`,
`3 arm(s), 12 block(s) of 25 trials.`, 12 dòng `block` có đủ ba mã `A`, `B`, `C`, và dòng cuối
`Dry run: nothing started, no key written.` Khác: chép lại lệnh, chạy lại; vẫn khác: DỪNG.

**Gửi thầy:** ảnh TOOL01, hai file `results\e1_tcp_*`, ảnh chụp màn hình kết quả mục 1, 2 và 4.
Thầy xác nhận rồi mới chạy `e3-1`.

## 3. CHẠY MỘT BUỔI E3

Buổi `e3-1` trước. Buổi `e3-2` chạy sau khi thầy xác nhận dữ liệu `e3-1`. Mỗi buổi làm bước 1 đến
9. Mỗi lượt ở bước 5, 6, 7: đặt vật đúng dòng máy gọi, rút tay khỏi cell, bấm ENTER; robot thả xong
thì trả lời `y` hoặc `n` như B1 của hướng dẫn đầy đủ. Vật dự phòng để ngoài bàn, ngoài khung hình
camera.

1. **Đặt biến buổi.** Dán dòng của buổi. Mở lại PowerShell giữa buổi thì `cd DTwinGP7`,
   `.venv\Scripts\activate`, dán lại dòng này.

   ```
   $B = "e3-1"; $F = 0
   ```

   ```
   $B = "e3-2"; $F = 100
   ```

2. **Kiểm đầu buổi.** Chạy lại mục 1 và 2 của phần 2 (đạt như ở đó), rồi:

   ```
   dir results\experiment_real_*
   dir results\telemetry_*
   ```

   **Đạt khi:** cả hai lệnh báo không tìm thấy file. Còn file: DỪNG, gửi tên file cho thầy.

3. **Dựng điều kiện khó** như ảnh `docs\anh_mau_bo_kho_1.jpg`, `docs\anh_mau_bo_kho_2.jpg`: tấm nền
   xanh lá (đã dán bộ thẻ thứ hai) phủ kín mặt bàn, cả trên lẫn dưới vạch đen ngang giữa bàn; giảm
   đèn phòng.

4. **Kiểm điều kiện ảnh.** Bàn như khi chạy nhưng không có vật, robot ở HOME. Chạy:

   ```
   python scripts/29_check_condition.py --session $B --condition hard
   ```

   **Đạt khi:** in `HARD CONDITION: PASS`. In `FAIL`: sửa theo các dòng dưới nó (`bare table`: phủ
   kín tấm nền; `not green enough`: sai tấm nền; `too dark`, `too bright`: chỉnh đèn), chạy lại tới
   khi `PASS`. Sửa hết mà vẫn `FAIL`: DỪNG. Từ lúc `PASS` tới hết bước 7 không đổi đèn, rèm, tấm nền.
   **Ghi** ra giấy: các dòng số đo ngay trên dòng `PASS`.

5. **Đối chứng đầu buổi, 20 lượt.** Tháo vít chỉ (ảnh `docs\anh_vit_chi.jpg`) nếu còn gắn; không
   còn ai trong cell; robot ở HOME; xoay chìa sang REMOTE, giữ REMOTE tới hết bước 7. Chạy, rồi dồn
   file:

   ```
   python scripts/03_run_experiment.py --mode real --depth-mode rgbd --pose-list config/pose_lists/hard_v2.csv --pose-slice 200:220 --session-id $B --block-id doichung-dau --operator-id PQM --ik-source yrc --tool-no 1 --confirm-each-trial --no-viewport-mirror --lighting dim
   ```

   ```
   mkdir results\$B\doichung
   move results\experiment_real_* results\$B\doichung\
   move results\telemetry_* results\$B\doichung\
   ```

   **Đạt khi:** qua `Trial 20/20`, in `RESULTS: success_rate=` rồi `CSV:`. Lệnh dừng trước lượt 1
   với dòng `STARTUP REFUSED`: sửa đúng điều dòng đó nêu, chạy lại lệnh.

6. **Chiến dịch chính, 300 lượt.** Chạy lệnh dưới (không có `--dry-run`), một lần mỗi buổi, rồi dồn
   file. Trong buổi không mở thư mục `results\blinding_keys\` và file `logs\experiment.log`.

   ```
   python tools/run_blinded_campaign.py --arm anchored "--depth-mode rgbd --model-path models/e3_anchored.pt" --arm wide "--depth-mode rgbd --model-path models/e3_wide.pt" --arm real_only "--depth-mode rgbd --model-path models/e2_seed1_best.pt" --pose-list config/pose_lists/hard_v2.csv --first-pose $F --trials 100 --block 25 --session $B --operator PQM --common "--mode real --ik-source yrc --tool-no 1 --confirm-each-trial --no-viewport-mirror --save-frames --frames-dir D:/Scientific/Dataset/DigitalTwin/frames --lighting dim"
   ```

   ```
   move results\experiment_real_* results\$B\
   move results\telemetry_* results\$B\
   ```

   **Đạt khi:** in `All 12 block(s) done.` Lệnh dừng giữa chừng (`Campaign stopped here`): buổi lỗi.

7. **Đối chứng cuối buổi, 20 lượt.** Chạy, rồi dồn file. Đạt như bước 5.

   ```
   python scripts/03_run_experiment.py --mode real --depth-mode rgbd --pose-list config/pose_lists/hard_v2.csv --pose-slice 200:220 --session-id $B --block-id doichung-cuoi --operator-id PQM --ik-source yrc --tool-no 1 --confirm-each-trial --no-viewport-mirror --lighting dim
   ```

   ```
   move results\experiment_real_* results\$B\doichung\
   move results\telemetry_* results\$B\doichung\
   ```

8. **Kiểm, gom file.**

   ```
   python -c "import csv,glob,sys;rows=[d for f in glob.glob(sys.argv[2],recursive=True) for d in csv.DictReader(open(f,encoding='utf-8'))];p={d['pose_id']:d['class_hint'] for d in csv.DictReader(open(sys.argv[1],encoding='utf-8'))};k=[d for d in rows if d['class_name']];m=sum(1 for d in k if d['class_name']==p.get(d['pose_id']));print('dung lop:',m,'/',len(k))" config/pose_lists/hard_v2.csv "results/$B/**/experiment_real_*.csv"
   ```

   ```
   python -c "import csv,glob,os,sys;r=[d for f in glob.glob(sys.argv[1]) for d in csv.DictReader(open(f,encoding='utf-8'))];k=[d for d in r if not (d.get('frame_path') and os.path.isfile(d['frame_path']) and os.path.isfile(d['frame_path'][:-4]+'_depth.png'))];print('thieu anh:',len(k),'/',len(r))" "results/$B/experiment_real_*.csv"
   ```

   **Đạt khi:** lệnh đầu in `dung lop: a / b` với a chia b từ 0,9 trở lên (ví dụ b = 340 thì a từ
   306 trở lên); lệnh thứ hai in `thieu anh: 0 / 300`. Không đạt: DỪNG. Đạt thì gom:

   ```
   copy results\blinding_keys\key_$B.json results\$B\
   copy logs\experiment.log results\$B\
   move results\do_sang_$B* results\$B\
   mkdir results\$B\anh
   Get-ChildItem results\$B -Recurse -Filter "experiment_real_*.csv" | ForEach-Object { Import-Csv $_.FullName } | Where-Object { $_.frame_path } | ForEach-Object { Copy-Item $_.frame_path results\$B\anh\; Copy-Item ($_.frame_path -replace '\.png$','_depth.png') results\$B\anh\ }
   ```

9. **Gửi.** Mở `notepad results\$B\nhat_ky.txt`, chép vào các dòng đã ghi ở bước 4 và mọi việc bất
   thường trong buổi; gửi cả thư mục `results\$B\` cho thầy như các buổi E2. Thư mục
   `D:\Scientific\Dataset\DigitalTwin\frames` không gửi, không xoá. Hết buổi: cất tấm nền, bật đèn.

**DỪNG nếu**, ở bước 5 đến 7: má kẹp hoặc vật chạm mặt bàn, thẻ hay vật khác; robot đi sai hướng.

**Lượt hỏng** (robot không chạy và vật còn trên thẻ; có alarm mà lệnh vẫn chạy tiếp; rơi vật) là
kết quả của chính lượt đó: không chạy lại lượt. Máy in `Real cell: robot left where it stopped; the
operator returns it to HOME`: trả lời `y`/`n` như B1; khi máy chờ ở câu `Placed it?`, xoá alarm, đưa
robot về HOME, chìa ở REMOTE, rồi mới bấm ENTER. Được nghỉ khi máy chờ ở câu `Placed it?`; không
đóng cửa sổ PowerShell.

**Buổi lỗi** (lệnh dừng giữa chừng, dừng khẩn, mất kết nối, camera mất tín hiệu, đã chạy lượt khi
bước 4 chưa `PASS`, bước 8 không đạt): DỪNG, gửi `logs\experiment.log` và các file của buổi, chờ trả
lời. Thầy xác nhận lỗi thì: làm bước 1 với tên buổi lỗi, chạy khối dưới (lệnh nào báo không tìm thấy
file thì bỏ qua), rồi chạy lại cả buổi từ bước 1, đúng tên buổi cũ.

```
$K = "results\khong_dung\${B}_$(Get-Date -Format yyyyMMdd_HHmm)"
mkdir results\khong_dung -Force
if (Test-Path results\$B) { Move-Item results\$B $K } else { mkdir $K }
move results\blinding_keys\key_$B.json $K\
move results\experiment_real_* $K\
move results\telemetry_* $K\
move results\do_sang_$B* $K\
copy logs\experiment.log $K\
```
