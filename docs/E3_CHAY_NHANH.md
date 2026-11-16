# CHẠY E3: TỜ NGẮN

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

- Trên tay dạy vào MAIN MENU → ROBOT → TOOL → chọn tool số 1, chụp ảnh màn hình (ảnh thấy
  đủ X, Y, Z, Rx, Ry, Rz). Không sửa số nào.
- Chìa ở TEACH, chọn TOOL01, hệ Robot, rồi chạy:

  ```
  python scripts/30_measure_tcp_twin_vs_controller.py --hse-ip 192.168.125.100
  ```

  Dòng đầu phải có `controller reports tool 1; expected 1`. Số tool khác 1: chọn TOOL01 trên tay dạy.
- Jog tay, tốc độ thấp, lần lượt tới 20 tư thế trong bảng dưới. Mỗi tư thế:
  1. đưa má kẹp tới trên chỗ ghi trong bảng, cao khoảng số mm ghi trong bảng (đo bằng thước từ mặt
     bàn tới đầu dưới má kẹp), má kẹp hướng thẳng xuống mặt bàn như khi chạm ở A4;
  2. dòng nào ghi phím thì bấm phím jog đó (hệ ROBOT) tới khi cổ tay xoay hay nghiêng khoảng số độ
     ghi trong bảng, ước bằng mắt (45 độ là một phần tám vòng);
  3. buông phím, robot đứng yên thì bấm ENTER trên máy tính.

  | Tư thế | Chỗ | Cao | Rồi bấm phím |
  |---|---|---|---|
  | 1 đến 5 | thẻ 1, 3, 11, 19, 21 | 50 mm | không bấm |
  | 6 đến 10 | thẻ 1, 3, 11, 19, 21 | 200 mm | xoay quanh Z (Rz), khoảng 45 độ |
  | 11 đến 14 | thẻ 2, 10, 12, 20 | 150 mm | thẻ 2 và 20: nghiêng quanh X (Rx); thẻ 10 và 12: nghiêng quanh Y (Ry); khoảng 15 độ |
  | 15 đến 17 | dấu điểm thả | 50, 150, 250 mm | không bấm |
  | 18 | dấu điểm thả | 150 mm | xoay quanh Z (Rz), khoảng 45 độ, ngược chiều với tư thế 6 đến 10 |
  | 19 | dấu điểm thả | 150 mm | nghiêng quanh X (Rx), khoảng 15 độ |
  | 20 | dấu điểm thả | 150 mm | nghiêng quanh Y (Ry), khoảng 15 độ |

  Độ cao và góc chỉ cần gần đúng: ở mỗi tư thế phần mềm đọc vị trí thật của robot.

  Sau mỗi lần bấm ENTER, máy đọc vị trí robot ba lần liền, cách nhau 0,2 giây. Hai trường hợp phải
  làm lại tư thế đang đọc:
  - máy in `NOT STILL`: robot còn nhích trong lúc đọc (thường do vừa buông phím jog), tư thế này
    không được ghi. Không jog thêm, đợi vài giây cho robot dừng hẳn, bấm ENTER lại;
  - máy in `TOOL MISMATCH`: tay dạy đang chọn tool khác TOOL01. Chọn lại TOOL01 trên tay dạy, bấm
    ENTER lại.
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

2. **Kiểm đầu buổi.** Buổi nào cũng kiểm lại ba điều dưới, vì máy có thể đã bị đổi từ lần trước.

   Ba file mô hình vẫn đúng. **Đạt khi:** ba dòng đều `True`.

   ```
   "e3_anchored.pt 6d5e2d7bf969f05163a325d2c10d582302850ca0f893f737fc8dce45902a9f95", "e3_wide.pt 311af74bdcc3a27e1b398058492a81acbf2c00fbe9d27c90439ac9f6025bd2cd", "e2_seed1_best.pt 52e25b0349d1586b45f7a0077e4b38479d845621435e85722555ea65cf5e702a" | ForEach-Object { $ten, $ma = $_ -split ' '; "{0}  {1}" -f $ten, ((Get-FileHash "models\$ten").Hash -eq $ma) }
   ```

   Mã chạy vẫn không đổi. **Đạt khi:** cả hai lệnh không in dòng nào.

   ```
   git diff --stat 3d06434 HEAD -- scripts src config tools
   git status --porcelain --untracked-files=no
   ```

   Không còn file kết quả của lần chạy trước nằm rời trong `results\` (để file các buổi không lẫn
   vào nhau). **Đạt khi:** cả hai lệnh không in dòng nào.

   ```
   dir results\experiment_real_*
   dir results\telemetry_*
   ```

   Không đạt điều nào: DỪNG, chụp màn hình gửi thầy.

3. **Dựng điều kiện khó** như ảnh `docs\anh_mau_bo_kho_1.jpg`, `docs\anh_mau_bo_kho_2.jpg`: tấm nền
   xanh lá (đã dán bộ thẻ thứ hai) phủ kín mặt bàn, cả trên lẫn dưới vạch đen ngang giữa bàn; giảm
   đèn phòng. Từ buổi thứ hai, tấm nền đã nằm sẵn trên bàn: chỉ kiểm nó còn phẳng, phủ kín, không
   xê dịch.

4. **Kiểm điều kiện ảnh.** Bàn như khi chạy nhưng không có vật, robot ở HOME. Chạy:

   ```
   python scripts/29_check_condition.py --session $B --condition hard
   ```

   Lệnh in các dòng số đo rồi dòng kết luận, ví dụ:

   ```
   Brightness, whole frame: 30.2/255, cards: 104.0
   TABLE, UPPER PART: green share 0.453  brightness 22.1  bare 0%
   TABLE, LOWER PART: green share 0.398  brightness 26.1  bare 0%
   HARD CONDITION: PASS
   ```

   Số đo và kết luận được lưu tự động vào `results\do_sang_<tên buổi>.txt`, không phải chép tay.

   **Đạt khi:** in `HARD CONDITION: PASS`. In `HARD CONDITION: FAIL`: các dòng bắt đầu bằng `-` ngay
   dưới nó nêu việc phải sửa (`bare table`: phủ kín tấm nền; `not green enough`: sai tấm nền;
   `too dark`, `too bright`: chỉnh đèn); sửa xong chạy lại lệnh, tới khi `PASS`. Sửa hết mà vẫn `FAIL`: DỪNG. Từ lúc `PASS`
   tới hết bước 7 không đổi đèn, rèm, tấm nền.

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

   **Đạt khi:** máy chạy hết 20 lượt, cuối cùng in một dòng bắt đầu bằng `RESULTS: success_rate=` và
   một dòng bắt đầu bằng `CSV:`.

   Máy không chạy lượt nào mà in dòng bắt đầu bằng `STARTUP REFUSED:` (máy từ chối bắt đầu): sửa theo
   bảng dưới, rồi chạy lại đúng lệnh trên.

   | Dòng `STARTUP REFUSED:` có chữ | Nghĩa | Sửa |
   |---|---|---|
   | `has active alarm` | tủ điều khiển còn alarm | xoá alarm trên tay dạy |
   | `reports controller error` | tủ điều khiển báo lỗi | xoá lỗi trên tay dạy |
   | `HOLD is active` | đang HOLD | nhả HOLD |
   | `already has a running JOB` | có job đang chạy | dừng job trên tay dạy |
   | `not in REMOTE mode` | chìa chưa ở REMOTE | xoay chìa sang REMOTE |

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

9. **Ghi nhật ký, gửi, dọn.**
   1. Mở nhật ký bằng lệnh dưới (Notepad hỏi tạo file mới: chọn Yes):

      ```
      notepad results\$B\nhat_ky.txt
      ```

      Chỉ ghi việc bất thường mà máy không tự biết: thẻ bị xê dịch, camera bị chạm, có người vào cell
      giữa buổi, vật dự phòng lọt vào khung hình. Không có gì thì ghi `khong co`. Lưu, đóng Notepad.
      Số lượt, kết quả, giờ chạy, người đặt vật và số đo ánh sáng máy đã tự ghi.
   2. Gửi thầy cả thư mục `results\$B\` (ví dụ `results\e3-1\`).
   3. Thư mục ảnh lớn `D:\Scientific\Dataset\DigitalTwin\frames` để nguyên: không gửi, không xoá.
      Ảnh cần cho phân tích đã được chép vào `results\$B\anh\` ở bước 8.
   4. Để nguyên tấm nền xanh và bộ thẻ trên bàn cho buổi sau (E3, E4 đều chạy điều kiện khó.
   
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

## 4. SAU BUỔI `e3-2`

E3 xong. Các buổi tiếp theo trên cell là E4 (`e4adapt-1`, `e4adapt-2`, `e4-1`, `e4-2`), chạy theo
tờ hướng dẫn E4 thầy gửi sau, vì các buổi đó cần mô hình huấn luyện giữa các buổi. Không có buổi E5
trên cell.
