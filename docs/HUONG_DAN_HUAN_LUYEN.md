# HƯỚNG DẪN CHO NGƯỜI HUẤN LUYỆN: PHÂN TÍCH, SINH ẢNH, HUẤN LUYỆN

Bản 22/09/2026, tách từ hướng dẫn thí nghiệm để bản cho người chạy cell ngắn lại. Nội dung
khớp Methods/Results của bài báo ngày 21/09/2026. Người chạy cell dùng
`HUONG_DAN_THI_NGHIEM_CELL_THAT.md`; file này là phần việc trên máy GPU và phần phân tích.

---

## Mã trong bài báo nằm ở pha nào

Bài báo gọi tên bốn đóng góp là C1 đến C4, và sáu thí nghiệm là E1 đến E6. Hướng dẫn này chia
theo pha, nên đây là bảng tra:

| Mã | Là gì | Chạy ở đâu trong hướng dẫn này |
|---|---|---|
| **C1** | Digital twin của cell, kèm cách đo độ trễ đồng bộ, sai số hình học, chi phí lập quỹ đạo | Pha 3 (E1) |
| **C2** | Ảnh tổng hợp neo theo hiệu chuẩn, so với ngẫu nhiên hoá dải rộng cùng ngân sách | Pha 5 (E3) |
| **C3** | Phân bổ ảnh theo lỗi, so với tăng cường ngẫu nhiên cùng ngân sách | Pha 5 (E4, có E5 bổ trợ) |
| **C4** | Ba chế độ độ sâu `rgbd`, `plane`, `fusion`, gồm cả vật inox phản chiếu | **Pha 4 (E2)**, 600 lượt của lệnh chiến dịch mù |

E5 và E6 không mang mã đóng góp riêng: E5 khảo sát yếu tố của recipe sinh ảnh, E6 cho khoảng cách mô phỏng với thật và
thời gian chu kỳ.

---

## So sánh E2: nộp cả họ một lần

Làm sau khi đã nhận đủ dữ liệu của các buổi. Người chạy cell dồn mỗi buổi vào `results/<tên buổi>/`,
khối đối chứng vào `results/<tên buổi>/doichung/`, file khoá ở `results/blinding_keys/key_<tên buổi>.json`.
Mỗi chiến dịch mù chia hai buổi: buổi 1 chạy tư thế 0:100, buổi 2 chạy 100:200, mọi nhánh trong
cùng buổi (cờ `--first-pose` của `run_blinded_campaign.py`), nên mỗi `pose_id` gặp mỗi nhánh đúng
một lần. Tên buổi và lịch ở B5, B6 của hướng dẫn cell.

So sánh cả họ trong một lệnh:

```
python scripts/04_analyze_results.py --csv "results/e2chuan-*/experiment_real_*.csv" --split-col depth_mode --baseline rgbd --pair-key pose_id --score-col human_ok > results/e2_phan_tich.txt
```

Mẫu `results/e2chuan-*/experiment_real_*.csv` lấy file của hai buổi, không lấy thư mục con
`doichung`. `--split-col depth_mode` tự gom các khối rời của cùng một nhánh lại, nên không phải nối
file bằng tay. Chương trình áp Holm cho cả họ. Chạy từng cặp riêng thì mỗi cặp tự thấy mình có ý
nghĩa. Ví dụ: hai kiểm định p = 0,03 và 0,04, sau Holm cả hai đều không.

Khối đối chứng đầu buổi so với cuối buổi, từng buổi một (hai khối phân biệt bằng cột `block_id`):

```
python scripts/04_analyze_results.py --csv "results/e2chuan-1/doichung/experiment_real_*.csv" --split-col block_id --baseline doichung-dau --pair-key pose_id --score-col human_ok > results/e2chuan-1_doichung.txt
```

`--score-col human_ok` dùng đánh giá toàn bộ thao tác; chấm đạt chỉ khi đúng vật được gắp,
vận chuyển và thả đúng tiêu chí đã khóa. Chạy thêm một lần **bỏ cờ đó** để lấy số của máy vào
`results/e2_phan_tich_may.txt`. Báo cáo hai cách ghi nhận và đối chiếu từng lượt bất đồng với
log; chênh lệch tổng không tự xác định nguyên nhân là rơi vật hay sai vị trí thả.

Script này **không tự ghi log ra file**, nên phần chuyển hướng `> results/e2_phan_tich.txt` là bắt
buộc: không có nó thì p sau Holm chỉ nằm trên màn hình.

Nếu một lượt nào đó không được chấm, chương trình **từ chối chạy** và nói còn bao nhiêu dòng
trống: chấm một nửa rồi so sánh sẽ lặng lẽ bỏ bớt lượt của đúng một nhánh.

**DỪNG nếu:** các lần chạy không dùng cùng danh sách và cùng `pose_id`. Phân tích tự từ chối, nhưng
lúc đó đã muộn.

**GHI SỐ:** tỷ lệ thành công từng cấu hình; chênh lệch bộ chuẩn với bộ khó. Bộ khó không thấp hơn
rõ rệt là một kết quả, không phải lần trượt: ghi lại, **không dựng lại điều kiện rồi chạy lại**.

---


## Pha 5 — E3 đến E6: sinh ảnh và huấn luyện trên máy GPU Linux

**Mục tiêu:** sinh ảnh tổng hợp, huấn luyện lại mô hình, chọn mô hình bằng tập validation ảnh
thật, rồi đo lại tỷ lệ gắp trên chính cell.

**Kết quả ra:** mỗi cấu hình một bộ `data/synth/<tên>/specs`, `/render`, `/dataset`; một gói
`data/packages/<tên>/dsv<k>/` (kèm `manifest.json`, `training.json`, `train.py`); trên máy GPU,
mỗi seed một thư mục `runs/seed<n>/` chứa `weights/best.pt`, `results.csv`, `run_record.json`;
`results/adaptation/<vòng>/failure_modes.json` của E4. Mô hình đem về cell phải kèm SHA-256.

**Phần nào cần robot:** sinh ảnh, render, gán nhãn, huấn luyện, chọn κ đều **không** cần robot.
Nhưng E3, E4, E5 chỉ có mAP là chưa đủ: mỗi mô hình mới phải mang về cell gắp thật.

**Số lượt gắp thật của Pha 5, chốt ngày 21/09/2026 theo hướng ít lượt nhất**, đã sửa bài báo cho
khớp: E3 gắp ở một κ đã chọn (400); E4 chạy adaptation 80 lượt mỗi vòng và chỉ đánh giá sau vòng
cuối (400); E5 gắp ba yếu tố neo (600). Lịch từng buổi, tên file mô hình và bảng tổng số lượt ở B5, B6 của hướng dẫn cell. Mọi lượt Pha 5 đều chạy trên
**bộ khó** (`hard_v2.csv`, nền xanh và thiếu sáng như Pha 4).

### E3: wide-range so với anchored, và quét κ

Mỗi recipe dùng cùng ngân sách **3000 ảnh tổng hợp** đưa vào train, cộng tập ảnh thật đã khóa.
`--mode blind` trong lệnh là recipe wide-range của bài. Sinh bốn cấu hình, cùng seed sinh cảnh:

```bash
python scripts/20_generate_synth.py --mode blind --n 3000 --seed 0 --out data/synth/blind
python scripts/20_generate_synth.py --mode anchored --kappa 1 --n 3000 --seed 0 --out data/synth/anchored_k1
python scripts/20_generate_synth.py --mode anchored --kappa 2 --n 3000 --seed 0 --out data/synth/anchored_k2
python scripts/20_generate_synth.py --mode anchored --kappa 4 --n 3000 --seed 0 --out data/synth/anchored_k4
```

Bộ sinh đọc `config/calibration/T_base_camera.npy` và `T_base_camera_sigma.json` (bản 18/09/2026,
đã trên git) để neo; thiếu sigma thì nó dùng số dự phòng trong YAML và **báo cảnh báo**, lô như
vậy không được dùng cho bài. Render rồi gán nhãn từng cấu hình:

```bash
blenderproc run --custom-blender-path <thư mục blender> src/synthgen/render_blenderproc.py -- --scenes data/synth/anchored_k2/specs --out data/synth/anchored_k2/render --samples 24 --device gpu
python scripts/20_generate_synth.py --make-labels --val-frac 0 --out data/synth/anchored_k2
```

`--val-frac 0` giữ đủ 3000 ảnh cho train; validation luôn là ảnh thật. Sau mỗi lô đếm lại số
ảnh và số nhãn, số spec không phải số ảnh đã render xong.

### Huấn luyện: đóng gói ở đây, chạy trên Linux

`23_launch_retrain.py` **không huấn luyện**; nó kiểm và đóng một gói mang sang máy GPU, val chỉ
gồm ảnh thật, năm lớp đúng thứ tự (kể cả `inox_box`):

```powershell
python scripts/23_launch_retrain.py --real D:/Scientific/Dataset/DigitalTwin/_work/yolo --synth data/synth/anchored_k2/dataset --experiment E3 --version 1 --out data/packages/e3_k2 --expected-real-train 1318 --expected-synth-train 3000 --zip
```

Số 1318 là số ảnh đang có trong `_work/yolo/images/train` (đếm 21/09/2026): 1256 ảnh của
`splits/train.txt` cộng 62 ảnh negative mà `export_yolo.py` nối thêm. Chưa đối chiếu được
baseline trên Linux đã học bộ nào; đối chiếu log Linux rồi mới khóa con số này.

Trên máy GPU, giải nén gói rồi:

```bash
python -m pip install -r requirements-training.txt
python train.py --dry-run
python train.py --device 0 --seeds 0 1 2 3 4
```

Recipe cố định theo bài báo: Ultralytics 8.4.66, YOLOv8s-seg, 130 epoch, `imgsz=1280`,
`batch=8`, AdamW `lr0=0.001111`, `warmup_epochs=3`, `close_mosaic=10`; gói đã ghi sẵn trong
`training.json`, không sửa tay. E2 và E3 dùng năm seed `0..4`, E4 và E5 dùng ba seed `0..2`.
Mỗi seed một thư mục `runs/seed<n>/`, không ghi đè run đã có.

### Chọn κ và mô hình bằng validation, luật viết trước khi nhìn số

Luật đúng theo Methods, chép vào nhật ký trước khi chạy đánh giá:

1. Trong mỗi run, giữ checkpoint có **tổng box mAP@0.5:0.95 + mask mAP@0.5:0.95** trên
   validation ảnh thật cao nhất (thang 0–1, không cộng phần trăm).
2. Với mỗi κ, lấy trung bình điểm đó của đủ năm seed. **κ có trung bình cao nhất thắng; bằng nhau
   thì lấy κ nhỏ hơn.** Không có vùng hoà, không làm tròn trước khi so.
3. Trong cấu hình đã chọn, seed đem đi gắp là seed có điểm cao nhất; bằng nhau thì seed đứng
   trước trong `0, 1, 2, 3, 4`.
4. Không dùng bất kỳ kết quả gắp thật hay ảnh test nào để chọn.

`e2_seed1_best.pt` được chọn từ trước theo mask mAP, không phải theo tổng box + mask; ghi đúng
như vậy, không viết lại lịch sử.

### Đo mAP

Chấm `best.pt` đã chọn của từng run, không lấy dòng cuối `results.csv`:

```bash
yolo segment val model=runs/seed0/weights/best.pt data=dataset.yaml imgsz=1280 | tee val_seed0.txt
```

Lấy dòng `all`, cột `mAP50-95` của phần **Mask** cho bảng kết quả; điểm box + mask chỉ để chọn
mô hình. Bộ ảnh `test-hard` cũ đã dùng khi phát triển bộ sinh, nên kết quả trên nó là phân tích
phát triển, không phải kiểm định độc lập; bài báo ghi rõ giới hạn này.

### E4: vòng lặp học từ lỗi và nhánh đối chứng cùng ngân sách

Từ checkpoint E3 đã chọn `f0`, hai nhánh: guided (ảnh sinh theo lỗi) và control (ảnh sinh
ngẫu nhiên, **cùng số ảnh**). Mỗi vòng thêm 1000 ảnh mỗi nhánh; hai vòng.

Lỗi để khai thác lấy từ **buổi adaptation**, chạy trên đoạn 220:300 của `hard_v2.csv` (80 tư thế
mà chiến dịch chính 0:200 và khối đối chứng 200:220 không đụng tới), bằng mô hình guided của vòng
trước (vòng 1 là `f0` = `e3_anchored.pt`, vòng 2 là `e4_guided1.pt`), một nhánh, không cần mù:
buổi `e4adapt-1` và `e4adapt-2`, lệnh D của hướng dẫn cell.

Chỉ khai thác đúng CSV chính của buổi đó, không lấy thư mục `doichung`:

```bash
python scripts/21_mine_failures.py --runs "results/e4adapt-1/experiment_real_*.csv" --n-budget 1000 --out results/adaptation/e4_k1/failure_modes.json
python scripts/20_generate_synth.py --from-failures results/adaptation/e4_k1/failure_modes.json --kappa <κ đã chọn> --seed 10000 --out data/synth/loop_k1
python scripts/20_generate_synth.py --mode anchored --kappa <κ đã chọn> --n 1000 --seed 20000 --out data/synth/control_k1
```

Seed ở đây là seed **sinh cảnh**, và bộ sinh dùng `seed + chỉ số ảnh`: E3 đã chiếm 0–2999, nên
vòng 1 dùng 10000 (guided) và 20000 (control), vòng 2 dùng 30000 và 40000. Dùng seed 2000 cho
1000 ảnh control là sinh lại đúng 1000 ảnh cuối của E3, không phải ảnh mới.

Đóng gói E4 bắt buộc truyền `--model` (checkpoint cha) và `--epochs` (đã khóa); từ vòng 2 mỗi
seed nối tiếp checkpoint của chính nó nên mỗi cặp nhánh/seed một gói riêng, `--seeds <n>`.
Không có lỗi trong buổi adaptation thì ghi "không cập nhật" và dừng, không đọc lại file lỗi cũ.

Đánh giá gắp thật **một lần, sau vòng 2**: hai nhánh guided-2 và control-2 chạy chung một chiến
dịch mù trên `hard_v2.csv` 0:200; `f0` không chạy lại, lấy đúng lượt của nhánh anchored ở E3.
Checkpoint vòng 1 chỉ chấm bằng mAP validation.

### E5: bỏ từng yếu tố

Cờ `--ablation` nhận `illumination`, `background`, `distractors`, `camera`, `pose`, chỉ đi với
`--mode anchored`; mỗi cấu hình 3000 ảnh, cùng κ đã chọn và cùng seed sinh cảnh với recipe đầy đủ:

```bash
python scripts/20_generate_synth.py --mode anchored --kappa <κ đã chọn> --ablation illumination --n 3000 --seed 0 --out data/synth/e5_illumination
```

Định nghĩa "tắt một yếu tố" của từng cờ ghi trong `docs/E5_ABLATION_PROTOCOL.md`; đọc trước khi
sinh lô lớn. Huấn luyện ba seed mỗi cấu hình. Hiệu ứng báo cáo là `ablation − full` theo điểm
phần trăm, âm hay dương đều ghi, các phép so sánh hiệu chỉnh Holm chung một họ.

**Gắp thật cho ba yếu tố neo**: `camera`, `illumination`, `background`, ba nhánh trong một chiến
dịch mù trên `hard_v2.csv` 0:200; recipe đầy đủ không chạy lại, lấy lượt của nhánh anchored ở E3.
`distractors` và `pose` chỉ so bằng mAP: hai yếu tố đó cả hai recipe đều ngẫu nhiên hoá như nhau
(bảng yếu tố trong bài), nên không nói gì về việc neo theo hiệu chuẩn.

### Mang mô hình mới về cell

Đổi tên `best.pt` đã chọn theo đúng bảng dưới (lệnh chiến dịch của người chạy cell đã ghi sẵn các
tên này), gửi kèm SHA-256 (`sha256sum <file>`); người chạy cell kiểm bằng `Get-FileHash`.

| File | Là gì | Buổi dùng |
|---|---|---|
| `e3_anchored.pt` | anchored ở κ đã chọn, seed đã chọn; cũng là `f0` của E4 | `e3-1`, `e3-2`, `e4adapt-1` |
| `e3_wide.pt` | wide-range, seed đã chọn | `e3-1`, `e3-2` |
| `e4_guided1.pt` | guided sau vòng 1 | `e4adapt-2` |
| `e4_guided2.pt`, `e4_control2.pt` | hai nhánh sau vòng 2 | `e4-1`, `e4-2` |
| `e5_camera.pt`, `e5_illumination.pt`, `e5_background.pt` | ba ablation neo | `e5-1`, `e5-2` |

`model_path` trong `config\experiment.yaml` **giữ nguyên** `e2_seed1_best.pt`: khối đối chứng mọi
buổi chạy bằng nó, nên đối chứng so được giữa các buổi. Mô hình của từng nhánh đi qua cờ
`--model-path` trong tham số nhánh (`--arm anchored "--depth-mode rgbd --model-path models/e3_anchored.pt"`),
SHA-256 của file ghi vào metadata của từng lần chạy. Bước này bắt buộc có robot; không có nó thì
E3, E4, E5 chỉ còn mAP. Nút Experiment trong giao diện đồ hoạ không thay được lệnh này: nó không
chạy preflight hiệu chuẩn và độ sâu.

Các nhánh Pha 5 khác nhau ở mô hình, CSV không có cột nào nói điều đó, nên nhánh được đọc từ file
khoá theo `block_id` (`--key`, tạo cột `arm`); `--only` giữ đúng các nhánh của một họ so sánh:

```
python scripts/04_analyze_results.py --csv "results/e3-*/experiment_real_*.csv" "results/e2kho-*/experiment_real_*.csv" --key "results/blinding_keys/key_e3-*.json" "results/blinding_keys/key_e2kho-*.json" --split-col arm --baseline anchored --pair-key pose_id --score-col human_ok > results/e3_phan_tich.txt
python scripts/04_analyze_results.py --csv "results/e4-*/experiment_real_*.csv" --key "results/blinding_keys/key_e4-*.json" --split-col arm --baseline control2 --pair-key pose_id --score-col human_ok > results/e4_phan_tich.txt
python scripts/04_analyze_results.py --csv "results/e4-*/experiment_real_*.csv" "results/e3-*/experiment_real_*.csv" --key "results/blinding_keys/key_e4-*.json" "results/blinding_keys/key_e3-*.json" --split-col arm --only guided2 control2 anchored --baseline anchored --pair-key pose_id --score-col human_ok > results/e4_so_voi_f0.txt
python scripts/04_analyze_results.py --csv "results/e5-*/experiment_real_*.csv" "results/e3-*/experiment_real_*.csv" --key "results/blinding_keys/key_e5-*.json" "results/blinding_keys/key_e3-*.json" --split-col arm --only camera illumination background anchored --baseline anchored --pair-key pose_id --score-col human_ok > results/e5_phan_tich.txt
```

Họ Holm theo Methods: E3 gồm mọi so sánh của E3 (anchored với wide-range, anchored với real-only
lấy từ nhánh `real_only` của E2 khó); E4 là so sánh loop với control (lệnh thứ hai, so hai nhánh
với `f0` là nhánh `anchored` của E3, để đọc kèm); E5 là các yếu tố so với recipe đầy đủ (nhánh
`anchored` của E3), gắp thật ba yếu tố, hai yếu tố còn lại chỉ có mAP.

### E6: khoảng cách mô phỏng với thật, và thời gian chu kỳ

Khoảng cách mô phỏng với thật: chấm cùng một checkpoint trên một tập ảnh tổng hợp **sinh riêng
để đánh giá** (dải seed không giao dải huấn luyện) và trên tập ảnh thật đánh giá, báo
`100 × (mAP tổng hợp − mAP thật)`.

Thời gian chu kỳ: kế hoạch **100 chu kỳ hoàn chỉnh**. Lấy từ telemetry và CSV của Pha 4 nếu đủ
mốc; `05_analyze_telemetry.py` suy đoạn chuyển động từ vận tốc khớp, chưa có mốc riêng cho nhận
dạng, định vị, lập quỹ đạo và kẹp, nên bảng theo từng công đoạn cần thêm mốc ghi trước khi đo.
P50 và P95 tính trên tổng thời gian của từng chu kỳ, không cộng percentile từng đoạn. Thống kê
chỉ trên lượt thành công phải ghi rõ số lượt hỏng đã bỏ. Mốc 10 giây là mục tiêu ứng dụng, chưa
phải số đo.

**Những điều dễ sai ở pha này:**

- **Luôn thêm `--device gpu`.** Thiếu cờ đó, Blender có thể lặng lẽ render bằng CPU và chạy
  lâu gấp đôi. Có cờ này, nếu máy không có GPU dùng được nó sẽ dừng hẳn và báo, thay vì âm
  thầm chạy chậm.
- **Chạy thử 20 cảnh trước** rồi mới chạy 3000. Đo thời gian một khung để biết lô lớn mất bao
  lâu. Tốc độ đo trên laptop (24 mẫu/px): GPU 11,2 giây/khung, CPU 18,0 giây/khung — nghĩa là
  cả bốn cấu hình E3 (12000 ảnh) mất khoảng 37 giờ trên GPU laptop, 60 giờ trên CPU.
- **Sau mỗi lô render phải kiểm tay:** mở vài file nhãn xem có rỗng không, mở vài ảnh xem có
  đen hoặc trắng bệt không. Bốn lỗi render từng gặp đều kết thúc "thành công" với mã thoát 0
  trong khi nhãn rỗng hoặc ảnh sai.
- **Đừng chạy `pytest` trong lúc đang render.** Có một bài kiểm đo nhịp thời gian sẽ trượt oan
  vì hai việc tranh CPU.
- Ổ C: chỉ còn 15 GB trống trên 150 GB (đo ngày 12/09/2026). Đặt `data/synth` sang ổ D:
  trước khi render, và kiểm lại dung lượng trống trước mỗi lô.

---

## Phụ lục: ghi chú Blender

Máy hiện tại bị chặn `download.blender.org`, nên `blenderproc` không tự tải Blender được. Đã
tải sẵn qua mirror và giải nén tại `C:\Users\manhh\blender\blender-4.2.1-windows-x64`; khi
chạy render nhớ trỏ `--custom-blender-path` vào đó. Máy GPU mới cũng có thể vướng như vậy:
lấy `blender-4.2.1-linux-x64.tar.xz` từ mirror (clarkson, nluug, dotsrc, freedif).
