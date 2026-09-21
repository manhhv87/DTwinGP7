# Chuẩn bị dữ liệu và huấn luyện trên Linux

Đã có công cụ đóng gói và kiểm tra offline. Chưa chạy các lần train E3/E4/E5 trong
đợt sửa này; chưa có kết quả robot mới. Checkpoint E2 lịch sử có thể dùng để kiểm
tra phần mềm, nhưng việc dùng nó cho chiến dịch chính thức cần được ghi trong
[campaign manifest](CAMPAIGN_MANIFEST.md).

## 1. Đối chiếu dữ liệu trước khi đóng gói

Tại thời điểm kiểm tra 21/09/2026, thư mục
`D:/Scientific/Dataset/DigitalTwin/_work/yolo` có **1.318 ảnh train và 223 ảnh val**.
Danh sách `_work/splits/train.txt` có 1.256 ảnh; exporter thêm 62 ảnh từ
`negatives.txt`. Paper đang giữ số liệu theo các split gốc và đánh dấu việc đối
chiếu này. Không suy ra bộ dữ liệu của lần train Linux chỉ từ thư mục Windows.

Trước chiến dịch, lấy danh sách ảnh/hash hoặc cache/log dữ liệu từ lần train Linux
để xác định có dùng 62 ảnh bổ sung hay không. Cùng một real-training manifest phải
được giữ ở tất cả nhánh so sánh. Không tự bỏ ảnh để làm cho số file khớp paper.

Từ thư mục repo trên Windows, kiểm tra **chỉ đọc**, không tạo package hay train:

```powershell
.venv/Scripts/python.exe scripts/23_launch_retrain.py `
  --real D:/Scientific/Dataset/DigitalTwin/_work/yolo `
  --experiment E2 --version 0 --inspect-only
```

Để kiểm tra riêng tập khai báo 1.256 ảnh, thêm hai đối số dưới đây. Đây là kiểm tra
một tập ứng viên, không xác nhận rằng baseline lịch sử đã dùng nó:

```text
--real-train-list D:/Scientific/Dataset/DigitalTwin/_work/splits/train.txt
--expected-real-train 1256
```

Sau khi xác minh, lưu quyết định vào campaign manifest. Khi đóng gói thật, truyền
`--expected-real-train` bằng số đã khóa; nếu chỉ chọn một tập con, luôn truyền
allowlist. Mặc định công cụ lấy toàn bộ ảnh trong `images/train`.

## 2. Chuẩn bị synthetic

E3 và mỗi nhánh E5 cần **3.000 ảnh training**, không phải 3.000 ảnh trước khi chia
train/val. Dùng `--make-labels --val-frac 0` và real validation chung trong bước
đóng gói. Các ảnh synthetic cho E6 phải được sinh riêng.

Xem [E5_ABLATION_PROTOCOL.md](E5_ABLATION_PROTOCOL.md) về năm can thiệp, ghép cặp
scene và các lệnh render. Chỉ render chiến dịch sau khi calibration, kích thước
mesh, camera, vị trí vật và mẫu ảnh thử đã được kiểm tra. Calibration placeholder
không phải phép đo thật.

Sampler dùng `seed + index`. E3/full E5 có thể dùng chung dải 0..2999 để ghép cặp
recipe. E4 phải dùng dải mới: ví dụ update 1 dùng guided seed 10000 và control
20000; update 2 dùng 30000 và 40000, mỗi đợt 1000 ảnh. Lưu cả seed, start index,
số ảnh và scene manifest. Không dùng seed 1000/2000 với E3 seed 0 vì các dải giao nhau.

## 3. Tạo package chuyển máy

Ví dụ dưới đây giả sử đã xác minh và khóa toàn bộ **1.318 ảnh** của export hiện tại.
Nếu quyết định đã khóa khác, thay count và truyền allowlist tương ứng trước khi chạy.
`--out` là thư mục cha; `--version 1` tạo thư mục `dsv1`. Mỗi cấu hình dùng một
thư mục cha riêng để tránh nhầm cùng số phiên bản giữa các nhánh.

```powershell
.venv/Scripts/python.exe scripts/23_launch_retrain.py `
  --real D:/Scientific/Dataset/DigitalTwin/_work/yolo `
  --synth data/synth/anchored_k2/dataset `
  --experiment E3 --version 1 --out data/packages/e3_k2 `
  --expected-real-train 1318 --expected-synth-train 3000 --zip
```

`k2` chỉ là tên ví dụ, không phải kết quả lựa chọn κ. Tạo các package wide-range,
κ=1/2/4 với cùng real train/validation và training recipe. Real-only E2 dùng
`--experiment E2` và không có `--synth`; không cần train lại baseline chỉ để kiểm
tra checkpoint đang có. Nếu dữ liệu/recipe lịch sử khác chiến dịch mới, phải ghi
rõ hoặc chạy baseline so sánh tương ứng.

Package giữ đủ **5 lớp theo đúng thứ tự**: carton, plastic_box, wood_box, metal_box,
inox_box. Công cụ kiểm nhãn polygon, giá trị hữu hạn, file nhãn rỗng cho ảnh âm
tính, ảnh trùng nội dung, số ảnh và SHA-256 của từng ảnh/nhãn. Nó không tự sửa
dataset. Tên file giống nhau từ nhiều nguồn được đặt trong thư mục nguồn riêng.
Validation chỉ lấy từ ảnh thật. File nguồn ngoài package không cần có trên Linux.

Các file quan trọng:

| File | Vai trò |
|---|---|
| `manifest.json` | File ảnh/nhãn, split, nguồn và hash; tổng số ảnh và instance |
| `training.json` | Recipe, seed, initialization và quy tắc chọn model |
| `dataset.yaml` | Mô tả dataset dùng đường dẫn tương đối |
| `train.py` | Runner độc lập, không cần checkout repo trên Linux |
| `requirements-training.txt` | Pin Ultralytics 8.4.66 |
| `initial.pt` | Chỉ có ở E4; checkpoint cha đã băm nội dung |

Hash phát hiện file trùng hoặc bị đổi, không phát hiện mọi ảnh gần giống và không
chứng minh độc lập capture session. Cần lưu session/pose manifest riêng.

## 4. Chạy trên Linux GPU

Giải nén/copy toàn bộ thư mục `dsvN`. Tạo môi trường Python, cài PyTorch phù hợp
CUDA của máy, rồi từ thư mục package:

```bash
python -m pip install -r requirements-training.txt
python train.py --dry-run
python train.py --device 0
python -m pip freeze > environment.freeze.txt
```

`--dry-run` băm lại các file, không import torch và không train. Khi train,
`dataset.runtime.yaml` được tạo với đường dẫn của máy hiện tại. Không dùng trực
tiếp YAML Windows cũ. Runner yêu cầu một GPU cụ thể hoặc `cpu`; nhiều GPU cần
protocol và kiểm tra riêng. GPU/runtime thực tế được ghi trong hồ sơ run.

Recipe mặc định E2/E3/E5: 130 epochs, `imgsz=1280`, batch 8, `nbs=64`,
`patience=0`, AdamW, `lr0=0.001111`, `momentum=0.9`, `warmup_bias_lr=0`.
E2/E3 dùng seed 0..4; E4/E5 dùng seed 0..2. Recipe đầy đủ nằm trong `training.json`.
Đây là recipe đã chỉ định cho so sánh sắp tới; không khẳng định mọi file log
lịch sử đều ghi các tham số giống nhau.

Có thể chia việc chạy theo các seed đã khai:

```bash
python train.py --device 0 --seeds 0
python train.py --device 0 --seeds 1 2 3 4
```

Runner từ chối ghi đè thư mục run cũ. Nếu run hỏng, giữ hồ sơ lỗi và tạo package/run
mới có ghi nhận lý do. Không có resume tự động. Nếu framework tự giảm batch do
thiếu VRAM hoặc bỏ ảnh lỗi, runner dừng để tránh báo một run khác ngân sách đã khóa.
Điều chỉnh recipe phải áp dụng nhất quán cho cả so sánh và ghi nhận thay đổi.

## 5. E4 cần checkpoint cha và ngân sách riêng

Khóa số epochs fine-tune trước khi xem kết quả. `--epochs` bắt buộc ở E4; không
điền một số tùy ý chỉ để lệnh chạy. Ví dụ khung lệnh dưới đây dùng các biến do
người thực hiện điền từ campaign manifest:

```bash
python scripts/23_launch_retrain.py --experiment E4 --version 1 \
  --real "$REAL" --synth "$INITIAL_SYNTH" "$GUIDED_UPDATE1" \
  --model "$SELECTED_E3_CHECKPOINT" --epochs "$E4_EPOCHS" \
  --expected-real-train "$REAL_COUNT" --expected-synth-train 4000 \
  --out data/packages/e4_guided
```

Control có cùng f0, cùng số ảnh mới và training schedule; chỉ dùng ảnh unguided
của chính nó. Update 2 giữ dữ liệu tích lũy của nhánh tương ứng, tổng 5.000 ảnh
synthetic, và tiếp tục checkpoint update 1 của **cùng nhánh và training seed**.
Vì mỗi package nhận một `--model`, tạo package riêng cho từng seed ở update 2,
với `--seeds 0`, `--seeds 1` hoặc `--seeds 2` và đúng checkpoint cha.

## 6. Hồ sơ trả về và lựa chọn model

Giữ `runs/seedN/run_record.json`, `args.yaml`, `results.csv`, `weights/best.pt`,
`environment.freeze.txt` cùng package manifest. Run record có SHA-256 checkpoint,
seed, recipe thực tế, số bước optimizer thực thi, thời gian, và điểm validation
chưa làm tròn tại mỗi lần lưu. Run thiếu epochs/bằng chứng được ghi `failed`.
Đây là kiểm soát thực thi, không thay thế đánh giá model.

Giữ checkpoint theo **tổng box + mask mAP@0.5:0.95**. Với E3, so κ bằng trung
bình điểm đó trên đủ năm seed; hòa thì κ nhỏ hơn. Chọn deployed seed trong cấu
hình bằng cùng điểm; hòa dùng thứ tự seed đã khai. Không dùng CSV đã làm tròn
để giải quyết chênh lệch rất nhỏ; dùng `selected_validation` trong run record.
Không dùng kết quả cuối cùng để chọn κ/seed hay quyết định dừng E4.

Seed 1 E2 hiện có là lựa chọn lịch sử theo mask-only giữa các seed; không đổi
mô tả đó thành composite hồi tố. Image mAP và task success là hai kết quả riêng.
Theo [mã nguồn Ultralytics 8.4.66](https://github.com/ultralytics/ultralytics/blob/v8.4.66/ultralytics/utils/metrics.py),
fitness segmentation cộng fitness box và mask; runner kiểm tra sự nhất quán đó
và lưu score lúc checkpoint được ghi, trước lượt re-validation cuối.

Sau khi chọn checkpoint, dùng CLI trong
[hướng dẫn cell thật](HUONG_DAN_THI_NGHIEM_CELL_THAT.md), kiểm calibration thật và
khóa cấu hình trước evaluation. GUI hiện chưa gộp đầy đủ real overlay/preflight
như CLI; không coi việc preview chạy được là xác nhận sẵn sàng thí nghiệm robot.
