# E5: protocol loại từng yếu tố của bộ sinh ảnh anchored

Trạng thái: đã có mã để tạo scene specification và kiểm tra tính nhất quán; chưa
render chiến dịch E5, chưa huấn luyện các model E5 và chưa thu kết quả robot. Đây là
protocol chuẩn bị trước thực nghiệm, không phải báo cáo kết quả.

E5 so sánh sáu nhánh: `none` (full recipe) và năm nhánh mỗi nhánh chỉ cố định hoặc
loại **một** yếu tố. Chỉ dùng recipe `anchored`, cùng giá trị `kappa` được chọn
trước từ E3, cùng calibration, cấu hình và ngân sách. Không kết hợp E5 với
`--from-failures`; tác động của failure conditioning thuộc E4.

## Can thiệp chính xác

| Giá trị `--ablation` | Thay đổi sau khi đã lấy mẫu toàn bộ scene |
|---|---|
| `none` | Giữ nguyên full recipe. Không đổi trình tự hoặc giá trị các draw cũ. |
| `illumination` | Giữ số đèn và vị trí tại `lighting.anchor_positions_mm`; mỗi đèn dùng `anchor_intensity_w`, `anchor_color_temp_k`. Bỏ jitter vị trí, công suất và nhiệt độ màu. |
| `background` | Mặt bàn dùng `background.anchor_albedo_rgb` và `anchor_roughness`, tại `objects.table_z_mm`. Trong recipe hiện tại chỉ albedo được randomize; roughness vốn đã cố định. |
| `distractors` | Thay danh sách distractor bằng `[]`. Không loại vật cần nhận diện. |
| `camera` | Cố định toàn bộ `T_BC_mm` tại file calibration đã cung cấp và nội tại tại `camera.intrinsics`; bỏ cả extrinsic và intrinsic jitter. |
| `pose` | Cố định vị trí và yaw theo số vật như mô tả bên dưới. Giữ nguyên draw về số vật, lớp, mesh/kích thước và vật liệu từng vật. |

Với `pose`, gọi `n` là số vật đã được lấy mẫu. Chọn trục X hoặc Y có khoảng tọa độ
rộng hơn; nếu bằng nhau chọn X. Chia khoảng đó thành `n` ô bằng nhau, đặt vật thứ
`i` tại tâm ô thứ `i`. Tọa độ trục còn lại là trung điểm của khoảng cấu hình;
`z = table_z_mm`; yaw là trung điểm `yaw_range_deg`. Thứ tự vật được giữ nguyên.
Đây là **một bố trí xác định có điều kiện theo số vật**, không phải tất cả ảnh có
một vị trí duy nhất khi số vật thay đổi.

Ví dụ với cấu hình hiện tại X = [510, 640] mm, Y = [-175, 215] mm (vùng đặt vật tính từ hiệu
chuẩn 18/09/2026, xem `config/synthgen.yaml`) và yaw = [-90, 90]°. Trục Y rộng hơn nên các ô
chia theo Y, X giữ ở trung điểm 575:

| Số vật | Tâm XY (mm) | Yaw |
|---|---|---|
| 1 | (575, 20) | 0° |
| 2 | (575, -77.5), (575, 117.5) | 0° cho mỗi vật |
| 3 | (575, -110), (575, 20), (575, 150) | 0° cho mỗi vật |

Đây là vị trí tham chiếu đề xuất trong bao lấy mẫu, **không phải vị trí đã đo hoặc
đã xác nhận khả thi tại robot**. Sampler từ chối nếu khoảng cách các ô không đạt
`min_separation_mm` ở số vật lớn nhất. Kiểm tra đó chỉ xét khoảng cách tâm; không
thay thế kiểm tra kích thước mesh, che khuất, ảnh nhìn thấy được hay an toàn robot.

Màu và độ nhám **vật thể** vẫn được randomize ở mọi nhánh. `background` không có
nghĩa là bỏ vật liệu vật thể. Tên cũ như “texture ablation” không diễn tả chính xác
can thiệp này và không nên dùng trong bảng kết quả.

## Ghép cặp và provenance

Sampler dùng `default_rng(seed + index)`. Mỗi nhánh lấy đầy đủ các draw theo đúng
thứ tự full recipe, rồi mới thay trường tương ứng với can thiệp. Vì vậy với cùng
`seed/index`, các yếu tố còn lại của scene giống nhau, kể cả lớp và vật liệu sau
yếu tố bị bỏ. Điều này ghép cặp **scene specification**; không bảo đảm ảnh render
giống bit hoặc triệt tiêu mọi tương tác thị giác giữa các yếu tố.

Mỗi `scene_*.json` lưu `ablation.protocol_version`, `factor` và `reference` đã giải
quyết thành giá trị cụ thể. `specs/manifest.json` lưu thêm toàn bộ cấu hình generator,
`T_BC_mm`, sigma extrinsic hiệu lực, nguồn sigma, `kappa`, seed và danh sách scene.
Các trường `camera/lights/background/objects/distractors` của scene là dữ liệu
thực tế renderer sử dụng. Manifest từ chối trộn nhánh, cấu hình, calibration,
sigma, seed hoặc ghi đè scene index đã có. Các manifest cũ thiếu provenance này
phải dùng thư mục chạy mới; không suy đoán cấu hình của một bộ ảnh lịch sử.

Sử dụng **một scene seed chung** cho cả sáu nhánh; ba training seed 0/1/2 dùng trên
cùng các bộ ảnh này. Không coi generation seed 0/1/2 là ba bộ ảnh độc lập: do quy
tắc `seed + index`, chúng có nhiều draw trùng nhau. Nếu cần chiến dịch scene độc
lập, phải định trước các dải seed/index không chồng nhau và ghi rõ estimand khác.

Giữ phiên bản Blender/BlenderProc, số samples/px, render device và cài đặt renderer
giống nhau; lưu thông tin này cùng manifest. Chọn `kappa`, anchor và protocol pose
trước khi xem điểm trên tập đánh giá cuối cùng. Giá trị fallback/proxy trong
checkout hiện tại chưa tự trở thành calibration thật chỉ vì sampler chạy được.

## Lệnh chuẩn bị trên Linux

Chạy từ thư mục gốc repo. Các đường dẫn calibration/cấu hình phải được thay bằng
bản đã đo, kiểm tra và khóa cho chiến dịch. Ví dụ sau dùng `kappa=2`; chỉ giữ giá trị
này nếu đúng lựa chọn đã chốt từ E3.

```bash
KAPPA=2
SCENE_SEED=0
for FACTOR in none illumination background distractors camera pose; do
  RUN="data/synth/e5_${FACTOR}"
  python scripts/20_generate_synth.py \
    --mode anchored --ablation "$FACTOR" --kappa "$KAPPA" \
    --n 3000 --seed "$SCENE_SEED" --out "$RUN" \
    --config config/synthgen.yaml \
    --calib config/calibration/T_base_camera.npy \
    --sigma config/calibration/T_base_camera_sigma.json
done
```

Lệnh trên chỉ lấy mẫu JSON, không gọi GPU hoặc robot. Render và chuyển nhãn là bước
riêng, thực hiện sau khi kiểm tra calibration và một lô ảnh nhỏ:

```bash
for FACTOR in none illumination background distractors camera pose; do
  RUN="data/synth/e5_${FACTOR}"
  blenderproc run src/synthgen/render_blenderproc.py -- \
    --scenes "$RUN/specs" --out "$RUN/render" --samples 64 --device gpu
  python scripts/20_generate_synth.py --make-labels --out "$RUN" \
    --val-frac 0 --seed 0
done
```

`--val-frac 0` giữ đủ 3000 ảnh cho **training**; dùng validation ảnh thật đã khóa
trong bước đóng gói huấn luyện. Không train trực tiếp bằng `dataset.yaml` vừa tạo
với synthetic validation rỗng. Mặc định `--val-frac 0.1` sẽ chỉ còn 2700 training
images, không đúng ngân sách 3000 ảnh training. Xác nhận đủ 3000 ảnh và nhãn mỗi
nhánh, `skipped = 0`; nếu render/conversion lỗi thì sửa hoặc hoàn tất các scene
thiếu, không âm thầm giảm ngân sách một nhánh. Giữ toàn bộ ảnh thật training giống
nhau giữa sáu nhánh; validation/test không được đưa vào training.

Đóng gói từng dataset cùng dữ liệu thật bằng `scripts/23_launch_retrain.py` và dùng
cùng training recipe đã khóa cho full/ablated. Dùng ba training seed 0/1/2; chọn
checkpoint bằng cùng quy tắc validation đã định trước. Đây là 18 lần huấn luyện
cho 6 nhánh × 3 seed. Chỉ tái sử dụng full-recipe run nếu khớp hoàn toàn dữ liệu,
calibration, `kappa`, recipe và quy tắc chọn model; ghi lại trường hợp tái sử dụng.

## Ghi nhận và diễn giải kết quả sau này

Mỗi run cần nối được: Git commit, manifest scene, cấu hình render, danh sách/hash
ảnh, training seed/arguments, checksum checkpoint, điểm validation và split đánh
giá. Đánh giá full và ablated trên cùng tập ảnh thật đã khóa. Với robot, dùng cùng
checkpoint-selection rule, pose IDs, chế độ định vị và protocol thử; không dùng
mAP để suy ra tỷ lệ gắp thành công.

Báo cáo `100 × (ablated − full)` theo điểm phần trăm cho mask mAP và task success,
kèm độ bất định và correction theo protocol thống kê của bài. Giá trị âm nghĩa
là metric giảm khi bỏ yếu tố; không định trước yếu tố nào sẽ thắng. Kết quả chỉ
mô tả can thiệp đã ghi trong bảng trên tại cell và recipe này. Ví dụ `camera`
gộp jitter nội/ngoại tại; `pose` thay đổi bố trí và các che khuất phát sinh. E5 này
không tách các tương tác đó thành tác động nhân quả độc lập của từng thông số.

## Kiểm tra phần mềm không dùng GPU/robot

```bash
python -m pytest tests/test_ablation.py tests/test_scene_sampler.py tests/test_synth_materials.py
```

Các test kiểm tra can thiệp đúng anchor, draw còn lại không đổi, pose theo số vật,
tính xác định, provenance, chặn trộn nhánh/ghi đè, chặn blind/failure conditioning
và CLI tạo JSON đúng nhánh. Chúng không đánh giá chất lượng ảnh render, detector
hoặc hiệu năng robot.
