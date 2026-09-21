# Khóa protocol trước khi thu kết quả chính thức

Mẫu [campaign_manifest.example.yaml](../config/campaign_manifest.example.yaml) là hồ sơ khoa
học để điền cho từng chiến dịch. Nó **chưa được các script tự nạp hoặc kiểm tra**; không truyền
nó thay cho `config/experiment.yaml`. `status: draft` và giá trị `null` đánh dấu thông tin
chưa có, không phải chấp nhận số 0, bỏ phép đo hoặc cho phép chạy cell.

Mẫu được đối chiếu với Methods/Results của bản thảo DigitalTwin ngày 21/09/2026. Những số
như 200 pose/điều kiện hay 100 chu kỳ là kế hoạch trong bản thảo, chưa phải số đã thu hoặc
cam kết đủ lực thống kê. Lớp/layout, số phiên, thời lượng fine-tuning E4 và các số đo cell
phải được xác nhận trước chiến dịch.

## Cách sử dụng

1. Sao chép mẫu thành file có ID chiến dịch riêng trong nơi lưu hồ sơ thí nghiệm. Ghi người
   chuẩn bị, commit code, phiên bản bản thảo, môi trường và các file liên quan. Dùng đường
   dẫn tương đối tới một gốc hồ sơ ổn định; ghi rõ gốc này khi chuyển máy.
2. Điền dữ liệu phát triển/hiệu chuẩn thực tế, manifest ảnh train/val, recipe sinh ảnh,
   tiêu chí lựa chọn và thiết kế pose/session. Khóa các quy tắc trước khi xem điểm của
   cấu hình mới. Mọi đổi quy tắc do đã nhìn dữ liệu phải được ghi nhận.
3. Sau huấn luyện, ghi quyết định validation đã áp đúng quy tắc, checkpoint, seed, điểm
   chưa làm tròn và SHA-256. Trước evaluation chính thức, điền toàn bộ trường cần dùng,
   lưu bản bất biến với `status: frozen`, người và thời điểm khóa. Hash chính manifest
   được lưu trong sổ đăng ký bên ngoài để tránh hash tự tham chiếu.
4. Thu dữ liệu theo hồ sơ đã khóa. Nếu phải thay đổi, giữ nguyên bản cũ, tạo phiên bản mới,
   ghi lý do, thời điểm và kết quả nào đã được xem. Phân biệt phân tích định trước với
   phân tích khám phá; không sửa protocol hồi tố để phù hợp kết quả.

Những trường không áp dụng cho phạm vi đã chốt cần ghi rõ lý do trong hồ sơ, không bỏ qua
bằng im lặng. Mẫu này giúp truy vết; preflight vật lý và kiểm tra operator vẫn là bước riêng
trong [hướng dẫn cell](HUONG_DAN_THI_NGHIEM_CELL_THAT.md).

## Hồ sơ dữ liệu và ranh giới giữa các tập

Manifest ảnh cần nêu **từng file ảnh và nhãn**, split, capture-session ID và hash; manifest
pose cần pose ID, class, size, layout, điều kiện, phiên và seed sinh danh sách. Số ảnh thư
mục hiện tại không chứng minh đó là tập train lịch sử: riêng pool negative bổ sung phải
được quyết định tường minh. Methods hiện ghi 1256 train và 223 validation. Kiểm tra Windows
ngày 21/09/2026 thấy 1318 ảnh train: script export đã nối thêm 62 negative vào 1256 ảnh
trong split list. Chưa xác nhận manifest của lần train baseline trên Linux. Ghi quan sát
cục bộ riêng với số lịch sử; không tự điền một số thành `verified_image_count`, tự bỏ
negative hoặc thay Results trước khi đối chiếu hồ sơ Linux.

| Nhóm | Vai trò được phép | Không dùng cho |
|---|---|---|
| Real train | Fit detector | Báo test độc lập |
| Real validation | Giữ checkpoint, chọn κ/deployed seed theo quy tắc đã khóa | Báo bằng chứng cuối cùng độc lập |
| Ảnh fit camera/appearance/region | Hiệu chuẩn và phát triển generator; ghi nguồn từng thông số | Bỏ nguồn gốc rồi gọi là held-out |
| `test-standard`/`test-hard` cũ | Development analysis vì đã ảnh hưởng generator | Tự gọi lại là confirmatory test |
| E4 adaptation | Mining context, phân bổ ảnh guided của đúng vòng | Final-evaluation score |
| Final evaluation ảnh/pose | Chấm model và pipeline đã khóa | Chọn κ, threshold, phân bổ ảnh hoặc quyết định dừng |
| Pilot và drift references | Kiểm tra quy trình và ổn định phiên | Tăng số evaluation độc lập |

Các điều kiện khó chọn sau khi đọc baseline vẫn là lựa chọn khám phá. Một tập ảnh xác nhận
mới cần độc lập với ảnh dùng train, validation và fitting. Danh sách adaptation và evaluation
phải tách pose ID lẫn phiên; chỉ đổi tên ID trên cùng một trial không tạo dữ liệu độc lập.
Liệt kê CSV được phép mining từng vòng bằng allowlist, không dùng wildcard quét cả `results`.

## Chọn model và kiểm soát ngân sách

Điểm lựa chọn là tổng **box + mask mAP@0.5:0.95 trên real validation**. Giữ checkpoint tốt
nhất trong mỗi run theo điểm này. E3 so κ bằng trung bình của đủ năm seed `[0,1,2,3,4]`;
hòa thì κ nhỏ hơn. Deployed seed có điểm cao nhất trong cấu hình; hòa thì theo thứ tự seed
đã khai, dùng điểm chưa làm tròn. E4/E5 có seed `[0,1,2]`. Mask mAP dùng trong bảng kết quả
là metric riêng, không thay bằng điểm composite.

`models/e2_seed1_best.pt` là ứng viên lịch sử: giữa các seed nó được chọn bằng mask-only
validation, còn checkpoint trong run được giữ theo composite. Ghi sự khác nhau đó thay vì
đổi tên quy tắc hồi tố. Quyết định dùng ứng viên này hay chọn model mới cho chiến dịch phải
được khóa trước final evaluation. E2 depth-mode dùng cùng model ở cả ba chế độ.

| Thí nghiệm | Dữ liệu synthetic và model |
|---|---|
| E3 | Real-only, wide-range và anchored κ=1/2/4; 3000 ảnh **train** cho mỗi recipe synthetic; năm seed |
| E4 | Cùng checkpoint E3 được chọn làm f0 cho guided/control; mỗi update +1000 ảnh mỗi nhánh; giữ dữ liệu tích lũy 3000→4000→5000; ba seed fine-tuning |
| E5 | Full và cả năm ablation, mỗi nhánh 3000 ảnh train; cùng generation seed; ba seed training |
| E6 | Cùng checkpoint chấm hai tập synthetic/real riêng; không chấm lại ảnh train để tính gap |

Giữ toàn bộ ảnh synthetic của ngân sách train bằng `--make-labels --val-frac 0`; bộ synthetic
evaluation được sinh riêng. Kiểm số ảnh/nhãn hợp lệ sau render/conversion. Với E4, mỗi nhánh
fine-tune checkpoint trước của chính nó và giữ synthetic data của chính nó; chỉ guided được
mining adaptation. Không có failure thì ghi update không chạy, không dùng allocation cũ.
Khóa số epochs fine-tune trước, lịch phải giống giữa guided và control. Ba seed của update 1
cùng bắt đầu từ f0; ở update 2, giữ lineage theo nhánh và seed, mỗi run nạp checkpoint trước
của chính nó. Package nhận một parent nên update 2 cần package riêng cho từng nhánh/seed
với `--seeds <seed> --model <parent_cua_seed> --epochs <so_da_khoa>`; không tái chọn một
parent chung bằng điểm update 1.

Vì RNG của scene là `seed + index`, các đợt bổ sung và synthetic evaluation phải dùng
dải không giao các dải training đã dùng. Ví dụ E3 là 0–2999; hai update E4 có thể dùng
guided/control lần lượt 10000/20000 rồi 30000/40000, mỗi dải dài 1000. Đổi seed từ 0
sang 1 hoặc dùng control seed 2000 vẫn trùng các draw E3. Manifest cần lưu seed cùng
start index và số scene; chỉ ghi seed là chưa đủ để kiểm tra trùng dải.

E3 hiện có so sánh vật lý anchored–wide-range ở từng κ; κ chọn bằng validation chỉ quyết định
khởi tạo E4/full E5. E5 đo cả mask mAP và task success cho đủ năm yếu tố. Muốn giảm nhánh vì
ngân sách phải đổi protocol và bản thảo trước khi nhìn kết quả liên quan, không tự bỏ yếu tố
vì dự đoán hiệu ứng nhỏ.

Các giá trị cố định E5 do sampler ghi trong `ablation.reference` và `generator_profile` của
manifest scene; lưu bản thực tế này cùng mã nguồn:

| Flag | Can thiệp trên recipe anchored |
|---|---|
| `none` | Full recipe |
| `illumination` | Ghim vị trí, công suất và nhiệt độ màu vào anchor trong cấu hình |
| `background` | Ghim màu/roughness nền vào anchor; giữ mặt phẳng bàn đã cấu hình |
| `distractors` | Danh sách distractor rỗng |
| `camera` | Ghim transform và intrinsics tham chiếu, không jitter |
| `pose` | Ghim vị trí theo số vật ở tâm các ô dọc trục dài của vùng đặt; yaw ở giữa khoảng; class/size/material vẫn lấy mẫu |

Seed sinh cảnh giống nhau giữ các lượt lấy mẫu yếu tố còn lại khi sampler áp ablation sau
khi lấy mẫu full scene. Vẫn kiểm render và nhãn: sự thay đổi occlusion là hậu quả của can thiệp,
không phải mọi pixel ngoài yếu tố bị tắt đều bất biến.

## Những thông tin chỉ có thể hoàn tất tại cell

- Transform, metadata, uncertainty, plane và intrinsics đo thật, hash của từng file, kích
  thước bảng in đo lại, TCP, điểm thả và kết quả touch-test. Không đưa giá trị mô phỏng vào
  trường `confirmed_real_measurement`.
- Phân bổ lớp × layout × điều kiện, các pose/session/block/operator, primary subgroup và
  số cặp hoàn chỉnh cần thu. N=200 pooled chưa xác nhận power cho inox-stacked hay họ đã
  chỉnh Holm. Không viết bảo đảm phát hiện mọi hiệu ứng lớn hơn một ngưỡng cố định.
- Mốc chấm placement là tâm hay mép, trục được kiểm, trạng thái băng tải và tiêu chí đạt.
  Dải 30 mm một chiều không chứng minh độ chính xác vị trí hai chiều. `human_ok` cần đánh
  giá đầy đủ đúng vật, vận chuyển không rơi và thả đúng vị trí. Lưu cách quan sát/cảm biến
  xác nhận giữ vật trong vận chuyển; một lần đọc sensor lúc đóng kẹp chưa chứng minh tín
  hiệu liên tục suốt hành trình.
- Cấu hình depth/geometry, checkpoint SHA-256 và threshold cố định. Một model triển khai
  mỗi cấu hình cho suy luận vật lý có điều kiện trên model đó; không ước lượng được biến
  thiên giữa các lần train độc lập chỉ từ số lượt gắp.
- Tiêu chí loại do phần cứng/hiệu chuẩn độc lập với hiệu ứng quan sát; giữ mọi bản ghi và
  phân tích toàn bộ-trial khi có loại. Failure kể cả safety rejection trước dispatch vẫn
  tính là trial không thành công, đồng thời báo riêng lý do.
- Mốc inference/localization/planning/motion/gripper và tổng cycle; ghi failures, resets,
  operator wait. P50/P95 tổng tính từ tổng đã đo, không cộng percentile thành phần. Script
  phân tích telemetry khớp chỉ cung cấp chẩn đoán chuyển động, không đủ các timestamp này.

## Hồ sơ phân tích phải chốt cùng protocol

Ghi danh sách contrast và họ Holm, primary class/layout và hướng hiệu ứng; báo successes/
attempts, complete pairs, `n01`/`n10`, chênh lệch, CI 95%, p gốc/đã chỉnh. `n01` là baseline
thất bại/comparator thành công. Bootstrap 10.000 lần theo pose hoàn chỉnh, seed 0; lần thử
lặp lại cùng pose không thành đơn vị độc lập. Lập phân tích độ nhạy theo session/block khi
có phụ thuộc cụm. CI điểm 95% không tự là CI đồng thời đã hiệu chỉnh nhiều phép so sánh.
Không bác bỏ khác biệt không chứng minh tương đương.

Nhật ký run và chọn model nên dùng cột: experiment, configuration, iteration, seed,
package/manifest hash, calibration hash, initial checkpoint hash, selected epoch,
validation box/mask score, final checkpoint hash, GPU/software, updates, wall time và
vai trò dữ liệu đánh giá. Quy trình tạo package/chạy train xem
[TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md).
