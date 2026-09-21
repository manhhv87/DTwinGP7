# HƯỚNG DẪN CHẠY THÍ NGHIỆM TRÊN CELL GP7

## 0. Chú ý

- Hướng dẫn này được đồng bộ với Methods/Results ngày 21/09/2026. Trước chiến dịch chính,
  điền và khóa [campaign manifest](CAMPAIGN_MANIFEST.md); các số lượt và ngày trong lệnh
  dưới đây là ví dụ lập lịch, chưa phải xác nhận phân bổ lớp/layout hoặc đủ lực thống kê.
- Quy trình đóng gói dữ liệu và huấn luyện Linux: [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md).
  Model dùng ở cell phải có ID, SHA-256, nguồn dữ liệu và biên bản chọn bằng validation.
- Ba mục đầu là chuẩn bị, sau đó là **5 pha, chặn nhau**: pha trước còn vướng mục DỪNG thì không chạy pha sau.
- IP tủ điều khiển: **192.168.1.100**.

**Cảnh báo:**

- **Đo sai cạnh ô bàn cờ** là lỗi nguy hiểm nhất: không báo lỗi, mọi khoảng cách co giãn sai
  tỷ lệ. Thước kẹp, đo hai lần, hai người đọc.
- **Sửa bố cục cell sau khi hiệu chuẩn = hiệu chuẩn lại.**

---

## Quy trình tổng thể

| Bước | Chạy ở đâu | Robot | Người làm gì | Ra cái gì |
|---|---|---|---|---|
| Chuẩn bị, mục 1–3 | máy tính cạnh cell | không | đo bốn số, in và gá bàn cờ | các file cấu hình đã điền số thật |
| Pha 1, hiệu chuẩn | cell, chế độ TEACH | người jog, robot không tự đi | 25–30 tư thế, rồi chạm thử 8 điểm | 4 file trong `config\calibration\` |
| Pha 2, chạy thử | cell, chế độ REMOTE | robot tự chạy | 1 lượt rồi 5 lượt, đo lệch điểm thả, dán vạch 30 mm | 1 CSV + 1 telemetry |
| Pha 3, E1 | cell | số mẫu theo manifest (lệnh ví dụ 50 lượt) | đặt vật, bấm ENTER, chấm y/n | `e1_*`, CSV, telemetry, 4 PNG |
| Pha 4, E2 | cell | dự kiến 200 pose/nhánh/điều kiện, phân bổ và đối chứng cần khóa | đặt vật, chấm y/n, không mở file khoá | 24 CSV khối mỗi buổi, file khoá, thư mục khung ảnh |
| Pha 5, sinh ảnh và huấn luyện | **máy GPU Linux** | không | khóa luật validation, train đủ seed, chọn κ cho E4/E5 | `specs`/`render`/`dataset`, package `dsv<k>/`, `runs/` |
| Pha 5, đo lại | cell | như Pha 4 | chép mô hình mới về rồi chạy lại đúng danh sách thẻ cũ | CSV gắp của mô hình mới |
| Phân tích | máy tính | không | gõ đúng lệnh có `>` hoặc `\| tee` | `e2_phan_tich.txt`, `results_summary.png` |

### Mã trong bài báo nằm ở pha nào

Bài báo gọi tên bốn đóng góp là C1 đến C4, và sáu thí nghiệm là E1 đến E6. Hướng dẫn này chia
theo pha, nên đây là bảng tra:

| Mã | Là gì | Chạy ở đâu trong hướng dẫn này |
|---|---|---|
| **C1** | Digital twin của cell, kèm cách đo độ trễ đồng bộ, sai số hình học, chi phí lập quỹ đạo | Pha 3 (E1) |
| **C2** | Ảnh tổng hợp neo theo hiệu chuẩn, so với ngẫu nhiên hoá dải rộng cùng ngân sách | Pha 5 (E3) |
| **C3** | Phân bổ ảnh theo lỗi, so với tăng cường ngẫu nhiên cùng ngân sách | Pha 5 (E4, có E5 bổ trợ) |
| **C4** | Ba chế độ độ sâu `rgbd`, `plane`, `fusion`, gồm cả vật inox phản chiếu | **Pha 4 (E2)**, ba nhánh ghép cặp theo manifest |

E5 và E6 không mang mã đóng góp riêng: E5 khảo sát yếu tố của recipe sinh ảnh, E6 cho khoảng cách mô phỏng với thật và
thời gian chu kỳ.

### Chia việc

Hai vai, không chồng lên nhau: **người chạy cell** làm mọi thứ có robot, **người huấn luyện** làm
mọi thứ trên máy GPU và phần phân tích.

| Việc | Người chạy cell | Người huấn luyện |
|---|---|---|
| Chuẩn bị mục 1–3, đo bốn số, in và gá bàn cờ | x | |
| Pha 1 hiệu chuẩn, chạm thử 8 điểm | x | |
| Pha 2, Pha 3, Pha 4: chạy cell, đặt vật, chấm y/n | x | |
| Nhật ký buổi, sao lưu `results\` và `logs\` | x | |
| Mở file khoá mù | | x |
| Phân tích: tỷ lệ thành công, McNemar, Holm | | x |
| Sinh ảnh, render, gán nhãn | | x |
| Viết luật chọn κ, chạy val, chọn ra κ | | x |
| Huấn luyện mọi cấu hình, mọi seed | | x |
| Pha 5 đo lại ở cell, sau khi nhận mô hình mới | x | |

**Sau mỗi buổi, người chạy cell gửi đi:** cả thư mục `results\<tên buổi>\`, file
`logs\experiment.log`, thư mục khung ảnh trên ổ D, file khoá
`results\blinding_keys\key_<buổi>.json` **còn nguyên chưa mở**, và trang nhật ký của buổi.

**Người huấn luyện gửi lại:** file `.pt` của mô hình mới kèm SHA-256, κ đã chọn kèm luật validation đã ghi trước,
dòng `model_path` cần sửa trong `config\experiment.yaml`, và ID danh sách pose evaluation đã khóa; không thay danh sách theo kết quả.

**Hai ranh giới không được vượt.** Người chạy cell không mở file khoá và không xem tỷ lệ thành
công đang chạy: biết cấu hình nào đang chạy là hỏng phần chấm bằng mắt. Người huấn luyện không
đổi danh sách thẻ giữa chiến dịch: đổi là mất ghép cặp, và mọi kiểm định McNemar đã chạy thành vô
nghĩa.

Ba việc chi phối cả quy trình:

1. **Pha trước còn vướng mục DỪNG thì không sang pha sau.** Gặp mục DỪNG thì dừng và ghi vào nhật
   ký, không tự đổi cấu hình rồi chạy tiếp.
2. **Chỉ Pha 5 phần sinh ảnh và huấn luyện là rời cell.** Mọi con số gắp đều phải đo ở cell thật,
   kể cả sau khi huấn luyện lại.
3. **Kết quả tự vào file, người chỉ ghi thứ đo bằng thước.** Chi tiết ở mục Ghi nhật ký.

---

## 1. An toàn — đọc trước

1. **Nút dừng khẩn luôn trong tầm tay.**
2. Không đưa tay vào vùng làm việc khi servo bật, kể cả robot đứng yên.
3. **Không đổi giới hạn tốc độ** cho tới khi chạy trơn 50 lượt.
4. Cấu hình mới: chạy 1 lượt, đứng nhìn, rồi mới chạy nhiều.
5. Tiếng lạ, cánh tay đi sai hướng, má kẹp sắp chạm bàn: **dừng ngay**.

**Hai cờ bắt buộc cho mọi lệnh chạy thật**:

- `--confirm-each-trial`: dừng chờ ENTER trước mỗi lượt. Thiếu nó, robot chạy lượt sau 1 giây
  sau lượt trước, lúc tay người còn trong cell.
- `--no-viewport-mirror`: tắt gương 3D. Khi gương bật, đoạn mã bắt Ctrl+C nằm sai luồng và
  **không bao giờ chạy**.

E6 vẫn giữ xác nhận từng lượt khi có người đặt vật. Chỉ bỏ cờ trong bố trí cấp vật phù hợp
và cell không có người; tách thời gian xác nhận/đặt vật khỏi chu kỳ tự động theo protocol.

---

## 2. Chuẩn bị

Máy cần Python 3.10 trở lên và Git. Mở **PowerShell**. **Lần đầu** trên một máy, lấy mã nguồn về
và cài:

```
git clone -b feat/synthgen-c2-c3-dataset https://github.com/manhhv87/DTwinGP7.git
cd DTwinGP7
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Những lần sau**, vào đúng thư mục đó, lấy bản mới nhất rồi bật môi trường:

```
cd DTwinGP7
git pull
.venv\Scripts\activate
```

Nếu `git pull` báo `config/synthgen.yaml` đã bị sửa ở máy, chạy `git checkout -- config/synthgen.yaml`
rồi `git pull` lại: bản trên git đã có sẵn thông số camera thật của cell.

Sau bước này đầu dòng lệnh sẽ hiện `(.venv)`. Nếu không hiện, môi trường ảo chưa bật, mọi lệnh phía sau sẽ báo thiếu thư viện.

**Baseline `models/e2_seed1_best.pt` đã được đưa lên Git ở commit `d736b75`.**
Clone đúng nhánh có commit này sẽ có file. Đây là ứng viên triển khai E2 lịch sử, chọn theo
validation mask mAP, chưa phải bằng chứng đã chạy robot thành công. Các trọng số E3–E5 khác
vẫn được chuyển riêng từ Linux; xem [models/README.md](../models/README.md).

Kiểm tra máy chạy được, chưa cần robot:

```
pytest tests/ -q
```

**DỪNG nếu:** có test thất bại hoặc không hoàn tất. Ghi commit, môi trường, số test passed/
skipped và lý do skip; không dùng số lượng test của một phiên bản cũ làm điều kiện đạt.

```
python scripts/03_run_experiment.py --mode sim --headless --trials 5
```

**DỪNG nếu:** không chạy hết, hoặc không tạo được file mới trong `results\`. Đây là chế độ mô phỏng, robot **không** di chuyển.
Tỷ lệ gắp in ra ở bước này **không có ý nghĩa**: cấu hình còn giữ số giữ chỗ ở mục 3, nên báo
`success_rate=0.0%` và các lượt `unreachable` là bình thường.

> Thư mục làm việc: `scripts\` là các lệnh chạy, `config\` là cấu hình, `results\` là nơi > mọi kết quả rơi vào, `logs\` là nhật ký chi tiết khi cần tra lỗi.

---

## 3. Bốn số phải đo, và một số phải khoá

Chưa có đủ thì phần mềm **từ chối chạy** chế độ thật.

**Bốn số phải đo:**

| # | Việc | Ghi vào đâu | Hiện tại |
|---|---|---|---|
| 1 | Đo lại tấm bàn cờ **đã in** bằng thước kẹp: cạnh một ô, cạnh một dấu. Máy in có thể co giãn vài phần trăm | gõ vào lệnh ở Pha 1 | Tấm in sẵn 7×5 ô, ô 45 mm, dấu 34 mm. **Gõ số đo được, không gõ 45/34** |
| 2 | TCP má kẹp thật, khai TOOL01 trên teach pendant (cách làm ở mục Khai TOOL01 ngay dưới) | `config/cell_layout_real.yaml` → `gripper.tcp_offset_xyz_mm` | `[0, 0, 100]`, số giữ chỗ |
| 3 | Điểm thả trên băng tải. Mặt băng tải **không được cao hơn** mặt bàn; thấp hơn thì vật rơi đúng phần chênh, chỉ chấp nhận vài cm | `config/experiment.yaml` → `place_position` | `[700, 120, 700]`, số giữ chỗ |
| 4 | Thông số camera D455 thật | `config/synthgen.yaml` → `camera.intrinsics` | **đã điền sẵn** từ hiệu chuẩn 18/09/2026 (fx 645,007); chỉ sửa khi thay camera |

**Không sửa `robot.pose.xyz_mm`** (đang là `[0, 0, 630]`). Giá trị này phải giống nhau lúc
hiệu chuẩn và lúc chạy; preflight tự kiểm và từ chối chạy nếu khác. Đã sửa thì hiệu chuẩn lại.

Lệnh đọc thông số camera (cắm camera vào máy trước):

```
python -c "from src.perception.camera import D455Camera; print(D455Camera().intrinsics)"
```

### Khai TOOL01

Robot phải biết đầu má kẹp nằm cách mặt bích bao xa. Khai sai thì mọi lệnh gắp đưa sai điểm
xuống bàn, và không có gì báo lỗi.

1. **Đo.** Kẹp đóng, dùng thước kẹp đo từ mặt bích tới đầu má kẹp, dọc theo trục cổ tay: đó là
   Z. Má kẹp nằm đúng tâm mặt bích thì X và Y bằng 0.
2. **Khai trên teach pendant.** Chìa khoá ở MAINTENANCE hoặc MANAGEMENT, servo tắt. Vào
   MAIN MENU → ROBOT → TOOL, chọn TOOL: 1, bấm SELECT, sửa Z bằng số vừa đo, rồi
   COMPLETE/REGISTER. Bấm DISP xem lại.
3. **Ghi đúng số đó** vào `config/cell_layout_real.yaml` → `gripper.tcp_offset_xyz_mm`. Hai
   chỗ phải cùng một số.
4. **Kiểm bằng cách xoay quanh đầu kẹp.** Chế độ TEACH, tốc độ thấp. Dựng một đầu bút đứng yên
   trên bàn, jog cho đầu má kẹp chạm đúng đầu bút. Chọn hệ toạ độ Robot rồi bấm các phím xoay:
   khai đúng thì má kẹp chỉ xoay quanh đầu bút, đầu kẹp không rời khỏi đầu bút. Đầu kẹp văng ra
   xa là số khai sai, đo lại.

Đừng kiểm bằng cách cho robot nâng lên hạ xuống: đi thẳng không xoay thì mọi điểm trên má kẹp
đi y hệt nhau, số TCP sai cũng không lộ ra.

### In và gá tấm bàn cờ

In file `charuco_a3_o45mm.pdf`. Không tải bảng ChArUco từ web: bảng kiểu cũ trông y hệt nhưng
bộ nhận dạng đọc sai.

| Bước | Yêu cầu |
|---|---|
| In | **100% / actual size**, không *fit to page*. Đo vạch 100 mm in sẵn; sai là in lại |
| Giấy | Mặt mờ, không cán bóng |
| Tấm nền | Alu composite 3 mm, dán phẳng (tiệm in dán decal, hoặc keo xịt). Không keo nước |
| Đo lại | Thước kẹp đo cạnh một ô và cạnh dấu. **Số đo được là số gõ vào lệnh** |

### Gá tấm bàn cờ lên má kẹp

![Bố trí chung và kích thước tấm](ban_ve_ga_ban_co.png)

![Chi tiết gá và trình tự lắp](ban_ve_ga_3d.png)

> **Bộ kẹp trong hai bản vẽ vẽ phỏng theo ảnh chụp, chưa đo.** Hình để hiểu cách lắp; mọi số đi
> mua vật tư phải lấy từ thước kẹp trên kẹp thật.

**Cách gá:** dán hai thanh nhôm song song vào lưng tấm, cho hai má kẹp ép vào hai mặt ngoài của chúng.

**Mặt in phải NGỬA LÊN, hướng về camera.** Camera nhìn từ trên xuống, mặt in úp xuống thì nó chỉ
thấy lưng tấm. Nghĩa là kẹp nằm dưới tấm: xoay cổ tay cho trục kẹp hướng lên trời. Càng ngang có
thò ra ngoài mép tấm cũng không sao, nó nằm dưới tấm.

Độ cao tấm phải thoả **hai ràng buộc ngược nhau**:

- *Camera:* trên 505 mm hoa văn tràn khung, nghiêng tấm còn ăn thêm biên nên lấy trần **470 mm**. Dưới 250 mm hoa văn chỉ chiếm một phần ba bề ngang, đọc góc kém.
- *Cơ khí:* lật kẹp lên thì mặt bích, xi lanh, càng ngang, giá đỡ đều nằm **dưới** tấm. Tấm ở độ cao h thì mặt bích ở **h − H**. Muốn mặt bích cách bàn ít nhất 50 mm thì **h ≥ H + 50**.

H đo được khoảng 180 mm (mặt bích tới mặt dưới kẹp, người dùng báo 21/09/2026), nên ràng buộc cơ
khí là h ≥ 230 mm; cộng giới hạn camera thì cửa sổ là **250 đến 470 mm**.

**Về độ nghiêng:** nghiêng càng đa dạng phép giải càng chắc, nhưng nghiêng quá thì dấu bị bóp méo và
bộ nhận dạng bỏ qua tư thế đó. Không có ngưỡng đo được, nên lấy chính chương trình làm thước:
nghiêng thêm tới khi nó thôi in `Captured pose` rồi lùi lại một chút. Điều bắt buộc là đừng để
cả 25 tư thế nằm ngang y hệt nhau.

---

## Pha 1 — Hiệu chuẩn tay–mắt và đo mặt bàn

**Mục tiêu:** cho phần mềm biết camera nằm ở đâu so với robot, và mặt bàn cao bao nhiêu.

**Kết quả ra:** bốn file trong `config\calibration\`: `T_base_camera.npy`,
`T_base_camera_meta.json`, `table_plane.json`, `T_base_camera_sigma.json`. Chép `tilt_deg` và
`rms_mm` từ `table_plane.json` vào nhật ký. Đo tay: bảng chạm thử 8 điểm.

**Cần có trước:** bốn số đo ở mục 3. Bàn dọn trống. Bàn cờ đã gá chắc lên má kẹp.
Robot để chế độ **TEACH** (chỉ đọc khớp, robot không tự chạy).

> **Bàn cờ gắn lên robot, không đặt trên bàn.** Camera đứng yên, nên thứ phải di chuyển giữa
> các lần chụp là bàn cờ; robot là thứ duy nhất mà ta biết chính xác nó đi đâu.
>
> **Gá lệch tâm không sao**, phép giải tự tìm ra khoảng lệch. Nhưng tấm **xê dịch giữa chừng**
> thì hỏng cả 25 tư thế, và hỏng lặng lẽ.

Thay `<cạnh ô>` và `<cạnh dấu>` bằng số vừa đo (đơn vị mm):

```
python scripts/02_run_calibration.py --hse-ip 192.168.1.100 --squares 7 5 --square-mm <cạnh ô> --marker-mm <cạnh dấu> --dict DICT_4X4_50 --method park --bootstrap 200
```

**Làm gì:**

1. Dùng teach pendant đưa robot tới một tư thế sao cho camera **nhìn thấy rõ cả tấm bàn cờ**.
2. Quay lại máy tính, bấm **ENTER**. Màn hình in `Captured pose #n`. Nếu không in, camera
   chưa thấy đủ ô bàn cờ: đổi tư thế rồi thử lại.
3. Lặp lại cho đủ **25 đến 30 tư thế**. Yêu cầu quan trọng: **xoay mặt bích đa dạng** theo
   nhiều hướng, và rải khắp mặt bàn. Nếu 25 tư thế đều na ná nhau thì kết quả sai mà không
   báo lỗi. Nghiêng tới mức nào thì xem đoạn về độ nghiêng ở mục 4.
4. Gõ **s** rồi ENTER để giải.
5. Chương trình hỏi tiếp: **dọn sạch bàn, đưa robot ra khỏi tầm nhìn camera**, rồi bấm ENTER
   để đo mặt bàn.

**DỪNG nếu:**
- Không sinh đủ bốn file trong `config\calibration\`.
- Dưới 25 tư thế (phần mềm chỉ đòi 10, nhưng 25 là số đã chốt cho bài).
- 25 tư thế na ná nhau. Không có gì tự bắt, phải tự nhìn.

**GHI SỐ:**
- `tilt_deg` mặt bàn. Phần mềm cảnh báo trên 2 nhưng không chặn; bàn nghiêng 2,3 độ là sự
  thật về cái bàn. Lớn thì kiểm xem bàn có nghiêng thật không.
- `rms_mm` mặt bàn. Chưa có ngưỡng; dùng để so giữa các lần hiệu chuẩn.
- **Chạm thử ít nhất 8 điểm** rải khắp vùng gắp, ghi trung bình, RMS, lớn nhất. Mục tiêu 3 mm
  là mục tiêu, không phải cửa chặn: cell đạt 3,4 mm vẫn chạy, báo cáo đúng 3,4.

Sau này chỉ xê dịch **bàn**: chạy lại với `--table-only`. Đụng vào **camera**: làm lại toàn bộ Pha 1.

---


## Pha 2 — Chạy thử, một lượt rồi năm lượt

**Mục tiêu:** chắc chắn robot đi đúng chỗ trước khi chạy hàng trăm lượt.

**Kết quả ra:** `results\experiment_real_<thời điểm>.csv` và `results\telemetry_<thời điểm>.csv`
của lượt chạy thử. Đo tay: vật rơi lệch bao nhiêu so với điểm dạy. Ghi vào nhật ký: số lượt hỏng
trên 5 và lý do từng lượt.

> **Chuyển TEACH sang REMOTE. Đây là khoảnh khắc nguy hiểm nhất cả quy trình.** Từ Pha 1 tới giờ
> robot ở chế độ TEACH, chỉ đi khi có người bóp công tắc an toàn. Từ đây nó **tự đi khi máy tính
> bảo**. Trước khi xoay chìa khoá, làm đủ bốn việc: dọn hết vật lạ khỏi vùng làm việc, đếm đủ số
> người và xác nhận không còn tay ai trong cell, kiểm nút dừng khẩn còn trong tầm tay, và nói to
> cho cả phòng biết. Xoay chìa xong mới bật servo.

**Cần có trước:** Pha 1 xong, không vướng mục DỪNG nào. Đặt sẵn một vật lên bàn. Tay đặt trên nút dừng khẩn.

```
python scripts/03_run_experiment.py --mode real --trials 1 --depth-mode rgbd --ik-source yrc --tool-no 1 --confirm-each-trial --no-viewport-mirror
```

Trước khi robot nhúc nhích, phần mềm tự kiểm tra một loạt điều kiện (gọi là *preflight*). Nếu
thiếu thứ gì nó sẽ **từ chối chạy và in rõ lý do kèm cách sửa** — đọc dòng đó, đừng bỏ qua.

**DỪNG nếu:**
- Preflight báo bất kỳ lỗi nào.
- Đầu má kẹp xuống thấp hơn mặt bàn cộng khoảng an toàn, dù chỉ một lần. Đây là an toàn.
- Robot đi sai hướng rõ rệt, hoặc có `motion_error` lặp lại.

**GHI SỐ:**
- **Vật rơi lệch bao nhiêu so với điểm dạy.** Đo bằng thước, ghi số. Con số ±15 mm là dung sai
  đã chốt cho bài báo, nhưng nếu cell thật cho 22 mm thì đó là một phát hiện về cell, không
  phải một lần trượt. Ghi lại; lệch nhiều thì dạy lại điểm thả.
- Số lượt hỏng trong 5 lượt đầu và lý do từng lượt.

**Xong Pha 2, dán vạch chấm:** hai vạch băng dính ở điểm thả, cách nhau **30 mm**, tức ±15 mm mỗi
bên tâm. Từ Pha 3 trở đi mắt chấm đạt hay hỏng là chấm theo vạch này, không chấm theo cảm tính.
Ghi vào nhật ký chọn mốc nào của vật để so với vạch (tâm vật hay mép vật) và giữ nguyên mốc đó cả
chiến dịch.

> **Người chấm bằng mắt không được biết đang chạy cấu hình nào.** Từ Pha 4 trở đi, chính người
> đặt vật là người chấm, nên buổi nào cũng phải chạy mù (xem mục làm mù ở dưới). Người biết cấu
> hình mà lại đi chấm là chỗ phản biện sẽ chỉ ra ngay.

**Vướng DỪNG:** không sang Pha 3. Preflight tự in lý do và cách sửa. TOOL01 xem mục Khai TOOL01 ở mục 3.

---

## Pha 3 — E1: đo nền tảng

**Mục tiêu:** lấy số về tốc độ và độ chính xác của hệ, chưa cần gắp nhiều.

**Kết quả ra:** `results\e1_tests.txt`; `figures\compare_fk_ik_<thời điểm>.csv` và `.png`;
`results\e1_hse_rtt.csv`; `results\experiment_real_<thời điểm>.csv` và
`results\telemetry_<thời điểm>.csv` của 50 lượt; bốn ảnh PNG telemetry trong `figures\`
(`joint_trajectory`, `joint_velocity`, `drift_events`, `cycle_time`); hai bản tóm tắt màn hình
`results\e1_rtt_tomtat.txt` và `results\e1_telemetry_tomtat.txt` (nơi có `p50` / `p95` / `p99` và
tần số telemetry thật đạt được).

**Cần có trước:** Pha 2 xong, không vướng mục DỪNG nào.

```
pytest tests/ -q > results/e1_tests.txt
```
```
python scripts/17_compare_fk_ik.py --samples 500 --fair
```
```
python tools/hse_rtt.py 192.168.1.100 --n 1000 --csv results/e1_hse_rtt.csv | tee results/e1_rtt_tomtat.txt
```
```
python scripts/03_run_experiment.py --mode real --trials 50 --depth-mode rgbd --pose-list config/pose_lists/std_v2.csv --telemetry-hz 10 --confirm-each-trial --no-viewport-mirror
```
```
python scripts/05_analyze_telemetry.py latest | tee results/e1_telemetry_tomtat.txt
```

Lệnh `hse_rtt` chỉ **đọc** trạng thái, robot không di chuyển. Lệnh thứ tư có gắp 50 lượt.

**DỪNG nếu:** có lần tự động dừng ngoài ý muốn, hoặc mất kết nối giữa chừng.

**GHI SỐ:** tốc độ ghi telemetry thật sự đạt được, và `p50` / `p95` / `p99` của RTT. Công cụ
cảnh báo khi `p95` vượt 100 ms, và cảnh báo đó có nghĩa là **vòng telemetry 10 Hz không giữ nổi
nhịp** — tức phải hạ tần số ghi xuống, chứ không phải huỷ chiến dịch. Mạng của cell là mạng của
cell; bài báo báo cáo con số đo được.

---

## Pha 4 — E2: chiến dịch chính, ba chế độ độ sâu (đóng góp C4)

**Mục tiêu:** đo tỷ lệ gắp thành công của ba cách tính độ sâu. Đây là phần nhiều số liệu nhất.

**Kết quả ra:** mỗi khối 25 lượt của mỗi nhánh sinh một `results\experiment_real_<thời điểm>.csv`
kèm một `results\telemetry_<thời điểm>.csv`; file khoá `results\blinding_keys\key_<session>.json`;
thư mục khung ảnh trên ổ D; `figures\results_summary.png` và `results\e2_phan_tich.txt` do lệnh so
sánh sinh ra. Người chấm bằng mắt sau mỗi lượt, câu trả lời tự vào cột `human_ok` của CSV.

**Cần có trước:** Pha 3 xong, không vướng mục DỪNG nào.

**Thẻ** là mốc vị trí dán trên bàn, đánh số 1–20. Mỗi lượt chương trình gọi một thẻ và một góc:

```
▶ Trial 7 — PLACE OBJECT: card=14  x=575.0 mm  y=80.0 mm  yaw=85.0 deg  class=metal_box
```

Cùng một danh sách đã khóa được phát lại cho mọi cấu hình cần so sánh. Kế hoạch trong Methods
là 200 pose ghép cặp mỗi cấu hình/điều kiện; số thực tế theo manifest. Ghép cặp bằng `pose_id`,
không coi lần thử lặp lại tại cùng pose là một đơn vị độc lập mới. Danh sách adaptation của
E4 phải tách khỏi danh sách evaluation, kể cả các phiên và ảnh đi kèm.

**Chuẩn bị bàn:**

Lưới 21 thẻ (3 hàng × 7 cột) chỉ phủ phần bàn mà camera nhìn trọn cả vật cao lẫn chồng hai lớp,
tính cả dung sai ±15 mm, và robot với tới mọi tư thế tiếp cận. Lưới tính từ hiệu chuẩn 18/09/2026
và dụng cụ khoảng 180 mm; lý do ghi trong `config/synthgen.yaml`. Đổi camera hoặc đổi dụng cụ thì
lưới phải tính lại.

1. In `position_cards.pdf` (21 thẻ, hai trang A4) ở **100% / actual size**, đo vạch 50 mm
   in sẵn, cắt theo viền.

2. **Lấy dấu bằng robot.** Toạ độ của mỗi thẻ in sẵn trên chính thẻ đó (x từ 525 đến 625 mm,
   y từ −160 đến +200 mm), tính theo **gốc robot**, không phải mép bàn. Trên teach pendant bấm
   COORD chọn hệ Robot, jog tới đúng X, Y của thẻ, hạ Z sát mặt bàn, đánh dấu ngay dưới đầu TCP
   rồi nhấc lên. Cần TOOL01 đã khai đúng. Hai mươi mốt lần, hết một buổi.

![Toạ độ thẻ đo từ gốc robot](ban_ve_toa_do_the.png)

*Bản in: `ban_ve_toa_do_the.pdf`.*

3. **Dán thẻ vào dấu**, tâm thẻ trùng dấu, mũi tên trên thẻ hướng **+x, tức ra xa robot**. Dán
   băng dính trong phủ kín cả thẻ để nó không bong và không xê dịch. Dán xong cả 21, jog lại về
   toạ độ thẻ số 1: mũi TCP phải rơi đúng tâm thẻ đó.

4. Mỗi lượt, chương trình đọc to số thẻ và góc xoay, rồi **dừng chờ**. Người đặt vật:

   - lấy đúng loại vật được gọi, đặt tâm vật lên chữ thập giữa thẻ;
   - xoay cho cạnh dài trùng vạch góc, đếm từ mũi tên 0 độ;
   - **rút tay ra khỏi vùng làm việc**, rồi mới bấm ENTER.

   Bấm ENTER xong robot mới chụp ảnh và gắp. **Không nhìn màn hình kết quả nhận dạng** trong
   lúc đặt — nhìn vào sẽ vô tình đặt "cho dễ nhận", làm hỏng số liệu.

5. **Gắp xong, màn hình hỏi:** `Trial N — part in the right place? [y]=yes [n]=no`. Trả lời
   theo đúng cái mắt vừa thấy:

   - **y**: vật nằm gọn trong vạch 30 mm ở điểm thả.
   - **n**: kẹp trượt không lấy được vật, vật rơi giữa đường, hoặc vật nằm ngoài vạch.

   Câu trả lời vào cột `human_ok` của cùng file CSV. Cột `success` bên cạnh là máy tự chấm; máy
   chỉ biết lúc đóng kẹp có vật hay không, không biết vật có rơi giữa đường không.

6. **Lấy vật về khỏi điểm thả**, trong lúc đang chờ ENTER của lượt kế (robot chắc chắn đứng
   yên). Không với ra băng tải lúc robot đang về.

**Đặt góc:** góc trong danh sách đã làm tròn về bội số 5 độ, đặt đúng vạch gần nhất. Dung sai vị
trí ±15 mm.

> Vật dài 190 mm, thẻ cách nhau 51 mm, nên vật đặt xuống **phủ kín thẻ**. Cầm vật lơ lửng, ngắm
> từ trên xuống, xoay cạnh dài song song vạch cần, rồi hạ thẳng.

### Chạy: xen kẽ và mù, bằng một lệnh

Đừng chạy hết 200 lượt của một chế độ rồi mới sang chế độ khác: mọi trôi trong buổi (hiệu chuẩn,
tay người, ánh sáng) sẽ lẫn vào kết quả y hệt một hiệu ứng thật. Và người đặt vật cũng là người
chấm bằng mắt, nên không được biết đang chạy cấu hình nào.

Một lệnh làm cả hai việc:

```
python tools/run_blinded_campaign.py --arm rgbd "--depth-mode rgbd" --arm plane "--depth-mode plane" --arm fusion "--depth-mode fusion" --pose-list config/pose_lists/std_v2.csv --trials 200 --block 25 --session 2026-09-20-sang --operator AN --seed 7 --common "--mode real --ik-source yrc --tool-no 1 --confirm-each-trial --no-viewport-mirror --save-frames --frames-dir D:/Scientific/Dataset/DigitalTwin/frames"
```

- Chạy `--dry-run` trước để xem lịch.
- Màn hình chỉ hiện `[A] Trial 7/25 — PLACE: card=14 yaw=85 class=inox_box`, rồi câu hỏi chấm
  `[y]/[n]` sau khi gắp xong. Mã A, B, C bốc ngẫu nhiên; máy chấm ra sao thì không hiện, nên
  người chấm không bị số của máy kéo theo.
- File khoá `results/blinding_keys/key_<session>.json` ghi mã nào là cấu hình nào. **Không mở
  cho tới khi chạy xong và chấm xong.**
- Thứ tự nhánh đảo mỗi vòng 25 lượt; `session`, `block`, `operator` ghi vào từng dòng CSV.
- Lỗi và cảnh báo vẫn hiện (an toàn không giấu). Nếu một lỗi lộ tên chế độ, ghi vào nhật ký là
  khối đó đã lộ.

**Khối đối chứng:** mỗi buổi chạy thêm 20 lượt của **cùng một cấu hình, cùng 20 tư thế**, một lần
lúc đầu buổi và một lần lúc cuối. Dùng đoạn 200:220 của danh sách, đoạn mà chiến dịch chính không
đụng tới (chiến dịch chạy 0:200), nên hai khối này không đè `pose_id` của ai:

```
python scripts/03_run_experiment.py --mode real --depth-mode rgbd --pose-list config/pose_lists/std_v2.csv --pose-slice 200:220 --session-id 2026-09-20-sang --block-id doichung-dau --operator-id AN --confirm-each-trial --no-viewport-mirror
```

Cuối buổi chạy lại đúng lệnh đó, đổi `--block-id doichung-cuoi`. Chênh lệch đầu–cuối dùng để
đánh giá drift, không phải lý do loại buổi dựa trên độ lớn hiệu ứng giữa cấu hình. Chỉ loại
theo tiêu chí hợp lệ phần cứng/hiệu chuẩn đã khóa độc lập trước khi đo; giữ bản gốc, ghi lý do
và phân tích độ nhạy trên tất cả lượt đã ghi nếu có loại dữ liệu.

**Bộ khó** (một điều kiện duy nhất: nền lạ và thiếu sáng cùng lúc, dựng một lần giữ cả buổi):

```
python tools/run_blinded_campaign.py --arm real_only "--depth-mode rgbd" --pose-list config/pose_lists/hard_v2.csv --trials 200 --block 25 --session 2026-09-21-sang --operator AN --seed 8 --common "--mode real --ik-source yrc --tool-no 1 --confirm-each-trial --no-viewport-mirror --save-frames --frames-dir D:/Scientific/Dataset/DigitalTwin/frames --lighting dim"
```

### So sánh: nộp cả họ một lần

*Việc của người huấn luyện, làm sau khi đã nhận đủ dữ liệu và mở file khoá.*

Chiến dịch mù sinh **một file CSV cho mỗi khối 25 lượt**, không phải một file cho mỗi nhánh: ba
nhánh × 200 lượt là 24 file trong một buổi. Tên nhánh nằm trong **cột** `depth_mode` của từng
dòng, không nằm trong tên file. Vậy nên cuối buổi phải dồn file vào thư mục của buổi, tách khối
đối chứng ra riêng:

```
mkdir results\2026-09-20-sang
mkdir results\2026-09-20-sang-doichung
move results\experiment_real_*.csv results\2026-09-20-sang\
move results\telemetry_*.csv results\2026-09-20-sang\
```

Hai file của khối đối chứng nhận ra bằng giờ chạy (đầu buổi và cuối buổi); chuyển hai file đó
sang `results\2026-09-20-sang-doichung\`. Rồi so sánh cả họ trong một lệnh:

```
python scripts/04_analyze_results.py --csv "results/2026-09-20-sang/experiment_real_*.csv" --split-col depth_mode --baseline rgbd --pair-key pose_id --score-col human_ok > results/e2_phan_tich.txt
```

`--split-col depth_mode` tự gom các khối rời của cùng một nhánh lại, nên không phải nối file bằng
tay. Chương trình áp Holm cho cả họ. Chạy từng cặp riêng thì mỗi cặp tự thấy mình có ý nghĩa. Ví
dụ: hai kiểm định p = 0,03 và 0,04, sau Holm cả hai đều không.

Khối đối chứng đầu buổi so với cuối buổi thì so theo file, vì cùng một nhánh:

```
python scripts/04_analyze_results.py --csv results/2026-09-20-sang-doichung/<file đầu buổi>.csv --paired-with results/2026-09-20-sang-doichung/<file cuối buổi>.csv --label-a dau_buoi --labels-b cuoi_buoi --pair-key pose_id --score-col human_ok > results/e2_doichung.txt
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


## Pha 5 — E3 đến E6: sinh ảnh và huấn luyện trên Linux GPU

*Việc của người huấn luyện, trừ mục Mang mô hình mới về cell.*

Mỗi cấu hình phải lưu spec, render, nhãn, manifest ảnh, cấu hình đã giải quyết toàn bộ giá trị,
log huấn luyện, trọng số và SHA-256. Nguồn cấu hình khóa là
[campaign manifest](CAMPAIGN_MANIFEST.md); lệnh đóng gói/chạy GPU chi tiết ở
[TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md).

Có thể chuẩn bị package và chạy thử phần mềm trước khi E2 vật lý hoàn tất. Tuy nhiên, dữ liệu
anchored dùng cho bài phải dựa trên hiệu chuẩn thật đã kiểm tra, cùng phiên bản cell; không
sinh lô chính bằng transform mô phỏng hoặc sigma dự phòng rồi gọi là hiệu chuẩn đo được.
Sinh ảnh, render, huấn luyện và validation không cần robot chuyển động. Tỷ lệ hoàn thành
pick-and-place của E3–E5 vẫn phải đo trên cell.

### E3: so sánh wide-range với anchored và quét κ

Mỗi recipe dùng cùng ngân sách **3000 ảnh synthetic đưa vào train**, ngoài tập real train đã
khóa. Trong CLI, `--mode blind` là recipe wide-range của bài. Sinh bốn cấu hình:

```bash
python scripts/20_generate_synth.py --mode blind --n 3000 --seed 0 --out data/synth/blind
python scripts/20_generate_synth.py --mode anchored --kappa 1 --n 3000 --seed 0 --out data/synth/anchored_k1
python scripts/20_generate_synth.py --mode anchored --kappa 2 --n 3000 --seed 0 --out data/synth/anchored_k2
python scripts/20_generate_synth.py --mode anchored --kappa 4 --n 3000 --seed 0 --out data/synth/anchored_k4
```

Seed 0 ở đây là seed **sinh dữ liệu**, khác seed huấn luyện. Dùng chung seed sinh cảnh khi
so sánh recipe, lưu cả spec thực tế; seed giống nhau không chứng minh các cảnh hoàn toàn
khớp nhau khi các recipe lấy mẫu khác nhau.

Render rồi gán nhãn từng cấu hình, ví dụ:

```bash
blenderproc run --custom-blender-path <thu_muc_blender> src/synthgen/render_blenderproc.py -- --scenes data/synth/anchored_k2/specs --out data/synth/anchored_k2/render --samples 24 --device gpu
python scripts/20_generate_synth.py --make-labels --val-frac 0 --out data/synth/anchored_k2
```

Lặp lại cho `blind`, `anchored_k1`, `anchored_k4`. `--val-frac 0` giữ ngân sách train 3000;
validation để chọn model dùng **real validation đã khóa**, không lấy synthetic validation
thay thế. Kiểm tra số ảnh/nhãn sau chuyển đổi, không coi số spec là số ảnh train đã thành công.
Ảnh synthetic để chấm E6 phải được sinh riêng với manifest và dải `seed + index`
không giao dải training. Chỉ đổi seed từ 0 sang 1 không tạo một bộ ảnh độc lập.

### Chọn κ và checkpoint bằng validation

Khóa luật dưới đây **trước** khi đọc điểm của loạt cấu hình mới:

1. Với từng run, giữ checkpoint có tổng `box mAP@0.5:0.95 + mask mAP@0.5:0.95`
   trên real validation cao nhất. Điểm số dùng thang 0–1 của mỗi metric, không cộng nhầm %.
2. Với từng κ trong `{1, 2, 4}`, tính trung bình điểm validation này của đủ năm seed
   `[0, 1, 2, 3, 4]`. Chọn κ có trung bình cao nhất; điểm bằng nhau thì chọn κ nhỏ hơn.
   Không tự đặt vùng hòa 0,01, không làm tròn điểm trước khi chọn.
3. Trong mỗi cấu hình, chọn deployed seed có cùng điểm validation cao nhất; nếu hòa,
   chọn seed đứng trước trong thứ tự `[0, 1, 2, 3, 4]` (E4/E5 dùng `[0, 1, 2]`).
4. κ được chọn quyết định checkpoint E3 khởi tạo E4 và recipe đầy đủ cho E5. Theo Results
   hiện tại, **cả ba κ vẫn có so sánh vật lý với wide-range**; việc chọn κ không xóa các
   nhánh còn lại khỏi đánh giá E3. Mỗi cấu hình triển khai một checkpoint đã khóa.
5. Không dùng ảnh/kết quả của final evaluation để chọn κ, model, threshold hoặc thời điểm
   dừng. Baseline `e2_seed1_best.pt` là lựa chọn lịch sử theo **mask-only validation**;
   lưu riêng provenance này, không mô tả hồi tố rằng nó đã được chọn bằng tổng box+mask.

Các ảnh mang tên `test-standard`/`test-hard` cũ đã tham gia phát triển generator, nên kết quả
trên đó là **development analysis**. Bộ hard kết hợp nền lạ + thiếu sáng cũng được chọn sau
khi xem baseline. Không đổi dòng `val:` sang bộ hard cũ để chọn κ rồi báo bộ đó là test độc lập.
Tập ảnh xác nhận mới và danh sách pose evaluation phải tách khỏi mọi dữ liệu phát triển,
hiệu chuẩn và adaptation. Thiếu tập xác nhận độc lập thì báo rõ giới hạn đó.

### Huấn luyện nhiều seed

E2/E3 dùng năm seed `[0, 1, 2, 3, 4]`; E4/E5 dùng ba seed `[0, 1, 2]`. Giữ nguyên thứ tự
và ngân sách cho mọi nhánh. Mỗi run có thư mục riêng; không ghi đè một run đã có kết quả.

Recipe so sánh ban đầu: Ultralytics **8.4.66**, YOLOv8s-seg, 130 epochs, `imgsz=1280`,
`batch=8`, `nbs=64`, `patience=0`, optimizer AdamW, `lr0=0.001111`, `momentum=0.9`,
`warmup_epochs=3`, `warmup_bias_lr=0`, `lrf=0.01`, `cos_lr=False`, `close_mosaic=10`.
Giữ augmentation và các tham số còn lại giống nhau, lưu effective args và môi trường từng run.
Không để `optimizer=auto` thay optimizer khi tập ảnh tăng. Cùng epochs nhưng khác số ảnh
vẫn khác số cập nhật optimizer; lưu số cập nhật, thời gian và tỷ lệ real/synthetic thực tế.

Script `23_launch_retrain.py` chuẩn bị package portable, không tự chạy huấn luyện. Làm theo
[TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md) để chốt manifest ảnh thật, đóng gói đủ năm lớp
(kể cả `inox_box`), chuyển package sang Linux, chạy `train.py --dry-run` rồi mới chạy GPU.
Đối chiếu ngày 21/09/2026: thư mục Windows `_work/yolo/images/train` có **1.318 ảnh**,
trong khi split list và paper ghi **1.256 ảnh**; script export nối thêm **62 ảnh negative**
vào train. Chưa xác minh bộ nào đã dùng trên máy Linux cho baseline. Vì vậy không tự bỏ
62 ảnh, không tự đổi số trong Results và không lấy thư mục hiện tại làm bằng chứng lịch sử.
Đối chiếu manifest/log Linux rồi khóa allowlist cùng số ảnh chính xác trước khi đóng gói.

### Đo mAP và báo cáo

Phân biệt ba loại điểm: validation để chọn checkpoint/κ; development để phân tích các tập cũ;
final test độc lập để báo cáo xác nhận sau khi khóa lựa chọn. Lưu điểm từng seed và trung bình
± độ lệch chuẩn **mask mAP** cho các bảng Results. Điểm box+mask dùng chọn model là một đại
lượng khác, không điền vào cột mask mAP. Độ lệch chuẩn giữa seed không thay thế bất định do
lấy mẫu các cảnh test độc lập.

Dùng `best.pt` đã chọn của từng run khi đánh giá, không lấy dòng cuối `results.csv` làm điểm
của checkpoint tốt nhất. Lưu dataset YAML, manifest ảnh, metric/scoring settings, checkpoint
SHA-256 và toàn bộ output đánh giá; `tee` là một cách lưu terminal output trên Linux.

### E4: adaptation riêng và đối chứng cùng ngân sách

Từ cùng checkpoint E3 đã chọn `f0`, tạo hai nhánh guided và unguided-control. Cả hai giữ cùng
real train và 3000 ảnh synthetic ban đầu. Mỗi update hoàn tất thêm **1000 ảnh train mỗi nhánh**;
ngân sách tích lũy là 3000 → 4000 → 5000. Fine-tune từ checkpoint trước của chính nhánh đó,
không vô tình khởi động lại từ pretrained COCO hoặc từ checkpoint nhánh kia.

Trước update thứ k, chạy **adaptation** với model guided của vòng trước, trên pose/session
adaptation đã khóa. Miner chỉ được đọc các CSV thuộc adaptation của đúng vòng. Ví dụ, thay
các tên file mẫu bằng danh sách đã kiểm tra trong manifest:

```bash
python scripts/21_mine_failures.py --runs results/adaptation/e4_k1/block_001.csv results/adaptation/e4_k1/block_002.csv --n-budget 1000 --out results/adaptation/e4_k1/failure_modes.json
python scripts/20_generate_synth.py --from-failures results/adaptation/e4_k1/failure_modes.json --kappa <kappa_da_chon> --seed 10000 --out data/synth/loop_k1
python scripts/20_generate_synth.py --mode anchored --kappa <kappa_da_chon> --n 1000 --seed 20000 --out data/synth/control_k1
```

Không dùng wildcard toàn cục `results/experiment_real_*.csv`: nó có thể trộn final evaluation,
chạy thử và các vòng khác vào tập mining. Công cụ không tự chứng minh dữ liệu độc lập; người
lập manifest phải kiểm tra vai trò, pose ID và session của từng file. Không khóa κ thành 2;
thay `<kappa_da_chon>` bằng kết quả validation đã ghi trước khi refinement.

Sampler dùng `seed + index`, nên khóa các dải không chồng nhau: E3 ban đầu dùng 0–2999;
ví dụ update 1 guided dùng 10000–10999, control 20000–20999; update 2 dùng seed 30000/40000
tương ứng. Đừng dùng seed control 2000 cho 1000 ảnh: với cùng recipe nó lặp lại draw của
1000 ảnh cuối E3, không phải thêm dữ liệu mới. Ghi dải thật đã dùng trong manifest; dành
một dải riêng không giao các dải này cho synthetic evaluation E6.

Nếu adaptation không có failure, ghi update không thực hiện và dừng theo thuật toán; không
đọc lại `failure_modes.json` cũ. Sau render, `--make-labels --val-frac 0` cho cả hai nhánh.
Package vòng 1 chứa initial + batch vòng 1; vòng 2 chứa initial + batch 1 + batch 2 của đúng
nhánh. Ba seed fine-tuning dùng cùng lịch giữa hai nhánh; khóa số epochs cho E4 trước khi chạy
và truyền tường minh như hướng dẫn training. Equal budget không có nghĩa số ảnh là đủ để
chứng minh hiệu quả; cần đối chiếu outcomes và bất định.

Ở update 1, ba seed cùng bắt đầu từ `f0` đã chọn. Từ update 2, mỗi nhánh **và mỗi seed**
tiếp tục checkpoint của chính nó ở update trước. Vì một package nhận một `--model`, tạo
package riêng cho từng cặp nhánh/seed ở update 2, truyền `--seeds <seed>` và `--model`
trỏ đúng parent, cùng `--epochs` đã khóa. Không đưa checkpoint tốt nhất giữa các seed của
update 1 làm parent chung cho cả ba seed update 2; cách đó thay đổi thiết kế đang mô tả.

Chỉ chấm các checkpoint đã lưu trên **evaluation** sau khi đã cố định toàn bộ update.
Không dùng kết quả evaluation để phân bổ ảnh, đổi threshold hoặc dừng vì tăng ít. Ghi riêng
failure counts/denominators của adaptation và evaluation. Một lỗi cơ khí bị miner gom theo
context không có nghĩa retraining sẽ sửa được nguyên nhân cơ khí.

### E5: ablation đủ năm yếu tố

CLI dùng `--ablation none|illumination|background|distractors|camera|pose`, chỉ với
`--mode anchored`. Tạo full recipe và năm ablation, mỗi cấu hình **3000 ảnh train**, cùng
κ đã chọn và cùng generation seed. Ví dụ:

```bash
python scripts/20_generate_synth.py --mode anchored --kappa <kappa_da_chon> --ablation illumination --n 3000 --seed 0 --out data/synth/e5_illumination
```

Lặp cho `none`, `background`, `distractors`, `camera`, `pose`; render/gán nhãn với
`--val-frac 0`, rồi huấn luyện ba seed `[0, 1, 2]`. Lưu cấu hình và giá trị cố định thật của
mỗi ablation cùng spec; kiểm tra từng biến đổi trước lô render chính. Bỏ biến thiên pose phải
có phân bố pose tham chiếu đã xác định, không diễn giải là đặt mọi vật chồng lên một điểm.

Cả năm yếu tố đều có **mask mAP và task success** so với full recipe theo kế hoạch hiện tại.
Không dự đoán hai yếu tố nào yếu rồi bỏ phép đo. Hiệu ứng là `ablation − full` theo điểm phần
trăm, âm/dương đều báo cáo; kiểm định năm contrast cùng họ với Holm. Nếu ngân sách phải đổi,
đổi protocol và bản thảo trước khi nhìn kết quả, đồng thời ghi rõ phạm vi suy luận mới.

### Mang mô hình mới về cell

Chép file `.pt` đã chọn và ghi SHA-256 vào `models\`, tạo cấu hình riêng cho từng model với
`model_path` tương ứng. Chạy cùng danh sách **evaluation đã khóa** giữa các cấu hình, vẫn chia
khối xen kẽ và mù theo Pha 4. Không dùng danh sách adaptation làm final evaluation. Các cờ
CLI về depth mode, calibration và class geometry phải được giữ đúng; nút Experiment trong
GUI không mặc nhiên tương đương quy trình CLI này. Bước đo task success cần robot thật.

### E6: transfer và thời gian chu kỳ

Chấm cùng checkpoint trên tập synthetic đánh giá riêng và tập real đánh giá riêng, báo cáo
`100 × (mAP_synthetic − mAP_real)` theo điểm phần trăm cho từng metric. Số dương nghĩa là
điểm synthetic cao hơn; không dùng số này để chọn lại checkpoint. Lưu cả thành phần lớp,
điều kiện và nguồn gốc của hai tập, không lấy ảnh synthetic train làm tập đánh giá.

Kế hoạch thời gian là **100 chu kỳ hoàn chỉnh**. Chốt mốc bắt đầu/kết thúc, cấu hình, điều kiện
và cách xử lý failure/reset trước khi đo. Có thể tận dụng log E2 nếu các mốc và metadata đáp
ứng đúng định nghĩa đó. Tách thời gian người đặt vật khỏi chu kỳ tự động, nhưng giữ thông tin
reset và can thiệp để không gọi chu kỳ máy là throughput sản xuất.

`05_analyze_telemetry.py` suy ra đoạn chuyển động từ ngưỡng vận tốc khớp; biểu đồ của nó là
chẩn đoán, không tự cung cấp timestamp inference, localization, planning và gripper. Lưu raw
telemetry cùng log từng trial và instrument các mốc còn thiếu trước khi điền bảng stage timing.
Tính P50/P95 tổng từ tổng thời gian đo trực tiếp từng chu kỳ, không cộng percentile từng stage.
Thống kê chỉ trên lượt thành công phải ghi rõ số failure bị loại và là phân tích bổ sung.

Giữ `--confirm-each-trial` khi có người đặt vật. Chỉ xét chu kỳ không cần xác nhận trong một
bố trí cấp vật đã được chuẩn bị và cell không có người; không coi lệnh lặp lại trên một vật
đã được chuyển sang băng tải là một kế hoạch cấp vật hợp lệ. Mục tiêu 10 giây là mốc ứng dụng,
chưa phải kết quả đã đo.

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

## Ghi nhật ký sau mỗi buổi

**Máy tự ghi, không phải chép tay:**

| Ghi ở đâu | Có gì |
|---|---|
| `results\experiment_real_<thời điểm>.csv` | mỗi lượt một dòng: thành công hay không, lý do hỏng, thời gian chu kỳ, toạ độ và độ tin cậy của nhận dạng, `pose_id`, điều kiện, `session` / `block` / `operator` |
| `results\telemetry_<thời điểm>.csv` | góc sáu khớp theo thời gian |
| `logs\experiment.log`, `logs\calibration.log` | toàn bộ những gì hiện trên màn hình, kể cả buổi chạy mù |
| `config\calibration\` | kết quả hiệu chuẩn; `tilt_deg` và `rms_mm` nằm trong `table_plane.json` |

**Hai cột chấm, đừng lẫn.** `success` là máy chấm: ở `--mode real`, lúc đóng kẹp phần mềm đọc cảm
biến có vật trong kẹp, không có vật thì ghi `grasp_failed`. Nhưng cảm biến chỉ biết lúc đóng kẹp,
nên vật rơi giữa đường hoặc thả sai chỗ vẫn được máy ghi là thành công. `human_ok` là người chấm,
do chính người đặt vật bấm y hoặc n sau mỗi lượt. Bài báo báo cáo `human_ok`; giữ cả hai cột để so.

**Số chỉ in ra màn hình thì đã có chỗ hứng sẵn trong lệnh:** `04_analyze_results.py`,
`hse_rtt.py`, `05_analyze_telemetry.py` và `yolo segment val` đều không tự ghi log ra file, nên
các lệnh trong hướng dẫn đã kèm sẵn `>` hoặc `| tee`. Gõ thiếu phần đó thì p sau Holm, các mốc
RTT, tần số telemetry và mAP chỉ còn trên màn hình.

**Người phải ghi, vì không công cụ nào ghi hộ:** bốn số đo ở mục 3; bảng chạm thử 8 điểm; vật
rơi lệch bao nhiêu so với điểm dạy; điều kiện thật của phòng; mọi bất thường.

Chép bảng này vào file văn bản, điền sau mỗi buổi chạy:

| Mục | Ghi gì |
|---|---|
| Ngày, giờ bắt đầu và kết thúc | |
| Pha nào, cấu hình gì | ví dụ: Pha 4, chế độ plane, bộ chuẩn |
| Số lượt chạy, số lượt thành công | |
| Tên các file kết quả trong `results\` | |
| Điều kiện thật của phòng | đèn gì, có che nắng không, có ai đi qua chắn sáng không |
| Bất thường | robot dừng, vật rơi, thẻ bị xê dịch, camera bị chạm... |

Rồi **sao lưu cả thư mục `results\` và `logs\` ra ổ ngoài**, đặt tên theo ngày và cấu hình.

---

## Từ điển thuật ngữ

| Từ | Nghĩa |
|---|---|
| **preflight** | loạt kiểm tra tự động chạy **trước khi** nối tới robot; không đạt là không cho chạy |
| **pose list / thẻ** | danh sách vị trí đặt vật in sẵn; mọi cấu hình đem so sánh phải dùng chung một danh sách |
| **pose_id** | mã của một lần đặt vật, ví dụ `P0042`; dùng để ghép cặp kết quả giữa các lần chạy |
| **hiệu chuẩn tay–mắt** | phép đo cho biết camera nằm ở đâu so với gốc robot |
| **TCP** | điểm tác động cuối, tức đầu má kẹp; khai trên teach pendant |
| **chế độ độ sâu** | ba cách tính độ cao vật: `rgbd` đọc ảnh độ sâu, `plane` tính từ mặt bàn, `fusion` kết hợp |
| **telemetry** | file ghi trạng thái khớp robot theo thời gian, 10 lần mỗi giây |
| **canary** | lô nhỏ chạy thử trước lô lớn để bắt lỗi sớm |
| **thẻ** *(card)* | mốc vị trí đánh số dán trên bàn; chương trình gọi số thẻ, người đặt vật đặt đúng thẻ đó, để mọi cấu hình chạy trên cùng bộ vị trí |

---

## Phụ lục: ngân sách số lượt gắp — dự thảo cần khóa

Các số dưới đây mô tả phạm vi so sánh, không phải cam kết đã đủ lực thống kê. Methods đề xuất
200 pose ghép cặp mỗi cấu hình/điều kiện; phân bổ theo lớp/layout, số phiên, các đối chứng và
những lần thử lặp phải được điền vào manifest trước thu chính thức. Không suy ra 200 pose
pooled là 200 pose cho riêng inox hoặc riêng stacked.

| Thí nghiệm | Cấu hình cần có | Số lượt/phân bổ còn phải khóa |
|---|---|---|
| E1 | Đo RTT, telemetry, touch-test; benchmark phần mềm tách riêng | Số mẫu thật mỗi phép đo, mốc thời gian và hồ sơ hiệu chuẩn |
| E2 (C4) | Cùng detector ở RGB-D, plane, fusion | Đề xuất 200 pose mỗi cấu hình/điều kiện; lớp × layout chưa khóa |
| E3 | Real-only, wide-range, anchored κ=1/2/4 | Cùng evaluation list; đủ nhánh để so anchored–wide ở từng κ |
| E4 adaptation | Model guided trước mỗi update | Danh sách/phiên riêng; số adaptation trials chưa khóa |
| E4 evaluation | f0 và hai nhánh ở tối đa hai update | Chỉ chấm sau khi đã khóa update; cùng list giữa checkpoint |
| E5 | Full recipe và cả năm ablation | Cùng list; cả năm đều đo task success và mask mAP |
| E6 | Cùng model, 100 chu kỳ hoàn chỉnh | Xác định log nào được dùng lại và đủ mốc stage nào |

Không cộng một tổng gắp cuối cùng trước khi chốt phạm vi và tái sử dụng đối chứng hợp lệ.
Đối chứng chỉ dùng chung khi checkpoint, pose list, điều kiện, calibration và phiên/block
cho phép so sánh theo protocol; tên recipe giống nhau chưa đủ. Khối drift, chạy thử và
adaptation không được đếm thành evaluation. Thử nghiệm 80 lượt không chứng minh không thoái
lui; không có kế hoạch equivalence/non-inferiority đã xác định thì báo hiệu ứng và bất định.

Power phụ thuộc số cặp bất đồng, phân bố lớp/layout và hiệu chỉnh nhiều phép so sánh. Hiệu
ứng dưới 10 điểm phần trăm vẫn có thể ước lượng; không có ngưỡng phát hiện chắc chắn chỉ từ
n=200. Khi lịch/ngân sách buộc thay đổi, cập nhật manifest và Methods/Results trước khi thu
hoặc xem kết quả liên quan, ghi lý do và các phân tích chuyển thành exploratory.

Mỗi lượt cần đặt vật thủ công: đo thời gian của pilot riêng để dự trù lịch và nhân theo số
lượt đã khóa; không đưa pilot vào final evaluation sau khi đã dùng nó để điều chỉnh hệ thống.

---

## Phụ lục: ghi chú Blender

Máy hiện tại bị chặn `download.blender.org`, nên `blenderproc` không tự tải Blender được. Đã
tải sẵn qua mirror và giải nén tại `C:\Users\manhh\blender\blender-4.2.1-windows-x64`; khi
chạy render nhớ trỏ `--custom-blender-path` vào đó. Máy GPU mới cũng có thể vướng như vậy:
lấy `blender-4.2.1-linux-x64.tar.xz` từ mirror (clarkson, nluug, dotsrc, freedif).
