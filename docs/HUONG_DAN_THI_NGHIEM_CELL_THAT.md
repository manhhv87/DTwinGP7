# HƯỚNG DẪN CHẠY THÍ NGHIỆM TRÊN CELL GP7

## 0. Chú ý

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
| Pha 3, E1 | cell | 50 lượt | đặt vật, bấm ENTER, chấm y/n | `e1_*`, CSV, telemetry, 4 PNG |
| Pha 4, E2 | cell | 3 nhánh × 200 lượt, bộ khó 200 lượt, kèm đối chứng 20+20 mỗi buổi | đặt vật, chấm y/n, không mở file khoá | 24 CSV khối mỗi buổi, file khoá, thư mục khung ảnh |
| Pha 5, sinh ảnh và huấn luyện | **máy GPU Linux** | không | viết luật chọn κ, chạy val, chọn ra một κ | `specs`/`render`/`dataset`, `dsv<k>.yaml`, `runs/` |
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
| **C4** | Ba chế độ độ sâu `rgbd`, `plane`, `fusion`, gồm cả vật inox phản chiếu | **Pha 4 (E2)**, chính 600 lượt của lệnh chiến dịch mù |

E5 và E6 không mang mã đóng góp riêng: E5 bổ trợ cho C3, E6 cho khoảng cách mô phỏng với thật và
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

**Người huấn luyện gửi lại:** file `.pt` của mô hình mới, κ đã chọn kèm luật chọn đã ghi trước,
dòng `model_path` cần sửa trong `config\experiment.yaml`, và danh sách thẻ cho lượt đo lại nếu
khác lần trước.

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

Riêng E6 đo chu kỳ không dùng `--confirm-each-trial`, nhưng cũng không ai đứng đặt vật: lấy chu
kỳ từ telemetry E2, hoặc chạy với vật đặt sẵn và cell trống.

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

Sau bước này đầu dòng lệnh sẽ hiện `(.venv)`. Nếu không hiện, môi trường ảo chưa bật, mọi lệnh phía sau sẽ báo thiếu thư viện.

**File trọng số mô hình không có trên git**, vì git của dự án cố ý loại mọi file `.pt`. Người huấn
luyện gửi riêng file `e2_seed1_best.pt`; chép vào thư mục `models\` của mã nguồn. Thiếu file này
thì chạy thật báo lỗi thiếu mô hình.

Kiểm tra máy chạy được, chưa cần robot:

```
pytest tests/ -q
```

**DỪNG nếu:** dòng cuối không phải `766 passed`, hoặc có chữ `failed` ở bất kỳ đâu.

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
| 4 | Thông số camera D455 thật | `config/synthgen.yaml` → `camera.intrinsics` | `fx: 642`, số giữ chỗ |

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

Ước lượng H khoảng 250–330 mm, tức cửa sổ thật có thể chỉ còn **380 đến 470 mm**. Đo H rồi tính
lại, đừng lấy con số ước lượng này làm chuẩn.

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
python scripts/03_run_experiment.py --mode real --trials 50 --depth-mode rgbd --pose-list config/pose_lists/std_v1.csv --telemetry-hz 10 --confirm-each-trial --no-viewport-mirror
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
▶ Trial 7 — PLACE OBJECT: card=14  x=470.8 mm  y=116.7 mm  yaw=85.0 deg  class=metal_box
```

Cùng một danh sách 300 tư thế được phát lại cho mọi cấu hình, nên kết quả ghép cặp được từng
lượt; đó là điều kiện của kiểm định McNemar. Đặt đại là mất ghép cặp.

**Chuẩn bị bàn:**

1. In `position_cards.pdf` (20 thẻ, một trang A4) ở **100% / actual size**, đo vạch 50 mm
   in sẵn, cắt theo viền.

2. **Lấy dấu bằng robot.** Toạ độ của mỗi thẻ in sẵn trên chính thẻ đó (x từ 317 đến 522 mm,
   y từ −350 đến +350 mm), tính theo **gốc robot**, không phải mép bàn. Trên teach pendant bấm
   COORD chọn hệ Robot, jog tới đúng X, Y của thẻ, hạ Z sát mặt bàn, đánh dấu ngay dưới đầu TCP
   rồi nhấc lên. Cần TOOL01 đã khai đúng. Hai mươi lần, hết một buổi.

![Toạ độ thẻ đo từ gốc robot](ban_ve_toa_do_the.png)

*Bản in: `ban_ve_toa_do_the.pdf`.*

3. **Dán thẻ vào dấu**, tâm thẻ trùng dấu, mũi tên trên thẻ hướng **+x, tức ra xa robot**. Dán
   băng dính trong phủ kín cả thẻ để nó không bong và không xê dịch. Dán xong cả 20, jog lại về
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
python tools/run_blinded_campaign.py --arm rgbd "--depth-mode rgbd" --arm plane "--depth-mode plane" --arm fusion "--depth-mode fusion" --pose-list config/pose_lists/std_v1.csv --trials 200 --block 25 --session 2026-09-20-sang --operator AN --seed 7 --common "--mode real --ik-source yrc --tool-no 1 --confirm-each-trial --no-viewport-mirror --save-frames --frames-dir D:/Scientific/Dataset/DigitalTwin/frames"
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
python scripts/03_run_experiment.py --mode real --depth-mode rgbd --pose-list config/pose_lists/std_v1.csv --pose-slice 200:220 --session-id 2026-09-20-sang --block-id doichung-dau --operator-id AN --confirm-each-trial --no-viewport-mirror
```

Cuối buổi chạy lại đúng lệnh đó, đổi `--block-id doichung-cuoi`. Hai khối lệch nhau nhiều hơn
hiệu ứng đang tìm thì buổi đó bỏ.

**Bộ khó** (một điều kiện duy nhất: nền lạ và thiếu sáng cùng lúc, dựng một lần giữ cả buổi):

```
python tools/run_blinded_campaign.py --arm real_only "--depth-mode rgbd" --pose-list config/pose_lists/hard_v1.csv --trials 200 --block 25 --session 2026-09-21-sang --operator AN --seed 8 --common "--mode real --ik-source yrc --tool-no 1 --confirm-each-trial --no-viewport-mirror --save-frames --frames-dir D:/Scientific/Dataset/DigitalTwin/frames --lighting dim"
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

`--score-col human_ok` chấm theo mắt người, và đó là số bài báo báo cáo. Chạy thêm một lần **bỏ
cờ đó** để lấy số của máy vào `results/e2_phan_tich_may.txt`: chênh lệch giữa hai số chính là số
vật đã kẹp được nhưng rơi giữa đường hoặc thả sai chỗ. Báo cáo cả hai.

Script này **không tự ghi log ra file**, nên phần chuyển hướng `> results/e2_phan_tich.txt` là bắt
buộc: không có nó thì p sau Holm chỉ nằm trên màn hình.

Nếu một lượt nào đó không được chấm, chương trình **từ chối chạy** và nói còn bao nhiêu dòng
trống: chấm một nửa rồi so sánh sẽ lặng lẽ bỏ bớt lượt của đúng một nhánh.

**DỪNG nếu:** các lần chạy không dùng cùng danh sách và cùng `pose_id`. Phân tích tự từ chối, nhưng
lúc đó đã muộn.

**GHI SỐ:** tỷ lệ thành công từng cấu hình; chênh lệch bộ chuẩn với bộ khó. Bộ khó không thấp hơn
rõ rệt là một kết quả, không phải lần trượt: ghi lại, **không dựng lại điều kiện rồi chạy lại**.

---


## Pha 5 — E3 đến E6: cần máy GPU

*Việc của người huấn luyện, trừ mục Mang mô hình mới về cell.*

**Mục tiêu:** sinh ảnh tổng hợp, huấn luyện lại mô hình, rồi đo lại trên chính cell thật.

**Kết quả ra:** mỗi cấu hình một bộ ba thư mục `data/synth/<tên>/specs`, `/render`, `/dataset`;
`data/packages/dsv<k>.yaml`; trên máy GPU, mỗi seed một thư mục `runs/dsv<k>_s<seed>/` chứa
`weights/best.pt` và `results.csv`; `results/failure_modes.json` của E4. Chép từ màn hình vào nhật
ký: mAP mask của từng seed, trung bình và độ lệch chuẩn, và κ đã chọn.

**Cần có trước:** E2 xong, và **không đụng vào camera** từ lúc hiệu chuẩn tới giờ.

Pha này sinh ảnh tổng hợp bằng Blender rồi huấn luyện lại mô hình. Nặng, nên làm trên máy GPU
Linux, không làm trên laptop.

**Phần nào cần robot:** sinh ảnh, render, gán nhãn, huấn luyện, quét κ đều **không** cần robot,
làm hoàn toàn trên máy GPU. Nhưng E3 và E4 chỉ có mAP là chưa đủ: mỗi mô hình mới phải mang về
cell gắp thật thì mới có tỷ lệ gắp để so. E6 thì hoặc trích từ telemetry đã ghi, hoặc chạy thêm
một buổi ở cell.

### E3: hai nhánh tổng hợp

**Cả hai nhánh** đều phải đi hết bốn bước: sinh spec, render, gán nhãn, huấn luyện. Bỏ sót
nhánh blind ở bất kỳ bước nào là mất luôn nhánh so sánh chính của C2.

```
python scripts/20_generate_synth.py --mode anchored --kappa 2 --n 3000 --seed 0 --out data/synth/anchored_k2
python scripts/20_generate_synth.py --mode blind --n 3000 --seed 0 --out data/synth/blind
```

Render và gán nhãn **cả hai**:

```
blenderproc run --custom-blender-path <thư mục blender> src/synthgen/render_blenderproc.py -- --scenes data/synth/anchored_k2/specs --out data/synth/anchored_k2/render --samples 24 --device gpu
python scripts/20_generate_synth.py --make-labels --out data/synth/anchored_k2

blenderproc run --custom-blender-path <thư mục blender> src/synthgen/render_blenderproc.py -- --scenes data/synth/blind/specs --out data/synth/blind/render --samples 24 --device gpu
python scripts/20_generate_synth.py --make-labels --out data/synth/blind
```

### Quét κ bằng mAP, không bằng gắp vật lý

κ là tham số tự do duy nhất của phương pháp neo, chắc chắn sẽ bị hỏi. Nhưng trả lời bằng gắp
vật lý tốn 400 lượt. Sinh và huấn luyện cả ba κ, **chọn κ theo mAP trên bộ khó chính**, rồi chỉ
gắp vật lý ở κ đã chọn.

**Chọn lúc nào:** ngồi ở máy GPU, sau khi huấn luyện xong ba κ, và **trước khi** quay lại cell.
Không chọn trong lúc đang chạy cell, cũng không để tới lúc phân tích số liệu gắp. Thứ tự bốn
bước, cả bốn đều không cần robot:

1. Huấn luyện ba κ, mỗi κ 5 seed.
2. Viết luật chọn vào nhật ký.
3. Chạy `yolo segment val` cho cả ba κ, được 15 con số mAP.
4. Áp luật, ra đúng một κ.

Xong bước 4 mới mang mô hình của κ đó về cell gắp thật.

```
python scripts/20_generate_synth.py --mode anchored --kappa 1 --n 3000 --seed 0 --out data/synth/anchored_k1
python scripts/20_generate_synth.py --mode anchored --kappa 4 --n 3000 --seed 0 --out data/synth/anchored_k4
```

Render và gán nhãn hai thư mục đó y như trên.

**Chốt luật chọn trước khi nhìn số.** Ba κ cho ba con số mAP. Nhìn ba số rồi mới chọn là chọn
theo dữ liệu, và phản biện gọi đúng tên nó là dò. Nên trước khi chạy lệnh đánh giá **đầu tiên**,
chép nguyên khối này vào nhật ký và điền ngày, tên người viết:

```
Luật chọn kappa — chốt ngày ........, người viết ........
1. Số dùng để chọn: mAP50-95 của mask, đo trên bộ khó chính.
2. Mỗi kappa lấy TRUNG BÌNH 5 seed. Không lấy seed tốt nhất.
3. Kappa nào trung bình cao nhất thì chọn.
4. Hai kappa hơn kém nhau dưới 0,01 thì coi là hoà, chọn kappa NHỎ hơn.
5. Chỉ kappa được chọn mới đem đi gắp vật lý. Hai kappa còn lại dừng ở mAP.
```

Viết xong mới chạy `yolo segment val` cho cả ba κ. Rồi áp luật đúng như đã viết, kể cả khi nó ra
κ mình không thích. Muốn đổi luật thì đổi **trước** khi chạy đánh giá, và ghi rõ đã đổi cái gì.

Luật này quyết định 15 lần huấn luyện (3 κ × 5 seed). Máy GPU không kham nổi thì hạ số seed của
riêng vòng quét xuống, nhưng phải hạ **trong luật, trước khi chạy**, chứ không phải hạ giữa chừng
lúc thấy lâu.

### Huấn luyện nhiều seed

Bài báo báo cáo mAP dạng trung bình cộng độ lệch chuẩn, nên **mỗi cấu hình phải huấn luyện
nhiều lần với seed khác nhau**: 5 seed cho E2 và E3, 3 seed cho E4 và E5. Đây là khối lượng GPU
lớn nhất cả dự án, tính lịch trước.

```
python scripts/23_launch_retrain.py --synth data/synth/anchored_k2/dataset --real <thư mục dataset thật> --version 1
```

Lệnh này **không huấn luyện**. Nó gộp dataset thành `data/packages/dsv1.yaml` (train trộn thật với
tổng hợp, val giữ nguyên ảnh thật) rồi in ra lệnh huấn luyện. Huấn luyện chạy trên máy GPU, cùng
một file yaml, đổi seed:

```
for SEED in 1 2 3 4 5; do
  yolo segment train data=data/packages/dsv1.yaml model=yolov8s-seg.pt epochs=100 imgsz=1280 seed=$SEED project=runs name=dsv1_s$SEED
done
```

`--version` là số thứ tự vòng lặp dataset, `seed` mới là hạt giống huấn luyện. Đổi nhầm hai cái
này thì ra 5 phiên bản dataset chứ không phải 5 lần huấn luyện cùng một dataset.

Làm tương tự cho `blind`, `anchored_k1`, `anchored_k4`.

### Đo mAP

Huấn luyện xong một seed thì đánh giá ngay mô hình tốt nhất của seed đó:

```
yolo segment val model=runs/dsv1_s1/weights/best.pt data=data/packages/dsv1.yaml imgsz=1280 | tee results/map_dsv1_s1.txt
```

Lệnh in ra một bảng. Lấy dòng `all`, cột `mAP50-95` của phần **Mask**. Tập val khai trong
`dsv1.yaml` là **ảnh thật** (lấy ảnh tổng hợp làm val sẽ thổi phồng mAP), nên số này so sánh được
giữa các cấu hình.

`tee` vừa hiện lên màn hình vừa giữ lại thành file, nên không phải chép tay. Chạy cho đủ 5 seed,
đổi tên file theo seed, rồi tính trung bình và độ lệch chuẩn của 5 con số. Đó là số điền vào cột
mAP của bảng anchored và cột ΔmAP của bảng ablation.

**Đo trên bộ khó** (để quét κ): chép `dsv1.yaml` thành file mới, sửa dòng `val:` trỏ vào thư mục
ảnh bộ khó, rồi chạy lại lệnh trên với `data=` file mới đó.

`runs/dsv1_s1/results.csv` cũng ghi mAP từng epoch ở cột `metrics/mAP50-95(M)`. Đừng lấy dòng cuối
file này: đó là epoch cuối, còn `best.pt` là epoch tốt nhất, hai cái thường khác nhau.

### E4: vòng lặp học từ lỗi, và nhánh đối chứng

Vòng lặp một mình không chứng minh được gì: ảnh sinh thêm có thể giúp chỉ vì **nhiều ảnh hơn**,
không phải vì **hướng theo lỗi**. Phải có nhánh đối chứng dùng **đúng cùng số ảnh** nhưng sinh
ngẫu nhiên.

```
python scripts/21_mine_failures.py --runs "results/experiment_real_*.csv" --n-budget 1000
python scripts/20_generate_synth.py --from-failures results/failure_modes.json --out data/synth/loop_k1 --seed 1000
python scripts/20_generate_synth.py --mode anchored --kappa 2 --n 1000 --seed 2000 --out data/synth/control_k1
```

`--n-budget 1000` chính là **N_k**, số ảnh mỗi vòng. Nhánh đối chứng phải dùng đúng con số đó.
Render, gán nhãn, huấn luyện 3 seed cho **cả hai** nhánh, rồi làm tiếp vòng 2 y hệt.

### E5: bỏ từng yếu tố — CHƯA CHẠY ĐƯỢC

> **`20_generate_synth.py` chưa có cờ tắt một yếu tố ngẫu nhiên hoá.** Không xếp lịch và không
> tính ngân sách gắp cho E5 cho tới khi có cờ đó.
>
> Việc phải làm trước là việc khoa học, không phải việc lập trình: chốt **"tắt một yếu tố" nghĩa
> là gì.** Ghim yếu tố đó ở giá trị neo đo được, hay ghim ở một giá trị mặc định? Hai cách cho hai
> kết luận khác nhau về tầm quan trọng của yếu tố ấy. Ghi định nghĩa đã chốt vào nhật ký trước khi
> viết cờ.
>
> Chốt xong thì thêm cờ không khó: mỗi nhóm yếu tố đã là một hàm riêng trong
> `src/synthgen/scene_sampler.py` (`_sample_lights`, `_sample_material`, `_sample_background`,
> `_sample_camera`, `_sample_distractors`).

Khi có công cụ rồi thì chỉ ba yếu tố có gắp vật lý. Hai yếu tố còn lại chỉ báo cáo ΔmAP, vì với
n = 200 thì hiệu ứng dưới 10 điểm phần trăm không phát hiện được, mà hai yếu tố đó gần như chắc
chắn nằm dưới ngưỡng. Nói rõ lý do đó trong bài thay vì im lặng bỏ qua.

### Mang mô hình mới về cell

Chép file `.pt` về thư mục `models\`, sửa `model_path` trong `config\experiment.yaml`, rồi chạy
**đúng danh sách thẻ cũ**, vẫn chia khối xen kẽ và vẫn mù, bằng chính lệnh ở Pha 4. Bước này bắt
buộc phải có robot; không có nó thì E3 với E4 chỉ còn mAP.

### E6: đo chu kỳ

**Không dùng `--confirm-each-trial`**, nhưng cũng không đứng đặt vật. Hai cách, chọn một:

1. **Lấy từ telemetry đã có.** Chiến dịch C4 đã ghi telemetry mọi lượt; trích thời gian chu kỳ
   của các lượt thành công từ đó. Không tốn thêm lượt gắp nào, và không ai phải đứng trong cell.
2. **Chạy riêng với vật đặt sẵn.** Đặt sẵn vật, chạy, **không ai đứng trong cell**, robot lặp
   trên cùng một vật.

```
python scripts/05_analyze_telemetry.py latest
```

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

## Phụ lục: ngân sách số lượt gắp

| Thí nghiệm | Cấu hình | Số lượt | Ghi chú |
|---|---|---:|---|
| E2 (C4) | 3 chế độ độ sâu × 200 trên bộ chuẩn | 600 | 100 lượt inox mỗi chế độ, 40 lượt xếp chồng |
| E2 (bộ khó) | real-only trên bộ khó chính | 200 | điều kiện hai yếu tố |
| E3 | blind + anchored (κ đã chọn) trên bộ khó chính | 400 | κ còn lại quét bằng mAP, không gắp |
| E3 | kiểm không thoái lui trên bộ chuẩn, 2 nhánh × 80 | 160 | chỉ cần đủ để thấy không tụt |
| E4 | 2 vòng × 2 nhánh trên bộ khó chính | 800 | nhánh thứ hai là đối chứng cùng ngân sách |
| E5 | 3 yếu tố × 200 | 600 | hai yếu tố còn lại báo cáo bằng ΔmAP |
| E6 | đo chu kỳ | 100 chu kỳ | lấy từ telemetry E2 nếu được |
| | **Tổng** | **2760** | cộng 100 chu kỳ |

**Bốn thứ cố tình không làm, để khỏi ai đó thêm lại:**

- Không chạy plane và fusion trên bộ khó: bảng chế độ độ sâu không tách theo điều kiện sáng.
- Không quét κ bằng gắp vật lý: quét bằng mAP, chỉ gắp ở κ đã chọn.
- Không gắp cho hai yếu tố E5 yếu nhất: với n = 200 chúng nằm dưới ngưỡng phát hiện, báo cáo
  bằng ΔmAP.
- Không chạy E1 riêng: gộp vào chiến dịch C4-rgbd, giữ 10 lượt làm cổng an toàn.

Mỗi lượt phải đặt vật bằng tay. Trước khi bắt đầu, bấm giờ 10 lượt đầu rồi nhân lên để biết
200 lượt mất bao lâu, đừng ước lượng.

---

## Phụ lục: ghi chú Blender

Máy hiện tại bị chặn `download.blender.org`, nên `blenderproc` không tự tải Blender được. Đã
tải sẵn qua mirror và giải nén tại `C:\Users\manhh\blender\blender-4.2.1-windows-x64`; khi
chạy render nhớ trỏ `--custom-blender-path` vào đó. Máy GPU mới cũng có thể vướng như vậy:
lấy `blender-4.2.1-linux-x64.tar.xz` từ mirror (clarkson, nluug, dotsrc, freedif).
