# Đọc trước

Dành cho người chạy cell. Huấn luyện mô hình và phân tích số liệu không thuộc phần việc này.
Mọi file nhắc tới dưới đây nằm cùng thư mục với file này.

## Đọc gì trước

`HUONG_DAN_THI_NGHIEM_CELL_THAT.md`: đọc mục **Quy trình tổng thể** và **Chia việc** trước, rồi mới
đọc lần lượt từ mục 1. Cách lấy mã nguồn về máy và cài đặt nằm ở mục 2; cách khai TOOL01 trên
teach pendant nằm ở mục 3.

Trong thư mục `docs\` của mã nguồn có file `HUONG_DAN_CAI_DAT.md`, viết từ trước và nhiều chỗ đã cũ (thông số bàn
cờ, giá trị TOOL01, số test). Chỗ nào nó nói khác hướng dẫn thí nghiệm thì làm theo hướng dẫn thí
nghiệm.

Hướng dẫn có liên kết tới `CAMPAIGN_MANIFEST.md`, `TRAINING_WORKFLOW.md` và `models/README.md`.
Đó là tài liệu phía người huấn luyện, nằm trong mã nguồn chứ không nằm cạnh file này; người chạy
cell không cần mở để làm Pha 1 đến Pha 4.

## In gì, in thế nào

| File | Khổ | Cách in | Kiểm sau khi in |
|---|---|---|---|
| `charuco_a3_o45mm.pdf` | A3 ngang | **100% / actual size**, không *fit to page*, giấy mặt mờ | đo vạch 100 mm in sẵn; sai là in lại. Rồi đo cạnh ô và cạnh dấu bằng thước kẹp, số đo được là số gõ vào lệnh Pha 1 |
| `position_cards.pdf` | A4 dọc | **100% / actual size** | đo vạch 50 mm in sẵn, rồi cắt 21 thẻ theo viền (hai trang) |
| `ban_ve_ga_ban_co.pdf` | A4 ngang | in thường | bản vẽ để đọc, không đo trên giấy |
| `ban_ve_ga_3d.pdf` | A4 ngang | in thường | bản vẽ để đọc, không đo trên giấy |
| `ban_ve_toa_do_the.pdf` | A4 ngang | in thường | bản vẽ để đọc, không đo trên giấy |

Chỉ dùng tấm bàn cờ A3 ô 45 mm. Mọi con số trong hướng dẫn tính cho tấm này.

## Nộp lại sau mỗi buổi

Theo mục **Chia việc** trong hướng dẫn: thư mục `results\<tên buổi>\`, file
`logs\experiment.log`, thư mục khung ảnh trên ổ D, file khoá
`results\blinding_keys\key_<buổi>.json` **còn nguyên, không mở**, và trang nhật ký của buổi.
Không đẩy những thứ đó lên git: `results\` và `logs\` đã bị git bỏ qua có chủ đích.
