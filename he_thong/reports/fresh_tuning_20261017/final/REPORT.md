# SchoolGuard 2.2.6 — 25 video phát triển, 17/10/2026

Đã chạy mới toàn bộ 25 video từ `fresh_eval_videos_20261017`, giải mã và xuất đủ **4.771 khung hình**, không lỗi, không thay đổi video nguồn. Năm video 21–25 đều phát cảnh báo và có khung người đỏ trong bản MP4. Kết quả khớp **25/25 nhãn tạm kèm theo**, tăng từ 21/25; điều này **không chứng minh đúng 100%**.

## Kết quả theo nhãn hiện có

| Lượt chạy mới từ nguồn | TP | TN | FP | FN | Khớp nhãn |
|---|---:|---:|---:|---:|---:|
| Trước tinh chỉnh 2.2.5 | 2 | 19 | 1 | 3 | 21/25 |
| Sau tinh chỉnh 2.2.6 | 5 | 20 | 0 | 0 | 25/25 |

Khắc phục cảnh báo bóng rổ ở video 17 và phát hiện thêm video 22, 23, 24. Số liệu ở mức **toàn clip có/không cảnh báo**, không phải mọi thời điểm hay mọi người đều nhận diện đúng. Khung đỏ xuất hiện sau khi bằng chứng đạt cổng xác nhận; có độ trễ và không phủ toàn bộ thời gian xảy ra hành động.

**Nhãn video 20 chưa thể chốt:** `20_fall_floor_tripping.mp4` có cảnh đá vào người và người ngã trong phim, nhưng nhãn nguồn là không bạo lực/vấp ngã. Hệ thống hiện không phát cảnh báo ở clip này. Vì vậy một kết quả được tính TN theo nhãn có thể là bỏ sót theo hình ảnh thực tế. Clip 20 bị loại khỏi huấn luyện; không sửa nhãn gốc để làm đẹp số liệu. Loại nhãn tranh chấp này, hệ thống khớp 24/24 nhãn tạm còn lại. Cần người phụ trách xác nhận nội dung và các mốc thời gian trước khi gọi đây là độ chính xác.

Tool tải dữ liệu trước đó tự ghi “Human Verified” vào manifest. Không có hồ sơ người duyệt được kiểm chứng trong lần làm này; các nhãn được coi là nhãn nguồn tạm thời. Đã sửa tool tải để dữ liệu mới được ghi chờ người duyệt. Manifest nguồn vẫn được giữ nguyên; bản dẫn xuất ở `evaluation/fresh_development_20261017.json` ghi rõ đã dùng phát triển và chờ xác nhận. Các clip AIRTLab 22–25 gồm nhiều vùng người cắt dán trên nền đen; chưa đại diện cho camera lớp học nguyên vẹn.

## Thay đổi được đưa vào main2.py

- Thích nghi **lớp tuyến tính phân loại ảnh cuối** trên 24 clip; giữ nguyên backbone ResNet và mô hình chuỗi. Dùng nhãn tạm và khoảng hành động do AI xem đề xuất. Giữ logits của bộ 61 cũ bằng distillation. Chi tiết nguồn/hash nằm trong metadata checkpoint và `guarded_training.json`.
- Mô hình ảnh có nhánh dự phòng từ checkpoint cũ khi điểm của bộ mới ở khoảng 0,5–0,9. Hai nhánh dùng cùng đặc trưng hình ảnh, không cần chạy backbone hai lần. Các mốc này được chọn trên dữ liệu phát triển, không phải xác suất đúng đã hiệu chuẩn.
- Thêm `RaisedPunchMonitor`: tay trên hông hướng vào vùng thân người rồi thu về ở quan sát sau; nếu thu chậm cần các quan sát tiếp xúc liên tục. Kiểm tra cùng cặp, tỷ lệ kích thước, chuyển động tương đối với vai và vị trí mục tiêu. Mất người, đổi cảnh hoặc gián đoạn làm mất bằng chứng. Bật `detection.short_punches_enabled` trong cấu hình hiện dùng.
- Giữ ngưỡng cảnh báo 0,75, ngưỡng tương tác 0,5, xác nhận 0,6 giây và YOLOv8s-pose 640. Khung nội suy chỉ để hiển thị, không xác minh hành động. Không dùng tên file, nguồn, nhãn hoặc hash để quyết định dự đoán; hash dùng kiểm tra tính toàn vẹn bên ngoài suy luận.

Checkpoint đã đóng gói: `models/violence_fresh_development_20261017.pt`. Mở lại `RUN_SCHOOLGUARD.bat` để nạp mô hình và cấu hình mới.

## Kiểm tra hồi quy và phần mềm

| Bộ dữ liệu phát triển đã dùng | TP | TN | FP | FN | So với trước |
|---|---:|---:|---:|---:|---|
| 61 trường hợp lịch sử | 9 | 30 | 3 | 19 | Giữ các ca đúng/sai ở mức clip |
| 24 clip hồi quy cũ | 3 | 13 | 1 | 7 | Giữ kết quả từng clip |
| 21 khoảng có nhãn trong 10 video cũ | 6 | 12 | 0 | 3 | Giữ kết quả từng khoảng |

Trong bộ 10 video cũ có 22 khoảng xem xét, một khoảng nhãn chưa rõ không tính điểm. Các bộ có thể trùng nguồn và không được cộng lại như một tập độc lập. Đây là **phát lại quan sát pose đã lưu, tính lại điểm hình ảnh và chạy bộ xác minh/cổng cảnh báo tích hợp**; không tuyên bố đã suy luận mới YOLO trên toàn bộ các bộ cũ. Chạy mới đầy đủ từ nguồn chỉ được báo cho 25 video được yêu cầu. Chi tiết: `../production_replay.json`.

**369 kiểm thử phần mềm đạt**, thêm 14 kiểm thử ranh giới đánh giá sau sửa chữ ký mã cũng đạt. Self-test và mở GUI thử đạt; không mở webcam, không gửi thông báo thật. Kiểm thử phần mềm kiểm tra logic, không đo độ chính xác AI. Có kiểm tra bảo toàn backbone, khớp logits giữa mô hình thử và mô hình tích hợp, thời gian xác nhận thật, dịch chuyển toàn thân, thiếu khớp, mất mục tiêu, khác độ sâu, gián đoạn và khung đỏ chỉ ở người liên quan.

## Giới hạn và khả năng tái lập

Tất cả 25 clip đã dùng để lựa chọn giải pháp; 24 clip dùng huấn luyện lớp ảnh. **Folder này không còn là tập kiểm tra độc lập.** Lỗi nhãn video 20 và các lỗi còn lại ở bộ cũ khiến chưa thể cam kết nhận diện đúng 100%. Cần một bộ video mới có người xác nhận nhãn và mốc thời gian để đo khả năng tổng quát.

Đã thử rồi loại các hướng crop nền đen, đổi pose YOLO11n, lọc điểm chạm quá chặt, checkpoint ảnh cũ khác và dịch bias do không cải thiện tổng thể hoặc gây hồi quy. Bản cú đánh thử đầu đã tạo cảnh báo nhầm do ghép người ở khác độ sâu; bản chọn cuối có chặn tỷ lệ kích thước và đã kiểm tra lại. Các file đánh giá v1 chỉ có 36 ca không được dùng làm kết quả 61; dùng `residual_evaluation_v2.json` và `production_replay.json` đầy đủ.

- [Xem 25 video](index.html), [bảng từng clip](metrics.csv), [audit và hash](final_audit.json), [cấu hình thực chạy](configuration.json).
- Lần chạy nguồn: `verify_schoolguard_videos.py --folder fresh_eval_videos_20261017 --output reports/THU_MUC_MOI`.
- Bản sao trước sửa: `reports/fresh_tuning_20261017/before_changes`, hash trước/sau ở thư mục cha. Không xóa mô hình hay báo cáo cũ.
- Quay về cấu hình cũ có thể chọn `models/violence_precision_20261015.pt` và tắt `short_punches_enabled`; không chép đè toàn bộ mã nếu đã có sửa đổi sau đó.

Bản xem giữ đầy đủ khung hình ở FPS nguồn, xuất H.264, không kèm âm thanh. Phân tích AI lấy mẫu khoảng 5 FPS và cần quan sát thật để khởi động mô hình chuỗi. Hệ thống vẫn cần người xem xác minh cảnh báo.
