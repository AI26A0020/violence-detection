# Dữ liệu cải thiện nhận diện ngày 15/10/2026

`development.json` có 703 video: 483 được phân vào train, 220 validation theo nhóm nguồn tách biệt. Có 11 video quá ngắn để tạo cửa sổ chuỗi, nên huấn luyện thực tế dùng 476 train và 216 validation. Phần dữ liệu công khai mới gồm thể thao/tập luyện, tương tác bình thường, hỗ trợ người ngã và xô xát diễn lại. Bộ 25 clip kiểm tra trước đã được tiêu thụ; ở vòng này nó là dữ liệu phát triển, không còn là kiểm tra riêng.

`holdout.json` chứa 24 clip được chọn trước khi xem điểm mô hình: 10 xô xát, 6 tương tác bình thường/đùa giỡn, 5 thể thao, 1 đám đông, 2 hỗ trợ người ngã. Đã khóa mã, mô hình, cấu hình và nhãn, rồi chạy một lượt; xem `holdout.json.used.json`. Kết quả: **TP 3, TN 13, FP 1, FN 7**. Một clip âm tính phát sinh hai cảnh báo. Không dùng kết quả này để tiếp tục chỉnh mô hình.

`sources.json`, `supplements.json`, `help_sources.json`, `crowd_source.json` ghi URL, phiên bản nguồn khi có, SHA256 và khoảng cắt. Hai clip hỗ trợ và clip Shibuya được giải mã tuần tự, dựng lại timestamp từ 0. Clip crowd05 bị loại trước suy luận do có dấu hiệu trùng nguồn với SCVD; đã thay bằng Shibuya. `novelty.json` kiểm tra hash, nhóm nguồn và tám pHash mỗi video so với dữ liệu đã dùng; 24 clip cuối không có cờ trùng trong phép kiểm tra mẫu này.

Nhãn hiện là nhãn nguồn kết hợp Codex xem ảnh theo thời gian, còn chờ người phụ trách duyệt. AIRTLab chia sẻ phòng và diễn viên giữa các cảnh, nên phép thử không chứng minh độ chính xác tại một trường mới. Kiểm tra pHash lấy mẫu cũng không loại trừ được mọi bản cắt/đổi mã từ nguồn khác. Phép thử dùng cảnh báo hình ảnh, pose, chuyển động, tương tác và kịch bản, chế độ cố định standard; tắt audio/ASR đồng nhất vì nhiều nguồn không có tiếng.

Kết quả chi tiết: [báo cáo](../../reports/precision_20261015/REPORT.md), [từng video](../../reports/precision_20261015/holdout24/results.json).
