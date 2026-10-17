# Dữ liệu phát triển và phép thử 25 clip mới

`sources.json` ghi 140 clip từ bộ dữ liệu công khai và hai đoạn hướng dẫn đỡ ngã dành riêng cho phép thử. `help_development.json` ghi hai nguồn đỡ ngã khác dùng phát triển. Mỗi bản ghi có nguồn, nhóm cảnh và hash. Clip mirror AIRTLab bị cắt méo được loại; nơi lưu kho có trong `excluded_transformed_mirror`.

`holdout.json` chứa 25 clip tách theo nguồn/cảnh: 10 fight, 6 play, 5 sports, 2 crowd, 2 fall. Nhãn nguồn kết hợp xem các mốc hình theo thời gian; **chưa được người phụ trách xác nhận toàn clip**. Dữ liệu có chung diễn viên/phòng ở AIRTLab nên không được coi là đánh giá camera tại một trường mới.

Tập được khóa trước suy luận, kèm hash mã, trọng số, cấu hình và các manifest đã dùng. `holdout.json.used.json` đánh dấu đã bắt đầu sử dụng; không xóa khóa hoặc marker, không dùng kết quả để chỉnh rồi gọi lại là tập chưa dùng. `prepare_upgrade_holdout.py` từ chối ghi đè tập đã khóa.

`holdout_novelty.json` ghi SHA256 và so sánh pHash tám mốc hình với dữ liệu cũ, bộ 28 clip đã dùng và tập phát triển mới. Không có cờ trùng theo phương pháp này; điều đó không loại trừ mọi dẫn xuất video hoặc cùng cảnh ở góc quay khác. Manifest nhóm nguồn dùng để kiểm soát thêm sự trùng cảnh.

Kết quả một lượt tại `reports/upgrade_20261015/holdout25/`; báo cáo tổng tại `reports/upgrade_20261015/REPORT.md`. Các montage chỉ dùng xem nhãn, không chứa kết quả mô hình để chọn clip theo điểm AI.

Giữ video cục bộ cho nghiên cứu/đánh giá theo điều kiện nguồn. Không tự xuất bản lại tài liệu/video của bên khác. URL và revision chính xác được lưu cho từng clip.
