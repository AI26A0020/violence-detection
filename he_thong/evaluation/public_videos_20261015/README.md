# 28 clip kiểm tra đã tải ngày 15/10/2026

Các clip dùng kiểm tra nằm trong **clips/**: 24 clip HMDB51, 1 clip tương tác không bạo lực AIRTLab và 3 clip đám đông. Tổng 86.0 MiB, 21 nhóm nguồn (một số clip cùng phim/nguồn).

- HMDB51: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
- Bản lưu cố định: https://huggingface.co/datasets/innat/HMDB51
- AIRTLab: https://huggingface.co/datasets/DanJoshua/AIRTLAB_YOLO
- Đám đông: https://huggingface.co/datasets/parsafaramarzi/Small-Crowd-and-Traffic-Video-Footages

`sources.json` ghi URL, phiên bản kho, nhóm nguồn và các nguồn lỗi. `source_videos/` giữ bản gốc của clip bổ sung, `clips/` giữ đoạn dùng kiểm tra (tối đa 8 giây cho nguồn bổ sung). AVI HMDB51 được giữ nguyên. Các dữ liệu dùng cho nghiên cứu/đánh giá cục bộ; khi sử dụng lại cần xem điều khoản của nguồn gốc, không suy giấy phép của mọi video từ nhãn của bản mirror.

`novelty_check.json` ghi kiểm tra SHA256 và hash cảm nhận của 8 khung mỗi clip so với manifest 7.072 mẫu cũ: không thấy trùng nguyên tệp hoặc ứng viên khớp từ hai khung. Phép này không chứng minh loại được mọi bản cắt/đổi mã hoặc nguồn chưa được ghi lại.

`review_01.jpg` đến `review_06.jpg` và `review_supplement.jpg` là ảnh xem nội dung trước khi chạy AI. Nhãn đề xuất có đủ đánh/đá, thể thao, ngã, tương tác vui đùa/ôm, đông người và đi lại. Những đoạn đánh/đá chủ yếu lấy từ phim; nhóm đám đông không phải mọi kiểu chen lấn. Cần người phụ trách xem video và duyệt nhãn trước khi trích kết quả chính thức.

Danh sách đã khóa: `../public28_holdout.json`. Kết quả một lượt: `../../reports/improvements_20261015/public28_evaluation/REVIEW_SUMMARY.md`.

Một nguồn AIRTLab bị hỏng (hash khớp bản mirror nhưng thiếu moov atom) đã bị loại trước suy luận, được chuyển vào `archive_unused/20261015/collection_rejected/`. Wikimedia từ chối tải media với HTTP 429; không dùng các nguồn đó trong bộ 28 clip này.
