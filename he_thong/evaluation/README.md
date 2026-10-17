# Tập kiểm tra cuối gồm 20–30 video mới

Lượt mới của bản **2.2.4** ở `visual_only_20261016/holdout.json` đã khóa và chạy một lần trên 22 clip: **TP 4, TN 11, FP 3, FN 4, lỗi 0**. Nhãn nguồn/Codex còn chờ người phụ trách duyệt; ví dụ chơi/đám đông chưa xác nhận đủ đùa giỡn và chen lấn mạnh. Xem [báo cáo và giới hạn](../reports/visual_only_20261016/REPORT.md). Tập này đã dùng, không chạy lại để chọn ngưỡng hoặc coi là một phép thử mới.

Đã tải và thử 28 clip mới theo yêu cầu ở `public_videos_20261015/clips`, manifest `public28_holdout.json`. Kết quả sơ bộ: 18 đúng, 1 báo nhầm, 9 bỏ sót; nhãn đề xuất còn chờ người phụ trách duyệt. Bộ clip ngắn, có nhiều cảnh phim, không đại diện cho độ chính xác tại trường. Manifest này đã dùng một lượt, không chạy lại để chỉnh ngưỡng. Các hướng dẫn bên dưới dùng cho một tập mới tiếp theo. Từ bản 2.2.4, luồng đánh giá chỉ dùng hình ảnh; âm thanh/ASR đã bị gỡ.

Trong **Hướng dẫn**, dùng **Tạo tập kiểm tra riêng**, chọn thư mục video mới rồi lưu JSON. Mở JSON, chọn đúng 20–30 clip và điền:

- `label`: `true` nếu thực sự có xô xát, `false` nếu không.
- `category`: `fight`, `play`, `sports`, `crowd` hoặc `fall`; tập kiểm tra cần có đủ năm loại cảnh.
- `source_group`: nguồn gốc chung của các đoạn cắt từ cùng video/camera/phiên; không dùng tên học sinh.
- `reviewed`: `true` sau khi xem toàn clip; `previously_unused`: `true` chỉ khi chưa dùng clip hoặc bản cắt/đổi mã của nó để huấn luyện hay chỉnh hệ thống.
- `reviewer`: người phụ trách tập nhãn ở đầu JSON. Khai báo đầy đủ các `known_manifests` đã dùng trước đây.
- `annotation_status: "human_reviewed"`, `human_review_pending: false`: chỉ đặt sau khi người phụ trách thực sự xem và duyệt toàn bộ nhãn. Nhãn nguồn hoặc Codex xem khung hình phải giữ trạng thái tạm.
- `strata`: đánh dấu `low_motion` hoặc `occluded` trên clip đã xem có xô xát ít chuyển động hoặc bị che khuất. Cần có ví dụ dương tính cho cả hai tình huống; không đánh dấu cho đủ chỉ tiêu nếu hình không cho thấy tình huống đó.

Sau đó bấm **Khóa nhãn tập kiểm tra**, rồi **Đánh giá tập đã khóa**. Mã, cấu hình, checkpoint, video và manifest dữ liệu cũ được khóa bằng SHA256. Mỗi tập cuối chỉ chạy một lần; nếu dùng kết quả để chỉnh tiếp, cần tập cuối mới.

Lệnh tương đương:

```powershell
.venv\Scripts\python.exe -X utf8 evaluation_workflow.py init --folder E:\video_moi --output evaluation\holdout.json
.venv\Scripts\python.exe -X utf8 evaluation_workflow.py lock --manifest evaluation\holdout.json
.venv\Scripts\python.exe -X utf8 evaluation_workflow.py run --manifest evaluation\holdout.json --output reports\holdout_final
```

Báo cáo ghi số đúng, báo nhầm, bỏ sót, lỗi và từng loại cảnh. Đơn vị là video: có ít nhất một cảnh báo tự động trong video được tính là phát hiện. Số đo này chưa cho biết số lần xô xát hoặc độ trễ phát hiện. So khớp SHA256 phát hiện trùng nguyên tệp; nguồn gốc của các bản cắt/đổi mã vẫn cần người phụ trách kiểm tra.

Tạo manifest mới sẽ tự tìm danh sách nguồn cũ. Công cụ huấn luyện bảo vệ cả tập cuối đã khóa/đã dùng trong thư mục con bằng hash, nhóm nguồn và pHash khung hình. Kiểm tra gần trùng lấy mẫu không chứng minh hai clip độc lập hoàn toàn về diễn viên, phòng hay bối cảnh trường.

Chỉ để chạy đo sơ bộ trên nhãn nguồn/AI, có thể dùng `lock --allow-provisional`. Báo cáo sẽ ghi rõ nhãn tạm và tầng khó còn thiếu; tùy chọn này không biến nhãn AI thành xác nhận của thầy/cô. Sau một lượt chạy, tập đó đã được dùng dù kết quả còn chờ duyệt nhãn. Không tiếp tục chỉnh ngưỡng dựa trên kết quả rồi gọi lại là phép thử mới.
