# Driver Watch

Ứng dụng Tkinter + OpenCV + MediaPipe Face Landmarker theo dõi dấu hiệu buồn ngủ của tài xế.

## Cài đặt trên Windows

Mở PowerShell tại thư mục dự án. Không cần lệnh launcher `py`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Chạy giao diện Tkinter

Model landmark `models\face_landmarker.task` được dùng để tính EAR/MAR và phát hiện khuôn mặt. Sau khi cài dependencies, chạy:

```powershell
.\.venv\Scripts\python.exe .\run.py
```

Giao diện hiển thị đồng hồ cập nhật mỗi giây, EAR/MAR trực tiếp, thời gian nhắm mắt, số cảnh báo và số lần ngáp. Ba thanh trượt điều chỉnh ngưỡng EAR, thời gian nhắm mắt trước khi cảnh báo (mặc định 2 giây) và thời gian miệng mở để ghi nhận ngáp. Khi phát hiện ngủ gật, mất mặt quá ngưỡng hoặc ngáp, ảnh camera kèm khung trạng thái được lưu trong `snapshots` và đường dẫn được ghi vào `history\events.db`. Các thay đổi và sự kiện cũng được ghi vào lịch sử, nạp lại khi mở ứng dụng. Khung bám mặt có viền mảnh và bốn góc đậm: xanh khi tỉnh táo, vàng khi hiệu chỉnh, đỏ khi báo động. Khi chưa thấy mặt, khung hướng dẫn nét đứt màu vàng hiện giữa hình và chuyển đỏ khi báo động mất mặt. Trên Windows, âm báo lặp đến khi hết cảnh báo và có thể tắt tiếng trên giao diện. Camera thử các cổng 0–3 bằng DirectShow, Media Foundation và backend mặc định trên Windows; chọn webcam bằng `camera.index` trong `config.yaml`.

Nếu camera không mở được, đóng các ứng dụng khác đang dùng webcam, cho phép ứng dụng desktop truy cập camera tại **Settings > Privacy & security > Camera**, rồi thử lại. Lỗi kết nối hiện chi tiết trong giao diện và nhật ký.

**Giới hạn:** EAR/MAR được tính từ landmark khuôn mặt; ngáp được ghi nhận khi MAR vượt ngưỡng mở miệng trong đủ thời gian. Hệ thống chưa đo góc nghiêng đầu; bộ đếm gục đầu chỉ dựa trên việc mất khuôn mặt sau khi hiệu chỉnh và đã từng thấy mặt. Có thể chỉnh ngưỡng EAR và thời gian ngay trên giao diện.

Flask API cũ vẫn có thể chạy riêng bằng (API này dùng model YOLO tại `models\best.pt`):

```powershell
.\.venv\Scripts\python.exe -m drowsiness.app
```

## Test và train

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\train.py
```

Dataset dùng các thư mục lớp ảnh trong `dataset/Sub1`, `dataset/Sub2` và `dataset/Sub3`. Script huấn luyện model phân loại cho Flask API; giao diện Tkinter sử dụng MediaPipe Face Landmarker để đo EAR/MAR.
