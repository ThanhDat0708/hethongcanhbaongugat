# Driver Watch

Ứng dụng Tkinter + OpenCV + Ultralytics YOLO theo dõi dấu hiệu buồn ngủ của tài xế.

## Cài đặt trên Windows

Mở PowerShell tại thư mục dự án. Không cần lệnh launcher `py`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Chạy giao diện Tkinter

Đặt model đã train tại `models\best.pt` nếu có, sau đó chạy:

```powershell
.\.venv\Scripts\python.exe .\run.py
```

Giao diện có bố cục camera bên trái, trạng thái và thông số bên phải, nhật ký cùng thanh chỉnh ngưỡng phía dưới. Nhật ký được lưu trong `history\events.db` và nạp lại khi mở ứng dụng. Camera thử các cổng 0–3 bằng DirectShow, Media Foundation và backend mặc định trên Windows; chọn webcam phù hợp bằng `camera.index` trong `config.yaml`. Trên Windows, âm báo lặp khi có cảnh báo; có thể tắt tiếng trên giao diện.

Nếu camera không mở được, đóng các ứng dụng khác đang dùng webcam, cho phép ứng dụng desktop truy cập camera tại **Settings > Privacy & security > Camera**, rồi thử lại. Lỗi kết nối hiện chi tiết trong giao diện và nhật ký.

**Giới hạn của model hiện tại:** model phân loại chỉ trả nhãn, không cung cấp landmark khuôn mặt. Vì vậy EAR, MAR và góc nghiêng đầu hiện được đánh dấu không khả dụng; không có cảnh báo gục đầu theo góc. Bộ đếm gục đầu chỉ ghi nhận trường hợp mất khuôn mặt sau khi hiệu chỉnh và đã từng thấy mặt. Hiệu chỉnh một giây chỉ khởi động theo dõi mất mặt, không tạo mốc góc đầu. Thanh EAR/MAR bị khóa đến khi có model landmark; các ngưỡng thời gian vẫn dùng được.

Flask API cũ vẫn có thể chạy riêng bằng:

```powershell
.\.venv\Scripts\python.exe -m drowsiness.app
```

## Test và train

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\train.py
```

Dataset dùng các thư mục lớp ảnh trong `dataset/Sub1`, `dataset/Sub2` và `dataset/Sub3`. Script huấn luyện model phân loại; model này không tạo landmark hay chỉ số EAR/MAR.

Khi model phân loại có các nhãn mắt/miệng phù hợp, giao diện đọc các nhãn `eye_closed`/`eye_open`, `mouth_open`/`mouth_closed` hoặc `yawn`. Bộ đếm ngáp chỉ tăng khi miệng được nhận diện mở liên tục đủ ngưỡng đã chọn.
