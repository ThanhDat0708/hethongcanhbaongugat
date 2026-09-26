# Driver Watch

He thong Flask + OpenCV + Ultralytics YOLO nhan dien dau hieu tai xe ngu gat.

## Cai dat

Dung Python 3.14 trong virtual environment duoc yeu cau:

```powershell
& E:\study_python\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Chay

Dat model da train tai `models/best.pt`, sau do:

```powershell
& E:\study_python\.venv\Scripts\python.exe run.py
```

Mo `http://127.0.0.1:5000`. Neu chua co model, giao dien van chay de kiem tra API va hien trang thai MODEL OFFLINE.

## Test va train

```powershell
& E:\study_python\.venv\Scripts\python.exe -m pytest -q
& E:\study_python\.venv\Scripts\python.exe scripts/train.py
```

Dataset dung format YOLO trong `dataset/`, voi cac class `face`, `eye_open`, `eye_closed`, `mouth_open`, `mouth_closed`. Truoc khi train, moi anh phai co mot file label cung ten trong `labels/train` va `labels/val`.

Moi dong label co format:

```text
class_id center_x center_y width height
```

Tat ca toa do phai duoc chuan hoa trong khoang `0..1`. Script se kiem tra anh va label truoc khi tai model; neu dataset rong, no se bao dung thu muc can bo sung thay vi in traceback cua Ultralytics.
