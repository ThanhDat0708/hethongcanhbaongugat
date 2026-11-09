import base64
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from drowsiness import gui


class FakeLabel:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.configuration = None

    def winfo_width(self):
        return self.width

    def winfo_height(self):
        return self.height

    def configure(self, **kwargs):
        self.configuration = kwargs


def test_render_frame_encodes_image_in_tk_supported_png(monkeypatch):
    photo_images = []

    def create_photo_image(**kwargs):
        photo_images.append(kwargs)
        return object()

    monkeypatch.setattr(gui.tk, "PhotoImage", create_photo_image)
    window = object.__new__(gui.DrowsinessWindow)
    window.root = object()
    window.video_label = FakeLabel(324, 244)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    window._render_frame(frame, status="normal", face_box=(80, 60, 240, 300))

    photo = photo_images[0]
    assert photo["format"] == "PNG"
    encoded = base64.b64decode(photo["data"])
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (240, 320, 3)
    assert window.video_label.configuration == {"image": window._photo, "text": ""}


@pytest.mark.parametrize(
    ("status", "color"),
    [
        ("normal", (55, 190, 75)),
        ("calibrating", (0, 210, 255)),
        ("alert", (40, 40, 245)),
    ],
)
def test_face_box_uses_four_corner_lines_and_status_color(status, color):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    gui.DrowsinessWindow._draw_face_box(frame, (60, 40, 260, 210), color)

    assert tuple(frame[40, 60]) == color
    assert tuple(frame[40, 70]) == color
    assert tuple(frame[40, 250]) == color
    assert tuple(frame[200, 60]) == color
    assert tuple(frame[200, 260]) == color


@pytest.mark.parametrize("alert", [False, True])
def test_missing_face_draws_centered_dashed_guide(alert):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    color = (0, 210, 255) if not alert else (40, 40, 245)

    gui.DrowsinessWindow._draw_dashed_guide(frame, color)

    assert tuple(frame[41, 160]) == color
    assert tuple(frame[41, 168]) == (0, 0, 0)


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ("PHÁT HIỆN NGỦ GẬT - Phát cảnh báo!", True),
        ("PHÁT HIỆN NGÁP", True),
        ("MẤT KHUÔN MẶT - NGHI GỤC ĐẦU", True),
        ("HIỆU CHỈNH HOÀN TẤT", False),
        ("ĐÃ HẾT CẢNH BÁO", False),
    ],
)
def test_detection_event_filter(event, expected):
    assert gui.DrowsinessWindow._is_detection_event(event) is expected


def test_detection_snapshot_saves_annotated_image_and_logs_path(tmp_path):
    window = object.__new__(gui.DrowsinessWindow)
    window.settings = SimpleNamespace(snapshot_dir=tmp_path / "snapshots")
    logs = []
    window._append_log = logs.append
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    path = window._save_detection_snapshot(frame, "alert", (60, 40, 260, 210))

    assert path is not None
    assert path.parent == tmp_path / "snapshots"
    saved = cv2.imread(str(path))
    assert saved is not None
    assert np.allclose(saved[40, 60], (40, 40, 245), atol=10)
    assert logs == [f"ĐÃ LƯU ẢNH PHÁT HIỆN: {path}"]


def test_snapshot_write_failure_is_reported(tmp_path, monkeypatch):
    window = object.__new__(gui.DrowsinessWindow)
    window.settings = SimpleNamespace(snapshot_dir=tmp_path / "snapshots")
    logs = []
    window._append_log = logs.append
    monkeypatch.setattr(gui.cv2, "imwrite", lambda *_args: False)

    path = window._save_detection_snapshot(
        np.zeros((240, 320, 3), dtype=np.uint8), "missing", None
    )

    assert path is None
    assert logs[0].startswith("KHÔNG LƯU ĐƯỢC ẢNH PHÁT HIỆN:")


@pytest.mark.parametrize(
    ("setter", "field", "value", "message"),
    [
        ("_set_eye_threshold", "eye_closed_threshold", "3.2",
         "ĐIỀU CHỈNH NGƯỠNG NHẮM MẮT: 3.2 giây."),
        ("_set_ear_threshold", "eye_aspect_threshold", "0.3",
         "ĐIỀU CHỈNH NGƯỠNG EAR: 0.30."),
        ("_set_yawn_threshold", "yawn_threshold", "2.4",
         "ĐIỀU CHỈNH NGƯỠNG NGÁP: 2.4 giây."),
    ],
)
def test_threshold_adjustments_update_monitor_and_are_logged(
    monkeypatch, setter, field, value, message
):
    window = object.__new__(gui.DrowsinessWindow)
    window.monitor = type("Monitor", (), {})()
    logs = []
    monkeypatch.setattr(window, "_append_log", logs.append)

    getattr(window, setter)(value)

    assert getattr(window.monitor, field) == float(value)
    assert logs == [message]
