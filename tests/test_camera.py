import pytest

cv2 = pytest.importorskip("cv2")

from drowsiness import detector


class FakeCamera:
    def __init__(self, opened, frame_available=True):
        self.opened = opened
        self.frame_available = frame_available
        self.released = False
        self.settings = []

    def isOpened(self):
        return self.opened

    def set(self, property_id, value):
        self.settings.append((property_id, value))
        return True

    def read(self):
        return self.frame_available, object() if self.frame_available else None

    def release(self):
        self.released = True


def test_open_camera_tries_windows_backends_and_returns_first_working_camera(monkeypatch):
    failed = FakeCamera(opened=False)
    working = FakeCamera(opened=True)
    attempts = []

    def video_capture(index, backend):
        attempts.append((index, backend))
        if backend == cv2.CAP_DSHOW:
            return failed
        return working

    monkeypatch.setattr(detector.sys, "platform", "win32")
    monkeypatch.setattr(cv2, "VideoCapture", video_capture)

    result = detector.open_camera(0, width=800, height=600, fps=20)

    assert result is working
    assert attempts[:2] == [(0, cv2.CAP_DSHOW), (0, cv2.CAP_MSMF)]
    assert failed.released
    assert (cv2.CAP_PROP_FRAME_WIDTH, 800) in working.settings
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 600) in working.settings


def test_open_camera_reports_when_no_candidate_returns_a_frame(monkeypatch):
    cameras = []

    def video_capture(_index, _backend):
        camera = FakeCamera(opened=True, frame_available=False)
        cameras.append(camera)
        return camera

    monkeypatch.setattr(detector.sys, "platform", "win32")
    monkeypatch.setattr(cv2, "VideoCapture", video_capture)

    with pytest.raises(RuntimeError, match="Không mở/đọc được camera"):
        detector.open_camera(0)
    assert cameras
    assert all(camera.released for camera in cameras)
