from dataclasses import dataclass
from math import hypot
from pathlib import Path
import time

import cv2


@dataclass(frozen=True)
class FaceMeasurements:
    ear: float
    mar: float
    face_box: tuple[int, int, int, int]


class FaceLandmarkDetector:
    _LEFT_EYE = (33, 160, 158, 133, 153, 144)
    _RIGHT_EYE = (362, 385, 387, 263, 373, 380)

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy model landmark khuôn mặt: {path}. "
                "Hãy tải face_landmarker.task vào thư mục models."
            )
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                "Chưa cài MediaPipe. Chạy pip install -r requirements.txt."
            ) from error

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self._mp = mp
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def process(self, frame) -> FaceMeasurements | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(
            time.monotonic_ns() // 1_000_000,
            self._last_timestamp_ms + 1,
        )
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None
        return self.measure(
            result.face_landmarks[0],
            image_width=frame.shape[1],
            image_height=frame.shape[0],
        )

    def close(self) -> None:
        self._landmarker.close()

    @classmethod
    def measure(
        cls, landmarks, image_width: int = 1, image_height: int = 1
    ) -> FaceMeasurements:
        points = [
            (point.x * image_width, point.y * image_height)
            for point in landmarks
        ]
        left = min(point[0] for point in points)
        top = min(point[1] for point in points)
        right = max(point[0] for point in points)
        bottom = max(point[1] for point in points)
        padding_x = (right - left) * 0.08
        padding_y = (bottom - top) * 0.08
        face_box = (
            max(0, int(left - padding_x)),
            max(0, int(top - padding_y)),
            min(image_width, int(right + padding_x)),
            min(image_height, int(bottom + padding_y)),
        )

        def distance(first: int, second: int) -> float:
            return hypot(
                points[first][0] - points[second][0],
                points[first][1] - points[second][1],
            )

        def eye_aspect_ratio(indices: tuple[int, ...]) -> float:
            first, second, third, fourth, fifth, sixth = indices
            horizontal = distance(first, fourth)
            if horizontal == 0:
                return 0.0
            return (
                distance(second, sixth) + distance(third, fifth)
            ) / (2 * horizontal)

        ear = (
            eye_aspect_ratio(cls._LEFT_EYE)
            + eye_aspect_ratio(cls._RIGHT_EYE)
        ) / 2
        mouth_width = distance(78, 308)
        mar = (
            (
                distance(13, 14)
                + distance(82, 87)
                + distance(312, 317)
            )
            / (2 * mouth_width)
            if mouth_width
            else 0.0
        )
        return FaceMeasurements(ear=ear, mar=mar, face_box=face_box)
