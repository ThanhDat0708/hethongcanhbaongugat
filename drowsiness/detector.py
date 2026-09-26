from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class Prediction:
    label: str | None
    confidence: float
    annotated: object
    detections: list[Detection]
    focus_box: tuple[int, int, int, int] | None = None


class YoloDetector:
    def __init__(self, model_path: str | Path, confidence: float = 0.5,
                 device: str = "cpu", inference_size: int = 320):
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.device = device
        self.inference_size = inference_size
        self.task = self.model.task
        self.names = self.model.names
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade)

    def predict(self, frame) -> Prediction:
        results = self.model.predict(
            frame, conf=self.confidence, imgsz=self.inference_size,
            device=self.device, verbose=False,
        )
        if not results:
            return Prediction(None, 0.0, frame, [])
        result = results[0]
        if self.task == "classify" or result.probs is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                        minSize=(80, 80))
            focus_box = max(faces, key=lambda face: face[2] * face[3], default=None)
            index = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            if focus_box is not None:
                x, y, width, height = (int(value) for value in focus_box)
                focus_box = (x, y, x + width, y + height)
            return Prediction(result.names[index], confidence, frame, [], focus_box)

        detections = []
        for box in result.boxes:
            coords = tuple(int(value) for value in box.xyxy[0].tolist())
            detections.append(Detection(result.names[int(box.cls[0])], float(box.conf[0]), coords))
        best = max(detections, key=lambda detection: detection.confidence, default=None)
        return Prediction(best.label if best else None, best.confidence if best else 0.0,
                          result.plot(), detections)

    @staticmethod
    def annotate_focus(frame, prediction: Prediction, status: str):
        annotated = frame.copy()
        if prediction.focus_box is None:
            return annotated
        x1, y1, x2, y2 = prediction.focus_box
        color = (0, 0, 255) if status in {"DROWSINESS", "ALERT"} else (255, 150, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = f"{prediction.label or 'UNKNOWN'} {prediction.confidence:.2f}"
        cv2.putText(annotated, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, 2, cv2.LINE_AA)
        return annotated

    @staticmethod
    def annotate_detections(frame, prediction: Prediction, status: str):
        annotated = frame.copy()
        color = (0, 0, 255) if status in {"DROWSINESS", "ALERT"} else (255, 150, 0)
        for detection in prediction.detections:
            x1, y1, x2, y2 = detection.box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{detection.label} {detection.confidence:.2f}",
                        (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2, cv2.LINE_AA)
        return annotated

    def detect(self, frame) -> tuple[object, list[Detection]]:
        prediction = self.predict(frame)
        return prediction.annotated, prediction.detections


def classify_detection(detections: list[Detection]) -> tuple[str | None, str | None]:
    eye = next((d.label for d in detections if d.label in {"eye_open", "eye_closed"}), None)
    mouth = next((d.label for d in detections if d.label in {"mouth_open", "mouth_closed"}), None)
    return ("OPEN" if eye == "eye_open" else "CLOSED" if eye == "eye_closed" else None,
            "OPEN" if mouth == "mouth_open" else "CLOSED" if mouth == "mouth_closed" else None)


def open_camera(index: int, width: int = 640, height: int = 480, fps: int = 15):
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Cannot open camera index {index}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    return camera


def annotate_camera_focus(frame, status: str = "NORMAL"):
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    annotated = frame.copy()
    color = (0, 0, 255) if status in {"DROWSINESS", "ALERT"} else (255, 150, 0)
    for x, y, width, height in faces:
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 3)
    return annotated
