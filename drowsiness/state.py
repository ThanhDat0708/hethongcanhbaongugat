from dataclasses import dataclass
from enum import StrEnum


class Status(StrEnum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    DROWSINESS = "DROWSINESS"
    ALERT = "ALERT"


@dataclass(frozen=True)
class StateSnapshot:
    status: Status
    eye: str
    mouth: str
    eye_closed_for: float
    mouth_open_for: float
    abnormal_for: float
    alert_due: bool
    predicted_class: str
    confidence: float

    @property
    def alert(self) -> bool:
        return self.status is Status.ALERT


class DrowsinessStateMachine:
    def __init__(self, settings):
        self.settings = settings
        self._eye_closed_since = None
        self._mouth_open_since = None
        self._abnormal_since = None
        self._last_alert = None
        self._eye_closed_frames = 0
        self._mouth_open_frames = 0
        self._predicted_class = "UNKNOWN"
        self._confidence = 0.0

    def update(self, eye: str | None, mouth: str | None, now: float) -> StateSnapshot:
        return self._update(eye, mouth, now, self._predicted_class, self._confidence)

    def update_prediction(self, predicted_class: str | None, confidence: float, now: float) -> StateSnapshot:
        label = predicted_class or "UNKNOWN"
        normalized = label.lower()
        eye = "CLOSED" if normalized == "eyeclose" else "OPEN" if normalized == "neutral" else None
        mouth = "OPEN" if normalized == "yawn" else "CLOSED" if normalized == "neutral" else None
        return self._update(eye, mouth, now, label, confidence)

    def update_detection(self, eye: str | None, mouth: str | None,
                         predicted_class: str | None, confidence: float,
                         now: float) -> StateSnapshot:
        return self._update(eye, mouth, now, predicted_class or "UNKNOWN", confidence)

    def _update(self, eye: str | None, mouth: str | None, now: float,
                predicted_class: str, confidence: float) -> StateSnapshot:
        eye = eye or "UNKNOWN"
        mouth = mouth or "UNKNOWN"
        self._predicted_class = predicted_class
        self._confidence = confidence
        if eye == "CLOSED":
            self._eye_closed_frames += 1
            self._eye_closed_since = self._eye_closed_since if self._eye_closed_since is not None else now
        else:
            self._eye_closed_frames = 0
            self._eye_closed_since = None
        if mouth == "OPEN":
            self._mouth_open_frames += 1
            self._mouth_open_since = self._mouth_open_since if self._mouth_open_since is not None else now
        else:
            self._mouth_open_frames = 0
            self._mouth_open_since = None

        eye_closed_for = self._duration(self._eye_closed_since, now)
        mouth_open_for = self._duration(self._mouth_open_since, now)
        abnormal = eye_closed_for >= self.settings.eye_closed_threshold
        abnormal = abnormal or (eye_closed_for >= self.settings.eye_closed_threshold
                    and mouth_open_for >= self.settings.mouth_open_threshold)
        abnormal = abnormal and self._eye_closed_frames >= self.settings.consecutive_frames
        if abnormal and self._abnormal_since is None:
            self._abnormal_since = now
        if not abnormal:
            self._abnormal_since = None
        abnormal_for = self._duration(self._abnormal_since, now)
        if not abnormal:
            status = Status.NORMAL
        elif abnormal_for >= self.settings.alert_threshold:
            status = Status.ALERT
        elif abnormal_for >= self.settings.drowsiness_threshold:
            status = Status.DROWSINESS
        else:
            status = Status.WARNING
        alert_due = status is Status.ALERT and (
            self._last_alert is None or now - self._last_alert >= self.settings.alert_cooldown
        )
        if alert_due:
            self._last_alert = now
        return StateSnapshot(status, eye, mouth, eye_closed_for, mouth_open_for, abnormal_for,
                     alert_due, predicted_class, confidence)

    @staticmethod
    def _duration(start: float | None, now: float) -> float:
        return 0.0 if start is None else max(0.0, now - start)
