from dataclasses import dataclass


@dataclass(frozen=True)
class MonitorSnapshot:
    eye: str
    mouth: str
    eye_closed_for: float
    mouth_open_for: float
    face_missing_for: float
    calibrated: bool
    alert_reasons: tuple[str, ...]
    alert_count: int
    yawn_count: int
    head_drop_count: int
    events: tuple[str, ...]

    @property
    def alert(self) -> bool:
        return bool(self.alert_reasons)


class DrowsinessMonitor:
    def __init__(
        self,
        eye_closed_threshold: float = 2.0,
        yawn_threshold: float = 1.5,
        face_missing_threshold: float = 1.5,
        eye_aspect_threshold: float = 0.21,
        mouth_aspect_threshold: float = 0.5,
    ) -> None:
        self.eye_closed_threshold = eye_closed_threshold
        self.yawn_threshold = yawn_threshold
        self.face_missing_threshold = face_missing_threshold
        self.eye_aspect_threshold = eye_aspect_threshold
        self.mouth_aspect_threshold = mouth_aspect_threshold
        self.alert_count = 0
        self.yawn_count = 0
        self.head_drop_count = 0
        self._calibration_started: float | None = None
        self._calibrated = False
        self._seen_face = False
        self._eye_closed_since: float | None = None
        self._mouth_open_since: float | None = None
        self._face_missing_since: float | None = None
        self._yawn_counted = False
        self._eye_alert_active = False
        self._face_alert_active = False
        self._alert_active = False

    def calibrate(self, now: float) -> None:
        self._calibration_started = now
        self._calibrated = False
        self._seen_face = False
        self._eye_closed_since = None
        self._mouth_open_since = None
        self._face_missing_since = None
        self._yawn_counted = False
        self._eye_alert_active = False
        self._face_alert_active = False
        self._alert_active = False

    def update(
        self,
        eye: str | None,
        mouth: str | None,
        face_present: bool,
        now: float,
    ) -> MonitorSnapshot:
        events: list[str] = []
        if face_present:
            self._seen_face = True

        if self._calibration_started is None:
            self.calibrate(now)
        if (
            not self._calibrated
            and self._seen_face
            and now - self._calibration_started >= 1.0
        ):
            self._calibrated = True
            events.append("HIỆU CHỈNH HOÀN TẤT - BẮT ĐẦU THEO DÕI")

        eye = eye if eye in {"OPEN", "CLOSED"} else "UNKNOWN"
        mouth = mouth if mouth in {"OPEN", "CLOSED"} else "UNKNOWN"
        if eye == "CLOSED":
            if self._eye_closed_since is None:
                self._eye_closed_since = now
        else:
            self._eye_closed_since = None

        if mouth == "OPEN":
            if self._mouth_open_since is None:
                self._mouth_open_since = now
                self._yawn_counted = False
        else:
            self._mouth_open_since = None
            self._yawn_counted = False

        if (
            self._calibrated
            and self._seen_face
            and not face_present
        ):
            if self._face_missing_since is None:
                self._face_missing_since = now
        else:
            self._face_missing_since = None

        eye_closed_for = self._duration(self._eye_closed_since, now)
        mouth_open_for = self._duration(self._mouth_open_since, now)
        face_missing_for = self._duration(self._face_missing_since, now)

        if mouth_open_for >= self.yawn_threshold and not self._yawn_counted:
            self.yawn_count += 1
            self._yawn_counted = True
            events.append("PHÁT HIỆN NGÁP")

        eye_alert = eye_closed_for >= self.eye_closed_threshold
        face_alert = (
            self._face_missing_since is not None
            and face_missing_for >= self.face_missing_threshold
        )
        if eye_alert and not self._eye_alert_active:
            events.append("PHÁT HIỆN NGỦ GẬT - Phát cảnh báo!")
        if face_alert and not self._face_alert_active:
            self.head_drop_count += 1
            events.append("MẤT KHUÔN MẶT - NGHI GỤC ĐẦU - Phát cảnh báo!")

        active = eye_alert or face_alert
        if active and not self._alert_active:
            self.alert_count += 1
            events.append("BẮT ĐẦU PHÁT CẢNH BÁO")
        elif self._alert_active and not active:
            events.append("ĐÃ HẾT CẢNH BÁO")
        self._eye_alert_active = eye_alert
        self._face_alert_active = face_alert
        self._alert_active = active

        reasons: list[str] = []
        if eye_alert:
            reasons.append("NHẮM MẮT QUÁ LÂU")
        if face_alert:
            reasons.append("MẤT KHUÔN MẶT - NGHI GỤC ĐẦU")

        return MonitorSnapshot(
            eye=eye,
            mouth=mouth,
            eye_closed_for=eye_closed_for,
            mouth_open_for=mouth_open_for,
            face_missing_for=face_missing_for,
            calibrated=self._calibrated,
            alert_reasons=tuple(reasons),
            alert_count=self.alert_count,
            yawn_count=self.yawn_count,
            head_drop_count=self.head_drop_count,
            events=tuple(events),
        )

    @staticmethod
    def _duration(start: float | None, now: float) -> float:
        return 0.0 if start is None else max(0.0, now - start)
