from drowsiness.monitor import DrowsinessMonitor


def test_closed_eyes_trigger_after_configured_duration_and_recover():
    monitor = DrowsinessMonitor(eye_closed_threshold=2.0)
    monitor.calibrate(0.0)

    assert not monitor.update("CLOSED", "CLOSED", True, 0.0).alert
    snapshot = monitor.update("CLOSED", "CLOSED", True, 2.0)
    assert snapshot.alert_reasons == ("NHẮM MẮT QUÁ LÂU",)
    assert snapshot.alert_count == 1

    recovered = monitor.update("OPEN", "CLOSED", True, 2.1)
    assert not recovered.alert
    assert monitor.update("CLOSED", "CLOSED", True, 2.2).alert_count == 1


def test_yawn_counted_once_per_continuous_open_mouth():
    monitor = DrowsinessMonitor(yawn_threshold=1.5)
    monitor.calibrate(0.0)

    assert monitor.update("OPEN", "OPEN", True, 0.0).yawn_count == 0
    assert monitor.update("OPEN", "OPEN", True, 1.5).yawn_count == 1
    assert monitor.update("OPEN", "OPEN", True, 2.0).yawn_count == 1
    assert monitor.update("OPEN", "CLOSED", True, 2.1).yawn_count == 1
    assert monitor.update("OPEN", "OPEN", True, 2.2).yawn_count == 1
    assert monitor.update("OPEN", "OPEN", True, 3.7).yawn_count == 2


def test_face_loss_is_gated_by_calibration_and_previous_face_detection():
    monitor = DrowsinessMonitor(face_missing_threshold=1.5)
    monitor.calibrate(0.0)

    before_first_face = monitor.update("OPEN", "CLOSED", False, 10.0)
    assert not before_first_face.alert
    assert not before_first_face.calibrated

    monitor.update("OPEN", "CLOSED", True, 10.1)
    calibrated = monitor.update("OPEN", "CLOSED", True, 11.1)
    assert calibrated.calibrated

    monitor.update("OPEN", "CLOSED", False, 11.2)
    alert = monitor.update("OPEN", "CLOSED", False, 12.7)
    assert alert.alert_reasons == ("MẤT KHUÔN MẶT - NGHI GỤC ĐẦU",)
    assert alert.head_drop_count == 1


def test_eye_and_face_alerts_are_independent_and_combined():
    monitor = DrowsinessMonitor(
        eye_closed_threshold=2.0,
        face_missing_threshold=1.5,
    )
    monitor.calibrate(0.0)
    monitor.update("CLOSED", "CLOSED", True, 0.0)
    monitor.update("CLOSED", "CLOSED", True, 1.1)
    monitor.update("CLOSED", "CLOSED", False, 1.2)

    snapshot = monitor.update("CLOSED", "CLOSED", False, 2.7)
    assert snapshot.alert_reasons == (
        "NHẮM MẮT QUÁ LÂU",
        "MẤT KHUÔN MẶT - NGHI GỤC ĐẦU",
    )
    assert snapshot.alert_count == 1
    assert snapshot.head_drop_count == 1
