from types import SimpleNamespace

from drowsiness.state import DrowsinessStateMachine, Status


def machine():
    return DrowsinessStateMachine(SimpleNamespace(
        eye_closed_threshold=2.0, mouth_open_threshold=1.5,
        drowsiness_threshold=2.5, alert_threshold=5.0,
        alert_cooldown=5.0, consecutive_frames=3,
    ))


def test_blink_does_not_trigger_warning():
    state = machine()
    assert state.update("CLOSED", "CLOSED", 0).status is Status.NORMAL
    assert state.update("OPEN", "CLOSED", 0.4).status is Status.NORMAL


def test_closed_eye_needs_consecutive_frames_and_duration():
    state = machine()
    assert state.update("CLOSED", "CLOSED", 0).status is Status.NORMAL
    assert state.update("CLOSED", "CLOSED", 2.1).status is Status.NORMAL
    assert state.update("CLOSED", "CLOSED", 2.2).status is Status.WARNING
    assert state.update("CLOSED", "CLOSED", 4.8).status is Status.DROWSINESS


def test_alert_is_throttled_by_cooldown():
    state = machine()
    for now in (0, 1, 2, 3, 4, 5, 6, 7, 8):
        snapshot = state.update("CLOSED", "OPEN", now)
    assert snapshot.status is Status.ALERT
    assert snapshot.alert_due is False
    assert state.update("OPEN", "CLOSED", 9).status is Status.NORMAL


def test_mouth_alone_does_not_trigger_drowsiness():
    state = machine()
    for now in (0, 1.5, 3, 5):
        assert state.update("OPEN", "OPEN", now).status is Status.NORMAL
