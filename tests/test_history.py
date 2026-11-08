from drowsiness.history import EventHistory, HistoryEntry


def test_history_is_persistent_ordered_and_counted(tmp_path):
    path = tmp_path / "history" / "events.db"
    history = EventHistory(path)
    history.record(HistoryEntry("08/11/2026 10:00:00", "alert", "CẢNH BÁO"))
    history.record(HistoryEntry("08/11/2026 10:00:01", "yawn", "PHÁT HIỆN NGÁP"))

    reopened = EventHistory(path)
    assert reopened.recent() == [
        HistoryEntry("08/11/2026 10:00:00", "alert", "CẢNH BÁO"),
        HistoryEntry("08/11/2026 10:00:01", "yawn", "PHÁT HIỆN NGÁP"),
    ]
    assert reopened.count("alert") == 1
    assert reopened.count("yawn") == 1
