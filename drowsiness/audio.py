import sys
import threading
from pathlib import Path


def play_alert() -> None:
    if sys.platform == "win32":
        import winsound
        sound = Path(__file__).resolve().parents[1] / "sounds" / "alert.wav"
        if sound.exists():
            winsound.PlaySound(str(sound), winsound.SND_FILENAME)
        else:
            winsound.Beep(1100, 700)
    else:
        print("ALERT: driver drowsiness detected", flush=True)


def play_alert_async() -> None:
    threading.Thread(target=play_alert, daemon=True).start()
