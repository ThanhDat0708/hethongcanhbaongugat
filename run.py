from pathlib import Path

from drowsiness.gui import run_gui


if __name__ == "__main__":
    run_gui(Path(__file__).resolve().parent / "config.yaml")
