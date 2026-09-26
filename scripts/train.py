from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def validate_dataset() -> None:
    missing = []
    for split in ("train", "val"):
        image_dir = ROOT / "dataset" / "images" / split
        label_dir = ROOT / "dataset" / "labels" / split
        images = [path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        labels = {path.stem for path in label_dir.glob("*.txt")}
        if not images:
            missing.append(f"{image_dir} (no images)")
        unmatched = [path.name for path in images if path.stem not in labels]
        if unmatched:
            missing.append(f"{label_dir} (missing labels for: {', '.join(unmatched[:3])})")
    if missing:
        raise SystemExit(
            "Dataset is not ready. Add YOLO images and matching .txt labels to:\n- "
            + "\n- ".join(missing)
        )


validate_dataset()
model = YOLO("yolo11n.pt")
model.train(data=str(ROOT / "dataset" / "data.yaml"), epochs=50, imgsz=640, project=str(ROOT / "runs"), name="drowsiness")
best = ROOT / "runs" / "drowsiness" / "weights" / "best.pt"
target = ROOT / "models" / "best.pt"
if best.exists():
    target.write_bytes(best.read_bytes())
    print(f"Saved {target}")
