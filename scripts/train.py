import random
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUBJECTS = tuple(ROOT / "dataset" / name for name in ("Sub1", "Sub2", "Sub3"))
STAGED_DATASET = ROOT / "runs" / "classification_dataset"
MODEL_OUTPUT = ROOT / "models" / "best.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def stage_dataset(validation_ratio: float = 0.2) -> None:
    random.seed(42)
    images_by_class: dict[str, list[Path]] = {}
    for subject in SOURCE_SUBJECTS:
        for class_dir in subject.iterdir():
            if not class_dir.is_dir():
                continue
            images_by_class.setdefault(class_dir.name, []).extend(
                path for path in class_dir.iterdir()
                if path.suffix.lower() in IMAGE_EXTENSIONS
            )

    if not images_by_class or any(not images for images in images_by_class.values()):
        raise SystemExit("Sub1/Sub2/Sub3 must contain non-empty image class folders.")

    if STAGED_DATASET.exists():
        shutil.rmtree(STAGED_DATASET)
    for class_name, images in sorted(images_by_class.items()):
        random.shuffle(images)
        split_at = max(1, int(len(images) * (1 - validation_ratio)))
        split_at = min(split_at, len(images) - 1) if len(images) > 1 else len(images)
        for split, split_images in (("train", images[:split_at]), ("val", images[split_at:])):
            target_dir = STAGED_DATASET / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(split_images):
                target = target_dir / f"{source.stem}_{index}{source.suffix.lower()}"
                shutil.copy2(source, target)

    print("Staged classification classes:", ", ".join(sorted(images_by_class)))


stage_dataset()
model = YOLO("yolo11n-cls.pt")
model.train(
    data=str(STAGED_DATASET), task="classify", epochs=5, imgsz=224,
    batch=16, device="cpu", workers=0, project=str(ROOT / "runs"),
    name="drowsiness_classify", exist_ok=True,
)

best = ROOT / "runs" / "drowsiness_classify" / "weights" / "best.pt"
if not best.exists():
    raise SystemExit(f"Training finished but {best} was not produced.")
MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(best, MODEL_OUTPUT)
print(f"Saved trained classification model to {MODEL_OUTPUT}")
