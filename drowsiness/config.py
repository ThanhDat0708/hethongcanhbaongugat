from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    model_path: Path
    camera_index: int = 0
    confidence_threshold: float = 0.5
    device: str = "cpu"
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 15
    inference_size: int = 320
    snapshot_dir: Path = Path("snapshots")
    eye_closed_threshold: float = 2.0
    yawn_threshold: float = 1.5
    drowsiness_threshold: float = 2.5
    alert_threshold: float = 5.0
    alert_cooldown: float = 5.0
    consecutive_frames: int = 3

    @property
    def mouth_open_threshold(self) -> float:
        return self.yawn_threshold


def load_settings(path: str | Path = "config.yaml") -> Settings:
    config_path = Path(path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model = values.get("model", {})
    camera = values.get("camera", {})
    drowsiness = values.get("drowsiness", {})
    model_path = model.get("path", values.get("model_path", "models/best.pt"))
    if not Path(model_path).is_absolute():
        model_path = config_path.parent / model_path
    return Settings(
        model_path=Path(model_path),
        camera_index=values.get("camera_index", camera.get("index", 0)),
        confidence_threshold=model.get("confidence", values.get("confidence_threshold", 0.5)),
        device=model.get("device", "cpu"),
        camera_width=camera.get("width", 640),
        camera_height=camera.get("height", 480),
        camera_fps=camera.get("fps", 15),
        inference_size=model.get("imgsz", 320),
        snapshot_dir=config_path.parent / values.get("snapshot_dir", "snapshots"),
        eye_closed_threshold=drowsiness.get("eye_closed_time", values.get("eye_closed_threshold", 2.0)),
        yawn_threshold=drowsiness.get("yawn_time", values.get("mouth_open_threshold", 1.5)),
        drowsiness_threshold=drowsiness.get("drowsiness_time", values.get("drowsiness_threshold", 2.5)),
        alert_threshold=drowsiness.get("alert_time", values.get("alert_threshold", 5.0)),
        alert_cooldown=drowsiness.get("alert_cooldown", values.get("alert_cooldown", 5.0)),
        consecutive_frames=drowsiness.get("consecutive_frames", values.get("consecutive_frames", 3)),
    )
