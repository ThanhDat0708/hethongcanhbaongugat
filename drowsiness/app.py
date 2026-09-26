import time
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, Response, send_from_directory

from .audio import play_alert_async
from .config import load_settings
from .detector import YoloDetector, annotate_camera_focus, classify_detection, open_camera
from .state import DrowsinessStateMachine


def create_app(config_path: str | Path = "config.yaml") -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    settings = load_settings(config_path)
    machine = DrowsinessStateMachine(settings)
    detector = None
    camera = None
    latest_snapshot = None
    latest = machine.update(None, None, time.monotonic())

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        return jsonify({"status": latest.status, "predicted_class": latest.predicted_class,
                "confidence": round(latest.confidence, 3),
                "eye_closed_duration": round(latest.eye_closed_for, 2),
                "yawn_duration": round(latest.mouth_open_for, 2),
                "alert": latest.alert,
                "eye": latest.eye, "mouth": latest.mouth,
                        "eye_closed_for": round(latest.eye_closed_for, 2),
                        "mouth_open_for": round(latest.mouth_open_for, 2),
                        "abnormal_for": round(latest.abnormal_for, 2),
                        "model_ready": detector is not None,
                        "model_path": str(settings.model_path),
                        "snapshot": latest_snapshot})

    def frames():
        nonlocal detector, camera, latest, latest_snapshot
        if detector is None:
            if settings.model_path.exists():
                detector = YoloDetector(settings.model_path, settings.confidence_threshold,
                                        settings.device, settings.inference_size)
        camera = open_camera(settings.camera_index, settings.camera_width,
                             settings.camera_height, settings.camera_fps)
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                if detector is None:
                    annotated = annotate_camera_focus(frame, latest.status)
                else:
                    prediction = detector.predict(frame)
                if detector is not None and detector.task == "classify":
                    latest = machine.update_prediction(prediction.label, prediction.confidence,
                                                       time.monotonic())
                    annotated = detector.annotate_focus(frame, prediction, latest.status)
                elif detector is not None:
                    eye, mouth = classify_detection(prediction.detections)
                    latest = machine.update_detection(eye, mouth, prediction.label,
                                                      prediction.confidence, time.monotonic())
                    annotated = detector.annotate_detections(frame, prediction, latest.status)
                if latest.alert_due:
                    play_alert_async()
                    settings.snapshot_dir.mkdir(parents=True, exist_ok=True)
                    snapshot = settings.snapshot_dir / f"alert_{int(time.time())}.jpg"
                    if cv2.imwrite(str(snapshot), annotated):
                        latest_snapshot = f"/snapshots/{snapshot.name}"
                ok, encoded = cv2.imencode(".jpg", annotated)
                if ok:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n")
        finally:
            if camera is not None:
                camera.release()
                camera = None

    @app.get("/video_feed")
    def video_feed():
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/snapshots/<path:name>")
    def snapshot(name: str):
        return send_from_directory(settings.snapshot_dir, name)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
