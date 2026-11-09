import base64
import queue
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import cv2

from .config import load_settings
from .detector import open_camera
from .history import EventHistory, HistoryEntry
from .landmarks import FaceLandmarkDetector
from .monitor import DrowsinessMonitor, MonitorSnapshot


@dataclass(frozen=True)
class CameraFrame:
    image: object
    eye: str | None
    mouth: str | None
    face_present: bool
    ear: float | None
    mar: float | None
    face_box: tuple[int, int, int, int] | None


class AlertAudio:
    def __init__(self) -> None:
        self._active = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._failed = False

    def set_active(self, active: bool) -> str | None:
        if sys.platform != "win32":
            return None
        if active and not self._failed:
            self._active.set()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._beep_loop, name="driver-watch-alert", daemon=True
                )
                self._thread.start()
        else:
            self._active.clear()
        if self._error is not None:
            error, self._error = self._error, None
            return f"Không phát được âm thanh cảnh báo: {error}"
        return None

    def _beep_loop(self) -> None:
        import winsound

        while self._active.is_set():
            try:
                winsound.Beep(1100, 350)
            except RuntimeError as error:
                self._error = str(error)
                self._failed = True
                self._active.clear()
                return
            self._active.wait(0.08)


class DrowsinessWindow:
    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        self.settings = load_settings(config_path)
        self.monitor = DrowsinessMonitor(
            eye_closed_threshold=self.settings.eye_closed_threshold,
            yawn_threshold=self.settings.yawn_threshold,
        )
        self.history = EventHistory(self.settings.history_path)
        self.monitor.alert_count = self.history.count("alert")
        self.monitor.yawn_count = self.history.count("yawn")
        self.monitor.head_drop_count = self.history.count("head_drop")
        self.root = tk.Tk()
        self.root.title("Driver Watch | Giám sát tài xế")
        self.root.geometry("1280x940")
        self.root.minsize(980, 820)
        self.root.configure(bg="#10191f")
        self._frames: queue.Queue[CameraFrame] = queue.Queue(maxsize=1)
        self._messages: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._stop_camera = threading.Event()
        self._camera_thread: threading.Thread | None = None
        self._photo: tk.PhotoImage | None = None
        self._latest: MonitorSnapshot | None = None
        self._audio = AlertAudio()
        self._muted = tk.BooleanVar(value=False)
        self._running = False
        self._blink = False
        self._closed = False
        self._build_ui()
        self._restore_history()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._update_clock()
        self._poll_camera()

    def run(self) -> None:
        self.root.mainloop()

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "TButton", font=("Segoe UI", 10, "bold"), padding=(10, 7),
            background="#263943", foreground="#e7f0ed",
        )
        style.map("TButton", background=[("active", "#35525a")])
        style.configure(
            "TCheckbutton",
            background="#192831",
            foreground="#e7f0ed",
            font=("Segoe UI", 9),
        )

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0, minsize=240)

        header = tk.Frame(self.root, bg="#10191f", padx=20, pady=12)
        header.grid(row=0, column=0, sticky="ew")
        tk.Label(
            header,
            text="DRIVER WATCH\nGIÁM SÁT TỈNH TÁO",
            bg="#10191f",
            fg="#e7f0ed",
            font=("Segoe UI", 17, "bold"),
            justify="left",
        ).pack(side="left")
        self.clock_label = tk.Label(
            header,
            bg="#10191f",
            fg="#edf4f1",
            font=("Segoe UI", 11),
            justify="right",
        )
        self.clock_label.pack(side="right")

        content = tk.Frame(self.root, bg="#10191f", padx=16, pady=4)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=11)
        content.columnconfigure(1, weight=10)
        content.rowconfigure(0, weight=1)
        video_panel = tk.Frame(content, bg="#192831", padx=12, pady=10)
        video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        video_panel.rowconfigure(1, weight=1)
        video_panel.columnconfigure(0, weight=1)
        video_header = tk.Frame(video_panel, bg="#192831")
        video_header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(
            video_header,
            text="CAMERA TRỰC TIẾP",
            bg="#192831",
            fg="#e7f0ed",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left")
        self.camera_label = tk.Label(
            video_header,
            text="CHƯA KẾT NỐI",
            bg="#192831",
            fg="#f0bd69",
            font=("Segoe UI", 9, "bold"),
        )
        self.camera_label.pack(side="right")
        self.model_label = tk.Label(
            video_panel,
            text="MODEL LANDMARK: CHƯA TẢI",
            bg="#192831",
            fg="#f0bd69",
            font=("Segoe UI", 8, "bold"),
        )
        self.model_label.grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.video_label = tk.Label(
            video_panel,
            text="Nhấn BẮT ĐẦU để kết nối camera",
            bg="#080e12",
            fg="#91a5a0",
            font=("Segoe UI", 12),
            highlightthickness=2,
            highlightbackground="#263943",
        )
        self.video_label.grid(row=1, column=0, sticky="nsew")

        right = tk.Frame(content, bg="#10191f")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        status_panel = tk.Frame(right, bg="#192831", padx=14, pady=12)
        status_panel.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(
            status_panel, text="TRẠNG THÁI HỆ THỐNG",
            bg="#192831", fg="#9bb0b5", font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w")
        self.banner = tk.Label(
            status_panel,
            text="BẤM BẮT ĐẦU ĐỂ KẾT NỐI CAMERA",
            bg="#192831",
            fg="#42d7b1",
            font=("Segoe UI", 12, "bold"),
            anchor="w",
            wraplength=500,
            justify="left",
        )
        self.banner.pack(fill="x", pady=(6, 10))
        button_row = tk.Frame(status_panel, bg="#192831")
        button_row.pack(fill="x")
        self.start_button = ttk.Button(
            button_row, text="BẮT ĐẦU", command=self.start
        )
        self.start_button.pack(side="left", fill="x", expand=True)
        self.stop_button = ttk.Button(
            button_row, text="DỪNG", command=self.stop, state="disabled"
        )
        self.stop_button.pack(side="left", padx=(7, 0))
        ttk.Button(
            button_row,
            text="⟳ HIỆU CHỈNH",
            command=self.recalibrate,
        ).pack(side="left", padx=(7, 0))
        controls_row = tk.Frame(status_panel, bg="#192831")
        controls_row.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(
            controls_row, text="Tắt tiếng cảnh báo", variable=self._muted,
            command=self._toggle_mute,
        ).pack(side="left")
        ttk.Button(
            controls_row, text="Thoát", command=self.close
        ).pack(side="right")

        metrics_panel = tk.Frame(right, bg="#10191f")
        metrics_panel.grid(row=1, column=0, sticky="nsew")
        metrics_panel.columnconfigure(0, weight=1)
        metrics_panel.columnconfigure(1, weight=1)
        metric_items = (
            ("EAR - ĐỘ MỞ MẮT", "--"),
            ("MAR - ĐỘ MỞ MIỆNG", "--"),
            ("NHẮM MẮT LIÊN TỤC", "0.00 s"),
            ("SỐ LẦN CẢNH BÁO", "0"),
            ("SỐ LẦN NGÁP", "0"),
            ("SỐ LẦN GỤC ĐẦU*", "0"),
            ("MẤT MẶT", "0.00 s"),
            ("HIỆU CHỈNH", "Chưa thực hiện"),
        )
        self.metric_values: dict[str, tk.Label] = {}
        for index, (title, value) in enumerate(metric_items):
            card = tk.Frame(metrics_panel, bg="#192831", padx=12, pady=8)
            card.grid(
                row=index // 2, column=index % 2, sticky="nsew",
                padx=(0, 4) if index % 2 == 0 else (4, 0),
                pady=4,
            )
            metrics_panel.rowconfigure(index // 2, weight=1)
            tk.Label(
                card, text=title, bg="#192831", fg="#9bb0b5",
                font=("Segoe UI", 8, "bold"), anchor="w",
            ).pack(fill="x")
            value_label = tk.Label(
                card, text=value, bg="#192831", fg="#e7f0ed",
                font=("Segoe UI", 16, "bold"), anchor="w",
            )
            value_label.pack(fill="x", pady=(6, 0))
            self.metric_values[title] = value_label
        tk.Label(
            metrics_panel,
            text="* Bộ đếm gục đầu hiện dựa trên mất mặt; góc đầu chưa được đo.",
            bg="#10191f", fg="#9bb0b5", font=("Segoe UI", 8),
            anchor="w", justify="left",
        ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(3, 0))

        lower = tk.Frame(self.root, bg="#10191f", padx=16, pady=8)
        lower.grid(row=2, column=0, sticky="nsew")
        lower.columnconfigure(0, weight=11)
        lower.columnconfigure(1, weight=8)
        lower.rowconfigure(0, weight=1)
        log_panel = tk.Frame(lower, bg="#192831", padx=12, pady=9)
        log_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        log_panel.rowconfigure(1, weight=1)
        log_panel.columnconfigure(0, weight=1)
        tk.Label(
            log_panel, text="NHẬT KÝ SỰ KIỆN (LƯU LỊCH SỬ)",
            bg="#192831", fg="#e7f0ed", font=("Segoe UI", 9, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.log = tk.Text(
            log_panel,
            height=7,
            bg="#10191f",
            fg="#cbd8d4",
            insertbackground="#cbd8d4",
            font=("Consolas", 8),
            relief="flat",
            wrap="word",
        )
        scroll = ttk.Scrollbar(log_panel, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.grid(row=1, column=0, sticky="nsew")
        scroll.grid(row=1, column=1, sticky="ns")
        self.log.configure(state="disabled")

        threshold_panel = tk.Frame(lower, bg="#192831", padx=12, pady=9)
        threshold_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        tk.Label(
            threshold_panel, text="ĐIỀU CHỈNH NGƯỠNG",
            bg="#192831", fg="#e7f0ed", font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 4))
        slider_grid = tk.Frame(threshold_panel, bg="#192831")
        slider_grid.pack(fill="both", expand=True)
        self._add_slider(
            slider_grid, "Ngưỡng EAR (độ mở mắt)", 0.1, 0.4,
            self.monitor.eye_aspect_threshold, self._set_ear_threshold,
            row=0, column=0, resolution=0.01,
        )
        self._add_slider(
            slider_grid, "Nhắm mắt → cảnh báo (giây)", 0.5, 5.0,
            self.settings.eye_closed_threshold, self._set_eye_threshold,
            row=0, column=1,
        )
        self._add_slider(
            slider_grid, "Ngáp → ghi nhận (giây)", 0.5, 5.0,
            self.settings.yawn_threshold, self._set_yawn_threshold,
            row=1, column=0,
        )
        for column in range(2):
            slider_grid.columnconfigure(column, weight=1, uniform="threshold")

    @staticmethod
    def _add_slider(
        parent: tk.Widget,
        label: str,
        minimum: float,
        maximum: float,
        initial: float,
        command=None,
        row: int = 0,
        column: int = 0,
        resolution: float = 0.1,
    ) -> None:
        frame = tk.Frame(parent, bg="#172326")
        frame.grid(row=row, column=column, sticky="ew", padx=3, pady=1)
        tk.Label(
            frame, text=label, bg="#172326", fg="#cbd8d4",
            font=("Segoe UI", 8),
        ).pack(anchor="w")
        scale = tk.Scale(
            frame,
            from_=minimum,
            to=maximum,
            resolution=resolution,
            orient="horizontal",
            showvalue=True,
            bg="#172326",
            fg="#cbd8d4",
            troughcolor="#34484b",
            highlightthickness=0,
            activebackground="#72d5b2",
            font=("Segoe UI", 8),
        )
        scale.set(initial)
        if command is not None:
            scale.configure(command=command)
        scale.pack(fill="x")

    def _set_eye_threshold(self, value: str) -> None:
        self.monitor.eye_closed_threshold = float(value)
        self._append_log(f"ĐIỀU CHỈNH NGƯỠNG NHẮM MẮT: {float(value):.1f} giây.")

    def _set_ear_threshold(self, value: str) -> None:
        self.monitor.eye_aspect_threshold = float(value)
        self._append_log(f"ĐIỀU CHỈNH NGƯỠNG EAR: {float(value):.2f}.")

    def _set_yawn_threshold(self, value: str) -> None:
        self.monitor.yawn_threshold = float(value)
        self._append_log(f"ĐIỀU CHỈNH NGƯỠNG NGÁP: {float(value):.1f} giây.")

    def _toggle_mute(self) -> None:
        self._refresh_audio()
        message = (
            "ĐÃ TẮT TIẾNG CẢNH BÁO."
            if self._muted.get()
            else "ĐÃ BẬT ÂM THANH CẢNH BÁO."
        )
        self._append_log(message)

    def _update_clock(self) -> None:
        now = datetime.now()
        weekdays = (
            "Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu",
            "Thứ Bảy", "Chủ Nhật",
        )
        self.clock_label.configure(
            text=f"{now:%H:%M:%S}\n{weekdays[now.weekday()]}, ngày "
            f"{now.day} tháng {now.month} năm {now.year}"
        )
        if not self._closed:
            self.root.after(1000, self._update_clock)

    def start(self) -> None:
        if self._running:
            return
        if self._camera_thread is not None and self._camera_thread.is_alive():
            self._append_log("CAMERA ĐANG ĐÓNG - Vui lòng đợi rồi thử lại.")
            return
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break
        self._running = True
        self.monitor.calibrate(time.monotonic())
        self._stop_camera.clear()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.camera_label.configure(text="ĐANG KẾT NỐI...", fg="#f0bd69")
        self.model_label.configure(text="MODEL: ĐANG KHỞI TẠO", fg="#f0bd69")
        self.banner.configure(
            text="ĐANG KẾT NỐI CAMERA...",
            bg="#192831",
            fg="#42d7b1",
        )
        self._append_log("BẮT ĐẦU GIÁM SÁT - Đang dò camera và hiệu chỉnh.")
        self._camera_thread = threading.Thread(
            target=self._camera_worker, name="driver-watch-camera", daemon=True
        )
        self._camera_thread.start()

    def stop(self) -> None:
        was_running = self._running
        self._running = False
        self._stop_camera.set()
        self._audio.set_active(False)
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.banner.configure(
            text="ĐÃ DỪNG GIÁM SÁT",
            bg="#192831",
            fg="#d4dfdc",
        )
        self.video_label.configure(
            image="",
            text="Nhấn BẮT ĐẦU để kết nối camera",
            highlightbackground="#263943",
        )
        self.camera_label.configure(text="ĐÃ NGẮT KẾT NỐI", fg="#9bb0b5")
        self._photo = None
        if was_running:
            self._append_log("ĐÃ DỪNG GIÁM SÁT.")

    def recalibrate(self) -> None:
        if not self._running:
            self._append_log("CHƯA HIỆU CHỈNH - Hãy bắt đầu camera trước.")
            return
        self.monitor.calibrate(time.monotonic())
        self.metric_values["HIỆU CHỈNH"].configure(text="Đang hiệu chỉnh...")
        self._append_log("HIỆU CHỈNH LẠI - Giữ tư thế bình thường trong 1 giây.")

    def _camera_worker(self) -> None:
        camera = None
        landmark_detector = None
        try:
            camera = open_camera(
                self.settings.camera_index,
                self.settings.camera_width,
                self.settings.camera_height,
                self.settings.camera_fps,
            )
            self._messages.put("CAMERA:CONNECTED")
            landmark_detector = FaceLandmarkDetector(
                self.settings.face_landmarker_path
            )
            self._messages.put("MODEL: LANDMARK SẴN SÀNG")

            while not self._stop_camera.is_set():
                ok, frame = camera.read()
                if not ok:
                    self._messages.put("Camera không đọc được khung hình.")
                    break
                measurements = landmark_detector.process(frame)
                face_present = measurements is not None
                ear = measurements.ear if measurements else None
                mar = measurements.mar if measurements else None
                face_box = measurements.face_box if measurements else None
                eye = (
                    "CLOSED"
                    if ear is not None and ear < self.monitor.eye_aspect_threshold
                    else "OPEN" if ear is not None else None
                )
                mouth = (
                    "OPEN"
                    if mar is not None and mar >= self.monitor.mouth_aspect_threshold
                    else "CLOSED" if mar is not None else None
                )
                packet = CameraFrame(
                    image=frame,
                    eye=eye,
                    mouth=mouth,
                    face_present=face_present,
                    ear=ear,
                    mar=mar,
                    face_box=face_box,
                )
                self._put_latest_frame(packet)
        except (ImportError, OSError, RuntimeError, ValueError, cv2.error) as error:
            self._messages.put(f"Lỗi camera/nhận diện: {error}")
        finally:
            if landmark_detector is not None:
                landmark_detector.close()
            if camera is not None:
                camera.release()

    def _put_latest_frame(self, packet: CameraFrame) -> None:
        try:
            self._frames.put_nowait(packet)
            return
        except queue.Full:
            pass
        try:
            self._frames.get_nowait()
        except queue.Empty:
            return
        try:
            self._frames.put_nowait(packet)
        except queue.Full:
            pass

    def _poll_camera(self) -> None:
        if self._closed:
            return
        while not self._messages.empty():
            message = self._messages.get()
            if message == "CAMERA:CONNECTED":
                self.camera_label.configure(text="ĐÃ KẾT NỐI", fg="#42d7b1")
                self._append_log("ĐÃ KẾT NỐI CAMERA.")
            elif message.startswith("MODEL:"):
                ready = message.endswith("SẴN SÀNG")
                self.model_label.configure(
                    text=message, fg="#72d5b2" if ready else "#f0bd69"
                )
            else:
                self._append_log(message)
                if message.startswith(("Lỗi camera", "Camera không")):
                    self.model_label.configure(
                        text="MODEL LANDMARK: LỖI", fg="#ff7777"
                    )
                    self.stop()
                    self.camera_label.configure(
                        text="KHÔNG KẾT NỐI", fg="#ff7777"
                    )
                    self.banner.configure(text=message, bg="#7a2630", fg="#ffffff")

        try:
            packet = self._frames.get_nowait()
        except queue.Empty:
            packet = None
        if packet is not None and self._running:
            snapshot = self.monitor.update(
                packet.eye,
                packet.mouth,
                packet.face_present,
                time.monotonic(),
            )
            self._latest = snapshot
            for event in snapshot.events:
                self._append_log(event)
            self._render_snapshot(snapshot)
            self.metric_values["EAR - ĐỘ MỞ MẮT"].configure(
                text=f"{packet.ear:.3f}" if packet.ear is not None else "--"
            )
            self.metric_values["MAR - ĐỘ MỞ MIỆNG"].configure(
                text=f"{packet.mar:.3f}" if packet.mar is not None else "--"
            )
            if snapshot.alert:
                overlay_status = "alert"
            elif packet.face_present and not snapshot.calibrated:
                overlay_status = "calibrating"
            elif packet.face_present:
                overlay_status = "normal"
            else:
                overlay_status = "missing"
            if any(self._is_detection_event(event) for event in snapshot.events):
                self._save_detection_snapshot(
                    packet.image, overlay_status, packet.face_box
                )
            self._render_frame(packet.image, overlay_status, packet.face_box)
            audio_error = self._audio.set_active(
                snapshot.alert and not self._muted.get()
            )
            if audio_error:
                self._append_log(audio_error)
        if not self._closed:
            self.root.after(40, self._poll_camera)

    def _render_snapshot(self, snapshot: MonitorSnapshot) -> None:
        self.metric_values["NHẮM MẮT LIÊN TỤC"].configure(
            text=f"{snapshot.eye_closed_for:.2f} s"
        )
        self.metric_values["SỐ LẦN CẢNH BÁO"].configure(text=str(snapshot.alert_count))
        self.metric_values["SỐ LẦN NGÁP"].configure(text=str(snapshot.yawn_count))
        self.metric_values["SỐ LẦN GỤC ĐẦU*"].configure(
            text=str(snapshot.head_drop_count)
        )
        self.metric_values["MẤT MẶT"].configure(
            text=f"{snapshot.face_missing_for:.2f} s"
        )
        self.metric_values["HIỆU CHỈNH"].configure(
            text="Hoàn tất" if snapshot.calibrated else "Đang chờ khuôn mặt"
        )
        self._blink = not self._blink
        if snapshot.alert:
            self.banner.configure(
                text="CẢNH BÁO: BẠN ĐANG BUỒN NGỦ!",
                bg="#a62632" if self._blink else "#751b27",
                fg="#ffffff",
            )
            self.video_label.configure(
                highlightbackground="#ff364a" if self._blink else "#751b27"
            )
        elif self._running:
            text = (
                "HIỆU CHỈNH - GIỮ TƯ THẾ BÌNH THƯỜNG"
                if not snapshot.calibrated
                else "ĐANG GIÁM SÁT"
            )
            self.banner.configure(text=text, bg="#1a4d3e", fg="#e7fff5")
            self.video_label.configure(highlightbackground="#263943")

    def _render_frame(
        self,
        frame,
        status: str,
        face_box: tuple[int, int, int, int] | None = None,
    ) -> None:
        display = self._annotate_frame(frame, status, face_box)
        height, width = display.shape[:2]
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()
        if label_width > 2 and label_height > 2:
            ratio = min(
                (label_width - 4) / width,
                (label_height - 4) / height,
            )
            target_size = (
                max(1, int(width * ratio)),
                max(1, int(height * ratio)),
            )
            display = cv2.resize(
                display,
                target_size,
                interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR,
            )
        height, width = display.shape[:2]
        ok, encoded = cv2.imencode(".png", display)
        if not ok:
            raise RuntimeError("Không mã hóa được khung hình camera.")
        self._photo = tk.PhotoImage(
            master=self.root,
            data=base64.b64encode(encoded.tobytes()).decode("ascii"),
            format="PNG",
        )
        self.video_label.configure(image=self._photo, text="")

    def _annotate_frame(
        self,
        frame,
        status: str,
        face_box: tuple[int, int, int, int] | None = None,
    ):
        display = frame.copy()
        colors = {
            "normal": (55, 190, 75),
            "calibrating": (0, 210, 255),
            "missing": (0, 210, 255),
            "alert": (40, 40, 245),
        }
        color = colors[status]
        if face_box is not None and status != "missing":
            self._draw_face_box(display, face_box, color)
        else:
            guide_color = colors["alert"] if status == "alert" else colors["missing"]
            self._draw_dashed_guide(display, guide_color)
        now = datetime.now()
        timestamp = f"{now:%H:%M:%S}  {now:%d/%m/%Y}"
        cv2.putText(
            display,
            timestamp,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        return display

    @staticmethod
    def _is_detection_event(event: str) -> bool:
        return event.startswith(
            (
                "PHÁT HIỆN NGỦ GẬT",
                "PHÁT HIỆN NGÁP",
                "MẤT KHUÔN MẶT - NGHI GỤC ĐẦU",
            )
        )

    def _save_detection_snapshot(
        self,
        frame,
        status: str,
        face_box: tuple[int, int, int, int] | None,
    ) -> Path | None:
        timestamp = datetime.now()
        filename = f"detection_{timestamp:%Y%m%d_%H%M%S_%f}.jpg"
        path = self.settings.snapshot_dir / filename
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            annotated = self._annotate_frame(frame, status, face_box)
            if not cv2.imwrite(str(path), annotated):
                raise OSError(f"OpenCV không ghi được tệp {path}.")
        except (OSError, cv2.error) as error:
            self._append_log(f"KHÔNG LƯU ĐƯỢC ẢNH PHÁT HIỆN: {error}")
            return None
        self._append_log(f"ĐÃ LƯU ẢNH PHÁT HIỆN: {path}")
        return path

    @staticmethod
    def _draw_face_box(
        frame, box: tuple[int, int, int, int], color: tuple[int, int, int]
    ) -> None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width - 1, x2))
        y2 = max(y1 + 1, min(height - 1, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        corner = max(12, min(x2 - x1, y2 - y1) // 5)
        for start_x, direction_x in ((x1, 1), (x2, -1)):
            for start_y, direction_y in ((y1, 1), (y2, -1)):
                cv2.line(
                    frame,
                    (start_x, start_y),
                    (start_x + direction_x * corner, start_y),
                    color,
                    4,
                    cv2.LINE_AA,
                )
                cv2.line(
                    frame,
                    (start_x, start_y),
                    (start_x, start_y + direction_y * corner),
                    color,
                    4,
                    cv2.LINE_AA,
                )

    @staticmethod
    def _draw_dashed_guide(frame, color: tuple[int, int, int]) -> None:
        height, width = frame.shape[:2]
        box_width = max(40, min(int(width * 0.42), 320))
        box_height = max(50, min(int(height * 0.66), 400))
        x1 = (width - box_width) // 2
        y1 = (height - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height
        dash_length = 12
        gap_length = 8
        for x in range(x1, x2, dash_length + gap_length):
            end_x = min(x + dash_length, x2)
            cv2.line(frame, (x, y1), (end_x, y1), color, 2, cv2.LINE_AA)
            cv2.line(frame, (x, y2), (end_x, y2), color, 2, cv2.LINE_AA)
        for y in range(y1, y2, dash_length + gap_length):
            end_y = min(y + dash_length, y2)
            cv2.line(frame, (x1, y), (x1, end_y), color, 2, cv2.LINE_AA)
            cv2.line(frame, (x2, y), (x2, end_y), color, 2, cv2.LINE_AA)

    def _refresh_audio(self) -> None:
        if self._latest is not None:
            error = self._audio.set_active(self._latest.alert and not self._muted.get())
            if error:
                self._append_log(error)

    def _append_log(
        self,
        message: str,
        category: str | None = None,
        persist: bool = True,
        timestamp: str | None = None,
    ) -> None:
        recorded_at = timestamp or datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        if category is None:
            if message.startswith("PHÁT HIỆN NGÁP"):
                category = "yawn"
            elif message.startswith("MẤT KHUÔN MẶT"):
                category = "head_drop"
            elif message.startswith("BẮT ĐẦU PHÁT CẢNH BÁO"):
                category = "alert"
            else:
                category = "system"
        if persist:
            try:
                self.history.record(HistoryEntry(recorded_at, category, message))
            except sqlite3.Error as error:
                self.banner.configure(
                    text=f"KHÔNG LƯU ĐƯỢC LỊCH SỬ: {error}",
                    bg="#7a2630",
                    fg="#ffffff",
                )
        self.log.configure(state="normal")
        self.log.insert("end", f"[{recorded_at}] {message}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def close(self) -> None:
        self.stop()
        self._closed = True
        self.root.destroy()

    def _restore_history(self) -> None:
        for entry in self.history.recent():
            self._append_log(
                entry.message,
                category=entry.category,
                persist=False,
                timestamp=entry.timestamp,
            )
        self.metric_values["SỐ LẦN CẢNH BÁO"].configure(
            text=str(self.monitor.alert_count)
        )
        self.metric_values["SỐ LẦN NGÁP"].configure(
            text=str(self.monitor.yawn_count)
        )
        self.metric_values["SỐ LẦN GỤC ĐẦU*"].configure(
            text=str(self.monitor.head_drop_count)
        )
        self._append_log("ỨNG DỤNG SẴN SÀNG.")


def run_gui(config_path: str | Path = "config.yaml") -> None:
    DrowsinessWindow(config_path).run()
