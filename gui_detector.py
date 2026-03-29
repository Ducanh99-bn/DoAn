import sys
sys.path.append('ultralytics')

import os
import time
import numpy as np
import cv2
import tensorflow as tf
from ultralytics import YOLO
from PyQt5 import QtCore, QtGui, QtWidgets

# ================= CONFIG =================
YOLO_MODEL = "best.onnx"
CLASSIFIER_MODEL = "mobilenetv2_egg_classifier.keras"

IMGSZ = 640
CONF_THRESH = 0.45
CLASSIFIER_IMG_SIZE = (256, 256)
MIN_ROI_SIZE = 0

# Default threshold (video file)
BROKEN_THRESH_DEFAULT = 0.10
# Override threshold for Camera
BROKEN_THRESH_CAMERA = 0.05

ROI_PAD_RATIO = 0.0

# Stability rule:
STABLE_FRAMES = 3
ALLOW_RELOCK = False

# Camera probing
CAMERA_MAX_INDEX = 10
# =========================================


def pick_opencv_backend():
    if hasattr(cv2, "CAP_DSHOW"):
        return cv2.CAP_DSHOW
    if hasattr(cv2, "CAP_V4L2"):
        return cv2.CAP_V4L2
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        return cv2.CAP_AVFOUNDATION
    return 0


OPENCV_BACKEND = pick_opencv_backend()


def probe_cameras(max_index=10, backend=0, warmup_sec=0.2, read_timeout_sec=0.8):
    ok = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, backend) if backend else cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue

        t0 = time.time()
        got = False
        w = h = 0
        fps = 0.0
        while time.time() - t0 < read_timeout_sec:
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                got = True
                h, w = frame.shape[:2]
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                break
            time.sleep(warmup_sec)

        cap.release()
        if got:
            ok.append((i, w, h, fps))
    return ok


def choose_preferred_camera_index():
    cams = probe_cameras(CAMERA_MAX_INDEX, OPENCV_BACKEND)
    if not cams:
        return 0, []
    idx = cams[-1][0]
    return idx, cams


# ================= VIDEO WORKER =================
class VideoWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    status_ready = QtCore.pyqtSignal(str)
    stats_ready = QtCore.pyqtSignal(int, int, int)  # total, intact, broken  (ADDED)

    def __init__(self, source, broken_thresh: float):
        super().__init__()
        self.source = source
        self.broken_thresh = float(broken_thresh)

        self.running = True
        self.paused = False

        # lock state
        self.id_lock = {}  # track_id -> locked label 0/1
        self.id_run = {}   # track_id -> {'last':0/1, 'cnt':int}

        # stats (ADDED)
        self.total = 0
        self.intact = 0
        self.broken = 0
        self.counted_ids = set()      # track_id đã tính vào total
        self.counted_state = {}       # track_id -> 0/1 đã ghi nhận (để update 0->1)

    def stop(self):
        self.running = False

    def pause(self, p):
        self.paused = bool(p)

    def _open_capture(self):
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, OPENCV_BACKEND) if OPENCV_BACKEND else cv2.VideoCapture(self.source)
        else:
            cap = cv2.VideoCapture(self.source)
        return cap

    def run(self):
        try:
            yolo = YOLO(YOLO_MODEL)
        except Exception as e:
            self.status_ready.emit(f"❌ Lỗi YOLO: {e}")
            return

        try:
            clf = tf.keras.models.load_model(CLASSIFIER_MODEL)
        except Exception as e:
            self.status_ready.emit(f"❌ Lỗi Classifier: {e}")
            return

        cap = self._open_capture()
        if not cap.isOpened():
            self.status_ready.emit("❌ Không mở được nguồn video/camera.")
            return

        try:
            backend_name = cap.getBackendName()
        except Exception:
            backend_name = "UNKNOWN"

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1:
            fps = 30.0

        # reset
        self.id_lock.clear()
        self.id_run.clear()

        # stats reset (ADDED)
        self.total = self.intact = self.broken = 0
        self.counted_ids.clear()
        self.counted_state.clear()

        self.status_ready.emit(
            f"🟢 ĐANG CHẠY | source={self.source} | backend={backend_name} | thresh={self.broken_thresh:.2f}"
        )

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            out = frame.copy()
            h0, w0 = frame.shape[:2]
            if w0 != w or h0 != h:
                w, h = w0, h0

            results = yolo.track(frame, imgsz=IMGSZ, conf=CONF_THRESH, persist=True, verbose=False)

            if results and results[0].boxes:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id[0]) if box.id is not None else None

                    if (x2 - x1) < MIN_ROI_SIZE or (y2 - y1) < MIN_ROI_SIZE:
                        continue

                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w - 1, x2); y2 = min(h - 1, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    pad = int(ROI_PAD_RATIO * max(x2 - x1, y2 - y1))
                    xx1 = max(0, x1 - pad); yy1 = max(0, y1 - pad)
                    xx2 = min(w - 1, x2 + pad); yy2 = min(h - 1, y2 + pad)

                    roi = frame[yy1:yy2, xx1:xx2]
                    if roi.size == 0:
                        continue

                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi = cv2.resize(roi, CLASSIFIER_IMG_SIZE, interpolation=cv2.INTER_AREA)

                    x = roi.astype(np.float32)
                    x = np.expand_dims(x, axis=0)

                    prob = float(clf.predict(x, verbose=0)[0][0])
                    cur_state = 1 if prob > self.broken_thresh else 0  # 1=VỠ, 0=NGUYÊN

                    # ========== RULE: once BROKEN, always BROKEN ==========
                    show_state = cur_state
                    did_lock_now = False  # (ADDED) track lock event
                    if track_id is not None:
                        if track_id in self.id_lock:
                            locked = self.id_lock[track_id]

                            if locked == 1:
                                show_state = 1
                            else:
                                if cur_state == 1:
                                    run = self.id_run.get(track_id)
                                    if run is None:
                                        run = {"last": 1, "cnt": 1}
                                    else:
                                        if run["last"] == 1:
                                            run["cnt"] += 1
                                        else:
                                            run["last"] = 1
                                            run["cnt"] = 1
                                    self.id_run[track_id] = run

                                    if run["cnt"] >= STABLE_FRAMES:
                                        self.id_lock[track_id] = 1
                                        self.id_run[track_id] = {"last": 1, "cnt": 0}
                                        show_state = 1
                                        did_lock_now = True  # (ADDED) became broken now
                                    else:
                                        show_state = 0
                                else:
                                    self.id_run[track_id] = {"last": 0, "cnt": 0}
                                    show_state = 0
                        else:
                            run = self.id_run.get(track_id)
                            if run is None:
                                run = {"last": cur_state, "cnt": 1}
                            else:
                                if run["last"] == cur_state:
                                    run["cnt"] += 1
                                else:
                                    run["last"] = cur_state
                                    run["cnt"] = 1
                            self.id_run[track_id] = run

                            if run["cnt"] >= STABLE_FRAMES:
                                self.id_lock[track_id] = cur_state
                                self.id_run[track_id] = {"last": cur_state, "cnt": 0}
                                show_state = cur_state
                                did_lock_now = True  # (ADDED) first lock
                            else:
                                show_state = cur_state
                    # ======================================================

                    # ===== STATS (ADDED): count only when track_id has a LOCKED state =====
                    if track_id is not None:
                        # first time lock -> add total + class
                        if did_lock_now and (track_id in self.id_lock):
                            locked_state = self.id_lock[track_id]
                            if track_id not in self.counted_ids:
                                self.counted_ids.add(track_id)
                                self.counted_state[track_id] = locked_state
                                self.total += 1
                                if locked_state == 1:
                                    self.broken += 1
                                else:
                                    self.intact += 1
                            else:
                                # already counted, but could be 0->1
                                prev = self.counted_state.get(track_id, 0)
                                if prev == 0 and locked_state == 1:
                                    self.counted_state[track_id] = 1
                                    self.broken += 1
                                    if self.intact > 0:
                                        self.intact -= 1
                        else:
                            # if already counted as intact and later locks to broken (0->1), update immediately
                            if track_id in self.counted_ids and (track_id in self.id_lock):
                                locked_state = self.id_lock[track_id]
                                prev = self.counted_state.get(track_id, locked_state)
                                if prev == 0 and locked_state == 1:
                                    self.counted_state[track_id] = 1
                                    self.broken += 1
                                    if self.intact > 0:
                                        self.intact -= 1

                    label = "TRUNG_VO" if show_state == 1 else "TRUNG_NGUYEN"
                    color = (70, 140, 255) if show_state == 1 else (40, 220, 120)

                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    tag = f"{label}"
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    y_text = max(28, y1 - 10)
                    cv2.rectangle(out, (x1, y_text - th - 10), (x1 + tw + 14, y_text + 6), color, -1)
                    cv2.putText(out, tag, (x1 + 7, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15, 15, 15), 2)

            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                                QtGui.QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg)

            # emit stats (ADDED)
            self.stats_ready.emit(self.total, self.intact, self.broken)

            time.sleep(max(0.0, 1.0 / fps))

        cap.release()
        self.status_ready.emit("⛔ ĐÃ DỪNG")


# ================= MAIN WINDOW =================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EGG INSPECTION SYSTEM")
        self.resize(1200, 780)

        self.video_path = None
        self.worker = None

        self._build_ui()
        self._wire()

        self._set_status("CHƯA CHỌN VIDEO / CAMERA")
        self._set_running(False)

    def _build_ui(self):
        self.setStyleSheet("""
            QMainWindow { background: #0B0F17; }
            QLabel { color: #E6EDF7; font-family: Inter, Arial; }
            QPushButton {
                background: #1C2536;
                color: #E6EDF7;
                border: 1px solid #2B3B57;
                padding: 10px 12px;
                border-radius: 10px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover { background: #22304A; }
            QPushButton:disabled { color:#6C7A96; border-color:#1A2436; background:#121A28; }
            QPushButton#primary { background:#2E6BFF; border-color:#2E6BFF; }
            QPushButton#primary:hover { background:#255AE0; }
            QPushButton#danger { background:#E84C4C; border-color:#E84C4C; }
            QPushButton#danger:hover { background:#D44343; }
            QPushButton#warn { background:#F2B84B; border-color:#F2B84B; color:#0B0F17; }
            QPushButton#warn:hover { background:#E2A93F; }
            QPushButton#success { background:#31C48D; border-color:#31C48D; color:#0B0F17; }
            QPushButton#success:hover { background:#2AB07E; }
            QFrame#card {
                background: #0F1624;
                border: 1px solid #1D2A41;
                border-radius: 14px;
            }
            QLabel#h1 { font-size: 18px; font-weight: 800; }
            QLabel#muted { color: #9FB0CC; }
            QLabel#bigNum { font-size: 34px; font-weight: 900; }
            QLabel#pill {
                background:#121C2D;
                border:1px solid #1D2A41;
                padding:10px 12px;
                border-radius:12px;
                font-size: 14px;
                font-weight: 700;
            }
        """)

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(12)

        header = QtWidgets.QFrame()
        header.setObjectName("card")
        header.setProperty("id", "card")
        hL = QtWidgets.QHBoxLayout(header)
        hL.setContentsMargins(16, 14, 16, 14)

        lblTitle = QtWidgets.QLabel("🥚 EGG INSPECTION")
        lblTitle.setObjectName("h1")
        lblTitle.setProperty("id", "h1")

        lblSub = QtWidgets.QLabel("YOLO detect + MobileNetV2 classify | once broken always broken")
        lblSub.setObjectName("muted")
        lblSub.setProperty("id", "muted")

        tbox = QtWidgets.QVBoxLayout()
        tbox.setSpacing(2)
        tbox.addWidget(lblTitle)
        tbox.addWidget(lblSub)

        self.lblStatus = QtWidgets.QLabel("...")
        self.lblStatus.setObjectName("pill")
        self.lblStatus.setProperty("id", "pill")
        self.lblStatus.setAlignment(QtCore.Qt.AlignCenter)

        hL.addLayout(tbox, 1)
        hL.addWidget(self.lblStatus, 0)
        outer.addWidget(header)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setHandleWidth(10)
        outer.addWidget(splitter, 1)

        # left: video
        body = QtWidgets.QFrame()
        body.setObjectName("card")
        body.setProperty("id", "card")
        bL = QtWidgets.QVBoxLayout(body)
        bL.setContentsMargins(16, 16, 16, 16)
        bL.setSpacing(12)

        self.lblCamera = QtWidgets.QLabel("NO VIDEO / CAMERA")
        self.lblCamera.setAlignment(QtCore.Qt.AlignCenter)
        self.lblCamera.setMinimumSize(920, 620)
        self.lblCamera.setStyleSheet("""
            QLabel {
                background: #05070D;
                border: 1px solid #1D2A41;
                border-radius: 14px;
            }
        """)

        btnRow = QtWidgets.QHBoxLayout()
        btnRow.setSpacing(10)

        self.btnSelect = QtWidgets.QPushButton("📂 Chọn video")

        self.btnCam = QtWidgets.QPushButton("📷 Mở camera (tự chọn)")
        self.btnCam.setObjectName("success")
        self.btnCam.setProperty("id", "success")

        self.btnStart = QtWidgets.QPushButton("▶ Start video")
        self.btnStart.setObjectName("primary")
        self.btnStart.setProperty("id", "primary")

        self.btnPause = QtWidgets.QPushButton("⏸ Pause")
        self.btnPause.setObjectName("warn")
        self.btnPause.setProperty("id", "warn")

        self.btnStop = QtWidgets.QPushButton("⏹ Stop")
        self.btnStop.setObjectName("danger")
        self.btnStop.setProperty("id", "danger")

        btnRow.addWidget(self.btnSelect)
        btnRow.addWidget(self.btnCam)
        btnRow.addStretch(1)
        btnRow.addWidget(self.btnStart)
        btnRow.addWidget(self.btnPause)
        btnRow.addWidget(self.btnStop)

        bL.addWidget(self.lblCamera, 1)
        bL.addLayout(btnRow)

        # right: stats (ADDED)
        statsCard = QtWidgets.QFrame()
        statsCard.setObjectName("card")
        statsCard.setProperty("id", "card")
        sL = QtWidgets.QVBoxLayout(statsCard)
        sL.setContentsMargins(16, 16, 16, 16)
        sL.setSpacing(10)

        sTitle = QtWidgets.QLabel("📊 THỐNG KÊ")
        sTitle.setObjectName("h1")
        sTitle.setProperty("id", "h1")
        sL.addWidget(sTitle)

        def stat_box(title):
            card = QtWidgets.QFrame()
            card.setObjectName("card")
            card.setProperty("id", "card")
            v = QtWidgets.QVBoxLayout(card)
            v.setContentsMargins(14, 12, 14, 12)
            v.setSpacing(6)
            lt = QtWidgets.QLabel(title)
            lt.setObjectName("muted")
            lt.setProperty("id", "muted")
            ln = QtWidgets.QLabel("0")
            ln.setObjectName("bigNum")
            ln.setProperty("id", "bigNum")
            v.addWidget(lt)
            v.addWidget(ln)
            return card, ln

        self.boxTotal, self.lblTotal = stat_box("TỔNG")
        self.boxIntact, self.lblIntact = stat_box("NGUYÊN")
        self.boxBroken, self.lblBroken = stat_box("VỠ")

        sL.addWidget(self.boxTotal)
        sL.addWidget(self.boxIntact)
        sL.addWidget(self.boxBroken)
        sL.addStretch(1)

        splitter.addWidget(body)
        splitter.addWidget(statsCard)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

    def _wire(self):
        self.btnSelect.clicked.connect(self.select_video)
        self.btnCam.clicked.connect(self.open_camera_auto)
        self.btnStart.clicked.connect(self.start_video)
        self.btnPause.clicked.connect(self.pause)
        self.btnStop.clicked.connect(self.stop)

    def _set_status(self, s):
        self.lblStatus.setText(s)

    def _set_running(self, running: bool):
        self.btnStop.setEnabled(running)
        self.btnPause.setEnabled(running)
        self.btnStart.setEnabled(not running)
        self.btnCam.setEnabled(not running)
        self.btnSelect.setEnabled(not running)

    def _start_worker(self, source, status_text, broken_thresh: float):
        self.stop()
        self.worker = VideoWorker(source, broken_thresh=broken_thresh)
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.status_ready.connect(self._set_status)
        self.worker.stats_ready.connect(self.update_stats)  # (ADDED)
        self.worker.start()
        self._set_status(status_text)
        self._set_running(True)

    def select_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Chọn video", "", "Video Files (*.mp4 *.avi *.mkv)"
        )
        if path:
            self.video_path = path
            self._set_status(f"ĐÃ CHỌN: {os.path.basename(path)}")

    def open_camera_auto(self):
        self.video_path = None
        cam_index, cams = choose_preferred_camera_index()
        if cams:
            info = " | ".join([f"{i}:{w}x{h}" for (i, w, h, _) in cams])
            self._set_status(f"CAMERA FOUND: {info}")
        self._start_worker(
            cam_index,
            f"🟢 CAMERA ĐANG CHẠY (index={cam_index}) | BROKEN_THRESH={BROKEN_THRESH_CAMERA:.2f}",
            broken_thresh=BROKEN_THRESH_CAMERA
        )

    def start_video(self):
        if not self.video_path:
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Chưa chọn video!")
            return
        self._start_worker(
            self.video_path,
            f"🟢 VIDEO ĐANG CHẠY | BROKEN_THRESH={BROKEN_THRESH_DEFAULT:.2f}",
            broken_thresh=BROKEN_THRESH_DEFAULT
        )

    def pause(self):
        if not self.worker:
            return
        self.worker.pause(not self.worker.paused)
        self._set_status("⏸ PAUSED" if self.worker.paused else "🟢 ĐANG CHẠY")

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(1500)
            self.worker = None
        self._set_running(False)
        if self.video_path:
            self._set_status(f"ĐÃ CHỌN: {os.path.basename(self.video_path)}")
        else:
            self._set_status("⛔ ĐÃ DỪNG")

        # reset UI stats (ADDED)
        self.update_stats(0, 0, 0)

    def show_frame(self, img):
        pix = QtGui.QPixmap.fromImage(img)
        self.lblCamera.setPixmap(
            pix.scaled(self.lblCamera.size(),
                       QtCore.Qt.KeepAspectRatio,
                       QtCore.Qt.SmoothTransformation)
        )

    # (ADDED)
    def update_stats(self, total, intact, broken):
        self.lblTotal.setText(str(total))
        self.lblIntact.setText(str(intact))
        self.lblBroken.setText(str(broken))


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
