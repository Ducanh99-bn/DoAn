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

                   