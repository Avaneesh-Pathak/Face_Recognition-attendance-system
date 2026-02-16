import logging
import cv2
import numpy as np
from collections import deque
from django.conf import settings

logger = logging.getLogger('core')

# ✅ Check if MediaPipe is available
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    mp = None
    MP_AVAILABLE = False
    logger.warning("⚠ MediaPipe not found — Liveness detection disabled")


class LivenessDetector:
    """
    Detects liveness using MediaPipe FaceMesh.
    """

    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    LEFT_EYE_SIMPLE = [33, 133, 160, 144, 158, 153]
    RIGHT_EYE_SIMPLE = [362, 263, 385, 380, 387, 373]

    def __init__(self,
                 blink_threshold=0.22,
                 motion_threshold=0.002,
                 consec_frames=3,
                 blink_required=1,
                 require_both=False,
                 adaptive=True):

        cfg = getattr(settings, "LIVENESS", {})

        self.blink_threshold = float(cfg.get("BLINK_THRESHOLD", blink_threshold))
        self.motion_threshold = float(cfg.get("MOTION_THRESHOLD", motion_threshold))
        self.consec_frames = int(cfg.get("CONSEC_FRAMES", consec_frames))
        self.blink_required = int(cfg.get("BLINK_REQUIRED", blink_required))
        self.require_both = bool(cfg.get("REQUIRE_BOTH", require_both))
        self.adaptive = adaptive

        self.blink_frame_count = 0
        self.total_blinks = 0
        self.last_ear = 0.0
        self.blink_detected = False

        self.ear_samples = deque(maxlen=30)
        self.dynamic_threshold = self.blink_threshold
        self.centroid_history = deque(maxlen=10)

        self.mesh = None
        self._closed = False  # ✅ important

        if MP_AVAILABLE:
            try:
                self.mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    refine_landmarks=True,
                    max_num_faces=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("✅ MediaPipe FaceMesh initialized successfully")
            except Exception:
                logger.exception("❌ MediaPipe FaceMesh failed to initialize")

    def _euclid(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _ear_simple(self, eye_pts):
        v1 = self._euclid(eye_pts[1], eye_pts[5])
        v2 = self._euclid(eye_pts[2], eye_pts[4])
        h = self._euclid(eye_pts[0], eye_pts[3])
        return 0.0 if h == 0 else (v1 + v2) / (2.0 * h)

    def _update_dynamic_threshold(self, ear):
        if not self.adaptive:
            return
        if ear > 0.15:
            self.ear_samples.append(ear)
        if len(self.ear_samples) >= 10:
            avg_ear = np.mean(self.ear_samples)
            self.dynamic_threshold = max(0.18, min(avg_ear * 0.6, 0.25))
            logger.debug(
                f"📊 EAR: {ear:.3f}, Avg: {avg_ear:.3f}, Thr: {self.dynamic_threshold:.3f}"
            )

    def detect_detail(self, frame):
        result = {
            'blink': False,
            'motion': False,
            'ear': 0.0,
            'live': False,
            'face_detected': False
        }

        if frame is None or not MP_AVAILABLE or self.mesh is None:
            return result

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = self.mesh.process(rgb)

        if not out or not out.multi_face_landmarks:
            self.blink_frame_count = 0
            self.last_ear = 0.0
            return result

        result['face_detected'] = True
        pts = [(lm.x * w, lm.y * h) for lm in out.multi_face_landmarks[0].landmark]

        left_ear = self._ear_simple([pts[i] for i in self.LEFT_EYE_SIMPLE])
        right_ear = self._ear_simple([pts[i] for i in self.RIGHT_EYE_SIMPLE])
        ear = (left_ear + right_ear) / 2 if left_ear and right_ear else max(left_ear, right_ear)

        result['ear'] = round(float(ear), 3)
        self.last_ear = ear

        self._update_dynamic_threshold(ear)
        blink_thr = self.dynamic_threshold

        if ear < blink_thr:
            self.blink_frame_count += 1
        else:
            if self.blink_frame_count >= self.consec_frames:
                self.total_blinks += 1
                logger.info(f"👁️ Blink detected (total={self.total_blinks})")
            self.blink_frame_count = 0

        result['blink'] = self.total_blinks >= self.blink_required

        centroid = np.mean(np.array(pts), axis=0)
        self.centroid_history.append(centroid)

        if len(self.centroid_history) > 1:
            motion_val = np.mean(
                np.linalg.norm(np.diff(self.centroid_history, axis=0), axis=1)
            ) / np.hypot(w, h)
            result['motion'] = motion_val > self.motion_threshold
            logger.debug(f"🎯 Motion: {motion_val:.6f}")
        else:
            result['motion'] = False

        result['live'] = (
            result['blink'] and result['motion']
            if self.require_both
            else result['blink'] or result['motion']
        )

        if result['live']:
            logger.info("✅ Liveness confirmed - resetting counters")
            self.reset()

        return result

    def detect(self, frame):
        return self.detect_detail(frame)['live']

    def reset(self):
        self.total_blinks = 0
        self.blink_frame_count = 0
        self.blink_detected = False
        self.ear_samples.clear()
        self.centroid_history.clear()

    def close(self):
        """
        ✅ Safe to call multiple times
        """
        if self._closed:
            return
        self._closed = True

        try:
            if self.mesh is not None:
                self.mesh.close()
                self.mesh = None
        except Exception as e:
            logger.warning(f"Liveness cleanup ignored: {e}")
