import cv2
import numpy as np
from collections import deque
import logging
from django.conf import settings

logger = logging.getLogger("core")

# -------------------------------------------------
# Optional MediaPipe
# -------------------------------------------------
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    mp = None
    MP_AVAILABLE = False
    logger.warning("⚠️ MediaPipe not available — liveness will be bypassed")


# -------------------------------------------------
# Liveness Guard
# -------------------------------------------------
class LivenessGuard:
    """
    Fast blink + head-movement liveness detection.
    Returns (is_live, reason) for UI display.
    """

    # Reduced landmark set (faster)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self):
        cfg = getattr(settings, "LIVENESS", {})

        self.blink_threshold = float(cfg.get("BLINK_THRESHOLD", 0.21))
        self.motion_threshold = float(cfg.get("MOTION_THRESHOLD", 0.002))
        self.consec_frames = int(cfg.get("CONSEC_FRAMES", 2))
        self.blink_required = int(cfg.get("BLINK_REQUIRED", 1))
        self.require_both = bool(cfg.get("REQUIRE_BOTH", False))
        self.history_len = int(cfg.get("HISTORY", 6))

        # State
        self.blink_frames = 0
        self.total_blinks = 0
        self.centroids = deque(maxlen=self.history_len)

        self.mesh = None
        if MP_AVAILABLE:
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("✅ LivenessGuard initialized (MediaPipe)")
        else:
            logger.warning("⚠️ LivenessGuard disabled (MediaPipe missing)")

    # -------------------------------------------------
    # Eye Aspect Ratio (FAST)
    # -------------------------------------------------
    def _ear(self, pts):
        a = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        b = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        c = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return 0.0 if c == 0 else (a + b) / (2.0 * c)

    # -------------------------------------------------
    # Main liveness check
    # -------------------------------------------------
    def check(self, frame, bbox):
        """
        Args:
            frame: BGR image
            bbox: [x1, y1, x2, y2]

        Returns:
            (bool, str) → (is_live, reason_for_ui)
        """

        # -------------------------------------------------
        # Fail-open if MediaPipe missing
        # -------------------------------------------------
        if not MP_AVAILABLE or self.mesh is None:
            logger.debug("Liveness bypassed (fail-open)")
            return True, ""

        x1, y1, x2, y2 = bbox
        fh, fw = frame.shape[:2]

        # Clamp bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            logger.debug("Liveness fail: empty crop")
            return False, "Face not visible properly"

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)

        if not res.multi_face_landmarks:
            logger.debug("Liveness fail: landmarks not detected")
            return False, "Please face the camera properly"

        lm = res.multi_face_landmarks[0].landmark
        ch, cw = crop.shape[:2]
        pts = [(p.x * cw, p.y * ch) for p in lm]

        # -------------------------------------------------
        # Blink detection
        # -------------------------------------------------
        ear_left = self._ear([pts[i] for i in self.LEFT_EYE])
        ear_right = self._ear([pts[i] for i in self.RIGHT_EYE])
        ear = (ear_left + ear_right) / 2.0

        if ear < self.blink_threshold:
            self.blink_frames += 1
        else:
            if self.blink_frames >= self.consec_frames:
                self.total_blinks += 1
                logger.debug("👁️ Blink detected (total=%d)", self.total_blinks)
            self.blink_frames = 0

        blink_ok = self.total_blinks >= self.blink_required

        # -------------------------------------------------
        # Head motion detection (normalized)
        # -------------------------------------------------
        centroid = np.mean(np.array(pts), axis=0)
        self.centroids.append(centroid)

        motion = False
        if len(self.centroids) > 1:
            diffs = np.linalg.norm(
                np.diff(self.centroids, axis=0), axis=1
            )
            norm_motion = np.mean(diffs) / max(cw, ch)
            motion = norm_motion > self.motion_threshold

        # -------------------------------------------------
        # Decision
        # -------------------------------------------------
        live = (
            blink_ok and motion
            if self.require_both
            else blink_ok or motion
        )

        if live:
            logger.info("✅ Liveness confirmed")
            self.total_blinks = 0
            self.blink_frames = 0
            self.centroids.clear()
            return True, ""

        # -------------------------------------------------
        # Failure reason for UI
        # -------------------------------------------------
        if not blink_ok and not motion:
            return False, "Please blink your eyes or move your head"
        if not blink_ok:
            return False, "Please blink your eyes"
        if not motion:
            return False, "Please move your head slightly"

        return False, "Liveness verification failed"
