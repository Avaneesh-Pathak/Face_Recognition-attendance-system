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
    ✅ Fixed blink detection with better EAR calculation and debugging
    """

    # Updated eye landmarks for MediaPipe (more accurate)
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    # Simplified eye points for EAR calculation
    LEFT_EYE_SIMPLE = [33, 133, 160, 144, 158, 153]
    RIGHT_EYE_SIMPLE = [362, 263, 385, 380, 387, 373]

    def __init__(self,
                 blink_threshold=0.22,
                 motion_threshold=0.002,
                 consec_frames=3,  # Increased for better reliability
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

        self.ear_samples = deque(maxlen=30)  # Use deque for better performance
        self.dynamic_threshold = self.blink_threshold

        self.centroid_history = deque(maxlen=10)  # Increased for smoother motion detection

        self.mesh = None
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
        """Improved EAR calculation using 6 points"""
        # Vertical distances
        v1 = self._euclid(eye_pts[1], eye_pts[5])
        v2 = self._euclid(eye_pts[2], eye_pts[4])
        
        # Horizontal distance
        h = self._euclid(eye_pts[0], eye_pts[3])
        
        # Avoid division by zero
        if h == 0:
            return 0.0
            
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def _update_dynamic_threshold(self, ear):
        if not self.adaptive:
            return
            
        # Only add samples when eyes are open (above basic threshold)
        if ear > 0.15:
            self.ear_samples.append(ear)
            
        # Update threshold if we have enough samples
        if len(self.ear_samples) >= 10:
            avg_ear = np.mean(self.ear_samples)
            std_ear = np.std(self.ear_samples)
            
            # More conservative dynamic threshold
            self.dynamic_threshold = max(0.18, min(avg_ear * 0.6, 0.25))
            
            logger.debug(f"📊 EAR: {ear:.3f}, Avg: {avg_ear:.3f}, Threshold: {self.dynamic_threshold:.3f}")

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

        # Calculate EAR for both eyes
        left_eye = [pts[i] for i in self.LEFT_EYE_SIMPLE]
        right_eye = [pts[i] for i in self.RIGHT_EYE_SIMPLE]
        
        left_ear = self._ear_simple(left_eye)
        right_ear = self._ear_simple(right_eye)
        
        # Use the average EAR, but handle cases where one eye might not be visible
        if left_ear > 0 and right_ear > 0:
            ear = (left_ear + right_ear) / 2.0
        else:
            ear = max(left_ear, right_ear)
            
        result['ear'] = round(float(ear), 3)
        self.last_ear = ear

        # Update dynamic threshold
        self._update_dynamic_threshold(ear)
        blink_thr = self.dynamic_threshold

        # Blink detection logic
        if ear < blink_thr:
            self.blink_frame_count += 1
            self.blink_detected = False
        else:
            # Check if we just completed a blink
            if self.blink_frame_count >= self.consec_frames:
                self.total_blinks += 1
                self.blink_detected = True
                logger.info(f"👁️ Blink detected! Total: {self.total_blinks}, Frames: {self.blink_frame_count}")
            self.blink_frame_count = 0

        result['blink'] = (self.total_blinks >= self.blink_required)

        # Motion detection
        centroid = np.mean(np.array(pts), axis=0)
        self.centroid_history.append(centroid)

        if len(self.centroid_history) > 1:
            motion_val = np.mean(np.linalg.norm(np.diff(list(self.centroid_history), axis=0), axis=1)) / (np.hypot(w, h))
            result['motion'] = motion_val > self.motion_threshold
            logger.debug(f"🎯 Motion: {motion_val:.6f}, Threshold: {self.motion_threshold}")
        else:
            result['motion'] = False

        # Liveness decision
        if self.require_both:
            result['live'] = result['blink'] and result['motion']
        else:
            result['live'] = result['blink'] or result['motion']

        # Reset if liveness confirmed
        if result['live']:
            logger.info("✅ Liveness confirmed - resetting counters")
            self.total_blinks = 0
            self.blink_frame_count = 0
            self.blink_detected = False

        return result

    def detect(self, frame):
        return self.detect_detail(frame)['live']

    def reset(self):
        """Reset the detector state"""
        self.total_blinks = 0
        self.blink_frame_count = 0
        self.blink_detected = False
        self.ear_samples.clear()
        self.centroid_history.clear()

    def close(self):
        if self.mesh:
            self.mesh.close()

    def __del__(self):
        self.close()