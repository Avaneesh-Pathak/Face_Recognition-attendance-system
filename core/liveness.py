# ...existing code...
import logging
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from collections import deque
from typing import Optional

logger = logging.getLogger('core')


class LivenessDetector:
    def __init__(self, blink_threshold=0.22, motion_threshold=0.0009,
                 consec_frames=2, history=5, blink_required=1):
        self.blink_threshold = float(blink_threshold)
        self.motion_threshold = float(motion_threshold)
        self.consec_frames = int(consec_frames)
        self.blink_required = int(blink_required)

        self.blink_frame_count = 0
        self.total_blinks = 0

        # Initialize MediaPipe FaceMesh once and reuse; ensure it is closed on cleanup
        self.mp_face_mesh = mp.solutions.face_mesh
        try:
            self.mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception:
            logger.exception("Failed to initialize MediaPipe FaceMesh")
            self.mesh = None

        self.centroid_history = deque(maxlen=int(history))

    def close(self):
        try:
            if getattr(self, 'mesh', None) is not None:
                self.mesh.close()
                self.mesh = None
        except Exception:
            logger.exception("Error closing MediaPipe FaceMesh")

    def __del__(self):
        # Attempt to release resources on object deletion
        try:
            self.close()
        except Exception:
            pass

    def eye_aspect_ratio(self, eye) -> float:
        # expects sequence of 6 (x,y) points
        try:
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])
            C = dist.euclidean(eye[0], eye[3])
            if C == 0:
                return 0.0
            ear = (A + B) / (2.0 * C)
            return float(ear)
        except Exception:
            logger.exception("Error computing EAR")
            return 0.0

    def detect(self, frame: np.ndarray) -> bool:
        """
        Return True if liveness is detected (blink or sufficient head motion).
        Safe to call repeatedly; will not raise on MediaPipe failures.
        """
        if frame is None:
            return False

        if getattr(self, 'mesh', None) is None:
            logger.debug("FaceMesh not initialized; cannot perform liveness detection")
            return False

        try:
            # Ensure frame is BGR 3-channel
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.mesh.process(rgb)

            if not result or not getattr(result, 'multi_face_landmarks', None):
                # reset temporary blink count to avoid stale counts
                self.blink_frame_count = 0
                return False

            face = result.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            pts = []
            for lm in face.landmark:
                # guard against malformed landmarks
                if lm.x is None or lm.y is None:
                    continue
                pts.append((lm.x * w, lm.y * h))

            # Need at least the indices used for eyes
            LEFT = [33, 160, 158, 133, 153, 144]
            RIGHT = [362, 385, 387, 263, 373, 380]
            max_idx = max(LEFT + RIGHT)
            if len(pts) <= max_idx:
                logger.debug("Insufficient landmarks: got %d landmarks", len(pts))
                return False

            left_eye = [pts[i] for i in LEFT]
            right_eye = [pts[i] for i in RIGHT]

            ear = (self.eye_aspect_ratio(left_eye) + self.eye_aspect_ratio(right_eye)) / 2.0

            # --- Blink detection ---
            if ear < self.blink_threshold:
                self.blink_frame_count += 1
            else:
                if self.blink_frame_count >= self.consec_frames:
                    self.total_blinks += 1
                self.blink_frame_count = 0

            # --- Head movement detection ---
            centroid = np.mean(np.array(pts), axis=0)
            self.centroid_history.append(centroid)

            motion_detected = False
            if len(self.centroid_history) >= 2:
                diffs = np.linalg.norm(
                    np.diff(np.array(self.centroid_history), axis=0), axis=1
                )
                mean_diff = float(np.mean(diffs)) if diffs.size > 0 else 0.0
                if mean_diff > self.motion_threshold:
                    motion_detected = True

            # --- Final liveness logic ---
            if self.total_blinks >= self.blink_required or motion_detected:
                # reset blink counter after positive detection to avoid repeat triggers
                self.total_blinks = 0
                self.blink_frame_count = 0
                return True

            return False

        except Exception:
            logger.exception("Exception during liveness detection")
            # conservative default: treat as not live
            return False