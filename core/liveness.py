# core/liveness.py
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from collections import deque

class LivenessDetector:
    def __init__(self, blink_threshold=0.22, motion_threshold=0.0009,
                 consec_frames=2, history=5, blink_required=1):
        self.blink_threshold = blink_threshold
        self.motion_threshold = motion_threshold
        self.consec_frames = consec_frames
        self.blink_required = blink_required

        self.blink_frame_count = 0
        self.total_blinks = 0

        self.mp_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

        self.centroid_history = deque(maxlen=history)

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)

        if not result.multi_face_landmarks:
            return False

        h, w, _ = frame.shape
        face = result.multi_face_landmarks[0]
        pts = [(lm.x * w, lm.y * h) for lm in face.landmark]

        LEFT = [33, 160, 158, 133, 153, 144]
        RIGHT = [362, 385, 387, 263, 373, 380]

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
            if np.mean(diffs) > self.motion_threshold:
                motion_detected = True

        # --- Final liveness logic ---
        if self.total_blinks >= self.blink_required or motion_detected:
            self.total_blinks = 0
            return True
        
        return False
