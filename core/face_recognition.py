import os
import cv2
import json
import numpy as np
import face_recognition
import threading
from django.conf import settings
from .models import Employee

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


# -----------------------------
# Utility Functions
# -----------------------------
def normalize_encoding(encoding):
    """Normalize face embedding vector for consistent similarity."""
    return encoding / np.linalg.norm(encoding)


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -----------------------------
# Face Recognition System
# -----------------------------
class FaceRecognitionSystem:
    """
    Optimized Face Recognition System
    ---------------------------------
    ✅ Uses in-memory caching
    ✅ Cosine similarity for matching
    ✅ Thread-safe lazy loading
    ✅ Supports MediaPipe (optional, faster detection)
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.known_face_encodings = np.empty((0, 128))  # store as numpy array
        self.known_face_employee_ids = []
        self.confidence_threshold = 0.55  # Cosine similarity threshold
        self.initialized = False
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        ) if MP_AVAILABLE else None

        self.load_known_faces()

    # -----------------------------
    # Load Faces from Database
    # -----------------------------
    def load_known_faces(self):
        """Load all known face encodings into memory."""
        try:
            with self.lock:
                employees = Employee.objects.filter(is_active=True).exclude(face_encoding__isnull=True)
                encodings, ids = [], []

                for emp in employees:
                    encoding = emp.get_face_encoding()
                    if encoding is not None:
                        encodings.append(normalize_encoding(encoding))
                        ids.append(emp.employee_id)

                if encodings:
                    self.known_face_encodings = np.vstack(encodings)
                    self.known_face_employee_ids = ids
                    self.initialized = True
                    print(f"[INFO] Loaded {len(ids)} face encodings into memory.")
                else:
                    self.known_face_encodings = np.empty((0, 128))
                    self.known_face_employee_ids = []
                    print("[WARN] No face encodings found in DB.")

        except Exception as e:
            print(f"[ERROR] Face loading failed: {e}")

    # -----------------------------
    # Register New Face
    # -----------------------------
    def register_face(self, image_path, employee):
        """Register a new face for an employee."""
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            if not face_encodings:
                return False, "No face detected in the image."
            if len(face_encodings) > 1:
                return False, "Multiple faces detected. Please use a single-face image."

            encoding = normalize_encoding(face_encodings[0])
            employee.save_face_encoding(encoding)

            # Update in-memory cache only for this new user
            with self.lock:
                self.known_face_encodings = np.vstack([self.known_face_encodings, encoding])
                self.known_face_employee_ids.append(employee.employee_id)

            return True, "Face registered successfully."

        except Exception as e:
            return False, f"Error registering face: {str(e)}"

    # -----------------------------
    # Detect Faces Efficiently
    # -----------------------------
    def detect_faces(self, frame):
        """Detect faces using MediaPipe (fast) or fallback to dlib."""
        if self.mp_face:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face.process(rgb)
            if not results.detections:
                return []

            h, w, _ = frame.shape
            locations = []
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                top = int(bbox.ymin * h)
                left = int(bbox.xmin * w)
                bottom = int((bbox.ymin + bbox.height) * h)
                right = int((bbox.xmin + bbox.width) * w)
                locations.append((top, right, bottom, left))
            return locations

        # Fallback to dlib if MediaPipe unavailable
        rgb_frame = frame[:, :, ::-1]
        return face_recognition.face_locations(rgb_frame, model=settings.FACE_RECOGNITION_MODEL)

    # -----------------------------
    # Recognize Face From Frame
    # -----------------------------
    def recognize_face_from_frame(self, frame):
        """Recognize face(s) from a video frame efficiently."""
        try:
            if not self.initialized or len(self.known_face_employee_ids) == 0:
                print("[WARN] No faces loaded in memory.")
                return None, 0.0, None

            rgb_frame = frame[:, :, ::-1]
            face_locations = self.detect_faces(frame)
            if not face_locations:
                return None, 0.0, None

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if not face_encodings:
                return None, 0.0, face_locations[0]

            face_encoding = normalize_encoding(face_encodings[0])

            # Vectorized cosine similarity
            similarities = np.dot(self.known_face_encodings, face_encoding)
            best_idx = np.argmax(similarities)
            confidence = similarities[best_idx]

            if confidence >= self.confidence_threshold:
                return self.known_face_employee_ids[best_idx], float(confidence), face_locations[0]

            return None, float(confidence), face_locations[0]

        except Exception as e:
            print(f"[ERROR] Recognition failed: {e}")
            return None, 0.0, None

    # -----------------------------
    # Recognize Face From Image
    # -----------------------------
    def recognize_face(self, image_path):
        """Recognize face from static image."""
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                return None, 0.0

            emp_id, conf, _ = self.recognize_face_from_frame(frame)
            return emp_id, conf

        except Exception as e:
            print(f"[ERROR] Image recognition failed: {e}")
            return None, 0.0


# -----------------------------
# Singleton Getter
# -----------------------------
_face_system_instance = None

def get_face_system():
    global _face_system_instance
    if _face_system_instance is None:
        _face_system_instance = FaceRecognitionSystem()
    return _face_system_instance
