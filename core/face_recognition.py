import logging
import threading
import numpy as np
import cv2
from django.conf import settings
from django.db.utils import ProgrammingError, OperationalError  # ✅ Added

logger = logging.getLogger('core')

# Optional libraries
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    mp = None
    MP_AVAILABLE = False

try:
    import face_recognition
    FR_AVAILABLE = True
except Exception:
    face_recognition = None
    FR_AVAILABLE = False
    logger.warning("face_recognition library not available; fallback detection/encoding may fail.")


def _safe_normalize(arr):
    a = np.asarray(arr, dtype=np.float32).ravel()
    norm = np.linalg.norm(a)
    if norm <= 0 or np.isnan(norm):
        return a
    return (a / norm).astype(np.float32)


class FaceRecognitionSystem:
    """
    Face recognition system using face_recognition (dlib) or MediaPipe.
    
    Attributes:
        confidence_threshold (float): Threshold for face matching.
    """
    def __init__(self, confidence_threshold=None):
        self.lock = threading.Lock()
        self.confidence_threshold = float(getattr(settings, "FACE_RECOGNITION_THRESHOLD", 0.55)
                                          if confidence_threshold is None else confidence_threshold)
        self.known_face_encodings = None
        self.known_face_employee_ids = []
        self.initialized = False

        # MediaPipe face detection (optional)
        self.mp_detector = None
        if MP_AVAILABLE:
            try:
                self.mp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
            except Exception:
                logger.exception("Failed to initialize MediaPipe face detector")
                self.mp_detector = None

        self.load_known_faces()

    # ✅ Protect against missing table/column
    def load_known_faces(self):
        from .models import Employee
        with self.lock:
            try:
                employees = Employee.objects.filter(is_active=True).exclude(
                    face_encoding__isnull=True).exclude(face_encoding__exact='')
            except (ProgrammingError, OperationalError):
                logger.warning("⚠ Database not ready — skipping face loading.")
                self.initialized = False
                return

            encs, ids = [], []
            for emp in employees:
                try:
                    enc = emp.get_face_encoding()
                    if enc is not None:
                        encs.append(_safe_normalize(enc))
                        ids.append(emp.employee_id)
                except Exception:
                    logger.exception("Failed loading encoding for employee %s", emp)

            if encs:
                self.known_face_encodings = np.vstack(encs).astype(np.float32)
                self.known_face_employee_ids = ids
            else:
                self.known_face_encodings = None
                self.known_face_employee_ids = []

            self.initialized = True
            logger.info("Loaded %d face encodings", len(ids))

    def register_face(self, image_path, employee):
        try:
            if not FR_AVAILABLE:
                return False, "face_recognition library not available."
            img = face_recognition.load_image_file(image_path)
            encs = face_recognition.face_encodings(img)
            if not encs:
                return False, "No face detected."
            if len(encs) > 1:
                return False, "Multiple faces detected."

            enc = _safe_normalize(encs[0])
            if hasattr(employee, "save_face_encoding"):
                employee.save_face_encoding(enc)

            with self.lock:
                if self.known_face_encodings is None:
                    self.known_face_encodings = enc.reshape(1, -1).astype(np.float32)
                else:
                    self.known_face_encodings = np.vstack([self.known_face_encodings, enc.reshape(1, -1)])
                self.known_face_employee_ids.append(employee.employee_id)

            return True, "Face registered successfully."
        except Exception:
            logger.exception("Error registering face")
            return False, "Internal error."

    def detect_faces(self, frame_bgr):
        if self.mp_detector is not None:
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                res = self.mp_detector.process(rgb)
                if not getattr(res, "detections", None):
                    return []
                h, w = frame_bgr.shape[:2]
                return [
                    (
                        max(0, int(det.location_data.relative_bounding_box.ymin * h)),
                        min(w, int((det.location_data.relative_bounding_box.xmin + det.location_data.relative_bounding_box.width) * w)),
                        min(h, int((det.location_data.relative_bounding_box.ymin + det.location_data.relative_bounding_box.height) * h)),
                        max(0, int(det.location_data.relative_bounding_box.xmin * w)),
                    )
                    for det in res.detections
                ]
            except Exception:
                logger.exception("MediaPipe detection failed.")

        if FR_AVAILABLE:
            try:
                rgb = frame_bgr[:, :, ::-1]
                model_name = getattr(settings, "FACE_RECOGNITION_MODEL", "hog")
                return face_recognition.face_locations(rgb, model=model_name)
            except Exception:
                logger.exception("face_recognition detection failed")

        return []

    def recognize_face_from_frame(self, frame_bgr):
        if not self.initialized or not self.known_face_employee_ids or self.known_face_encodings is None:
            return None, 0.0, None

        locs = self.detect_faces(frame_bgr)
        if not locs:
            return None, 0.0, None

        try:
            rgb = frame_bgr[:, :, ::-1]
            encs = face_recognition.face_encodings(rgb, locs) if FR_AVAILABLE else []
        except Exception:
            encs = []

        if not encs:
            return None, 0.0, locs[0]

        enc = _safe_normalize(encs[0])
        db = self.known_face_encodings
        if enc.shape[0] != db.shape[1]:
            self.load_known_faces()
            db = self.known_face_encodings
            if db is None or enc.shape[0] != db.shape[1]:
                return None, 0.0, locs[0]

        sims = np.dot(db, enc)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= self.confidence_threshold:
            return self.known_face_employee_ids[best_idx], best_score, locs[0]
        return None, best_score, locs[0]

    # -----------------------------
    # Recognize from static image path
    # -----------------------------
    def recognize_face(self, image_path):
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                return None, 0.0
            emp_id, score, _ = self.recognize_face_from_frame(frame)
            return emp_id, score
        except Exception:
            logger.exception("Image recognition failed")
            return None, 0.0


# -----------------------------
# Singleton Getter
# -----------------------------
_face_system_instance = None
def get_face_system():
    global _face_system_instance
    if _face_system_instance is None:
        try:
            _face_system_instance = FaceRecognitionSystem()
        except (ProgrammingError, OperationalError):
            logger.warning("⚠ Face recognition skipped — DB not migrated yet.")
            return None
    return _face_system_instance
