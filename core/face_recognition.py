import logging
import threading
import numpy as np
import cv2

from django.conf import settings
from django.db.utils import ProgrammingError, OperationalError
from .models import Employee

logger = logging.getLogger("core")

# -----------------------------
# Optional libraries
# -----------------------------
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
    logger.warning("face_recognition library not available.")


# -----------------------------
# Face Recognition System
# -----------------------------
class FaceRecognitionSystem:
    def __init__(self, confidence_threshold=None):
        self.lock = threading.Lock()

        # IMPORTANT: this is a DISTANCE threshold (lower is better)
        self.confidence_threshold = float(
            getattr(settings, "FACE_RECOGNITION_THRESHOLD", 0.60)
            if confidence_threshold is None else confidence_threshold
        )

        self.known_face_encodings = None
        self.known_face_employee_ids = []
        self.initialized = False

        # Optional MediaPipe detector (only for face detection)
        self.mp_detector = None
        if MP_AVAILABLE:
            try:
                self.mp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=0.5
                )
            except Exception:
                logger.exception("MediaPipe face detector init failed")

        self.load_known_faces()

    # -----------------------------
    # Load known encodings from DB
    # -----------------------------
    def load_known_faces(self):
        with self.lock:
            try:
                employees = (
                    Employee.objects
                    .filter(is_active=True)
                    .exclude(face_encoding__isnull=True)
                    .exclude(face_encoding__exact="")
                )
            except (ProgrammingError, OperationalError):
                logger.warning("DB not ready — skipping face loading")
                self.initialized = False
                return

            encs, ids = [], []
            for emp in employees:
                try:
                    enc = emp.get_face_encoding()
                    if enc is not None:
                        encs.append(enc.astype(np.float32))
                        ids.append(emp.employee_id)
                except Exception:
                    logger.exception("Failed loading encoding for %s", emp)

            if encs:
                self.known_face_encodings = np.vstack(encs)
                self.known_face_employee_ids = ids
                self.initialized = True
                logger.info("Loaded %d face encodings", len(ids))
            else:
                self.known_face_encodings = None
                self.known_face_employee_ids = []
                self.initialized = False

    # -----------------------------
    # Face detection
    # -----------------------------
    def detect_faces(self, frame_bgr):
        if self.mp_detector:
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                res = self.mp_detector.process(rgb)
                if not res.detections:
                    return []

                h, w = frame_bgr.shape[:2]
                boxes = []
                for det in res.detections:
                    bb = det.location_data.relative_bounding_box
                    top = int(bb.ymin * h)
                    left = int(bb.xmin * w)
                    bottom = int((bb.ymin + bb.height) * h)
                    right = int((bb.xmin + bb.width) * w)
                    boxes.append((top, right, bottom, left))
                return boxes
            except Exception:
                logger.exception("MediaPipe detection failed")

        if FR_AVAILABLE:
            try:
                rgb = frame_bgr[:, :, ::-1]
                model = getattr(settings, "FACE_RECOGNITION_MODEL", "hog")
                return face_recognition.face_locations(rgb, model=model)
            except Exception:
                logger.exception("face_recognition detection failed")

        return []

    # -----------------------------
    # Recognize from frame
    # -----------------------------
    def recognize_face_from_frame(self, frame_bgr):
        if not self.initialized or self.known_face_encodings is None:
            return None, 0.0, None

        locs = self.detect_faces(frame_bgr)
        if not locs:
            return None, 0.0, None

        try:
            rgb = frame_bgr[:, :, ::-1]
            encs = face_recognition.face_encodings(rgb, locs)
        except Exception:
            encs = []

        if not encs:
            return None, 0.0, locs[0]

        enc = encs[0].astype(np.float32)

        # ---------- CORRECT MATCHING ----------
        distances = face_recognition.face_distance(
            self.known_face_encodings, enc
        )

        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])

        if best_distance <= self.confidence_threshold:
            confidence = max(0.0, 1.0 - best_distance)
            return self.known_face_employee_ids[best_idx], confidence, locs[0]

        return None, 0.0, locs[0]

    # -----------------------------
    # Recognize from image path
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
# Singleton
# -----------------------------
_face_system_instance = None

def get_face_system():
    global _face_system_instance
    if _face_system_instance is None:
        try:
            _face_system_instance = FaceRecognitionSystem()
        except (ProgrammingError, OperationalError):
            logger.warning("Face system not initialized (DB not ready)")
            return None
    return _face_system_instance


# -----------------------------
# VIEW WRAPPER (USE THIS)
# -----------------------------
def recognize_employee_from_frame(frame_bgr):
    system = get_face_system()
    if system is None:
        return None, 0.0

    emp_id, confidence, _ = system.recognize_face_from_frame(frame_bgr)

    if not emp_id:
        return None, 0.0

    try:
        employee = Employee.objects.select_related("user").get(employee_id=emp_id)
        return employee, float(confidence)
    except Employee.DoesNotExist:
        return None, 0.0
