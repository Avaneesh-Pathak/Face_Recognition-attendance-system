import logging
import threading
import numpy as np
import cv2
from django.conf import settings
from .models import Employee

logger = logging.getLogger('core')

# optional libraries
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


# -----------------------------
# Utility Functions
# -----------------------------
def _safe_normalize(arr):
    a = np.asarray(arr, dtype=np.float32).ravel()
    norm = np.linalg.norm(a)
    if norm <= 0 or np.isnan(norm):
        return a
    return (a / norm).astype(np.float32)


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# -----------------------------
# Face Recognition System
# -----------------------------
class FaceRecognitionSystem:
    """
    In-memory face recognition helper:
      - loads embeddings from DB
      - supports MediaPipe detection (optional) or face_recognition
      - uses cosine similarity on normalized vectors
    """

    def __init__(self, confidence_threshold=None):
        self.lock = threading.Lock()
        self.confidence_threshold = float(getattr(settings, "FACE_RECOGNITION_THRESHOLD", 0.55) if confidence_threshold is None else confidence_threshold)
        self.known_face_encodings = None  # numpy array shape (N, D) or None
        self.known_face_employee_ids = []
        self.initialized = False

        # MediaPipe detector (optional)
        self.mp_detector = None
        if MP_AVAILABLE:
            try:
                self.mp_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
            except Exception:
                logger.exception("Failed to initialize MediaPipe face detector")
                self.mp_detector = None

        self.load_known_faces()

    # -----------------------------
    # Load Faces from Database
    # -----------------------------
    def load_known_faces(self):
        """Load all known face encodings into memory (normalized)."""
        with self.lock:
            try:
                employees = Employee.objects.filter(is_active=True).exclude(face_encoding__isnull=True).exclude(face_encoding__exact='')
                encs = []
                ids = []
                for emp in employees:
                    try:
                        enc = emp.get_face_encoding() if hasattr(emp, "get_face_encoding") else None
                        if enc is None:
                            # try parsing stored JSON/list formats
                            raw = emp.face_encoding
                            if isinstance(raw, str):
                                try:
                                    import json
                                    arr = json.loads(raw)
                                    enc = np.asarray(arr, dtype=np.float32)
                                except Exception:
                                    enc = None
                            elif isinstance(raw, (list, tuple, np.ndarray)):
                                enc = np.asarray(raw, dtype=np.float32)
                            elif isinstance(raw, (bytes, bytearray)):
                                try:
                                    enc = np.frombuffer(raw, dtype=np.float32)
                                except Exception:
                                    enc = None
                        if enc is None:
                            continue
                        encs.append(_safe_normalize(enc))
                        ids.append(emp.employee_id)
                    except Exception:
                        logger.exception("Failed parsing embedding for employee %s", getattr(emp, "pk", None))
                        continue

                if encs:
                    self.known_face_encodings = np.vstack(encs).astype(np.float32)
                    self.known_face_employee_ids = ids
                    self.initialized = True
                    logger.info("Loaded %d face encodings (dim=%d)", len(ids), self.known_face_encodings.shape[1])
                else:
                    self.known_face_encodings = None
                    self.known_face_employee_ids = []
                    self.initialized = True
                    logger.info("No face encodings loaded from DB.")
            except Exception:
                logger.exception("Error loading known faces")
                self.known_face_encodings = None
                self.known_face_employee_ids = []
                self.initialized = True

    # -----------------------------
    # Register New Face (image_path) for employee
    # -----------------------------
    def register_face(self, image_path, employee):
        """
        Extract encoding from image_path and save to employee via model helper.
        Updates in-memory cache.
        """
        try:
            if not FR_AVAILABLE:
                return False, "face_recognition library not available."

            img = face_recognition.load_image_file(image_path)
            encs = face_recognition.face_encodings(img)
            if not encs:
                return False, "No face detected in the image."
            if len(encs) > 1:
                return False, "Multiple faces detected. Use a single-face image."

            enc = _safe_normalize(encs[0])
            # save via model helper (expects numpy array)
            if hasattr(employee, "save_face_encoding"):
                employee.save_face_encoding(enc)
            else:
                # fallback: save as JSON
                import json
                employee.face_encoding = json.dumps(enc.tolist())
                employee.save(update_fields=["face_encoding"])

            # update in-memory cache
            with self.lock:
                if self.known_face_encodings is None:
                    self.known_face_encodings = enc.reshape(1, -1).astype(np.float32)
                else:
                    if enc.shape[0] != self.known_face_encodings.shape[1]:
                        # reload DB to attempt fix
                        self.load_known_faces()
                        if self.known_face_encodings is None or enc.shape[0] != self.known_face_encodings.shape[1]:
                            # replace with single
                            self.known_face_encodings = enc.reshape(1, -1).astype(np.float32)
                            self.known_face_employee_ids = [employee.employee_id]
                        else:
                            self.known_face_encodings = np.vstack([self.known_face_encodings, enc.reshape(1, -1)])
                            self.known_face_employee_ids.append(employee.employee_id)
                    else:
                        self.known_face_encodings = np.vstack([self.known_face_encodings, enc.reshape(1, -1)])
                        self.known_face_employee_ids.append(employee.employee_id)

            return True, "Face registered successfully."
        except Exception:
            logger.exception("Error registering face")
            return False, "Internal error registering face."

    # -----------------------------
    # Detect faces (returns list of locations in (top,right,bottom,left) order)
    # -----------------------------
    def detect_faces(self, frame_bgr):
        # prefer MediaPipe detector (fast) if available
        if self.mp_detector is not None:
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                res = self.mp_detector.process(rgb)
                if not getattr(res, "detections", None):
                    return []
                h, w = frame_bgr.shape[:2]
                locs = []
                for det in res.detections:
                    bbox = det.location_data.relative_bounding_box
                    top = max(0, int(bbox.ymin * h))
                    left = max(0, int(bbox.xmin * w))
                    bottom = min(h, int((bbox.ymin + bbox.height) * h))
                    right = min(w, int((bbox.xmin + bbox.width) * w))
                    locs.append((top, right, bottom, left))
                return locs
            except Exception:
                logger.exception("MediaPipe detection failed; falling back")

        # fallback to face_recognition if available
        if FR_AVAILABLE:
            try:
                rgb = frame_bgr[:, :, ::-1]
                model_name = getattr(settings, "FACE_RECOGNITION_MODEL", "hog")
                return face_recognition.face_locations(rgb, model=model_name)
            except Exception:
                logger.exception("face_recognition detection failed")
                return []
        # no detector available
        return []

    # -----------------------------
    # Recognize face(s) from frame
    # -----------------------------
    def recognize_face_from_frame(self, frame_bgr):
        """
        Returns tuple: (employee_id_or_None, score_float, face_location_or_None)
        """
        try:
            if not self.initialized or not self.known_face_employee_ids or self.known_face_encodings is None:
                logger.debug("No known faces loaded; skipping recognition")
                return None, 0.0, None

            locs = self.detect_faces(frame_bgr)
            if not locs:
                return None, 0.0, None

            # compute encodings for first detected face
            if FR_AVAILABLE:
                try:
                    rgb = frame_bgr[:, :, ::-1]
                    encs = face_recognition.face_encodings(rgb, locs)
                except Exception:
                    logger.exception("face_recognition encoding failed")
                    encs = []
            else:
                encs = []

            if not encs:
                return None, 0.0, locs[0]

            enc = _safe_normalize(encs[0])

            # ensure dimension match
            db = self.known_face_encodings
            if enc.shape[0] != db.shape[1]:
                logger.warning("Embedding dimension mismatch; reloading DB")
                self.load_known_faces()
                db = self.known_face_encodings
                if db is None or enc.shape[0] != db.shape[1]:
                    return None, 0.0, locs[0]

            sims = np.dot(db, enc)  # vectorized dot (both sides normalized)
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            if best_score >= self.confidence_threshold:
                return self.known_face_employee_ids[best_idx], best_score, locs[0]
            return None, best_score, locs[0]
        except Exception:
            logger.exception("Recognition from frame failed")
            return None, 0.0, None

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
        _face_system_instance = FaceRecognitionSystem()
    return _face_system_instance
