import cv2
import numpy as np
import time
import threading
import logging
import warnings
import json
import os

from django.conf import settings
from django.db.utils import ProgrammingError, OperationalError
from insightface.app import FaceAnalysis

# Models
from .models import Employee

# Configure Logger
logger = logging.getLogger("core")
logger.setLevel(logging.INFO)   # 🔍 DEBUG ADDED

# =====================================================
# ⚡ Production Configuration
# =====================================================
MODEL_NAME = getattr(settings, "INSIGHTFACE_MODEL", "buffalo_l")
DET_SIZE = getattr(settings, "INSIGHTFACE_DET_SIZE", (640, 640))
MATCH_THRESHOLD = getattr(settings, "FACE_MATCH_THRESHOLD", 0.50)
AMBIGUITY_MARGIN = 0.08 
DB_REFRESH_SECONDS = 60
ACCUMULATOR_TARGET = 1.2 

logger.info("🔧 Face system configuration loaded")  # 🔍 DEBUG ADDED

# =====================================================
# 🛠 Math & Image Helpers
# =====================================================

def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalizes a vector to unit length for Cosine Similarity."""
    if vec is None:
        logger.warning("normalize(): vec is None")  # 🔍 DEBUG ADDED
        return None

    vec = vec.ravel().astype(np.float32)
    norm = np.linalg.norm(vec)

    if norm == 0:
        logger.warning("normalize(): zero-norm vector detected")  # 🔍 DEBUG ADDED

    return vec if norm == 0 else vec / norm


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    """
    try:
        if image is None:
            logger.warning("apply_clahe(): image is None")  # 🔍 DEBUG ADDED
            return None

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    except Exception as e:
        logger.exception(f"apply_clahe() failed: {e}")  # 🔍 DEBUG ADDED
        return image

# =====================================================
# 🧠 Face Recognition System (Singleton)
# =====================================================

class FaceRecognitionSystem:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        logger.debug("FaceRecognitionSystem.__new__ called")  # 🔍 DEBUG ADDED
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.debug("Creating new FaceRecognitionSystem instance")  # 🔍 DEBUG ADDED
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "initialized"):
            logger.debug("FaceRecognitionSystem already initialized")  # 🔍 DEBUG ADDED
            return

        warnings.filterwarnings("ignore", category=FutureWarning)
        logger.info("🚀 Initializing Face Recognition System...")

        self.app = FaceAnalysis(
            name=MODEL_NAME,
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=DET_SIZE)

        logger.info("✅ InsightFace model prepared")  # 🔍 DEBUG ADDED

        self.known_embeddings = np.empty((0, 512), dtype=np.float32)
        self.known_ids = []
        self.last_db_refresh = 0
        
        self.tracker_state = {}
        self.lock = threading.RLock()

        self.initialized = True
        logger.info("✅ Face System Ready.")

    # -------------------------------------------------
    # 1. Registration Helper
    # -------------------------------------------------
    def get_embedding(self, image_input):
        logger.info("get_embedding(): started")  # 🔍 DEBUG ADDED
        try:
            if isinstance(image_input, str):
                logger.debug(f"get_embedding(): image path = {image_input}")  # 🔍 DEBUG ADDED
                if not os.path.exists(image_input):
                    logger.error(f"Image file missing: {image_input}")
                    return None
                img = cv2.imread(image_input)
            else:
                logger.debug("get_embedding(): image array received")  # 🔍 DEBUG ADDED
                img = image_input

            if img is None:
                logger.error("get_embedding(): cv2.imread failed")  # 🔍 DEBUG ADDED
                return None

            faces = self.app.get(img)
            logger.debug(f"get_embedding(): faces detected = {len(faces)}")  # 🔍 DEBUG ADDED
            
            if not faces:
                logger.info("get_embedding(): retrying with CLAHE")  # 🔍 DEBUG ADDED
                img = apply_clahe(img)
                faces = self.app.get(img)

            if not faces:
                logger.warning("get_embedding(): no face detected")  # 🔍 DEBUG ADDED
                return None

            faces.sort(
                key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                reverse=True
            )

            logger.info("get_embedding(): embedding extracted successfully")  # 🔍 DEBUG ADDED
            return normalize(faces[0].embedding).tolist()

        except Exception as e:
            logger.exception(f"get_embedding() crashed: {e}")  # 🔍 DEBUG ADDED
            return None

    # -------------------------------------------------
    # 2. Database Loader
    # -------------------------------------------------
    def load_from_db(self):
        logger.debug("load_from_db(): called")  # 🔍 DEBUG ADDED
        with self.lock:
            if time.time() - self.last_db_refresh < DB_REFRESH_SECONDS and len(self.known_ids) > 0:
                logger.debug("load_from_db(): cache valid, skipping refresh")  # 🔍 DEBUG ADDED
                return

            try:
                qs = Employee.objects.filter(is_active=True).exclude(
                    face_encoding__isnull=True
                ).exclude(face_encoding__exact="")

                logger.info(f"load_from_db(): employees fetched = {qs.count()}")  # 🔍 DEBUG ADDED

                embs, ids = [], []

                for emp in qs:
                    try:
                        raw = emp.face_encoding
                        if isinstance(raw, str):
                            raw = json.loads(raw)

                        if raw and isinstance(raw, list):
                            if isinstance(raw[0], (list, np.ndarray)):
                                for e in raw:
                                    embs.append(normalize(np.array(e)))
                                    ids.append(emp.employee_id)
                            else:
                                embs.append(normalize(np.array(raw)))
                                ids.append(emp.employee_id)

                    except Exception as e:
                        logger.warning(f"load_from_db(): corrupt encoding {emp.employee_id}: {e}")  # 🔍 DEBUG ADDED

                if embs:
                    self.known_embeddings = np.vstack(embs).astype(np.float32)
                    self.known_ids = np.array(ids)
                    self.last_db_refresh = time.time()
                    logger.info(f"✅ DB refreshed: {len(ids)} vectors loaded")
                else:
                    logger.warning("⚠ load_from_db(): DB loaded but empty")  # 🔍 DEBUG ADDED

            except (ProgrammingError, OperationalError) as e:
                logger.error(f"load_from_db(): DB error suppressed: {e}")  # 🔍 DEBUG ADDED

    # -------------------------------------------------
    # 3. Vectorized Identification
    # -------------------------------------------------
    def identify_face(self, emb):
        if emb is None:
            logger.debug("identify_face(): embedding is None")  # 🔍 DEBUG ADDED
            return None, 0.0

        if self.known_embeddings.size == 0:
            logger.debug("identify_face(): no known embeddings loaded")  # 🔍 DEBUG ADDED
            return None, 0.0

        sims = np.dot(self.known_embeddings, emb)
        best_idx = np.argmax(sims)
        best_score = float(sims[best_idx])

        logger.debug(f"identify_face(): best_score = {best_score}")  # 🔍 DEBUG ADDED

        if best_score > MATCH_THRESHOLD:
            if len(sims) > 1:
                sims[best_idx] = -1.0
                second_best = float(np.max(sims))
                if (best_score - second_best) < AMBIGUITY_MARGIN:
                    logger.info("identify_face(): ambiguous match")  # 🔍 DEBUG ADDED
                    return None, best_score
            
            return self.known_ids[best_idx], best_score

        return None, best_score

    # -------------------------------------------------
    # 4. Main Recognition Pipeline
    # -------------------------------------------------
    def recognize_from_frame(self, frame):
        logger.debug("recognize_from_frame(): called")  # 🔍 DEBUG ADDED

        if frame is None:
            logger.warning("recognize_from_frame(): frame is None")  # 🔍 DEBUG ADDED
            return None, 0.0, {"reason": "INVALID_FRAME"}

        self.load_from_db()

        h, w = frame.shape[:2]
        if w > 1280:
            scale = 640 / w
            frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            logger.debug("recognize_from_frame(): frame resized")  # 🔍 DEBUG ADDED

        faces = self.app.get(frame)
        if not faces:
            logger.debug("recognize_from_frame(): retry CLAHE")  # 🔍 DEBUG ADDED
            faces = self.app.get(apply_clahe(frame))

        if not faces:
            logger.info("recognize_from_frame(): NO_FACE")  # 🔍 DEBUG ADDED
            return None, 0.0, {"reason": "NO_FACE"}

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

        if (face.bbox[2]-face.bbox[0]) < 50:
            logger.info("recognize_from_frame(): face too small")  # 🔍 DEBUG ADDED
            return None, 0.0, {"reason": "TOO_FAR"}

        emp_id, score = self.identify_face(normalize(face.embedding))
        bbox = face.bbox.astype(int).tolist()

        if emp_id:
            acc = self.tracker_state.get(emp_id, 0.0) + score
            self.tracker_state[emp_id] = acc

            for k in list(self.tracker_state.keys()):
                if k != emp_id:
                    self.tracker_state[k] = max(0, self.tracker_state[k] - 0.2)

            if acc > ACCUMULATOR_TARGET:
                logger.info(f"recognize_from_frame(): FACE CONFIRMED {emp_id}")  # 🔍 DEBUG ADDED
                self.tracker_state[emp_id] = 0.0
                return emp_id, score, {"bbox": bbox}

            return None, score, {"reason": "ACCUMULATING", "bbox": bbox}

        logger.info("recognize_from_frame(): UNKNOWN face")  # 🔍 DEBUG ADDED
        return None, score, {"reason": "UNKNOWN", "bbox": bbox}


# Factory Pattern
def get_face_system():
    logger.debug("get_face_system(): called")  # 🔍 DEBUG ADDED
    if FaceRecognitionSystem._instance is None:
        FaceRecognitionSystem._instance = FaceRecognitionSystem()
    return FaceRecognitionSystem._instance
