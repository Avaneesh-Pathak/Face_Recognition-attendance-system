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
logger.setLevel(logging.INFO)   ##

# =====================================================
# ⚡ Production Configuration
# =====================================================
MODEL_NAME = getattr(settings, "INSIGHTFACE_MODEL", "buffalo_l")
DET_SIZE = getattr(settings, "INSIGHTFACE_DET_SIZE", (640, 640))
MATCH_THRESHOLD = getattr(settings, "FACE_MATCH_THRESHOLD", 0.50)
AMBIGUITY_MARGIN = 0.08 
DB_REFRESH_SECONDS = 60
ACCUMULATOR_TARGET = 1.2 


# =====================================================
# 🛠 Math & Image Helpers
# =====================================================

def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalizes a vector to unit length for Cosine Similarity."""
    if vec is None:
        logger.warning("normalize(): vec is None")  ##
        return None

    vec = vec.ravel().astype(np.float32)
    norm = np.linalg.norm(vec)

    if norm == 0:
        logger.warning("normalize(): zero-norm vector detected")  ##

    return vec if norm == 0 else vec / norm


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    """
    try:
        if image is None:
            logger.warning("apply_clahe(): image is None")  ##
            return None

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    except Exception as e:
        logger.exception(f"apply_clahe() failed: {e}")  ##
        return image

# =====================================================
# 🧠 Face Recognition System (Singleton)
# =====================================================

class FaceRecognitionSystem:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        logger.debug("FaceRecognitionSystem.__new__ called")  ##
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.debug("Creating new FaceRecognitionSystem instance")  ##
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "initialized"):
            logger.debug("FaceRecognitionSystem already initialized")  ##
            return

        warnings.filterwarnings("ignore", category=FutureWarning)
        logger.info("Initializing Face Recognition System...")

        self.app = FaceAnalysis(
            name=MODEL_NAME,
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=DET_SIZE)

        logger.info("InsightFace model prepared")  ##

        self.known_embeddings = {}  # {organisation_id: np.ndarray}
        self.known_ids = {}         # {organisation_id: np.array}
        self.last_db_refresh = {}   # {organisation_id: float}
        
        self.tracker_state = {}
        self.lock = threading.RLock()

        self.initialized = True
        logger.info("Face System Ready.")

    # -------------------------------------------------
    # 1. Registration Helper
    # -------------------------------------------------
    def get_embedding(self, image_input):
        logger.info("get_embedding(): started")  ##
        try:
            if isinstance(image_input, str):
                logger.debug(f"get_embedding(): image path = {image_input}")  ##
                if not os.path.exists(image_input):
                    logger.error(f"Image file missing: {image_input}")
                    return None
                img = cv2.imread(image_input)
            else:
                logger.debug("get_embedding(): image array received")  ##
                img = image_input

            if img is None:
                logger.error("get_embedding(): cv2.imread failed")  ##
                return None

            faces = self.app.get(img)
            logger.debug(f"get_embedding(): faces detected = {len(faces)}")  ##
            
            if not faces:
                logger.info("get_embedding(): retrying with CLAHE")  ##
                img = apply_clahe(img)
                faces = self.app.get(img)

            if not faces:
                logger.warning("get_embedding(): no face detected")  ##
                return None

            faces.sort(
                key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                reverse=True
            )

            logger.info("get_embedding(): embedding extracted successfully")  ##
            return normalize(faces[0].embedding).tolist()

        except Exception as e:
            logger.exception(f"get_embedding() crashed: {e}")  ##
            return None

    # -------------------------------------------------
    # 2. Database Loader
    # -------------------------------------------------
    def load_from_db(self,organisation_id):
        logger.debug("load_from_db(): called")  ##
        with self.lock:
            now = time.time()
            # If loaded within refresh window and cache is non-empty, use existing data
            if (now - self.last_db_refresh.get(organisation_id, 0) < DB_REFRESH_SECONDS 
                    and organisation_id in self.known_ids):
                return

            try:
                qs = Employee.objects.filter(organisation_id=organisation_id,is_active=True).exclude(
                    face_encoding__isnull=True
                ).exclude(face_encoding__exact="")

                logger.info(f"load_from_db(): employees fetched = {qs.count()}")  ##

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
                        logger.warning(f"load_from_db(): corrupt encoding {emp.employee_id}: {e}")  ##

                if embs:
                    self.known_embeddings[organisation_id] = np.vstack(embs).astype(np.float32)
                    self.known_ids[organisation_id] = np.array(ids)
                    self.last_db_refresh[organisation_id] = time.time()
                    logger.info(f"Refreshed tenant {organisation_id}: {len(ids)} vectors loaded")
                else:
                    self.known_embeddings[organisation_id] = np.empty((0, 512), dtype=np.float32)
                    self.known_ids[organisation_id] = np.array([])
                    logger.warning("⚠ load_from_db(): DB loaded but empty")  ##

            except (ProgrammingError, OperationalError) as e:
                logger.error(f"load_from_db(): DB error suppressed: {e}")  ##

    # -------------------------------------------------
    # 3. Vectorized Identification
    # -------------------------------------------------
    def identify_face(self, emb, organisation_id):
        """Identifies a face strictly within the boundary vectors of a single organisation."""
        if emb is None:
            return None, 0.0

        org_embs = self.known_embeddings.get(organisation_id)
        org_ids = self.known_ids.get(organisation_id)

        if org_embs is None or org_embs.size == 0:
            return None, 0.0

        sims = np.dot(org_embs, emb)
        best_idx = np.argmax(sims)
        best_score = float(sims[best_idx])

        if best_score > MATCH_THRESHOLD:
            return org_ids[best_idx], best_score

        return None, best_score

    # -------------------------------------------------
    # 4. Main Recognition Pipeline
    # -------------------------------------------------
    def recognize_from_frame(self, frame, organisation_id):
        logger.debug("recognize_from_frame(): called")  ##

        if frame is None:
            logger.warning("recognize_from_frame(): frame is None")  ##
            return None, 0.0, {"reason": "INVALID_FRAME"}

        self.load_from_db(organisation_id)

        h, w = frame.shape[:2]
        if w > 1280:
            scale = 640 / w
            frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            logger.debug("recognize_from_frame(): frame resized")  ##

        faces = self.app.get(frame)
        if not faces:
            logger.debug("recognize_from_frame(): retry CLAHE")  ##
            faces = self.app.get(apply_clahe(frame))

        if not faces:
            logger.info("recognize_from_frame(): NO_FACE")  ##
            return None, 0.0, {"reason": "NO_FACE"}

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

        if (face.bbox[2]-face.bbox[0]) < 50:
            logger.info("recognize_from_frame(): face too small")  ##
            return None, 0.0, {"reason": "TOO_FAR"}

        emp_id, score = self.identify_face(normalize(face.embedding),organisation_id)
        bbox = face.bbox.astype(int).tolist()

        if emp_id:
            acc = self.tracker_state.get(emp_id, 0.0) + score
            self.tracker_state[emp_id] = acc

            for k in list(self.tracker_state.keys()):
                if k != emp_id:
                    self.tracker_state[k] = max(0, self.tracker_state[k] - 0.2)

            if acc > ACCUMULATOR_TARGET:
                logger.info(f"recognize_from_frame(): FACE CONFIRMED {emp_id}")  ##
                self.tracker_state[emp_id] = 0.0
                return emp_id, score, {"bbox": bbox}

            return None, score, {"reason": "ACCUMULATING", "bbox": bbox}

        logger.info("recognize_from_frame(): UNKNOWN face")  ##
        return None, score, {"reason": "UNKNOWN", "bbox": bbox}


# Factory Pattern
def get_face_system():
    logger.debug("get_face_system(): called")  ##
    if FaceRecognitionSystem._instance is None:
        FaceRecognitionSystem._instance = FaceRecognitionSystem()
    return FaceRecognitionSystem._instance
