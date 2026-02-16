import cv2
import numpy as np
import time
import threading
import logging
import warnings

from django.conf import settings
from django.db.utils import ProgrammingError, OperationalError
from insightface.app import FaceAnalysis

from .models import Employee
from .anti_spoof import LivenessGuard
from .trackers import IOUTracker

logger = logging.getLogger("core")


# =====================================================
# Utility
# =====================================================
def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).ravel()
    n = np.linalg.norm(vec)
    return vec if n == 0 else vec / n


# =====================================================
# Face Recognition System
# =====================================================
class FaceRecognitionSystem:
    def __init__(self):
        logger.info("🚀 Initializing FaceRecognitionSystem")
        self.lock = threading.Lock()

        # Silence InsightFace warnings
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module="insightface"
        )

        # -------------------------------------------------
        # InsightFace (CPU-safe)
        # -------------------------------------------------
        self.app = FaceAnalysis(
            name=getattr(settings, "INSIGHTFACE_MODEL", "buffalo_l")
        )
        self.app.prepare(
            ctx_id=int(getattr(settings, "INSIGHTFACE_CTX_ID", -1)),
            det_size=tuple(getattr(settings, "INSIGHTFACE_DET_SIZE", (640, 640)))
        )

        logger.info("✅ InsightFace ready")

        # -------------------------------------------------
        # Cached face DB
        # -------------------------------------------------
        self.known_ids: list[str] = []
        self.known_embeddings = np.empty((0, 512), dtype=np.float32)

        # Helpers
        self.tracker = IOUTracker(max_age=10, iou_threshold=0.3)
        self.liveness = LivenessGuard()

        self.last_det_ts = 0.0
        self.min_interval = float(
            getattr(settings, "FR_MIN_DET_INTERVAL", 0.35)
        )

        self.load_from_db()

    # =====================================================
    # Load encodings from DB
    # =====================================================
    def load_from_db(self):
        logger.info("📥 Loading face embeddings from database")

        with self.lock:
            try:
                qs = (
                    Employee.objects
                    .filter(is_active=True)
                    .exclude(face_encoding__isnull=True)
                    .exclude(face_encoding__exact="")
                )
            except (ProgrammingError, OperationalError):
                logger.warning("⚠️ Database not ready — skipping face load")
                return

            embs, ids = [], []

            for emp in qs:
                try:
                    enc = emp.get_face_encoding()
                    if enc is not None:
                        embs.append(normalize(enc))
                        ids.append(emp.employee_id)
                        logger.debug("➕ Loaded encoding: %s", emp.employee_id)
                except Exception:
                    logger.exception("❌ Failed loading encoding for %s", emp)

            if embs:
                self.known_embeddings = np.vstack(embs).astype(np.float32)
                self.known_ids = ids
                logger.info("✅ Loaded %d face embeddings", len(ids))
            else:
                self.known_embeddings = np.empty((0, 512), dtype=np.float32)
                self.known_ids = []
                logger.warning("⚠️ No face embeddings found")

    # =====================================================
    # Detect faces + embeddings
    # =====================================================
    def detect_faces(self, frame_bgr):
        logger.debug("👁️ Running InsightFace detection")

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)

        results = []
        for f in faces:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = getattr(f, "embedding", None)
            if emb is None:
                continue

            results.append({
                "bbox": list(map(int, f.bbox)),
                "embedding": normalize(emb),
                "score": float(f.det_score)
            })

        logger.debug("📦 Faces detected: %d", len(results))
        return results

    # =====================================================
    # Match embedding
    # =====================================================
    def match(self, emb: np.ndarray):
        if self.known_embeddings.size == 0:
            logger.warning("⚠️ No known embeddings loaded")
            return None, 0.0

        emb = normalize(emb)

        sims = np.dot(self.known_embeddings, emb)
        best_idx = int(np.argmax(sims))
        best_cos = float(sims[best_idx])
        best_id = self.known_ids[best_idx]

        euc = float(np.linalg.norm(self.known_embeddings[best_idx] - emb))
        euc_max = float(getattr(settings, "FACE_EUCLIDEAN_MAX", 1.15))
        cos_min = float(getattr(settings, "FACE_RECOGNITION_THRESHOLD", 0.60))

        logger.debug(
            "🔎 Match result → id=%s cos=%.3f euc=%.3f",
            best_id, best_cos, euc
        )

        if best_cos >= cos_min and euc <= euc_max:
            return best_id, best_cos

        return None, best_cos

    # =====================================================
    # Registration helper
    # =====================================================
    def get_embedding(self, image_path: str):
        logger.debug("📝 Extracting embedding from image: %s", image_path)

        img = cv2.imread(image_path)
        if img is None:
            logger.warning("❌ Image not readable")
            return None

        faces = self.app.get(img)
        if not faces:
            logger.warning("❌ No face found in registration image")
            return None

        faces.sort(
            key=lambda f: (
                f.det_score,
                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            ),
            reverse=True
        )

        emb = getattr(faces[0], "normed_embedding", None)
        if emb is None:
            emb = getattr(faces[0], "embedding", None)

        if emb is None:
            logger.warning("❌ Embedding extraction failed")
            return None

        logger.info("✅ Registration embedding extracted")
        return normalize(emb).astype(np.float32)

    # =====================================================
    # Live recognition (FAST PATH)
    # =====================================================
    def recognize_from_frame(self, frame_bgr):
        start = time.time()
        logger.debug("🎥 recognize_from_frame start")

        # Resize once for speed
        h, w = frame_bgr.shape[:2]
        if w > 640:
            scale = 640 / w
            frame_bgr = cv2.resize(frame_bgr, (640, int(h * scale)))
            logger.debug("⚡ Frame resized for speed")

        now = time.time()

        # Detection throttle
        if now - self.last_det_ts >= self.min_interval or self.tracker.empty:
            faces = self.detect_faces(frame_bgr)
            self.tracker.update([f["bbox"] for f in faces])
            self.last_det_ts = now
        else:
            faces = self.detect_faces(frame_bgr)

        if not faces:
            logger.debug("❌ No faces detected")
            return None, 0.0, None

        faces = [f for f in faces if f["score"] >= 0.60]
        if not faces:
            logger.debug("❌ Faces below confidence threshold")
            return None, 0.0, None

        face = max(
            faces,
            key=lambda f: f["score"] *
            ((f["bbox"][2] - f["bbox"][0]) *
             (f["bbox"][3] - f["bbox"][1]))
        )

        # -------------------------------------------------
        # Liveness check (WITH MESSAGE)
        # -------------------------------------------------
        live, reason = self.liveness.check(frame_bgr, face["bbox"])
        if not live:
            logger.warning("🚫 Liveness failed: %s", reason)
            return None, 0.0, {
                "type": "liveness",
                "reason": reason or "Please blink or move your head"
            }

        # -------------------------------------------------
        # Match
        # -------------------------------------------------
        emp_id, score = self.match(face["embedding"])

        logger.info(
            "⏱ Recognition completed in %.1f ms",
            (time.time() - start) * 1000
        )

        return emp_id, float(score), face["bbox"]


# =====================================================
# Singleton
# =====================================================
_instance = None


def get_face_system():
    global _instance
    if _instance is None:
        logger.info("🧠 Creating FaceRecognitionSystem singleton")
        _instance = FaceRecognitionSystem()
    return _instance


# =====================================================
# View helper
# =====================================================
def recognize_employee_from_frame(frame_bgr):
    system = get_face_system()
    emp_id, score, _ = system.recognize_from_frame(frame_bgr)

    if not emp_id:
        return None, 0.0

    try:
        employee = Employee.objects.select_related("user").get(employee_id=emp_id)
        return employee, float(score)
    except Employee.DoesNotExist:
        return None, 0.0
    


    
