import cv2
import numpy as np
import time
import threading
import logging
import warnings

from django.conf import settings
from django.db.utils import ProgrammingError, OperationalError
from django.utils import timezone
from datetime import timedelta

from insightface.app import FaceAnalysis

from .models import (
    Employee,
    AttendanceMatchLog,
    FaceMatchStats,
    EmployeeThreshold,
)
from .anti_spoof import LivenessGuard
from .trackers import IOUTracker

logger = logging.getLogger("core")

# =====================================================
# Utility helpers
# =====================================================
def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).ravel()
    n = np.linalg.norm(vec)
    return vec if n == 0 else vec / n


def confidence_level(cos, euc):
    if cos > 0.80 and euc < 1.30:
        return "HIGH"
    if cos > 0.60 and euc < 1.50:
        return "MED"
    return "LOW"


def recently_marked(emp_id, minutes=5):
    cutoff = timezone.now() - timedelta(minutes=minutes)

    exists = AttendanceMatchLog.objects.filter(
        employee_id=emp_id,
        created_at__gte=cutoff
    ).exists()

    if exists:
        logger.info(
            "recently_marked=True for emp_id=%s (within last %d minutes)",
            emp_id,
            minutes
        )
    else:
        logger.debug(
            "recently_marked=False for emp_id=%s",
            emp_id
        )

    return exists

def log_false_reject(emp_id):
    stats, _ = FaceMatchStats.objects.get_or_create(employee_id=emp_id)
    stats.false_rejects += 1
    stats.total_attempts += 1
    stats.save()


def get_employee_threshold(emp_id):
    obj, _ = EmployeeThreshold.objects.get_or_create(employee_id=emp_id)
    return obj.cos_max, obj.euc_max


def update_employee_threshold(emp_id, cos, euc):
    obj, _ = EmployeeThreshold.objects.get_or_create(employee_id=emp_id)
    obj.cos_max = max(obj.cos_max, cos * 0.98)
    obj.euc_max = min(obj.euc_max, euc * 1.02)
    obj.save()


def frame_quality(frame, bbox):
    x1, y1, x2, y2 = bbox
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return 0.0

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    size_score = (x2 - x1) * (y2 - y1)

    if blur < 30 or size_score < 2500:
        return 0.2
    return 1.0


# =====================================================
# Face Recognition System
# =====================================================
class FaceRecognitionSystem:
    def __init__(self):
        logger.info("Initializing FaceRecognitionSystem")
        self.lock = threading.Lock()

        warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

        self.app = FaceAnalysis(
            name=getattr(settings, "INSIGHTFACE_MODEL", "buffalo_l")
        )
        self.app.prepare(
            ctx_id=int(getattr(settings, "INSIGHTFACE_CTX_ID", -1)),
            det_size=tuple(getattr(settings, "INSIGHTFACE_DET_SIZE", (640, 640)))
        )

        self.known_ids = []
        self.known_embeddings = np.empty((0, 512), dtype=np.float32)

        self.tracker = IOUTracker(max_age=10, iou_threshold=0.3)
        self.liveness = LivenessGuard()

        self.last_det_ts = 0.0
        self.min_interval = float(getattr(settings, "FR_MIN_DET_INTERVAL", 0.35))

        # 🔒 Temporal smoothing buffer
        self.recent_ids = []
        self.max_votes = 3
        self.required_votes = 1

        self.load_from_db()

    # =====================================================
    # Registration helper
    # =====================================================
    def get_embedding(self, image_path: str):
        logger.debug("Extracting embedding from image: %s", image_path)

        img = cv2.imread(image_path)
        if img is None:
            logger.warning("Image not readable")
            return None

        faces = self.app.get(img)
        if not faces:
            logger.warning("No face found in registration image")
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
            logger.warning("Embedding extraction failed")
            return None

        return normalize(emb).astype(np.float32)
    
    # =====================================================
    # Load DB embeddings
    # =====================================================
    def load_from_db(self):
        logger.info("Loading face embeddings from database")
        with self.lock:
            try:
                qs = (
                    Employee.objects
                    .filter(is_active=True)
                    .exclude(face_encoding__isnull=True)
                    .exclude(face_encoding__exact="")
                )
            except (ProgrammingError, OperationalError):
                logger.warning("DB not ready")
                return

            embs, ids = [], []
            for emp in qs:
                enc = emp.get_face_encoding()
                if enc is not None:
                    embs.append(normalize(enc))
                    ids.append(emp.employee_id)

            if embs:
                self.known_embeddings = np.vstack(embs)
                self.known_ids = ids
                logger.info("Loaded %d embeddings", len(ids))

    # =====================================================
    # Detect faces
    # =====================================================
    def detect_faces(self, frame_bgr):
        logger.debug("Running InsightFace detection")

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

        logger.debug("Faces detected: %d", len(results))
        return results


    # =====================================================
    # Matching logic (CRITICAL)
    # =====================================================
    def match(self, emb: np.ndarray):
        if self.known_embeddings.size == 0:
            return None, 0.0, "NO_DB"

        emb = normalize(emb)

        sims = np.dot(self.known_embeddings, emb)
        dists = np.linalg.norm(self.known_embeddings - emb, axis=1)

        order = np.argsort(-sims)
        best = order[0]
        second = order[1] if len(order) > 1 else None

        best_id = self.known_ids[best]
        best_cos = float(sims[best])
        best_euc = float(dists[best])

        if second is not None:
            if best_cos - float(sims[second]) < 0.05:
                return None, best_cos, "AMBIGUOUS"

        conf = confidence_level(best_cos, best_euc)

        emp_cos_thr, emp_euc_thr = get_employee_threshold(best_id)

        if emp_cos_thr == 0:
            emp_cos_thr = 0.65
        if emp_euc_thr == 0:
            emp_euc_thr = 1.50

        self.recent_ids.append(best_id)
        self.recent_ids = self.recent_ids[-self.max_votes:]

        if self.recent_ids.count(best_id) < self.required_votes:
            return None, best_cos, "UNSTABLE"

        update_employee_threshold(best_id, best_cos, best_euc)

        AttendanceMatchLog.objects.create(
            employee_id=best_id,
            cosine=best_cos,
            euclidean=best_euc,
            confidence=conf,
        )

        return best_id, best_cos, conf

    # =====================================================
    # Live recognition
    # =====================================================
    def recognize_from_frame(self, frame_bgr):
        start = time.time()

        h, w = frame_bgr.shape[:2]
        if w > 640:
            frame_bgr = cv2.resize(frame_bgr, (640, int(h * 640 / w)))

        faces = self.detect_faces(frame_bgr)
        if not faces:
            return None, 0.0, None

        face = max(faces, key=lambda f: f["score"])
        live, reason = self.liveness.check(frame_bgr, face["bbox"])
        if not live:
            return None, 0.0, None

        if frame_quality(frame_bgr, face["bbox"]) < 0.5:
            return None, 0.0, None

        emp_id, score, reason = self.match(face["embedding"])

        logger.info("Processing time: %.1f ms", (time.time() - start) * 1000)

        

        return emp_id, score, {
            "bbox": face["bbox"],
            "confidence": score,
            "reason": reason,
        }



# =====================================================
# Singleton
# =====================================================
_instance = None

def get_face_system():
    global _instance
    if _instance is None:
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
        emp = Employee.objects.select_related("user").get(employee_id=emp_id)
        return emp, float(score)
    except Employee.DoesNotExist:
        return None, 0.0
