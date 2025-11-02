import threading
import numpy as np
import cv2
import os
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

# InsightFace
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except Exception as e:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("insightface not available: %s", e)


# --------------------------
# Helpers
# --------------------------
def _normalize(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32).ravel()
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.astype(np.float32)
    return (v / norm).astype(np.float32)


# --------------------------
# FaceRecognitionSystem (InsightFace: RetinaFace + ArcFace)
# --------------------------
class FaceRecognitionSystem:
    """
    Uses insightface FaceAnalysis for detection+embedding.
    Maintains an in-memory normalized embedding matrix and corresponding ids.
    This implementation is defensive: handles missing insightface, varying embedding sizes,
    and attempts to reload from DB on shape mismatch.
    """

    def __init__(self, model_name: str = "buffalo_l", det_size=(640, 640), confidence_threshold: float = 0.60):
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("insightface is required but not installed.")
        self.lock = threading.Lock()
        self.model_name = model_name
        self.det_size = det_size
        self.confidence_threshold = float(confidence_threshold)

        # Initialize insightface app
        try:
            self.app = FaceAnalysis(name=self.model_name)
            # allow override of ctx_id in settings; default to CPU (-1)
            ctx_id = getattr(settings, "INSIGHTFACE_CTX_ID", -1)
            self.app.prepare(ctx_id=ctx_id, det_size=self.det_size)
            logger.info("InsightFace initialized (model=%s, det_size=%s, ctx_id=%s)", self.model_name, self.det_size, ctx_id)
        except Exception as ex:
            logger.exception("Failed to initialize InsightFace FaceAnalysis: %s", ex)
            raise

        # in-memory store: start empty with zero columns (will set on first enc)
        self.known_ids = []                 # list of employee_id
        self.known_embeddings = np.empty((0, 0), dtype=np.float32)  # dynamic width
        self.initialized = False

        # lazy load from DB
        self.load_from_db()

    # --------------------------
    # load all embeddings from Django DB
    # --------------------------
    def load_from_db(self):
        from .models import Employee
        with self.lock:
            encs = []
            ids = []
            try:
                qs = Employee.objects.filter(is_active=True).exclude(face_encoding__isnull=True).exclude(face_encoding__exact='')
                for emp in qs:
                    enc = None
                    # prefer model helper if available
                    try:
                        enc = emp.get_face_encoding()
                    except Exception:
                        # fallback: try to parse JSON/list stored in face_encoding
                        try:
                            import json
                            raw = emp.face_encoding
                            if isinstance(raw, str):
                                arr = json.loads(raw)
                                enc = np.array(arr, dtype=np.float32)
                        except Exception:
                            enc = None

                    if enc is None:
                        continue

                    encn = _normalize(enc)
                    encs.append(encn)
                    ids.append(emp.employee_id)

                if encs:
                    # stack to a matrix of shape (N, D)
                    self.known_embeddings = np.vstack(encs).astype(np.float32)
                    self.known_ids = ids
                else:
                    # empty matrix with zero columns
                    self.known_embeddings = np.empty((0, 0), dtype=np.float32)
                    self.known_ids = []

                self.initialized = True
                logger.info("FaceSystem loaded %d embeddings (dim=%s)", len(self.known_ids),
                            self.known_embeddings.shape[1] if self.known_embeddings.size else 0)
            except Exception as e:
                logger.exception("Error loading embeddings from DB: %s", e)
                # keep initialized flag to avoid repeated noisy attempts
                self.initialized = True

    # --------------------------
    # Extract embedding directly from an image path
    # --------------------------
    def get_embedding(self, image_path: str):
        """
        Load an image from disk and return the first detected face embedding (normalized).
        Returns None if no face is found.
        """
        if not os.path.exists(image_path):
            logger.debug("get_embedding: image not found -> %s", image_path)
            return None

        img = cv2.imread(image_path)
        if img is None:
            logger.debug("get_embedding: failed to read image -> %s", image_path)
            return None

        faces = self.analyze_frame(img)
        if not faces:
            logger.debug("get_embedding: no face detected in %s", image_path)
            return None

        # Use first (highest det_score) face
        faces.sort(key=lambda x: x["det_score"], reverse=True)
        embedding = faces[0]["embedding"]
        return _normalize(embedding)

    # --------------------------
    # Append new encoding (avoid full reload)
    # --------------------------
    def append_embedding(self, employee_id: str, embedding: np.ndarray):
        """Append single normalized embedding to memory after registration."""
        emb = _normalize(embedding).astype(np.float32)
        with self.lock:
            try:
                if self.known_embeddings.size == 0:
                    # first embedding -> set shape accordingly
                    self.known_embeddings = emb.reshape(1, -1)
                else:
                    # verify dimension matches
                    if emb.shape[0] != self.known_embeddings.shape[1]:
                        logger.warning("Embedding dimension mismatch: new=%d current=%d; reloading DB",
                                       emb.shape[0], self.known_embeddings.shape[1])
                        # try reloading DB to attempt fix
                        self.load_from_db()
                        # if still mismatch, reset to only this embedding
                        if self.known_embeddings.size == 0 or emb.shape[0] != self.known_embeddings.shape[1]:
                            self.known_embeddings = emb.reshape(1, -1)
                        else:
                            self.known_embeddings = np.vstack([self.known_embeddings, emb.reshape(1, -1)])
                    else:
                        self.known_embeddings = np.vstack([self.known_embeddings, emb.reshape(1, -1)])
                self.known_ids.append(employee_id)
            except Exception:
                logger.exception("Failed appending embedding for %s", employee_id)

    # --------------------------
    # Extract faces+embeddings from frame using insightface
    # --------------------------
    def analyze_frame(self, frame_bgr):
        """
        Return list of detected faces with fields:
        - bbox: [x1,y1,x2,y2] (int)
        - kps: landmarks
        - embedding: numpy array (normalized)
        - det_score: float
        """
        if getattr(self, "app", None) is None:
            logger.debug("analyze_frame called but insightface app is not initialized")
            return []

        try:
            # Convert to RGB for InsightFace
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            faces = self.app.get(img)
            results = []

            for f in faces:
                try:
                    bbox = np.array(getattr(f, "bbox", [])).astype(int).tolist()
                except Exception:
                    bbox = []

                # ✅ Explicitly handle embedding retrieval
                emb = getattr(f, "normed_embedding", None)
                if emb is None:
                    emb = getattr(f, "embedding", None)

                if emb is None:
                    continue

                res = {
                    "bbox": bbox,
                    "kps": getattr(f, "kps", None),
                    "embedding": _normalize(emb),
                    "det_score": float(getattr(f, "det_score", 0.0))
                }
                results.append(res)

            return results

        except Exception as ex:
            logger.exception("analyze_frame error: %s", ex)
            return []


    # --------------------------
    # Recognize best match for single face embedding
    # --------------------------
    def match_embedding(self, emb: np.ndarray, return_score: bool = True):
        """Vectorized cosine match. Returns (employee_id or None, score)."""
        with self.lock:
            if self.known_embeddings.size == 0:
                return None, 0.0
            embn = _normalize(emb).astype(np.float32)
            # if embedding dimension mismatch, attempt reload
            if embn.shape[0] != self.known_embeddings.shape[1]:
                logger.warning("Embedding dimension mismatch (match): emb=%d db=%d; reloading DB", embn.shape[0],
                               self.known_embeddings.shape[1])
                # try to reload; if still mismatch return no match
                self.load_from_db()
                if self.known_embeddings.size == 0 or embn.shape[0] != self.known_embeddings.shape[1]:
                    return None, 0.0
            # since both sides normalized -> dot product = cosine similarity in [-1,1]
            sims = np.dot(self.known_embeddings, embn)   # shape (N,)
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            if best_score >= self.confidence_threshold:
                return self.known_ids[best_idx], best_score
            return None, best_score

    # --------------------------
    # Recognize from frame (first face only, convenience)
    # --------------------------
    def recognize_from_frame(self, frame_bgr):
        """
        Detect faces and match the first (largest) face.
        Returns tuple: (employee_id or None, score, bbox)
        """
        faces = self.analyze_frame(frame_bgr)
        if not faces:
            return None, 0.0, None

        # choose face with highest det_score or largest bbox area
        faces.sort(key=lambda x: (x.get("det_score", 0.0),
                                  (x.get("bbox", [0, 0, 0, 0])[2] - x.get("bbox", [0, 0, 0, 0])[0]) *
                                  (x.get("bbox", [0, 0, 0, 0])[3] - x.get("bbox", [0, 0, 0, 0])[1])),
                   reverse=True)
        top = faces[0]
        employee_id, score = self.match_embedding(top["embedding"])
        # convert bbox from [x1,y1,x2,y2] to (top,right,bottom,left) for drawing compatibility if bbox present
        bbox = top.get("bbox") or []
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox
            face_loc = (y1, x2, y2, x1)
        else:
            face_loc = None
        return employee_id, float(score), face_loc


# --------------------------
# Singleton accessor
# --------------------------
_instance = None


def get_face_system():
    global _instance
    if _instance is None:
        _instance = FaceRecognitionSystem(
            model_name=getattr(settings, "INSIGHTFACE_MODEL", "buffalo_l"),
            det_size=getattr(settings, "INSIGHTFACE_DET_SIZE", (640, 640)),
            confidence_threshold=getattr(settings, "FACE_RECOGNITION_THRESHOLD", 0.60),
        )
    return _instance