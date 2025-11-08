
import threading
import numpy as np
import cv2
import os
import logging
import time
from typing import List, Dict, Tuple, Optional
from django.conf import settings

logger = logging.getLogger(__name__)

# InsightFace
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except Exception as e:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("insightface not available: %s", e)

# Local helpers
from core.anti_spoof import LivenessGuard
from .trackers import IOUTracker

def _normalize(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32).ravel()
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.astype(np.float32)
    return (v / norm).astype(np.float32)


class FaceRecognitionSystem:
    """
    InsightFace-based detector+embedder with:
      - Lightweight IOU tracker to avoid re-detecting every frame
      - Liveness guard (heuristic) to reduce spoofing
      - In-memory DB of normalized embeddings
      - Hybrid scoring (cosine + euclidean) in match_embedding
      - Graceful fallbacks when models/dimensions mismatch
    """
    def __init__(self, model_name: str = "buffalo_l", det_size=(640, 640), confidence_threshold: float = 0.60):
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("insightface is required but not installed.")
        self.lock = threading.Lock()
        self.model_name = model_name
        self.det_size = det_size
        self.confidence_threshold = float(confidence_threshold)

        # insightface
        try:
            self.app = FaceAnalysis(name=self.model_name)
            ctx_id = getattr(settings, "INSIGHTFACE_CTX_ID", -1)
            self.app.prepare(ctx_id=ctx_id, det_size=self.det_size)
            logger.info("InsightFace initialized (model=%s, det_size=%s, ctx_id=%s)", self.model_name, self.det_size, ctx_id)
        except Exception as ex:
            logger.exception("Failed to initialize InsightFace FaceAnalysis: %s", ex)
            raise

        # stores
        self.known_ids: List[str] = []
        self.known_embeddings = np.empty((0, 0), dtype=np.float32)
        self.initialized = False

        # add-ons
        self.tracker = IOUTracker(max_age=10, iou_threshold=0.3)
        self.liveness = LivenessGuard()

        # perf
        self.min_det_interval = float(getattr(settings, "FR_MIN_DET_INTERVAL", 0.25))  # seconds between heavy detect
        self._last_det_ts = 0.0

        self.load_from_db()

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
                    try:
                        enc = emp.get_face_encoding()
                    except Exception:
                        try:
                            import json
                            raw = emp.face_encoding
                            if isinstance(raw, str):
                                obj = json.loads(raw)
                                arr = np.asarray(obj, dtype=np.float32)
                                if arr.ndim == 2:
                                    arr = arr.mean(axis=0)
                                enc = arr
                        except Exception:
                            enc = None
                    if enc is None:
                        continue
                    encn = _normalize(enc)
                    encs.append(encn)
                    ids.append(emp.employee_id)

                if encs:
                    self.known_embeddings = np.vstack(encs).astype(np.float32)
                    self.known_ids = ids
                else:
                    self.known_embeddings = np.empty((0, 0), dtype=np.float32)
                    self.known_ids = []

                self.initialized = True
                logger.info("FaceSystem loaded %d embeddings (dim=%s)", len(self.known_ids),
                            self.known_embeddings.shape[1] if self.known_embeddings.size else 0)
            except Exception as e:
                logger.exception("Error loading embeddings from DB: %s", e)
                self.initialized = True

    # --------------------------
    def get_embedding(self, image_path: str):
        if not os.path.exists(image_path):
            logger.debug("get_embedding: image not found -> %s", image_path)
            return None
        img = cv2.imread(image_path)
        if img is None:
            logger.debug("get_embedding: failed to read image -> %s", image_path)
            return None
        faces = self.analyze_frame(img, force_detect=True)
        if not faces:
            logger.debug("get_embedding: no face detected in %s", image_path)
            return None
        faces.sort(key=lambda x: x["det_score"], reverse=True)
        embedding = faces[0]["embedding"]
        return _normalize(embedding)

    # --------------------------
    def append_embedding(self, employee_id: str, embedding: np.ndarray):
        emb = _normalize(embedding).astype(np.float32)
        with self.lock:
            try:
                if self.known_embeddings.size == 0:
                    self.known_embeddings = emb.reshape(1, -1)
                else:
                    if emb.shape[0] != self.known_embeddings.shape[1]:
                        logger.warning("Embedding dimension mismatch: new=%d current=%d; reloading DB",
                                       emb.shape[0], self.known_embeddings.shape[1])
                        self.load_from_db()
                        if self.known_embeddings.size == 0 or emb.shape[0] != self.known_embeddings.shape[1]:
                            self.known_embeddings = emb.reshape(1, -1)
                        else:
                            self.known_embeddings = np.vstack([self.known_embeddings, emb.reshape(1, -1)])
                    else:
                        self.known_embeddings = np.vstack([self.known_embeddings, emb.reshape(1, -1)])
                self.known_ids.append(employee_id)
            except Exception:
                logger.exception("Failed appending embedding for %s", employee_id)

    def _detect_faces(self, frame_bgr) -> List[dict]:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = self.app.get(img)
        out = []
        for f in faces:
            bbox = np.array(getattr(f, "bbox", [])).astype(int).tolist()

            # ✅ Safe Fixed Embedding Extraction (avoids ValueError)
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = getattr(f, "embedding", None)
            if emb is None:
                continue

            out.append({
                "bbox": bbox,
                "kps": getattr(f, "kps", None),
                "embedding": _normalize(emb),
                "det_score": float(getattr(f, "det_score", 0.0))
            })
        return out


    # --------------------------
    def analyze_frame(self, frame_bgr, force_detect: bool = False) -> List[dict]:
        """
        Detection+embedding with a throttle and IOU tracking.
        If we detected recently, we track existing boxes to save compute.
        """
        now = time.time()
        if force_detect or (now - self._last_det_ts) >= self.min_det_interval or self.tracker.empty:
            faces = self._detect_faces(frame_bgr)
            tracks = self.tracker.update([f["bbox"] for f in faces])
            # attach track ids
            for f, tid in zip(faces, tracks):
                f["track_id"] = tid
            self._last_det_ts = now
            return faces

        # use tracker to predict and avoid heavy detect
        tracks = self.tracker.predict()
        faces = []
        for tid, bbox in tracks.items():
            x1, y1, x2, y2 = bbox
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # lightweight check: reuse previous embedding if available
            # For simplicity, recompute embedding because InsightFace does both quickly.
            try:
                img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                fs = self.app.get(img)
                if not fs:
                    continue
                f = sorted(fs, key=lambda ff: float(getattr(ff, "det_score", 0.0)), reverse=True)[0]
                emb = getattr(f, "normed_embedding", None) or getattr(f, "embedding", None)
                if emb is None:
                    continue
                faces.append({
                    "bbox": [x1, y1, x2, y2],
                    "kps": getattr(f, "kps", None),
                    "embedding": _normalize(emb),
                    "det_score": float(getattr(f, "det_score", 0.0)),
                    "track_id": tid
                })
            except Exception as ex:
                logger.debug("track embed error: %s", ex)
                continue
        return faces

    # --------------------------
    def match_embedding(self, emb: np.ndarray):
        """Hybrid cosine + euclidean decision."""
        with self.lock:
            if self.known_embeddings.size == 0:
                return None, 0.0
            embn = _normalize(emb).astype(np.float32)
            if embn.shape[0] != self.known_embeddings.shape[1]:
                logger.warning("Embedding dimension mismatch (match): emb=%d db=%d; reloading DB", embn.shape[0],
                               self.known_embeddings.shape[1])
                self.load_from_db()
                if self.known_embeddings.size == 0 or embn.shape[0] != self.known_embeddings.shape[1]:
                    return None, 0.0

            sims = np.dot(self.known_embeddings, embn)
            best_idx = int(np.argmax(sims))
            best_cos = float(sims[best_idx])
            best_id = self.known_ids[best_idx]
            # euclidean guard
            euc = float(np.linalg.norm(self.known_embeddings[best_idx] - embn))
            dim = embn.shape[0]
            EUC_MAX = float(getattr(settings, "FACE_EUCLIDEAN_MAX", 1.15 if dim >= 256 else 0.85))
            if best_cos >= self.confidence_threshold and euc <= EUC_MAX:
                return best_id, best_cos
            return None, best_cos

    # --------------------------
    def recognize_from_frame(self, frame_bgr):
        """
        Returns tuple: (employee_id or None, score, bbox)
        Includes a liveness check to reduce spoofing.
        """
        faces = self.analyze_frame(frame_bgr)
        if not faces:
            return None, 0.0, None

        # choose face with highest det score & area
        faces.sort(key=lambda x: (x.get("det_score", 0.0),
                                  (x.get("bbox", [0,0,0,0])[2]-x.get("bbox", [0,0,0,0])[0])*
                                  (x.get("bbox", [0,0,0,0])[3]-x.get("bbox", [0,0,0,0])[1])),
                   reverse=True)
        top = faces[0]

        # liveness guard
        if not self.liveness.is_live(frame_bgr, top["bbox"]):
            return None, 0.0, None

        employee_id, score = self.match_embedding(top["embedding"])
        bbox = top.get("bbox") or []
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox
            face_loc = (y1, x2, y2, x1)
        else:
            face_loc = None
        return employee_id, float(score), face_loc


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
