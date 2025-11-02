import os
import json
import logging
from typing import Tuple, Optional

import cv2
import numpy as np
from django.conf import settings

from core.models import Employee, AttendanceSettings

logger = logging.getLogger('core')

_insight_model = None


def _init_model():
    global _insight_model
    if _insight_model is not None:
        return _insight_model
    try:
        import insightface
        model_name = getattr(settings, "INSIGHTFACE_MODEL", "buffalo_l")
        det_size = getattr(settings, "INSIGHTFACE_DET_SIZE", (640, 640))
        ctx_id = getattr(settings, "INSIGHTFACE_CTX_ID", -1)
        app = insightface.app.FaceAnalysis(name=model_name)
        app.prepare(ctx_id=ctx_id, det_size=det_size)
        _insight_model = app
        logger.info("InsightFace initialized (model=%s, det_size=%s, ctx_id=%s)", model_name, det_size, ctx_id)
    except Exception:
        logger.exception("InsightFace not available or failed to initialize")
        _insight_model = None
    return _insight_model


def _normalize(vec: np.ndarray) -> Optional[np.ndarray]:
    if vec is None:
        return None
    a = np.asarray(vec, dtype=np.float32).ravel()
    if a.size == 0:
        return None
    norm = np.linalg.norm(a)
    if norm <= 0 or np.isnan(norm):
        return a.astype(np.float32)
    return (a / norm).astype(np.float32)


def _parse_db_embedding(raw) -> Optional[np.ndarray]:
    """Parse Employee.face_encoding stored in various formats and return normalized vector."""
    if raw is None:
        return None
    try:
        if isinstance(raw, np.ndarray):
            return _normalize(raw)
        if isinstance(raw, (list, tuple)):
            return _normalize(np.asarray(raw, dtype=np.float32))
        if isinstance(raw, (bytes, bytearray)):
            try:
                arr = np.frombuffer(raw, dtype=np.float32)
                return _normalize(arr)
            except Exception:
                # fallthrough to try decode as utf-8 json
                try:
                    s = raw.decode('utf-8')
                    obj = json.loads(s)
                    return _normalize(np.asarray(obj, dtype=np.float32))
                except Exception:
                    return None
        if isinstance(raw, str):
            # json string or python repr of list
            try:
                obj = json.loads(raw)
                return _normalize(np.asarray(obj, dtype=np.float32))
            except Exception:
                # try eval-like fallback but avoid eval - try comma split
                try:
                    parts = [float(x) for x in raw.strip('[]() ').split(',') if x.strip()]
                    return _normalize(np.asarray(parts, dtype=np.float32))
                except Exception:
                    return None
    except Exception:
        logger.exception("Failed to parse DB embedding")
    return None



def get_face_embedding(image) -> Optional[np.ndarray]:
    """
    Accepts:
      - numpy BGR image (cv2 frame)
      - path string to an image file
      - bytes/bytearray of image data
    Returns L2-normalized numpy.float32 embedding or None.
    """
    model = _init_model()
    if model is None:
        logger.error("No insightface model available to extract embeddings")
        return None

    img = None
    try:
        if isinstance(image, str):
            if not os.path.exists(image):
                logger.debug("get_face_embedding: path does not exist: %s", image)
                return None
            img = cv2.imread(image)
        elif isinstance(image, (bytes, bytearray)):
            arr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            img = image

        if img is None:
            return None

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = model.get(rgb)
        if not faces:
            return None

        faces = sorted(faces, key=lambda f: float(getattr(f, "det_score", 0.0)), reverse=True)
        f = faces[0]

        # avoid truth-testing numpy arrays (which raises ValueError)
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            emb = getattr(f, "embedding", None)

        if emb is None:
            return None

        emb_arr = np.asarray(emb, dtype=np.float32).ravel()
        if emb_arr.size == 0:
            return None

        return _normalize(emb_arr)
    except Exception:
        logger.exception("Error extracting face embedding")
        return None


def recognize_face(frame, threshold: Optional[float] = None) -> Tuple[Optional[Employee], float]:
    """
    Recognize an employee from a BGR frame (numpy array) or image path.
    Returns (Employee or None, similarity_score).
    Uses AttendanceSettings.confidence_threshold as default threshold when available.
    """
    emb = get_face_embedding(frame)
    if emb is None:
        return None, 0.0

    try:
        settings_obj = AttendanceSettings.objects.first()
        default_thresh = settings_obj.confidence_threshold if settings_obj else getattr(settings, "FACE_RECOGNITION_THRESHOLD", 0.60)
    except Exception:
        default_thresh = getattr(settings, "FACE_RECOGNITION_THRESHOLD", 0.60)

    if threshold is None:
        threshold = float(default_thresh)

    # Load employees with embeddings
    qs = Employee.objects.filter(is_active=True).exclude(face_encoding__isnull=True).exclude(face_encoding__exact='')
    emps = list(qs)
    if not emps:
        return None, 0.0

    db_embs = []
    db_map = []
    for emp in emps:
        db_emb = None
        try:
            if hasattr(emp, "get_face_encoding"):
                db_emb = emp.get_face_encoding()
            if db_emb is None:
                db_emb = _parse_db_embedding(emp.face_encoding)
        except Exception:
            logger.exception("Failed obtaining DB embedding for emp id=%s", getattr(emp, "pk", None))
            db_emb = None

        if db_emb is None:
            continue
        db_embs.append(db_emb)
        db_map.append(emp)

    if not db_embs:
        return None, 0.0

    try:
        db_matrix = np.vstack(db_embs).astype(np.float32)  # shape (N, D)
        # both sides normalized -> dot product = cosine similarity
        sims = np.dot(db_matrix, emb)  # shape (N,)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score >= float(threshold):
            return db_map[best_idx], best_score
        return None, best_score
    except Exception:
        logger.exception("Error comparing embeddings")
        return None, 0.0