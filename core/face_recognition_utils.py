
import os
import json
import logging
from typing import Tuple, Optional, List, Dict, Any

import cv2
import numpy as np
from django.conf import settings

from core.models import Employee, AttendanceSettings

logger = logging.getLogger('core')

# --------------------------
# Global insightface singleton
# --------------------------
_insight_model = None


def _init_model():
    """
    Initialize and cache a single global InsightFace FaceAnalysis app.
    Respects settings:
      - INSIGHTFACE_MODEL (default "buffalo_l")
      - INSIGHTFACE_DET_SIZE (default (640, 640))
      - INSIGHTFACE_CTX_ID (default -1 for CPU, 0+ for GPU)
    """
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


# --------------------------
# Math helpers
# --------------------------
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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize(a)
    b = _normalize(b)
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size == 0 or b.size == 0 or a.shape[0] != b.shape[0]:
        return 1e9
    return float(np.linalg.norm(a - b))


# --------------------------
# DB embedding parsing
# --------------------------
def _parse_db_embedding(raw) -> Optional[np.ndarray]:
    """Parse Employee.face_encoding stored in various formats and return normalized vector.
       Supports: numpy array, list, tuple, bytes (float32), json string, comma list string,
       or a JSON list of lists (will return mean embedding)."""
    if raw is None:
        return None
    try:
        # numpy passthrough
        if isinstance(raw, np.ndarray):
            arr = raw.astype(np.float32)
            if arr.ndim == 2:  # multiple -> mean
                arr = arr.mean(axis=0)
            return _normalize(arr)

        # python list / tuple (either 1D or list of lists)
        if isinstance(raw, (list, tuple)):
            arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim == 2:  # multiple
                arr = arr.mean(axis=0)
            return _normalize(arr)

        # raw bytes -> try float32 buffer
        if isinstance(raw, (bytes, bytearray)):
            try:
                arr = np.frombuffer(raw, dtype=np.float32)
                return _normalize(arr)
            except Exception:
                try:
                    s = raw.decode('utf-8')
                    obj = json.loads(s)
                    arr = np.asarray(obj, dtype=np.float32)
                    if arr.ndim == 2:
                        arr = arr.mean(axis=0)
                    return _normalize(arr)
                except Exception:
                    return None

        # string -> json or comma separated
        if isinstance(raw, str):
            try:
                obj = json.loads(raw)
                arr = np.asarray(obj, dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr.mean(axis=0)
                return _normalize(arr)
            except Exception:
                try:
                    parts = [float(x) for x in raw.strip('[]() ').split(',') if x.strip()]
                    return _normalize(np.asarray(parts, dtype=np.float32))
                except Exception:
                    return None
    except Exception:
        logger.exception("Failed to parse DB embedding")
    return None


def _face_align_by_eyes(img_bgr: np.ndarray, kps: Optional[np.ndarray], desired_size: int = 112) -> np.ndarray:
    """
    Lightweight alignment using eye keypoints if available (InsightFace provides kps).
    If kps missing, returns original image region.
    """
    if kps is None or len(kps) < 2:
        return img_bgr

    left_eye = kps[0]
    right_eye = kps[1]
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eyes_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (img_bgr.shape[1], img_bgr.shape[0]), flags=cv2.INTER_LINEAR)
    # center crop to square then resize
    h, w = rotated.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = rotated[y0:y0+side, x0:x0+side]
    if crop.size == 0:
        crop = img_bgr
    return cv2.resize(crop, (desired_size, desired_size), interpolation=cv2.INTER_LINEAR)


# --------------------------
# Embedding extractor
# --------------------------
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

        # optional alignment
        try:
            bbox = getattr(f, "bbox", None)
            kps = getattr(f, "kps", None)
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox[:4])
                face_crop = img[y1:y2, x1:x2].copy()
                face_crop = _face_align_by_eyes(face_crop, kps)
            else:
                face_crop = img
        except Exception:
            face_crop = img

        # Re-run just embedding on aligned crop if available
        try:
            rgb2 = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            faces2 = model.get(rgb2)
            if faces2:
                f2 = sorted(faces2, key=lambda ff: float(getattr(ff, "det_score", 0.0)), reverse=True)[0]
                emb = getattr(f2, "normed_embedding", None) or getattr(f2, "embedding", None)
            else:
                emb = getattr(f, "normed_embedding", None) or getattr(f, "embedding", None)
        except Exception:
            emb = getattr(f, "normed_embedding", None) or getattr(f, "embedding", None)

        if emb is None:
            return None

        emb_arr = np.asarray(emb, dtype=np.float32).ravel()
        if emb_arr.size == 0:
            return None
        return _normalize(emb_arr)
    except Exception:
        logger.exception("Error extracting face embedding")
        return None


# --------------------------
# Recognition (hybrid scoring + mean templates)
# --------------------------
def _collect_db_embeddings() -> Tuple[List[np.ndarray], List[Employee]]:
    qs = Employee.objects.filter(is_active=True).exclude(face_encoding__isnull=True).exclude(face_encoding__exact='')
    emps = list(qs)
    db_embs: List[np.ndarray] = []
    db_map: List[Employee] = []
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
        db_embs.append(_normalize(db_emb))
        db_map.append(emp)
    return db_embs, db_map


def recognize_face(frame, threshold: Optional[float] = None) -> Tuple[Optional[Employee], float]:
    """
    Recognize an employee from a BGR frame (numpy array) or image path.
    Returns (Employee or None, similarity_score).
    Uses AttendanceSettings.confidence_threshold as default threshold when available.
    Hybrid decision uses cosine >= threshold and euclidean <= EUC_MAX (auto tuned per dim).
    """
    emb = get_face_embedding(frame)
    if emb is None:
        return None, 0.0

    # thresholds
    try:
        settings_obj = AttendanceSettings.objects.first()
        default_thresh = settings_obj.confidence_threshold if settings_obj else getattr(settings, "FACE_RECOGNITION_THRESHOLD", 0.60)
    except Exception:
        default_thresh = getattr(settings, "FACE_RECOGNITION_THRESHOLD", 0.60)

    if threshold is None:
        threshold = float(default_thresh)

    # euclidean guardrail (typical ArcFace 512D genuine ~0.8-1.2; impostor ~1.3-1.8 after L2)
    dim = emb.shape[0]
    EUC_MAX = float(getattr(settings, "FACE_EUCLIDEAN_MAX", 1.15 if dim >= 256 else 0.85))

    db_embs, db_map = _collect_db_embeddings()
    if not db_embs:
        return None, 0.0

    try:
        db_matrix = np.vstack(db_embs).astype(np.float32)  # shape (N, D)
        sims = np.dot(db_matrix, emb)  # cosine
        best_idx = int(np.argmax(sims))
        best_cos = float(sims[best_idx])
        best_emp = db_map[best_idx]
        best_euc = _euclidean(db_matrix[best_idx], emb)

        # hybrid decision
        if best_cos >= float(threshold) and best_euc <= EUC_MAX:
            return best_emp, best_cos
        return None, best_cos
    except Exception:
        logger.exception("Error comparing embeddings")
        return None, 0.0
