# attendance/face_system.py
import threading
import numpy as np
import cv2
import os
from django.conf import settings

# InsightFace
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except Exception as e:
    INSIGHTFACE_AVAILABLE = False
    print(f"[WARN] insightface not available: {e}")

# --------------------------
# Helpers
# --------------------------
def _normalize(vec: np.ndarray) -> np.ndarray:
    v = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return (v / norm).astype(np.float32)


# --------------------------
# FaceRecognitionSystem (InsightFace: RetinaFace + ArcFace)
# --------------------------
class FaceRecognitionSystem:
    """
    Uses insightface FaceAnalysis (e.g. 'buffalo_l' or 'antelope') for detection+embedding.
    Loads embeddings from Django Employee model into an in-memory normalized numpy array
    and performs vectorized cosine matching for very fast lookups.
    """

    def __init__(self, model_name="buffalo_l", det_size=(640, 640), confidence_threshold=0.60):
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("insightface is required but not installed.")
        self.lock = threading.Lock()
        self.model_name = model_name
        self.det_size = det_size
        self.confidence_threshold = float(confidence_threshold)  # cosine threshold (0..1)
        self.app = FaceAnalysis(name=self.model_name)
        # CPU mode
        self.app.prepare(ctx_id=-1, det_size=self.det_size)   # use CPU (-1). Recommended for CPU-only. :contentReference[oaicite:4]{index=4}

        # in-memory store
        self.known_ids = []                 # list of employee_id
        self.known_embeddings = np.empty((0, 512), dtype=np.float32)  # insightface embeddings usually 512-d
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
                qs = Employee.objects.filter(is_active=True).exclude(face_encoding__isnull=True)
                for emp in qs:
                    enc = emp.get_face_encoding()
                    if enc is None:
                        continue
                    encn = _normalize(enc)
                    encs.append(encn)
                    ids.append(emp.employee_id)
                if encs:
                    self.known_embeddings = np.vstack(encs).astype(np.float32)
                    self.known_ids = ids
                else:
                    self.known_embeddings = np.empty((0, 512), dtype=np.float32)
                    self.known_ids = []
                self.initialized = True
                print(f"[INFO] FaceSystem loaded {len(self.known_ids)} embeddings")
            except Exception as e:
                print(f"[ERROR] loading embeddings from DB: {e}")
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
            print(f"[WARN] get_embedding: image not found -> {image_path}")
            return None

        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARN] get_embedding: failed to read image -> {image_path}")
            return None

        faces = self.analyze_frame(img)
        if not faces:
            print(f"[INFO] get_embedding: no face detected in {image_path}")
            return None

        # Use first (largest/confident) face
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
            if self.known_embeddings.size == 0:
                self.known_embeddings = emb.reshape(1, -1)
            else:
                self.known_embeddings = np.vstack([self.known_embeddings, emb])
            self.known_ids.append(employee_id)

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
        # insightface expects RGB
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = self.app.get(img)  # returns list-like face objects
        results = []
        for f in faces:
            # f.bbox, f.kps, f.det_score, f.normed_embedding
            bbox = f.bbox.astype(int).tolist()
            res = {
                "bbox": bbox,                              # [x1,y1,x2,y2]
                "kps": getattr(f, "kps", None),
                "embedding": _normalize(getattr(f, "normed_embedding", f.embedding)),
                "det_score": float(getattr(f, "det_score", 0.0))
            }
            results.append(res)
        return results

    # --------------------------
    # Recognize best match for single face embedding
    # --------------------------
    def match_embedding(self, emb: np.ndarray, return_score=True):
        """Vectorized cosine match. Returns (employee_id or None, score)."""
        with self.lock:
            if self.known_embeddings.size == 0:
                return None, 0.0
            embn = _normalize(emb).astype(np.float32)
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
        faces.sort(key=lambda x: (x["det_score"], (x["bbox"][2]-x["bbox"][0])*(x["bbox"][3]-x["bbox"][1])), reverse=True)
        top = faces[0]
        employee_id, score = self.match_embedding(top["embedding"])
        # convert bbox from [x1,y1,x2,y2] to (top,right,bottom,left) for drawing compatibility
        x1,y1,x2,y2 = top["bbox"]
        face_loc = (y1, x2, y2, x1)
        return employee_id, float(score), face_loc


# --------------------------
# Singleton accessor
# --------------------------
_instance = None

def get_face_system():
    global _instance
    if _instance is None:
        _instance = FaceRecognitionSystem()
    return _instance
