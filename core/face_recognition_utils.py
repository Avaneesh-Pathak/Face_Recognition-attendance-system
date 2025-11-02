import cv2
import numpy as np
import insightface
from django.conf import settings
from core.models import Employee

model = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
model.prepare(ctx_id=0)

def get_face_embedding(image):
    faces = model.get(image)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding

def recognize_face(frame, threshold=0.38):
    emb = get_face_embedding(frame)
    if emb is None:
        return None

    best_match = None
    best_score = threshold

    for emp in Employee.objects.exclude(face_encoding=None):
        db_emb = np.frombuffer(emp.face_encoding, dtype=np.float32)
        sim = np.dot(emb, db_emb)
        print(f"[DEBUG] Comparing with {emp.user.username}: similarity = {sim:.3f}")  # 👈 Add this
        if sim > best_score:
            best_match = emp
            best_score = sim

    if best_match:
        print(f"[MATCH] Found {best_match.user.username} with {best_score:.3f}")
    else:
        print("[INFO] No match found")
    return best_match

