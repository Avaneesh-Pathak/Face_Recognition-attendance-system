import os
import json
import logging

import cv2
import numpy as np
from django.core.management.base import BaseCommand
from django.conf import settings

from core.models import Employee

logger = logging.getLogger('core')

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except Exception as e:
    INSIGHTFACE_AVAILABLE = False
    logger.exception("insightface not available: %s", e)


class Command(BaseCommand):
    help = "Rebuild all employee face embeddings using InsightFace (ArcFace)"

    def handle(self, *args, **options):
        if not INSIGHTFACE_AVAILABLE:
            self.stdout.write(self.style.ERROR("insightface is not installed or failed to import. Install insightface to use this command."))
            return

        model_name = getattr(settings, "INSIGHTFACE_MODEL", "buffalo_l")
        det_size = getattr(settings, "INSIGHTFACE_DET_SIZE", (640, 640))
        ctx_id = getattr(settings, "INSIGHTFACE_CTX_ID", -1)  # -1 = CPU

        try:
            app = FaceAnalysis(name=model_name)
            app.prepare(ctx_id=ctx_id, det_size=det_size)
        except Exception as e:
            logger.exception("Failed to initialize InsightFace FaceAnalysis: %s", e)
            self.stdout.write(self.style.ERROR(f"Failed to initialize insightface: {e}"))
            return

        updated = 0
        for emp in Employee.objects.all():
            try:
                user_label = emp.user.get_full_name() or getattr(emp.user, "username", str(getattr(emp, "pk", "unknown")))
                if not emp.face_image:
                    self.stdout.write(self.style.WARNING(f"Skipping {user_label} — no face image found"))
                    continue

                img_path = getattr(emp.face_image, "path", None) or getattr(emp.face_image, "name", None)
                if not img_path or not os.path.exists(img_path):
                    # if storage path is different, try using storage.path if available
                    self.stdout.write(self.style.WARNING(f"Image not found for {user_label}: {img_path}"))
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    self.stdout.write(self.style.WARNING(f"Failed to read image for {user_label}: {img_path}"))
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = app.get(rgb)
                if not faces:
                    self.stdout.write(self.style.WARNING(f"No face detected for {user_label}"))
                    continue

                # pick best face by det_score (if available)
                faces = sorted(faces, key=lambda f: float(getattr(f, "det_score", 0.0)), reverse=True)
                f = faces[0]
                embedding = getattr(f, "normed_embedding", None) or getattr(f, "embedding", None)
                if embedding is None:
                    self.stdout.write(self.style.WARNING(f"No embedding produced for {user_label}"))
                    continue

                arr = np.asarray(embedding, dtype=np.float32).ravel()
                emb_list = [float(x) for x in arr.tolist()]

                # store as JSON for portability
                emp.face_encoding = json.dumps(emb_list)
                emp.save(update_fields=["face_encoding"])

                updated += 1
                self.stdout.write(self.style.SUCCESS(f"Updated embedding for {user_label}"))
            except Exception as e:
                logger.exception("Failed processing employee %s: %s", getattr(emp, "pk", None), e)
                self.stdout.write(self.style.ERROR(f"Error processing {getattr(emp.user, 'username', getattr(emp, 'pk', 'unknown'))}: {e}"))

        self.stdout.write(self.style.SUCCESS(f"Rebuilt embeddings for {updated} employees."))