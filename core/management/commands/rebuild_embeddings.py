from django.core.management.base import BaseCommand
from core.models import Employee
from insightface.app import FaceAnalysis
import numpy as np
import cv2
import os

class Command(BaseCommand):
    help = "Rebuild all employee face embeddings using InsightFace (ArcFace)"

    def handle(self, *args, **options):
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        count = 0
        for emp in Employee.objects.all():
            if not emp.face_image:
                self.stdout.write(self.style.WARNING(f"Skipping {emp.user.get_full_name()} — no face image found"))
                continue

            img_path = emp.face_image.path
            if not os.path.exists(img_path):
                self.stdout.write(self.style.WARNING(f"Image not found: {img_path}"))
                continue

            img = cv2.imread(img_path)
            faces = app.get(img)
            if not faces:
                self.stdout.write(self.style.WARNING(f"No face detected for {emp.user.get_full_name()}"))
                continue

            embedding = faces[0]['embedding']
            emp.face_encoding = embedding.tolist()
            emp.save()
            count += 1
            self.stdout.write(self.style.SUCCESS(f"✅ Updated embedding for {emp.user.get_full_name()}"))

        self.stdout.write(self.style.SUCCESS(f"\n✨ Rebuilt embeddings for {count} employees."))
