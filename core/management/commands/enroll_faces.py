
import json
import os
import time
import logging
from typing import List

import cv2
import numpy as np
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from django.conf import settings
from core.models import Employee
from core.liveness import LivenessDetector

# Preferred fast path
try:
    from core.face_system import get_face_system
    _FS = get_face_system()
except Exception:
    _FS = None

# Fallback
try:
    from core.face_recognition_utils import get_face_embedding
except Exception:
    get_face_embedding = None

logger = logging.getLogger('core')

def _extract_embedding(frame_bgr: np.ndarray) -> np.ndarray:
    if _FS is not None:
        # Use analyzer to get the top face
        faces = _FS.analyze_frame(frame_bgr, force_detect=True)
        if faces:
            faces.sort(key=lambda x: x.get("det_score", 0.0), reverse=True)
            return faces[0]["embedding"]
        return None
    if get_face_embedding is not None:
        return get_face_embedding(frame_bgr)
    return None


class Command(BaseCommand):
    help = "Enroll multiple face templates for an employee using a webcam or video source."

    def add_arguments(self, parser):
        parser.add_argument('--employee', '-e', required=True, help='Employee ID or username to enroll')
        parser.add_argument('--samples', '-n', type=int, default=12, help='Number of samples to capture (default: 12)')
        parser.add_argument('--source', '-s', default=0, help='Video source (default: 0)')
        parser.add_argument('--append', action='store_true', help='Append to existing embeddings instead of overwrite')
        parser.add_argument('--no-liveness', action='store_true', help='Do not require liveness (blink/motion)')
        parser.add_argument('--output-dir', default='', help='Optional directory to save captured face frames for audit')
        parser.add_argument('--interval', type=float, default=0.25, help='Seconds between auto-captures')
        parser.add_argument('--min-score', type=float, default=0.6, help='Min detection score to accept a sample')

    def handle(self, *args, **options):
        emp_key = options['employee']
        samples = int(options['samples'])
        source = options['source']
        append = bool(options['append'])
        require_liveness = not bool(options['no_liveness'])
        out_dir = options['output_dir'].strip()
        interval = float(options['interval'])
        min_score = float(options['min_score'])

        emp = Employee.objects.filter(employee_id=emp_key).first()
        if emp is None:
            emp = Employee.objects.filter(user__username=emp_key).first()
        if emp is None:
            raise CommandError(f"Employee not found: {emp_key}")

        # Prepare capture
        try:
            src = int(source)
        except Exception:
            src = source

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise CommandError(f"Unable to open video source: {source}")

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        liveness = LivenessDetector()

        collected: List[np.ndarray] = []

        self.stdout.write(self.style.SUCCESS(
            f"Enrolling {samples} samples for {emp_key} "
            f"({'append' if append else 'overwrite'}) "
            f"{'(liveness required)' if require_liveness else '(liveness disabled)'}"
        ))
        self.stdout.write("Press 'q' to quit early.")

        last_ts = 0.0
        try:
            while len(collected) < samples:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                # Recognize face region to get embedding and score
                score = 0.0
                face_ok = False
                if _FS is not None:
                    faces = _FS.analyze_frame(frame, force_detect=True)
                    if faces:
                        faces.sort(key=lambda x: x.get("det_score", 0.0), reverse=True)
                        top = faces[0]
                        score = float(top.get("det_score", 0.0))
                        emb = top["embedding"]
                        # draw bbox
                        bb = top.get("bbox")
                        if bb and len(bb) == 4:
                            x1,y1,x2,y2 = bb
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                        face_ok = emb is not None and score >= min_score
                    else:
                        emb = None
                        face_ok = False
                else:
                    emb = _extract_embedding(frame)
                    face_ok = emb is not None

                live_ok = True
                if require_liveness:
                    try:
                        live_ok = liveness.detect(frame)
                    except Exception:
                        live_ok = False

                ready = face_ok and live_ok and (time.time() - last_ts) >= interval

                label = f"samples: {len(collected)}/{samples}  face:{'OK' if face_ok else '...'}  live:{'OK' if live_ok else '...'}  det:{score:.2f}"
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                cv2.imshow("Enroll (press q to stop)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                if ready and emb is not None:
                    collected.append(emb.astype(np.float32))
                    last_ts = time.time()
                    if out_dir:
                        fn = os.path.join(out_dir, f"{emp_key}_{int(last_ts*1000)}.jpg")
                        try:
                            cv2.imwrite(fn, frame)
                        except Exception:
                            pass

            cap.release()
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            cap.release()
            cv2.destroyAllWindows()

        if not collected:
            raise CommandError("No samples collected.")

        # Save to DB
        emb_list = [e.tolist() for e in collected]
        try:
            with transaction.atomic():
                if append and emp.face_encoding:
                    try:
                        prev = json.loads(emp.face_encoding)
                        if isinstance(prev, list):
                            # flatten nested lists if needed
                            if prev and isinstance(prev[0], list):
                                prev_list = prev
                            else:
                                prev_list = [prev]
                            emb_list = prev_list + emb_list
                    except Exception:
                        pass
                emp.face_encoding = json.dumps(emb_list)
                emp.save(update_fields=['face_encoding'])
        except Exception as e:
            raise CommandError(f"Failed to save embeddings: {e}")

        self.stdout.write(self.style.SUCCESS(f"Saved {len(emb_list)} templates to employee {emp_key}"))
