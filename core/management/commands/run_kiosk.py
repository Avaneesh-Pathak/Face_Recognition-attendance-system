# core/management/commands/run_kiosk.py
import cv2
import time
import os
from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils import timezone
from datetime import timedelta
from core.models import Employee, Attendance
from core.face_recognition_utils import recognize_face
from core.liveness import LivenessDetector

# -------------------------------------------------------------------
# Capture folder for saving images
# -------------------------------------------------------------------
CAPTURE_DIR = os.path.join(settings.MEDIA_ROOT, "attendance_captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)

COOLDOWN_MINUTES = 10
last_seen = {}

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def save_capture(frame, emp_id):
    ts = timezone.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{emp_id}_{ts}.jpg"
    path = os.path.join(CAPTURE_DIR, filename)
    cv2.imwrite(path, frame)
    return os.path.join("attendance_captures", filename)

def on_cooldown(emp):
    now = timezone.now()
    last = last_seen.get(emp.employee_id)
    if last and (now - last).total_seconds() < COOLDOWN_MINUTES * 60:
        return True
    return False

def mark_attendance(emp, frame, conf=0.0):
    if on_cooldown(emp):
        print(f"[INFO] Cooldown active for {emp.employee_id}, skipping.")
        return
    relpath = save_capture(frame, emp.employee_id)
    Attendance.objects.create(
        employee=emp,
        attendance_type="check_in",
        confidence_score=conf,
        image_capture=relpath
    )
    last_seen[emp.employee_id] = timezone.now()
    print(f"[INFO] ✅ Marked check-in for {emp.name} ({conf:.2f}) at {timezone.now().strftime('%H:%M:%S')}")

# -------------------------------------------------------------------
# Command
# -------------------------------------------------------------------
class Command(BaseCommand):
    help = "Run real-time attendance kiosk (CPU only, blink + motion liveness)."

    def add_arguments(self, parser):
        parser.add_argument("--src", type=int, default=0, help="Camera source index (default 0)")
        parser.add_argument("--process-every", type=int, default=3, help="Process every Nth frame")
        parser.add_argument("--show", action="store_true", help="Show video window")

    def handle(self, *args, **options):
        src = options["src"]
        every = max(1, int(options["process_every"]))
        show = bool(options["show"])

        print("[INFO] Starting Face Attendance Kiosk...")
        cap = cv2.VideoCapture(src)
        liveness = LivenessDetector(blink_required=1, motion_required=True)
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame_idx += 1
                if frame_idx % every != 0:
                    continue

                emp, conf = recognize_face(frame)
                if emp:
                    if liveness.detect(frame):
                        mark_attendance(emp, frame, conf)
                        cv2.putText(frame, f"{emp.name} LIVE ✅", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        print(f"[WARN] Liveness failed for {emp.employee_id} (possible spoof).")
                        cv2.putText(frame, "SPOOF ⚠️", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "UNKNOWN", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if show:
                    cv2.imshow("Face Attendance Kiosk", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")
        finally:
            cap.release()
            if show:
                cv2.destroyAllWindows()
            print("[INFO] Kiosk stopped.")
