import cv2
import time
from django.utils import timezone
from core.face_recognition_utils import recognize_face
from core.liveness import LivenessDetector
from core.models import Attendance

COOLDOWN_SECONDS = 600  # 10 min

last_seen = {}

def mark_attendance(emp):
    now = timezone.now()
    last_entry = last_seen.get(emp.emp_id)

    if last_entry and (now - last_entry).total_seconds() < COOLDOWN_SECONDS:
        return  # still in cooldown

    Attendance.objects.create(employee=emp, timestamp=now)
    last_seen[emp.emp_id] = now
    print(f"[INFO] ✅ Attendance marked for {emp.name} at {now.strftime('%H:%M:%S')}")

def run_video_stream(src=0):
    cap = cv2.VideoCapture(src)
    liveness_detector = LivenessDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emp = recognize_face(frame)
        if emp:
            if liveness_detector.detect(frame):
                mark_attendance(emp)
                cv2.putText(frame, f"{emp.name}: LIVE ✅", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            else:
                print(f"[WARN] Liveness failed for {emp.emp_id} (possible spoof).")
                cv2.putText(frame, "SPOOF ⚠️", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        else:
            cv2.putText(frame, "UNKNOWN", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        cv2.imshow("Face Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
