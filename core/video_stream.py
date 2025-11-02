import logging
import cv2
import time
import numpy as np

from django.utils import timezone
from django.conf import settings

from core.liveness import LivenessDetector
from core.models import Attendance, Employee, AttendanceSettings

# Prefer normalized embedding extractor; provide fallback recognize_face if available
try:
    from core.face_recognition_utils import recognize_face  # may be provided elsewhere
except Exception:
    from core.face_recognition_utils import get_face_embedding
    recognize_face = None  # will provide local implementation below

logger = logging.getLogger('core')

# Read cooldown from Django settings (fallback to 600s)
COOLDOWN_SECONDS = getattr(settings, 'COOLDOWN_SECONDS', None)
if COOLDOWN_SECONDS is None:
    try:
        COOLDOWN_SECONDS = settings.LIVENESS.get("COOLDOWN_SECONDS", 600)
    except Exception:
        COOLDOWN_SECONDS = 600

# track last-marked time per employee pk
last_seen = {}


def _fallback_recognize_face(frame):
    """
    Fallback recognition using embeddings stored on Employee model.
    Returns Employee instance or None.
    """
    emb = get_face_embedding(frame)
    if emb is None:
        return None

    try:
        settings_obj = AttendanceSettings.objects.first()
        threshold = settings_obj.confidence_threshold if settings_obj else 0.60
    except Exception:
        threshold = 0.60

    employees = list(Employee.objects.exclude(face_encoding__isnull=True).exclude(face_encoding__exact=''))
    if not employees:
        return None

    db_embeddings = []
    db_emps = []
    for emp in employees:
        db_emb = emp.get_face_encoding()
        if db_emb is None:
            continue
        db_embeddings.append(db_emb)
        db_emps.append(emp)

    if not db_embeddings:
        return None

    db_matrix = np.vstack(db_embeddings).astype(np.float32)  # shape (N, D)
    sims = np.dot(db_matrix, emb)  # (N,)
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    if best_score >= threshold:
        return db_emps[best_idx]
    return None


def mark_attendance(emp):
    """
    Create a check-in or check-out Attendance record for emp respecting cooldown.
    Uses employee.pk as key in last_seen and logs errors instead of crashing.
    """
    now = timezone.now()
    key = getattr(emp, 'pk', None)
    if key is None:
        logger.warning("mark_attendance called with invalid employee object: %r", emp)
        return

    last_entry = last_seen.get(key)
    if last_entry and (now - last_entry).total_seconds() < COOLDOWN_SECONDS:
        logger.debug("Cooldown active for employee %s (%.1fs remaining)",
                     key, COOLDOWN_SECONDS - (now - last_entry).total_seconds())
        return  # still in cooldown

    try:
        # Decide check-in vs check-out based on last attendance today
        latest = Attendance.objects.filter(employee=emp, timestamp__date=now.date()).order_by('-timestamp').first()
        if latest and latest.attendance_type == 'check_in':
            attendance_type = 'check_out'
        else:
            attendance_type = 'check_in'

        Attendance.objects.create(employee=emp, attendance_type=attendance_type, timestamp=now)
        last_seen[key] = now

        name = getattr(emp.user, 'get_full_name', lambda: None)()
        if not name:
            name = getattr(emp.user, 'username', str(emp.pk))
        logger.info("Attendance marked: %s - %s (%s)", name, attendance_type, now.strftime('%Y-%m-%d %H:%M:%S'))
    except Exception:
        logger.exception("Failed to mark attendance for employee %s", key)


def run_video_stream(src=0):
    """
    Main loop to run video stream and perform recognition + liveness + attendance marking.
    This function is safe to call from a dedicated process / management command.
    """
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error("Unable to open video source: %s", src)
        return

    liveness_detector = LivenessDetector()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Empty frame received from source %s; retrying...", src)
                time.sleep(0.1)
                continue

            try:
                emp = None
                if recognize_face:
                    try:
                        emp = recognize_face(frame)
                    except Exception:
                        logger.exception("recognize_face failed, falling back to internal recognizer")
                        emp = _fallback_recognize_face(frame)
                else:
                    emp = _fallback_recognize_face(frame)

                if emp:
                    try:
                        is_live = liveness_detector.detect(frame)
                    except Exception:
                        logger.exception("Liveness detection error; assuming not live")
                        is_live = False

                    if is_live:
                        mark_attendance(emp)
                        label = f"{emp.user.get_full_name() or emp.user.username}: LIVE"
                        color = (0, 255, 0)
                    else:
                        logger.warning("Liveness failed for employee %s", getattr(emp, 'employee_id', emp.pk))
                        label = "SPOOF"
                        color = (0, 0, 255)
                else:
                    label = "UNKNOWN"
                    color = (0, 255, 255)

                # Draw label on frame (safe even if GUI not available)
                try:
                    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                except Exception:
                    # ignore drawing errors on headless systems
                    pass

                # Show frame if possible (useful for local testing)
                try:
                    cv2.imshow("Face Attendance", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quit key pressed; exiting video stream.")
                        break
                except Exception:
                    # cv2.imshow may fail on headless servers; continue without display
                    pass

            except Exception:
                logger.exception("Error processing frame")

    finally:
        try:
            cap.release()
        except Exception:
            logger.exception("Failed releasing capture")
        try:
            cv2.destroyAllWindows()
        except Exception:
            # ignore on headless systems
            pass