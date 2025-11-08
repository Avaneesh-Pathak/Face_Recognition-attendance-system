import logging
import cv2
import time
import numpy as np
from typing import Optional, Dict

from django.utils import timezone
from django.conf import settings
from django.db.models import Q

from core.liveness import LivenessDetector
from core.models import Attendance, Employee, AttendanceSettings

# Preferred fast recognizer (InsightFace-based)
try:
    from core.face_system import get_face_system
    _FACE_SYS_AVAILABLE = True
except Exception as e:
    _FACE_SYS_AVAILABLE = False

# As a fallback we can still use embedding utils
try:
    from core.face_recognition_utils import get_face_embedding
except Exception:
    get_face_embedding = None

logger = logging.getLogger('core')

COOLDOWN_SECONDS = getattr(settings, 'COOLDOWN_SECONDS', None)
if COOLDOWN_SECONDS is None:
    try:
        COOLDOWN_SECONDS = settings.LIVENESS.get("COOLDOWN_SECONDS", 600)
    except Exception:
        COOLDOWN_SECONDS = 600

# Track last marked time per employee pk
_last_seen: Dict[int, timezone.datetime] = {}
_last_seen_lock = None  # lazy import threading to avoid overhead in Django import path


def _mark_attendance(emp: Employee):
    """Safely mark attendance with cooldown and check-in/out toggling."""
    global _last_seen, _last_seen_lock
    if _last_seen_lock is None:
        import threading
        _last_seen_lock = threading.Lock()

    now = timezone.now()
    pk = getattr(emp, 'pk', None)
    if pk is None:
        logger.warning("mark_attendance called with invalid employee object: %r", emp)
        return

    with _last_seen_lock:
        last = _last_seen.get(pk)
        if last and (now - last).total_seconds() < COOLDOWN_SECONDS:
            return

        try:
            latest = Attendance.objects.filter(employee=emp, timestamp__date=now.date()).order_by('-timestamp').first()
            if latest and latest.attendance_type == 'check_in':
                att_type = 'check_out'
            else:
                att_type = 'check_in'

            Attendance.objects.create(employee=emp, attendance_type=att_type, timestamp=now)
            _last_seen[pk] = now

            name = getattr(emp.user, 'get_full_name', lambda: None)() or getattr(emp.user, 'username', str(pk))
            logger.info("Attendance marked: %s - %s", name, att_type)
        except Exception:
            logger.exception("Failed to mark attendance for employee %s", pk)


def _find_employee_by_employee_id(emp_id: str) -> Optional[Employee]:
    try:
        return Employee.objects.filter(Q(employee_id=emp_id) | Q(user__username=emp_id)).first()
    except Exception:
        logger.exception("Employee lookup failed for id=%s", emp_id)
        return None


def _fallback_match(frame_bgr) -> Optional[Employee]:
    """Very simple fallback if face_system is not available."""
    if get_face_embedding is None:
        return None
    emb = get_face_embedding(frame_bgr)
    if emb is None:
        return None

    try:
        settings_obj = AttendanceSettings.objects.first()
        threshold = float(settings_obj.confidence_threshold) if settings_obj else 0.60
    except Exception:
        threshold = 0.60

    emps = list(Employee.objects.exclude(face_encoding__isnull=True).exclude(face_encoding__exact=''))
    if not emps:
        return None

    db = []
    idx = []
    for e in emps:
        enc = None
        try:
            enc = e.get_face_encoding()
        except Exception:
            enc = None
        if enc is None:
            continue
        db.append(enc)
        idx.append(e)
    if not db:
        return None

    db_mat = np.vstack(db).astype(np.float32)
    sims = np.dot(db_mat, emb)
    k = int(np.argmax(sims))
    if float(sims[k]) >= threshold:
        return idx[k]
    return None


def run_video_stream(src=0, show_window: bool = True):
    """
    Main loop: capture -> recognize -> liveness -> mark attendance -> draw HUD.
    Uses FaceRecognitionSystem when available for speed and accuracy.
    """
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error("Unable to open video source: %s", src)
        return

    # Configure camera (optional safe settings)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    liveness = LivenessDetector()

    fs = get_face_system() if _FACE_SYS_AVAILABLE else None

    prev_ts = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            # --- Recognition ---
            emp = None
            bbox_vis = None
            score = 0.0
            try:
                if fs is not None:
                    emp_id, score, bbox = fs.recognize_from_frame(frame)
                    bbox_vis = bbox
                    if emp_id:
                        emp = _find_employee_by_employee_id(emp_id)
                else:
                    emp = _fallback_match(frame)
            except Exception:
                logger.exception("recognition failed; trying fallback")
                emp = _fallback_match(frame)

            # --- Liveness + attendance ---
            label = "UNKNOWN"
            color = (0, 255, 255)
            if emp:
                try:
                    live = liveness.detect(frame)
                except Exception:
                    logger.exception("Liveness error")
                    live = False

                if live:
                    _mark_attendance(emp)
                    uname = getattr(emp.user, 'get_full_name', lambda: None)() or getattr(emp.user, 'username', str(emp.pk))
                    label = f"{uname} ({score:.2f}) LIVE"
                    color = (0, 200, 0)
                else:
                    label = "SUSPECTED SPOOF"
                    color = (0, 0, 255)

            # --- HUD ---
            try:
                # bbox
                if bbox_vis and len(bbox_vis) == 4:
                    (top, right, bottom, left) = bbox_vis
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                # fps
                now = time.time()
                dt = now - prev_ts
                prev_ts = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)
                cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(frame, f"FPS: {fps:.1f}", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            except Exception:
                pass

            # --- display ---
            if show_window:
                try:
                    cv2.imshow("Face Attendance", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quit pressed; exiting.")
                        break
                except Exception:
                    # headless: ignore
                    pass

    finally:
        try:
            cap.release()
        except Exception:
            logger.exception("Failed to release capture")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
