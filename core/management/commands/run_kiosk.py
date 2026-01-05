import os
import time
import logging

import cv2
from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils import timezone

from core.models import Employee, Attendance, AttendanceSettings
from core.face_recognition_utils import recognize_face
from core.liveness import LivenessDetector

logger = logging.getLogger('core')

# -------------------------------------------------------------------
# Capture folder for saving images
# -------------------------------------------------------------------
CAPTURE_DIR = os.path.join(getattr(settings, "MEDIA_ROOT", "."), "attendance_captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)

# default cooldown (seconds) if not provided via settings
DEFAULT_COOLDOWN_SECONDS = int(getattr(settings, "KIOSK_COOLDOWN_SECONDS", 10 * 60))
_last_seen = {}  # keyed by employee.pk


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def save_capture(frame, emp_identifier):
    ts = timezone.now().strftime("%Y%m%d_%H%M%S")
    safe_id = str(emp_identifier).replace(" ", "_")
    filename = f"{safe_id}_{ts}.jpg"
    path = os.path.join(CAPTURE_DIR, filename)
    try:
        cv2.imwrite(path, frame)
        return os.path.join("attendance_captures", filename)
    except Exception:
        logger.exception("Failed to save capture to %s", path)
        return None


def on_cooldown(emp, cooldown_seconds=DEFAULT_COOLDOWN_SECONDS):
    now = timezone.now()
    key = getattr(emp, "pk", None)
    if key is None:
        return True
    last = _last_seen.get(key)
    if last and (now - last).total_seconds() < float(cooldown_seconds):
        return True
    return False


def mark_attendance(emp, frame, conf=0.0):
    """
    Create check-in / check-out Attendance for emp.
    Respects cooldown and AttendanceSettings.min_hours_before_checkout.
    """
    try:
        if on_cooldown(emp):
            logger.debug("Cooldown active for employee %s; skipping mark.", getattr(emp, "employee_id", emp.pk))
            return

        now = timezone.now()
        # determine today's latest attendance for this employee
        latest = Attendance.objects.filter(employee=emp, timestamp__date=now.date()).order_by("-timestamp").first()

        settings_obj = AttendanceSettings.objects.first()
        min_hours = float(settings_obj.min_hours_before_checkout) if settings_obj and getattr(settings_obj, "min_hours_before_checkout", None) is not None else float(getattr(settings, "MIN_HOURS_BEFORE_CHECKOUT", 3.0))

        if latest and latest.attendance_type == "check_in":
            # attempt check-out, enforce min hours
            elapsed = now - latest.timestamp
            if elapsed.total_seconds() < min_hours * 3600:
                logger.info("Employee %s attempted checkout too early (%.1f/%.1f hours)", getattr(emp, "employee_id", emp.pk), elapsed.total_seconds() / 3600.0, min_hours)
                return
            a_type = "check_out"
        else:
            a_type = "check_in"

        relpath = save_capture(frame, getattr(emp, "employee_id", getattr(emp, "pk", "unknown")))
        attendance_kwargs = {
            "employee": emp,
            "attendance_type": a_type,
            "confidence_score": float(conf),
            "timestamp": now
        }
        if relpath:
            attendance_kwargs["image_capture"] = relpath

        Attendance.objects.create(**attendance_kwargs)
        _last_seen[getattr(emp, "pk")] = now

        name = (emp.user.get_full_name() or emp.user.username) if getattr(emp, "user", None) else str(getattr(emp, "employee_id", emp.pk))
        logger.info("Marked %s for %s (score %.3f) at %s", a_type, name, float(conf), now.isoformat())
    except Exception:
        logger.exception("Failed to mark attendance for employee %s", getattr(emp, "pk", None))


# -------------------------------------------------------------------
# Command
# -------------------------------------------------------------------
class Command(BaseCommand):
    help = "Run real-time attendance kiosk (runs camera loop; intended to be run as a separate process)."

    def add_arguments(self, parser):
        parser.add_argument("--src", type=int, default=0, help="Camera source index (default 0)")
        parser.add_argument("--process-every", type=int, default=3, help="Process every Nth frame (default 3)")
        parser.add_argument("--show", action="store_true", help="Show preview window")

    def handle(self, *args, **options):
        src = int(options.get("src", 0))
        every = max(1, int(options.get("process_every", 3)))
        show = bool(options.get("show", False))

        cooldown_seconds = int(getattr(settings, "KIOSK_COOLDOWN_SECONDS", DEFAULT_COOLDOWN_SECONDS))
        logger.info("Starting kiosk (src=%s, every=%d, cooldown=%ds, show=%s)", src, every, cooldown_seconds, show)

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            logger.error("Unable to open video source: %s", src)
            return

        liveness = LivenessDetector()  # uses internal defaults
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                frame_idx += 1
                if frame_idx % every != 0:
                    continue

                try:
                    emp, conf = recognize_face(frame)
                except Exception:
                    logger.exception("Error during face recognition")
                    emp, conf = None, 0.0

                if emp:
                    try:
                        live = liveness.detect(frame)
                    except Exception:
                        logger.exception("Liveness detection failed; skipping marking")
                        live = False

                    if live:
                        mark_attendance(emp, frame, conf)
                        label = f"{emp.user.get_full_name() or emp.user.username}: LIVE"
                        color = (0, 255, 0)
                    else:
                        logger.warning("Liveness failed for employee %s", getattr(emp, "employee_id", emp.pk))
                        label = "SPOOF"
                        color = (0, 0, 255)
                else:
                    label = "UNKNOWN"
                    color = (0, 255, 255)

                # draw label for preview (safe on headless servers; exceptions ignored)
                try:
                    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                except Exception:
                    pass

                if show:
                    try:
                        cv2.imshow("Face Attendance Kiosk", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except Exception:
                        # ignore imshow errors on headless systems
                        pass

        except KeyboardInterrupt:
            logger.info("Kiosk stopped by user (KeyboardInterrupt)")
        finally:
            try:
                cap.release()
            except Exception:
                logger.exception("Error releasing camera")
            if show:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            logger.info("Kiosk stopped")