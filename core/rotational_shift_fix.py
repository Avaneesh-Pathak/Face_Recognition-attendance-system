# core/shifts.py
from datetime import date, datetime, timedelta
from django.utils import timezone
from collections import defaultdict

def get_work_date_for_checkin(checkin_dt: datetime, employee) -> date:
    """
    Dynamically determines the correct 'work date' for a check-in, taking into 
    account standard day shifts, night shifts, and dynamic rotational shift assignments.
    """
    # Lazy import inside the function to prevent circular dependency loops
    from core.models import get_shift_for_day

    # Ensure datetime is local
    local_checkin = timezone.localtime(checkin_dt)
    calendar_date = local_checkin.date()
    # Use the actual prior shift end, rather than a universal 06:00 cutoff.
    yesterday = calendar_date - timedelta(days=1)
    yesterday_rule = get_shift_for_day(employee, yesterday)
    
    if yesterday_rule:
        # A night shift crosses midnight (end time <= start time)
        is_yesterday_night_shift = yesterday_rule.shift_end_time <= yesterday_rule.shift_start_time
        if is_yesterday_night_shift and local_checkin.time() < yesterday_rule.shift_end_time:
            # Belongs to yesterday's night shift
            return yesterday

    # Otherwise, fallback to the current calendar date
    return calendar_date


def build_sessions(logs):
    """
    Pair check-in / check-out logs into (start, end) sessions.
    Unmatched check-ins (employee still on duty) are returned with end=None.
    """
    sessions = []
    open_in = None

    for log in sorted(logs, key=lambda x: x.timestamp):
        if log.attendance_type == "check_in":
            open_in = timezone.localtime(log.timestamp)
        elif log.attendance_type == "check_out" and open_in is not None:
            out_time = timezone.localtime(log.timestamp)
            if out_time > open_in:
                sessions.append({"start": open_in, "end": out_time})
            open_in = None

    if open_in is not None:
        sessions.append({"start": open_in, "end": None})

    return sessions


def split_sessions_by_shift(sessions, employee):
    """
    Attributes each session to a single 'work date' using dynamic rotation lookups.
    """
    daily_map = defaultdict(lambda: {"hours": 0.0, "segments": [], "on_duty": False})
    now = timezone.now()

    for s in sessions:
        start = s["start"]
        end   = s["end"]

        # Uses the dynamic date resolver we defined above
        work_date = get_work_date_for_checkin(start, employee)

        if end is None:
            hours = (now - start).total_seconds() / 3600
            daily_map[work_date]["on_duty"] = True
        else:
            hours = (end - start).total_seconds() / 3600

        daily_map[work_date]["hours"] += hours
        daily_map[work_date]["segments"].append(
            {"start": start, "end": end or now}
        )

    return daily_map


def get_shift_from_time(dt: datetime) -> str:
    """Detect shift label from the check-in datetime."""
    hour = dt.hour
    if hour >= 19 or hour < 6:
        return "Night"
    if 6 <= hour < 14:
        return "Day"
    if 14 <= hour < 19:
        return "Evening"
    return "Shift Change"
