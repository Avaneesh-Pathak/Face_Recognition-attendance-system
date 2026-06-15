"""
ROTATIONAL SHIFT FIX — Drop-in replacement for views.py session logic
=====================================================================

ROOT CAUSE:
-----------
The original `split_sessions_by_shift` splits sessions at calendar midnight or a
fixed boundary_time.  This breaks for rotational employees because:

  • Employee B: check-in 14-Jun 20:00 → check-out 15-Jun 14:00   (night shift, 18 h)
    → midnight split credits 4 h to Jun 14 AND 14 h to Jun 15
    → Jun 15 now looks "present" for B even though B's work day was Jun 14

  • Employee A: check-in 15-Jun 14:00 → check-out 16-Jun 08:00   (day→night boundary)
    → midnight split credits 10 h to Jun 15 AND 8 h to Jun 16
    → Jun 16 looks "present" for A even though A's work day was Jun 15

FIX STRATEGY:
-------------
Assign the ENTIRE session to the "work date" determined by when the CHECK-IN
(session start) falls relative to each day's shift window.  No hour-splitting
across calendar days for the attendance status decision.

We still track total hours per work-date accurately by summing session durations,
but the "present/absent" attribution key is the work-date of the check-in.
"""

from datetime import date, datetime, time, timedelta
from collections import defaultdict
from django.utils import timezone


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: determine the "work date" (shift ownership date) for a given datetime
# ──────────────────────────────────────────────────────────────────────────────

def get_work_date_for_checkin(checkin_dt: datetime, work_rule) -> date:
    """
    Returns the canonical 'work date' that the session STARTING at checkin_dt
    belongs to, based on the employee's shift window and actual check-in time.
    """
    if work_rule is None:
        return checkin_dt.date()

    hour = checkin_dt.hour

    # 1. Day or Evening shift check-ins (between 6:00 AM and 7:00 PM / 19:00)
    #    always belong to the calendar day of the check-in.
    if 6 <= hour < 19:
        return checkin_dt.date()

    # 2. Night shift check-ins starting in the evening (7:00 PM to midnight)
    #    also belong to the calendar day of the check-in.
    if hour >= 19:
        return checkin_dt.date()

    # 3. Night shift check-ins after midnight (12:00 AM to 6:00 AM):
    #    If the employee's default work rule is a night shift, this is likely a 
    #    late check-in for a shift that started yesterday evening.
    shift_start = work_rule.shift_start_time
    shift_end = work_rule.shift_end_time
    is_rule_night_shift = shift_end <= shift_start

    if is_rule_night_shift:
        return checkin_dt.date() - timedelta(days=1)

    # Otherwise, fallback to the current calendar date
    return checkin_dt.date()

# ──────────────────────────────────────────────────────────────────────────────
# FIXED: build_sessions  (same interface as before, no change needed here)
# ──────────────────────────────────────────────────────────────────────────────

def build_sessions(logs):
    """
    Pair check-in / check-out logs into (start, end) sessions.
    Unmatched check-ins (employee still on duty) are returned as open sessions
    with end=None so the calendar can show "On Duty" correctly.
    """
    sessions = []
    open_in = None

    for log in sorted(logs, key=lambda x: x.timestamp):
        if log.attendance_type == "check_in":
            # Close any previously unclosed check-in (shouldn't happen normally)
            open_in = timezone.localtime(log.timestamp)

        elif log.attendance_type == "check_out" and open_in is not None:
            out_time = timezone.localtime(log.timestamp)
            if out_time > open_in:
                sessions.append({"start": open_in, "end": out_time})
            open_in = None

    # Open (unclosed) session → employee currently on duty
    if open_in is not None:
        sessions.append({"start": open_in, "end": None})   # end=None = on duty

    return sessions


# ──────────────────────────────────────────────────────────────────────────────
# FIXED: split_sessions_by_shift  ← THIS IS THE KEY REPLACEMENT
# ──────────────────────────────────────────────────────────────────────────────

def split_sessions_by_shift(sessions, work_rule):
    """
    Attribute each session to a single 'work date' based on when the check-in
    (session start) occurred relative to the shift window.

    Returns:
        daily_map: dict[date -> {"hours": float, "segments": list, "on_duty": bool}]

    Replaces the original split_sessions_by_shift that split at midnight
    boundary_time, which broke rotational / 12-hour shift workers.
    """
    daily_map = defaultdict(lambda: {"hours": 0.0, "segments": [], "on_duty": False})

    now = timezone.now()

    for s in sessions:
        start = s["start"]
        end   = s["end"]          # None = still on duty

        # ── Determine work date from check-in time ──────────────────────────
        work_date = get_work_date_for_checkin(start, work_rule)

        # ── Hours calculation ───────────────────────────────────────────────
        if end is None:
            # Employee is currently on duty — show hours so far
            hours = (now - start).total_seconds() / 3600
            daily_map[work_date]["on_duty"] = True
        else:
            hours = (end - start).total_seconds() / 3600

        daily_map[work_date]["hours"] += hours
        daily_map[work_date]["segments"].append(
            {"start": start, "end": end or now}
        )

    return daily_map


# ──────────────────────────────────────────────────────────────────────────────
# FIXED: shift type detector (used for chip display in the calendar)
# ──────────────────────────────────────────────────────────────────────────────

def get_shift_from_time(dt: datetime) -> str:
    """
    Detect shift label from the check-in datetime.
    Kept simple — returns Night / Day / Evening / Shift Change.
    """
    hour = dt.hour
    if hour >= 19 or hour < 6:
        return "Night"
    if 6 <= hour < 14:
        return "Day"
    if 14 <= hour < 19:
        return "Evening"
    return "Shift Change"


# ──────────────────────────────────────────────────────────────────────────────
# HOW TO INTEGRATE INTO attendance_calendar VIEW
# ──────────────────────────────────────────────────────────────────────────────
#
# 1. In views.py, replace the two local function definitions:
#       def build_sessions(logs):  ...
#       def split_sessions_by_shift(sessions, rule):  ...
#    with imports from this file (or copy the bodies in).
#
# 2. The rest of the attendance_calendar view stays EXACTLY the same.
#    The daily_map it receives now has correct work-date attribution.
#
# 3. In the template, the "on_duty" flag per day is now available if you want
#    to show a "🟢 On Duty" badge instead of only using check_out == "—".
#
# EXAMPLE IMPORT (top of views.py):
#   from .rotational_shift_fix import (
#       build_sessions,
#       split_sessions_by_shift,
#       get_shift_from_time,
#   )
#
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# ALSO FIX: EnterprisePayrollManager.generate_salary  (models.py)
# ──────────────────────────────────────────────────────────────────────────────
#
# The payroll engine does the same midnight-split for daily_hours_map.
# Replace the session → daily_hours section with this logic:
#
#   daily_hours_map = {}
#   for start, end in sessions:                # sessions = list of (datetime, datetime)
#       work_date = get_work_date_for_checkin(start, emp.work_rule)
#       hours = (end - start).total_seconds() / 3600
#       daily_hours_map[work_date] = daily_hours_map.get(work_date, 0) + hours
#
# That single change makes payroll consistent with the calendar.
# ──────────────────────────────────────────────────────────────────────────────