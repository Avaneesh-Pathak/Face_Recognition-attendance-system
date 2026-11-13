"""Shared attendance normalization for calendar and payroll.

A completed check-in/out pair belongs to the date of its check-in (respecting
assigned overnight shifts). Open check-ins are shown as pending, not payable
hours; duplicate/inconsistent events do not fabricate paid shifts.
"""
from collections import defaultdict
from datetime import timedelta
from decimal import Decimal
from django.utils import timezone
from core.rotational_shift_fix import get_work_date_for_checkin

MAX_SESSION_HOURS = 20  # Flag implausible sessions for human correction.


def workday_sessions(logs, employee):
    result = defaultdict(lambda: {'hours': Decimal('0'), 'segments': [], 'pending': False,
                                  'check_in': None, 'check_out': None, 'logs': [], 'issues': []})
    checkin = None
    for log in sorted(logs, key=lambda rec: (rec.timestamp, rec.pk or 0)):
        if log.attendance_type == 'check_in':
            # A second check-in cannot silently close a prior one.
            if checkin is not None:
                old = result[get_work_date_for_checkin(checkin.timestamp, employee)]
                old['pending'] = True
                old['issues'].append('Check-in without a matching checkout')
            checkin = log
            day = get_work_date_for_checkin(log.timestamp, employee)
            result[day]['check_in'] = result[day]['check_in'] or timezone.localtime(log.timestamp)
            result[day]['logs'].append(log)
        elif log.attendance_type == 'check_out' and checkin is not None:
            duration = Decimal(str((log.timestamp - checkin.timestamp).total_seconds())) / Decimal('3600')
            day = get_work_date_for_checkin(checkin.timestamp, employee)
            result[day]['logs'].append(log)
            if 0 < duration <= MAX_SESSION_HOURS:
                entry = result[day]
                entry['hours'] += duration
                entry['segments'].append({'start': timezone.localtime(checkin.timestamp),
                                          'end': timezone.localtime(log.timestamp)})
                entry['check_out'] = timezone.localtime(log.timestamp)
            else:
                result[day]['pending'] = True
                result[day]['issues'].append('Invalid session duration; review required')
            checkin = None
        elif log.attendance_type == 'check_out':
            entry = result[timezone.localtime(log.timestamp).date()]
            entry['logs'].append(log)
            entry['check_out'] = timezone.localtime(log.timestamp)
            entry['pending'] = True
            entry['issues'].append('Checkout without a matching check-in')
    if checkin is not None:
        result[get_work_date_for_checkin(checkin.timestamp, employee)]['pending'] = True
    return dict(result)


def day_status(day, employee, entry, rule, holiday=False, leave=None):
    """One attendance classification for the HTML, API and payroll policy."""
    from core.models import is_working_day, get_employee_rule_from_rule
    if day < employee.date_of_joining or (employee.date_of_resignation and day > employee.date_of_resignation):
        return 'not_employed', 'Not employed'
    if day > timezone.localdate():
        return 'future', 'Upcoming'
    if entry.get('pending'):
        return 'pending', 'Needs review' if entry.get('issues') else 'Pending checkout'
    if leave is not None:
        return ('paid_leave', 'Paid leave') if leave else ('unpaid_leave', 'Unpaid leave')
    working = is_working_day(day, get_employee_rule_from_rule(rule)) if rule else day.weekday() < 5
    if holiday:
        return 'holiday', 'Holiday · worked' if entry.get('hours') else 'Holiday'
    if not working:
        return 'rest', 'Rest day · worked' if entry.get('hours') else 'Rest day'
    hours = entry.get('hours', Decimal('0'))
    if hours >= Decimal(str(rule.full_day_hours if rule else 8)) * Decimal('0.90'):
        return 'present', 'Present'
    if hours >= Decimal(str(rule.half_day_hours if rule else 4)):
        return 'half', 'Half day'
    if not hours and day == timezone.localdate():
        return 'awaiting', 'Not checked in'
    return 'absent', 'Absent' if not hours else 'Insufficient hours'
