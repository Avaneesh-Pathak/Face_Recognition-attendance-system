"""Tenant-scoped calendar context, with overnight events attached to work dates."""
import calendar
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from django.db.models import Q
from django.utils import timezone
from core.models import Attendance, EmployeeShiftAssignment, Holiday, LeaveApplication
from core.services.attendance_engine import workday_sessions, day_status


def calendar_days(employee, first, last):
    bounds = (timezone.make_aware(datetime.combine(first - timedelta(days=2), time.min)),
              timezone.make_aware(datetime.combine(last + timedelta(days=2), time.max)))
    logs = Attendance.global_objects.filter(organisation_id=employee.organisation_id,
        employee=employee, timestamp__range=bounds).order_by('timestamp', 'pk')
    shifts = list(EmployeeShiftAssignment.global_objects.filter(
        organisation_id=employee.organisation_id, employee=employee, start_date__lte=last
    ).filter(Q(end_date__isnull=True) | Q(end_date__gte=first - timedelta(days=2)))
      .select_related('work_rule').order_by('-start_date', '-pk'))
    # The same resolver is reused for early-morning work date attribution.
    employee._calendar_shifts = shifts
    daily = workday_sessions(logs, employee)
    holidays = set(Holiday.global_objects.filter(organisation_id=employee.organisation_id,
        date__range=(first, last)).values_list('date', flat=True))
    leaves = {}
    for leave in LeaveApplication.global_objects.filter(organisation_id=employee.organisation_id,
            employee=employee, status='approved', start_date__lte=last, end_date__gte=first).select_related('leave_type'):
        day = max(first, leave.start_date)
        while day <= min(last, leave.end_date):
            leaves[day] = leaves.get(day, True) and leave.leave_type.is_paid
            day += timedelta(days=1)
    result = []
    day = first
    while day <= last:
        rule = next((s.work_rule for s in shifts if s.start_date <= day and
                     (s.end_date is None or s.end_date >= day)), employee.work_rule)
        entry = daily.get(day, {})
        status, label = day_status(day, employee, entry, rule, day in holidays, leaves.get(day))
        hours = entry.get('hours', Decimal('0'))
        result.append(dict(entry, date=day, day=day.day, status=status, label=label,
            hours=hours, hours_display=f'{hours:.2f}',
            present=status == 'present', pending=bool(entry.get('pending')),
            is_future=day > timezone.localdate(), is_today=day == timezone.localdate(),
            is_weekend=status == 'rest', rule=rule))
        day += timedelta(days=1)
    return result
