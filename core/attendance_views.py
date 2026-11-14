import calendar
from datetime import date, datetime, timedelta
from decimal import Decimal
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Q
from django.http import JsonResponse, HttpResponseForbidden
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST
from core.models import Attendance, Employee, Payroll, AuditEvent
from core.security.attendance_permissions import can_correct_attendance
from core.services.calendar import calendar_days
from core.utils.access import get_visible_employees


def visible_employee(request, pk):
    return get_object_or_404(get_visible_employees(request.user).select_related('user', 'work_rule'), pk=pk)


@login_required
@require_GET
def attendance_calendar(request, employee_id, year, month):
    if not 2000 <= year <= 2100 or not 1 <= month <= 12:
        return JsonResponse({'error': 'Invalid calendar period'}, status=400)
    employee = visible_employee(request, employee_id)
    first = date(year, month, 1)
    last = date(year, month, calendar.monthrange(year, month)[1])
    days = calendar_days(employee, first, last)
    previous = first - timedelta(days=1)
    following = last + timedelta(days=1)
    totals = {key: sum(d['status'] == key for d in days)
              for key in ('present', 'half', 'absent', 'pending', 'paid_leave', 'unpaid_leave')}
    return render(request, 'attendance_calendar.html', {
        'employee': employee, 'data': days, 'year': year, 'month': month,
        'month_name': calendar.month_name[month], 'padding': range(first.weekday()),
        'today_year': timezone.localdate().year, 'today_month': timezone.localdate().month,
        'today_date': timezone.localdate().isoformat(),
        'display_timezone': timezone.get_current_timezone_name(), 'total_days': last.day,
        'day_names': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        'previous_year': previous.year, 'previous_month': previous.month,
        'next_year': following.year, 'next_month': following.month,
        'total_working_hours': sum((d['hours'] for d in days), Decimal('0')),
        'totals': totals, 'can_edit_attendance': can_correct_attendance(request.user, employee),
        'all_employees': get_visible_employees(request.user).select_related('user'),
    })


@login_required
@require_GET
def attendance_calendar_data(request, employee_id, year, month):
    if not 2000 <= year <= 2100 or not 1 <= month <= 12:
        return JsonResponse({'error': 'Invalid calendar period'}, status=400)
    employee = visible_employee(request, employee_id)
    days = calendar_days(employee, date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1]))
    return JsonResponse({'employee': employee.user.get_full_name(), 'year': year, 'month': month,
        'month_name': calendar.month_name[month], 'data': [
            {key: value for key, value in day.items() if key in (
                'day', 'date', 'status', 'label', 'present', 'pending', 'hours',
                'is_weekend', 'is_future', 'check_in', 'check_out', 'issues')}
            for day in days]})


@login_required
@require_GET
def attendance_day_detail(request, emp_id, date):
    try:
        target = datetime.strptime(date, '%Y-%m-%d').date()
    except ValueError:
        return JsonResponse({'error': 'Invalid date'}, status=400)
    employee = visible_employee(request, emp_id)
    day = calendar_days(employee, target, target)[0]
    return JsonResponse({
        'date': target.strftime('%d %B %Y'), 'status': day['label'],
        'total_hours': day['hours_display'], 'pending': day['pending'], 'issues': day.get('issues', []),
        'can_edit': can_correct_attendance(request.user, employee),
        'timeline': [{'start': x['start'].isoformat(), 'end': x['end'].isoformat()}
                     for x in day.get('segments', [])],
        'logs': [{'id': rec.pk, 'type': rec.get_attendance_type_display(),
                  'attendance_type': rec.attendance_type,
                  'timestamp': timezone.localtime(rec.timestamp).strftime('%I:%M %p'),
                  'date': timezone.localtime(rec.timestamp).strftime('%d-%m-%Y'),
                  'local_timestamp': timezone.localtime(rec.timestamp).strftime('%Y-%m-%dT%H:%M'),
                  'location': rec.location or '', 'notes': rec.notes or ''}
                 for rec in day.get('logs', [])],
    })


def correction(request, att_id=None, deleting=False):
    if not can_correct_attendance(request.user):
        return JsonResponse({'error': 'Only HR and Admin can correct attendance'}, status=403)
    record = get_object_or_404(Attendance.objects, pk=att_id) if att_id else None
    employee = record.employee if record else get_object_or_404(Employee.objects, pk=request.POST.get('employee') or 0)
    if not can_correct_attendance(request.user, employee):
        return JsonResponse({'error': 'Only Admin can correct their own attendance'}, status=403)
    reason = request.POST.get('reason', '').strip()
    if len(reason) < 5 or len(reason) > 1000:
        return JsonResponse({'error': 'Provide a correction reason (5–1000 characters)'}, status=400)
    from core.security.audit import audit_context
    audit_context.set({**audit_context.get(), 'reason': reason})
    try:
        stamp = None
        if not deleting:
            stamp = datetime.fromisoformat(request.POST.get('timestamp', ''))
            if timezone.is_naive(stamp):
                stamp = timezone.make_aware(stamp)
            if stamp > timezone.now():
                raise ValueError('Attendance cannot be in the future')
            if request.POST.get('attendance_type') not in ('check_in', 'check_out'):
                raise ValueError('Choose Check In or Check Out')
            if timezone.localtime(stamp).date() < employee.date_of_joining:
                raise ValueError('Attendance predates employment')
            if employee.date_of_resignation and timezone.localtime(stamp).date() > employee.date_of_resignation:
                raise ValueError('Attendance is after employment ended')
        with transaction.atomic():
            Employee.global_objects.select_for_update().get(pk=employee.pk)
            if record:
                record = get_object_or_404(Attendance.objects.select_for_update(), pk=record.pk)
            dates = [timezone.localtime(s).date() for s in (stamp, record.timestamp if record else None) if s]
            # Neighbouring dates protect overnight sessions on a month boundary.
            months = {d.replace(day=1) for day in dates for d in (day - timedelta(days=1), day, day + timedelta(days=1))}
            affected = Payroll.global_objects.filter(organisation_id=employee.organisation_id,
                employee=employee, month__in=months)
            if deleting:
                record.delete()
            else:
                record = record or Attendance(employee=employee, organisation_id=employee.organisation_id)
                record.timestamp = stamp
                record.attendance_type = request.POST['attendance_type']
                record.location = request.POST.get('location', '')
                record.notes = reason
                record.full_clean()
                record.save()
            # Paid payrolls are immutable snapshots. Corrections can still be
            # recorded for historical attendance, while processed (unpaid)
            # payrolls must be recalculated from the corrected source data.
            for payroll in affected.filter(status='processed'):
                payroll.status = 'pending'
                payroll.save(update_fields=['status'])
        return JsonResponse({'success': True})
    except (ValueError, TypeError, ValidationError) as exc:
        return JsonResponse({'error': '; '.join(exc.messages) if isinstance(exc, ValidationError) else str(exc)}, status=400)


@login_required
@require_POST
def create_attendance(request):
    return correction(request)


@login_required
@require_POST
def update_attendance(request, att_id):
    return correction(request, att_id)


@login_required
@require_POST
def delete_attendance(request, att_id):
    return correction(request, att_id, deleting=True)


@login_required
@require_GET
def audit_log(request):
    if not request.user.is_superuser:
        return HttpResponseForbidden('Audit history is restricted to system superusers')
    events = AuditEvent.objects.select_related('organisation').all()
    entity = request.GET.get('entity', '')
    action = request.GET.get('action', '')
    if entity:
        events = events.filter(entity=entity)
    if action:
        events = events.filter(action=action)
    actor = request.GET.get('actor', '').strip()[:150]
    if actor:
        events = events.filter(actor_name__icontains=actor)
    for param, lookup in (('start', 'created_at__date__gte'), ('end', 'created_at__date__lte')):
        value = request.GET.get(param)
        if value:
            try:
                events = events.filter(**{lookup: date.fromisoformat(value)})
            except ValueError:
                return JsonResponse({'error': 'Invalid audit date filter'}, status=400)
    from core.signals import AUDITED
    filters = request.GET.copy()
    filters.pop('page', None)
    page_obj = Paginator(events, 50).get_page(request.GET.get('page'))
    action_choices = (
        ('create', 'Create'), ('update', 'Update'), ('delete', 'Delete'),
        ('login', 'Login'), ('logout', 'Logout'), ('lock', 'Lock'),
        ('unlock', 'Unlock'), ('unlock_failed', 'Unlock failed'),
    )
    action_labels = dict(action_choices)
    field_labels = {
        'employee_id': 'Employee code', 'employee_name': 'Employee name',
        'employee_code': 'Employee code',
        'organisation_id': 'Organisation ID',
        'attendance_type': 'Attendance event', 'timestamp': 'Attendance time',
        'actor_id': 'Actor ID', 'actor_name': 'Actor name', 'request_id': 'Request ID',
    }
    page_events = list(page_obj.object_list)
    employee_pks = set()
    employee_user_ids = set()
    for event in page_events:
        if event.actor_id:
            employee_user_ids.add(event.actor_id)
        if event.entity == 'Authentication':
            try:
                employee_user_ids.add(int(event.object_id))
            except (TypeError, ValueError):
                pass
        elif event.entity == 'Employee':
            try:
                employee_pks.add(int(event.object_id))
            except (TypeError, ValueError):
                pass
        else:
            before = event.before if isinstance(event.before, dict) else {}
            after = event.after if isinstance(event.after, dict) else {}
            employee_id = after.get('employee_id') or before.get('employee_id')
            try:
                if employee_id:
                    employee_pks.add(int(employee_id))
            except (TypeError, ValueError):
                pass

    employee_profiles = Employee.global_objects.filter(
        Q(pk__in=employee_pks) | Q(user_id__in=employee_user_ids)
    ).select_related('user')
    profiles_by_pk = {employee.pk: employee for employee in employee_profiles}
    profiles_by_user = {employee.user_id: employee for employee in employee_profiles}

    for event in page_events:
        before = event.before if isinstance(event.before, dict) else {}
        after = event.after if isinstance(event.after, dict) else {}
        snapshot_values = before if event.action == 'delete' else after
        subject_profile = None
        if event.entity == 'Employee':
            try:
                subject_profile = profiles_by_pk.get(int(event.object_id))
            except (TypeError, ValueError):
                pass
        elif event.entity == 'Authentication':
            try:
                subject_profile = profiles_by_user.get(int(event.object_id))
            except (TypeError, ValueError):
                pass
        else:
            employee_id = snapshot_values.get('employee_id') or before.get('employee_id')
            try:
                if employee_id:
                    subject_profile = profiles_by_pk.get(int(employee_id))
            except (TypeError, ValueError):
                pass

        actor_profile = profiles_by_user.get(event.actor_id)
        if actor_profile and actor_profile.organisation_id != event.organisation_id:
            actor_profile = None
        if subject_profile and subject_profile.organisation_id != event.organisation_id:
            subject_profile = None
        event.actor_display_name = (
            actor_profile.user.get_full_name() or actor_profile.user.get_username()
            if actor_profile else event.actor_name
        )
        event.actor_employee_code = actor_profile.employee_id if actor_profile else ''
        event.subject_employee_name = (
            (subject_profile.user.get_full_name() or subject_profile.user.get_username())
            if subject_profile else snapshot_values.get('employee_name', '')
        )
        event.subject_employee_code = (
            subject_profile.employee_id if subject_profile
            else (snapshot_values.get('employee_code') or
                  (snapshot_values.get('employee_id', '') if event.entity == 'Employee' else ''))
        )

        if event.action == 'create':
            changed_fields = after.keys()
        elif event.action == 'delete':
            changed_fields = before.keys()
        elif event.action == 'update':
            changed_fields = {key for key in before.keys() | after.keys()
                              if before.get(key) != after.get(key)}
        else:
            changed_fields = set()
        event.action_label = action_labels.get(event.action, event.action.replace('_', ' ').title())
        event.change_rows = [
            {'label': ('Employee code' if field == 'employee_id' and event.entity == 'Employee'
                       else 'Employee record reference' if field == 'employee_id'
                       else field_labels.get(field, field.replace('_', ' ').capitalize())),
             'before': before.get(field), 'after': after.get(field)}
            for field in sorted(changed_fields)
        ]
    return render(request, 'audit_log.html', {
        'page_obj': page_obj,
        'action_choices': action_choices,
        'entities': sorted(AUDITED | {'User', 'Authentication'}), 'filter_query': filters.urlencode(),
    })
