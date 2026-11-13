"""Request identity for durable, deliberately redacted business audit events."""
from contextvars import ContextVar
import uuid

audit_context = ContextVar('audit_context', default={})


class AuditContextMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        token = audit_context.set({
            'actor_id': request.user.pk if request.user.is_authenticated else None,
            'actor_name': request.user.get_username() if request.user.is_authenticated else 'System',
            'request_id': getattr(request, 'request_id', uuid.uuid4().hex),
        })
        try:
            return self.get_response(request)
        finally:
            audit_context.reset(token)


def snapshot(instance):
    # Never retain passwords, embeddings, photos, documents or contact details.
    fields = {
        'employee_id', 'organisation_id', 'attendance_type', 'timestamp', 'notes',
        'role', 'is_active', 'employment_status', 'department_id', 'manager_id',
        'work_rule_id', 'assigned_location_id', 'date_of_joining', 'date_of_resignation',
        'base_salary', 'hra', 'allowances', 'deductions', 'effective_from',
        'month', 'basic_pay', 'net_salary', 'status', 'paid_date', 'present_days',
        'half_days', 'absent_days', 'overtime_hours', 'holiday_overtime_hours',
        'regular_overtime_pay', 'holiday_overtime_pay', 'start_date', 'end_date',
        'leave_type_id', 'full_day_hours', 'half_day_hours', 'overtime_rate',
        'working_days', 'shift_start_time', 'shift_end_time', 'date', 'name',
        'count_weekend_overtime', 'count_holiday_overtime', 'flexible_attendance',
    }
    return {field.attname: str(getattr(instance, field.attname))
            if getattr(instance, field.attname) is not None else None
            for field in instance._meta.concrete_fields if field.attname in fields}
