"""Audit business changes; payroll is generated only by an explicit pay run."""
import uuid
from django.db import transaction
from django.db.models.signals import pre_save, post_save, post_delete, pre_delete
from django.dispatch import receiver
from django.core.cache import cache
from core.models import AuditEvent
from core.security.audit import audit_context, snapshot
from django.contrib.auth.signals import user_logged_in, user_logged_out

AUDITED = {'Employee', 'Attendance', 'Payroll', 'SalaryStructure', 'WorkRule',
           'EmployeeShiftAssignment', 'LeaveApplication', 'LeaveApproval',
           'Holiday', 'OfficeLocation', 'Department', 'Resignation', 'AttendanceSettings'}


@receiver(pre_delete, sender='core.Employee')
def preserve_employee_history(sender, instance, **kwargs):
    from django.core.exceptions import ValidationError
    from core.models import Attendance, Payroll
    if Attendance.global_objects.filter(employee=instance).exists() or Payroll.global_objects.filter(employee=instance).exists():
        raise ValidationError('Deactivate this employee to preserve attendance and payroll history')


@receiver(user_logged_in)
@receiver(user_logged_out)
def audit_authentication(sender, request, user, signal, **kwargs):
    if user is None:
        return
    profile = getattr(user, 'employee', None)
    if profile and profile.organisation_id:
        AuditEvent.global_objects.create(organisation_id=profile.organisation_id,
            actor_id=user.pk, actor_name=user.get_username(), action='login' if signal is user_logged_in else 'logout',
            entity='Authentication', object_id=str(user.pk),
            request_id=getattr(request, 'request_id', ''))


@receiver(pre_save)
def remember_before(sender, instance, raw=False, **kwargs):
    if raw or sender.__name__ not in AUDITED or sender._meta.app_label != 'core':
        return
    previous = sender.global_objects.filter(pk=instance.pk).first() if instance.pk else None
    instance._audit_before = snapshot(previous) if previous else {}
    instance._biometric_changed = (sender.__name__ == 'Employee' and previous is not None
                                   and previous.face_encoding != instance.face_encoding)


def record(instance, action):
    if instance.organisation_id:
        after = {} if action == 'delete' else snapshot(instance)
        if action != 'delete' and getattr(instance, '_biometric_changed', False):
            after['face_enrolment_changed'] = True
        AuditEvent.global_objects.create(
            organisation_id=instance.organisation_id, entity=type(instance).__name__,
            object_id=str(instance.pk), action=action,
            before=snapshot(instance) if action == 'delete' else getattr(instance, '_audit_before', {}),
            after=after, **audit_context.get())


def invalidate_gallery(instance):
    org_id = instance.organisation_id
    def invalidate():
        # Shared version invalidates every worker, without caching biometrics in Redis.
        cache.set(f'face-gallery-version:{org_id}', uuid.uuid4().hex, timeout=None)
    transaction.on_commit(invalidate, robust=True)


@receiver(post_save)
def audit_saved(sender, instance, created, raw=False, **kwargs):
    if raw or sender.__name__ not in AUDITED or sender._meta.app_label != 'core':
        return
    if created or getattr(instance, '_audit_before', {}) != snapshot(instance) or getattr(instance, '_biometric_changed', False):
        record(instance, 'create' if created else 'update')
    if sender.__name__ == 'Employee':
        invalidate_gallery(instance)
    invalidate_payroll(sender, instance)


@receiver(post_delete)
def audit_deleted(sender, instance, **kwargs):
    if sender.__name__ in AUDITED and sender._meta.app_label == 'core':
        record(instance, 'delete')
        if sender.__name__ == 'Employee':
            invalidate_gallery(instance)
        invalidate_payroll(sender, instance)


def invalidate_payroll(sender, instance):
    from datetime import timedelta
    from django.utils import timezone
    from core.models import Payroll
    if sender.__name__ not in {'Attendance', 'SalaryStructure', 'WorkRule',
                               'EmployeeShiftAssignment', 'LeaveApplication', 'Holiday', 'Employee'}:
        return
    if sender.__name__ == 'Employee':
        before = getattr(instance, '_audit_before', {})
        after = snapshot(instance)
        if not any(before.get(f) != after.get(f) for f in (
                'work_rule_id', 'date_of_joining', 'date_of_resignation', 'employment_status')):
            return
    records = Payroll.global_objects.filter(organisation_id=instance.organisation_id, status='processed')
    employee_id = instance.pk if sender.__name__ == 'Employee' else getattr(instance, 'employee_id', None)
    if employee_id:
        records = records.filter(employee_id=employee_id)
    if sender.__name__ == 'Attendance':
        from datetime import datetime
        stamps = [instance.timestamp]
        previous = getattr(instance, '_audit_before', {}).get('timestamp')
        if previous:
            stamps.append(datetime.fromisoformat(previous))
        months = {(timezone.localtime(stamp).date() + timedelta(days=offset)).replace(day=1)
                  for stamp in stamps for offset in (-1, 0, 1)}
        records = records.filter(month__in=months)
    for payroll in records.order_by('employee_id', 'pk'):
        payroll.status = 'pending'
        payroll.save(update_fields=['status'])
