"""Rebuild missing/pending face templates from existing employee photos."""
import json
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from core.models import Employee
from core.face_system import get_face_system, normalize
from core.security.audit import audit_context


def has_valid_template(raw):
    try:
        values = json.loads(raw) if raw else None
        if not isinstance(values, list) or not values:
            return False
        if isinstance(values[0], (int, float)):
            values = [values]
        return any((vector := normalize(value)) is not None and vector.size == 512 for value in values[:12])
    except (ValueError, TypeError):
        return False


class Command(BaseCommand):
    help = 'Repair missing or pending face enrolment from stored photos. Defaults to dry run.'

    def add_arguments(self, parser):
        parser.add_argument('--organisation', type=int, required=True)
        parser.add_argument('--employee', type=int, help='Internal employee primary key')
        parser.add_argument('--apply', action='store_true')

    def handle(self, *args, **options):
        employees = Employee.global_objects.filter(organisation_id=options['organisation'],
            organisation__is_active=True, is_active=True, user__is_active=True, employment_status='active')
        if options['employee']:
            employees = employees.filter(pk=options['employee'])
        matcher = get_face_system()
        repaired = 0
        for employee in employees.order_by('pk'):
            if has_valid_template(employee.face_encoding):
                continue
            if not employee.face_image or not Path(employee.face_image.path).is_file():
                self.stdout.write(f'Employee {employee.pk}: no saved photo; enrol a face first.')
                continue
            original_encoding, original_photo = employee.face_encoding, employee.face_image.name
            embedding = matcher.get_embedding(employee.face_image.path)
            if embedding is None:
                self.stdout.write(f'Employee {employee.pk}: photo needs replacement (one clear face required).')
                continue
            if options['apply']:
                token = audit_context.set({'actor_name': 'Enrolment repair',
                    'reason': 'Rebuilt missing face template from the employee stored photo'})
                try:
                    with transaction.atomic():
                        locked = Employee.global_objects.select_for_update().get(pk=employee.pk,
                            organisation_id=options['organisation'], is_active=True)
                        if locked.face_encoding != original_encoding or locked.face_image.name != original_photo:
                            raise CommandError('Employee enrolment changed during repair; retry the command')
                        locked.face_encoding = json.dumps(embedding)
                        locked.save(update_fields=['face_encoding'])
                finally:
                    audit_context.reset(token)
                matcher.invalidate_tenant(employee.organisation_id)
                repaired += 1
                self.stdout.write(f'Employee {employee.pk}: repaired.')
            else:
                self.stdout.write(f'Employee {employee.pk}: valid photo, ready to repair with --apply.')
        self.stdout.write(f'Repaired {repaired} enrolment(s).')
