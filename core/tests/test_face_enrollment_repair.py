from io import StringIO
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import numpy as np
from django.core.files.base import ContentFile
from django.core.management import call_command
from django.test import TestCase, SimpleTestCase, override_settings
from core.models import Employee, AuditEvent, Attendance
from core.tests import test_payroll_calendar_concurrency as fixtures
from core.face_system import FaceRecognitionSystem


class EnrollmentRepairTests(TestCase):
    setUpTestData = classmethod(fixtures.PayrollCalendarTests.setUpTestData.__func__)

    def test_pending_enrollment_repair_is_scoped_idempotent_and_audited(self):
        with TemporaryDirectory() as media, override_settings(MEDIA_ROOT=media):
            for employee in (self.emp, self.other_emp):
                employee.face_image.save('photo.jpg', ContentFile(b'fixture-photo'), save=False)
                employee.face_encoding = json.dumps({'status': 'FACE_PENDING'})
                employee.save()
            with patch('core.management.commands.repair_face_enrollments.get_face_system') as matcher:
                matcher.return_value.get_embedding.return_value = [1.0] + [0.0] * 511
                call_command('repair_face_enrollments', organisation=self.org.pk, stdout=StringIO())
                self.emp.refresh_from_db()
                self.assertIsInstance(json.loads(self.emp.face_encoding), dict)
                call_command('repair_face_enrollments', organisation=self.org.pk, apply=True, stdout=StringIO())
                self.emp.refresh_from_db(); self.other_emp.refresh_from_db()
                self.assertEqual(len(json.loads(self.emp.face_encoding)), 512)
                self.assertIsInstance(json.loads(self.other_emp.face_encoding), dict)
                event = AuditEvent.global_objects.latest('pk')
                self.assertEqual(event.actor_name, 'Enrolment repair')
                self.assertTrue(event.after['face_enrolment_changed'])
                self.assertNotIn('face_encoding', event.after)
                count = AuditEvent.global_objects.count()
                call_command('repair_face_enrollments', organisation=self.org.pk, apply=True, stdout=StringIO())
                self.assertEqual(AuditEvent.global_objects.count(), count)
                self.assertFalse(Attendance.global_objects.exists())


@override_settings(REDIS_URL='')
class MissingEnrollmentTests(SimpleTestCase):
    def test_empty_gallery_has_actionable_reason(self):
        matcher = FaceRecognitionSystem()
        face = SimpleNamespace(bbox=[0, 0, 200, 200], det_score=.99, embedding=np.ones(512))
        with patch.object(matcher, '_infer', return_value=[face]), patch.object(
                matcher, 'load_from_db', return_value=(np.empty((0, 512)), np.array([]))):
            result = matcher.recognize_from_frame(np.zeros((320, 320, 3)), 1, 'device')
        self.assertIsNone(result[0])
        self.assertEqual(result[2]['reason'], 'NO_ENROLLED_FACES')
