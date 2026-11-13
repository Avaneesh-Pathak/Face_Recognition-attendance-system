"""Staging regression tests: run under configured dev DB/Redis and Django."""
from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse
from django.core.files.base import ContentFile
from core.models import Organisation, Employee, Attendance
from core.security.crypto import decrypt_token, encrypt_id


class MultiTenantSecurityTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.a = Organisation.objects.create(name='Test Organisation A')
        cls.b = Organisation.objects.create(name='Test Organisation B')
        cls.ua = User.objects.create_user('org_a', password='safe-test-password-12345')
        cls.ub = User.objects.create_user('org_b', password='safe-test-password-12345')
        cls.ea = Employee.global_objects.create(organisation=cls.a, user=cls.ua,
                                                employee_id='EMP100', position='Nurse', role='Employee',
                                                phone_number='1234567890')
        cls.eb = Employee.global_objects.create(organisation=cls.b, user=cls.ub,
                                                employee_id='EMP100', position='Nurse', role='Employee',
                                                phone_number='1234567890')

    def test_tenant_a_cannot_view_tenant_b_profile(self):
        self.client.force_login(self.ua)
        response = self.client.get(reverse('employee_profile', args=[self.eb.pk]))
        self.assertIn(response.status_code, (403, 404))

    def test_signed_employee_url_scoped_to_organisation(self):
        token = encrypt_id('Employee', self.eb.pk, self.b.pk)
        self.assertEqual(decrypt_token(token)['i'], self.eb.pk)
        self.client.force_login(self.ua)
        response = self.client.get(reverse('employee_profile_secure', args=[token]))
        self.assertEqual(response.status_code, 404)

    def test_no_anonymous_scan(self):
        response = self.client.post(reverse('mark_attendance'))
        self.assertIn(response.status_code, (301, 302, 403))

    @override_settings(REQUIRE_VERIFIED_LIVENESS=True, FACE_LIVENESS_VERIFIER='')
    def test_no_unverified_liveness(self):
        # Attendance endpoint deliberately denies biometric marking by default.
        self.client.force_login(self.ua)
        response = self.client.post(reverse('mark_attendance'), data={
            'capture': ContentFile(b'not-an-image', name='fake.jpg')})
        self.assertIn(response.status_code, (400, 503))
        self.assertEqual(Attendance.global_objects.count(), 0)

    @override_settings(REQUIRE_VERIFIED_LIVENESS=False)
    def test_face_only_mode_reaches_recognition_and_is_labelled(self):
        from unittest.mock import patch
        from io import BytesIO
        from PIL import Image
        import uuid
        self.client.force_login(self.ua)
        page = self.client.get(reverse('attendance_page'))
        self.assertContains(page, 'Face-only mode')
        self.assertContains(page, 'no anti-spoofing')
        image = BytesIO()
        Image.new('RGB', (320, 320), 'white').save(image, format='JPEG')
        with patch('core.views.get_face_system') as get_matcher:
            get_matcher.return_value.recognize_from_frame.return_value = (None, 0.0, {'reason': 'UNKNOWN'})
            response = self.client.post(reverse('mark_attendance'), {
                'capture': ContentFile(image.getvalue(), name='scan.jpg'),
                'scan_session': str(uuid.uuid4()),
            })
        self.assertEqual(response.status_code, 200)
        get_matcher.return_value.recognize_from_frame.assert_called_once()
        self.assertEqual(response.json()['message'], 'Face not recognized')
        self.assertEqual(Attendance.global_objects.count(), 0)

    @override_settings(REQUIRE_VERIFIED_LIVENESS=True, FACE_LIVENESS_VERIFIER='')
    def test_missing_pad_is_explained_before_scanning(self):
        self.client.force_login(self.ua)
        response = self.client.get(reverse('attendance_page'))
        self.assertContains(response, 'Scanner setup incomplete')
        self.assertContains(response, 'let scannerBlocked = true;')

    def test_tenant_employee_ids_can_overlap(self):
        self.assertNotEqual(self.ea.pk, self.eb.pk)
        self.assertEqual(self.ea.employee_id, self.eb.employee_id)

    def test_face_templates_encrypted_on_disk(self):
        import json
        from django.db import connection
        self.ea.face_encoding = json.dumps([1.0] + [0.0] * 511)
        self.ea.save(update_fields=['face_encoding'])
        with connection.cursor() as cursor:
            cursor.execute('SELECT face_encoding FROM core_employee WHERE id=%s', [self.ea.pk])
            stored = cursor.fetchone()[0]
        self.assertTrue(stored.startswith('enc:v1:'))
        self.assertNotIn('[1.0, 0.0', stored)
        reloaded = Employee.global_objects.get(pk=self.ea.pk)
        self.assertEqual(json.loads(reloaded.face_encoding)[0], 1.0)

    def test_recognition_cannot_match_other_organisation(self):
        import json
        import numpy as np
        from core.face_system import FaceRecognitionSystem
        self.ea.face_encoding = json.dumps([1.0] + [0.0] * 511)
        self.eb.face_encoding = json.dumps([0.0, 1.0] + [0.0] * 510)
        self.ea.save(update_fields=['face_encoding'])
        self.eb.save(update_fields=['face_encoding'])
        matcher = FaceRecognitionSystem()
        org_a, score_a = matcher.identify_face(np.array([1.0] + [0.0] * 511), self.a.pk)
        org_b, score_b = matcher.identify_face(np.array([1.0] + [0.0] * 511), self.b.pk)
        self.assertEqual(org_a, self.ea.pk)
        self.assertIsNone(org_b)

    def test_no_csrf_bypass_for_capture(self):
        from django.test import Client
        guarded = Client(enforce_csrf_checks=True)
        guarded.force_login(self.ua)
        resp = guarded.post(reverse('mark_attendance'), data={
            'capture': ContentFile(b'not-an-image', name='fake.jpg')})
        self.assertEqual(resp.status_code, 403)
