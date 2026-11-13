from unittest.mock import patch

from django.contrib.auth.hashers import check_password
from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from core.models import AuditEvent, Employee, Organisation
from core.security.kiosk import SESSION_KEY


@override_settings(CACHES={'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}})
class AttendanceKioskTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.org = Organisation.objects.create(name='Kiosk organisation')
        cls.user = User.objects.create_user('kiosk-admin', password='test-admin-password')
        cls.employee = Employee.global_objects.create(
            organisation=cls.org, user=cls.user, employee_id='K-1', role='Admin')

    def setUp(self):
        cache.clear()
        self.client.force_login(self.user)

    def lock(self, client=None):
        return (client or self.client).post(reverse('kiosk_lock'), {'code': 'unique-648201', 'confirm_code': 'unique-648201'})

    def test_lock_stores_only_hash_and_survives_reload(self):
        self.assertEqual(self.lock().status_code, 200)
        state = self.client.session[SESSION_KEY]
        self.assertTrue(check_password('unique-648201', state['code_hash']))
        response = self.client.get(reverse('attendance_page'))
        self.assertContains(response, 'data-kiosk-locked="true"')
        self.assertNotContains(response, state['code_hash'])
        self.assertNotContains(response, 'unique-648201')
        self.assertIn('no-store', response['Cache-Control'])
        self.assertTrue(self.client.get(reverse('kiosk_status')).json()['locked'])

    def test_locked_session_blocks_navigation_mutations_and_logout(self):
        self.lock()
        for url in ('dashboard', 'payroll_list', 'attendance_settings', 'logout', 'login'):
            self.assertRedirects(self.client.get(reverse(url)), reverse('attendance_page'), fetch_redirect_response=False)
        for url in ('attendance_create', 'generate_payroll', 'logout'):
            self.assertEqual(self.client.post(reverse(url)).status_code, 423)
        self.assertEqual(self.client.get(reverse('dashboard'), HTTP_ACCEPT='application/json').status_code, 423)

    def test_scanner_and_logs_remain_available(self):
        self.lock()
        self.assertEqual(self.client.get(reverse('attendance_log_api')).status_code, 200)
        response = self.client.post(reverse('mark_attendance'))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()['message'], 'No image data')

    def test_wrong_code_cannot_unlock_or_replace_code(self):
        self.lock()
        self.assertEqual(self.client.post(reverse('kiosk_unlock'), {'code': 'wrong-code'}).status_code, 403)
        self.assertEqual(self.lock().status_code, 409)
        self.assertIn(SESSION_KEY, self.client.session)
        self.assertEqual(self.client.post(reverse('kiosk_unlock'), {'code': 'unique-648201'}).status_code, 200)
        self.assertNotIn(SESSION_KEY, self.client.session)
        self.assertEqual(self.client.get(reverse('dashboard')).status_code, 200)

    def test_attempt_limit_survives_reload_and_ip_change(self):
        self.lock()
        for attempt in range(5):
            self.assertEqual(self.client.post(reverse('kiosk_unlock'), {'code': 'wrong-code'}, REMOTE_ADDR=f'10.0.0.{attempt}').status_code, 403)
        self.client.get(reverse('attendance_page'))
        response = self.client.post(reverse('kiosk_unlock'), {'code': 'unique-648201'})
        self.assertEqual(response.status_code, 429)
        self.assertIn(SESSION_KEY, self.client.session)

    def test_cache_failure_keeps_kiosk_locked(self):
        self.lock()
        with patch('core.security.kiosk.cache') as unavailable_cache:
            unavailable_cache.add.side_effect = RuntimeError('offline')
            response = self.client.post(reverse('kiosk_unlock'), {'code': 'unique-648201'})
        self.assertEqual(response.status_code, 503)
        self.assertIn(SESSION_KEY, self.client.session)

    def test_employee_cannot_enable_and_codes_must_match(self):
        for values in ({'code': '123', 'confirm_code': '123'}, {'code': '123456', 'confirm_code': '654321'}):
            self.assertEqual(self.client.post(reverse('kiosk_lock'), values).status_code, 400)
        self.employee.role = 'Employee'
        self.employee.save(update_fields=['role'])
        self.assertEqual(self.lock().status_code, 403)

    def test_lock_is_per_browser_session(self):
        other = Client()
        other.force_login(self.user)
        self.lock()
        self.assertFalse(other.get(reverse('kiosk_status')).json()['locked'])
        self.assertEqual(other.get(reverse('dashboard')).status_code, 200)

    def test_code_is_never_audited(self):
        self.lock()
        self.client.post(reverse('kiosk_unlock'), {'code': 'wrong-code'})
        self.client.post(reverse('kiosk_unlock'), {'code': 'unique-648201'})
        events = AuditEvent.global_objects.filter(entity='AttendanceKiosk')
        self.assertEqual(set(events.values_list('action', flat=True)), {'lock', 'unlock_failed', 'unlock'})
        self.assertNotIn('unique-648201', str(list(events.values())))
        self.assertNotIn('code_hash', str(list(events.values())))
        self.assertTrue(all(event.organisation_id == self.org.pk for event in events))

    def test_lock_and_unlock_require_csrf_and_post(self):
        guarded = Client(enforce_csrf_checks=True)
        guarded.force_login(self.user)
        self.assertEqual(self.lock(guarded).status_code, 403)
        self.assertEqual(self.client.get(reverse('kiosk_lock')).status_code, 405)
        self.assertEqual(self.client.get(reverse('kiosk_unlock')).status_code, 405)
        self.assertEqual(guarded.post(reverse('kiosk_unlock'), {'code': 'anything'}).status_code, 403)
