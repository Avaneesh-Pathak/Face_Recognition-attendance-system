from datetime import date, time
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch
import json
import numpy as np
from django.contrib.auth.models import User
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.db import transaction
from django.test import TestCase, SimpleTestCase, override_settings
from django.urls import reverse
from core.models import Attendance, AuditEvent, Employee, Payroll, SalaryStructure, WorkRule, EmployeeShiftAssignment
from core.tests import test_payroll_calendar_concurrency as fixtures
from core.face_system import FaceRecognitionSystem, BusyInference


class ProductionWorkflowTests(TestCase):
    setUpTestData = classmethod(fixtures.PayrollCalendarTests.setUpTestData.__func__)
    mark = fixtures.PayrollCalendarTests.mark

    def setUp(self):
        cache.clear()

    def role(self, role):
        self.emp.role = role
        self.emp.save(update_fields=['role'])
        self.client.force_login(self.user)

    def correction(self, **extra):
        return dict(employee=self.emp.pk, attendance_type='check_in',
                    timestamp='2026-10-07T09:00', reason='Corrected missed scan', **extra)

    def generate_closed(self):
        with patch('django.utils.timezone.localdate', return_value=date(2026, 11, 1)):
            Payroll.objects.generate_salary(2026, 10, organisation=self.org)
        return Payroll.global_objects.get(employee=self.emp)

    def test_only_admin_can_create_update_delete_own_attendance(self):
        record = self.mark(date(2026, 10, 7), 9, 'check_in')
        for role in ('Employee', 'Manager', 'Finance', 'HR'):
            self.role(role)
            for name, args in [('attendance_create', []), ('attendance_update', [record.pk]), ('attendance_delete', [record.pk])]:
                with self.subTest(role=role, action=name):
                    self.assertEqual(self.client.post(reverse(name, args=args), self.correction()).status_code, 403)
        self.role('Admin')
        self.assertEqual(self.client.post(reverse('attendance_update', args=[record.pk]), self.correction()).status_code, 200)
        self.assertEqual(self.client.post(reverse('attendance_delete', args=[record.pk]), {'reason': 'Duplicate entry removed'}).status_code, 200)
        self.assertEqual(self.client.post(reverse('attendance_create'), self.correction()).status_code, 200)

    def test_hr_can_correct_other_employee_but_not_other_tenant(self):
        self.role('HR')
        colleague = Employee.global_objects.create(organisation=self.org,
            user=User.objects.create_user('colleague'), employee_id='B2', date_of_joining=date(2026, 1, 1))
        payload = self.correction(); payload['employee'] = colleague.pk
        self.assertEqual(self.client.post(reverse('attendance_create'), payload).status_code, 200)
        payload['employee'] = self.other_emp.pk
        self.assertEqual(self.client.post(reverse('attendance_create'), payload).status_code, 404)

    def test_invalid_event_and_missing_reason_do_not_write(self):
        self.role('Admin')
        for values in ({'attendance_type': 'present'}, {'timestamp': 'bad'}, {'reason': ''}, {'timestamp': '2099-01-01T12:00'}):
            payload = self.correction(); payload.update(values)
            self.assertEqual(self.client.post(reverse('attendance_create'), payload).status_code, 400)
        self.assertFalse(Attendance.global_objects.exists())

    def test_audit_actor_before_after_reason_and_tenant_visibility(self):
        self.role('Admin')
        self.client.post(reverse('attendance_create'), self.correction())
        event = AuditEvent.global_objects.filter(entity='Attendance').latest('pk')
        self.assertEqual(event.actor_id, self.user.pk)
        self.assertEqual(event.reason, 'Corrected missed scan')
        self.assertEqual(event.after['attendance_type'], 'check_in')
        self.assertTrue(event.request_id)
        self.assertEqual(self.client.get(reverse('audit_log')).status_code, 200)
        self.other_emp.role = 'Admin'; self.other_emp.save()
        self.client.force_login(self.user_other)
        response = self.client.get(reverse('audit_log'))
        self.assertNotContains(response, 'Corrected missed scan')
        self.assertTrue(all(e.organisation_id == self.other.pk for e in response.context['page_obj']))
        self.role('HR')
        self.assertEqual(self.client.get(reverse('audit_log')).status_code, 403)

    def test_audit_history_cannot_be_updated_or_deleted(self):
        event = AuditEvent.global_objects.first()
        with self.assertRaises(ValidationError):
            AuditEvent.global_objects.filter(pk=event.pk).update(actor_name='Someone else')
        with self.assertRaises(ValidationError):
            event.delete()

    def test_business_write_rolls_back_if_audit_fails(self):
        with patch.object(AuditEvent, 'save', side_effect=RuntimeError('audit unavailable')):
            with self.assertRaises(RuntimeError):
                self.mark(date(2026, 10, 7), 9, 'check_in')
        self.assertFalse(Attendance.global_objects.exists())

    def test_calendar_half_day_pending_and_overnight_detail(self):
        self.mark(date(2026, 10, 6), 9, 'check_in')
        self.mark(date(2026, 10, 6), 13, 'check_out')
        self.mark(date(2026, 10, 7), 22, 'check_in')
        self.mark(date(2026, 10, 8), 6, 'check_out')
        self.mark(date(2026, 10, 9), 9, 'check_in')
        self.client.force_login(self.user)
        days = self.client.get(reverse('attendance_calendar_data', args=[self.emp.pk, 2026, 10])).json()['data']
        self.assertEqual(days[5]['status'], 'half')
        self.assertEqual(days[6]['status'], 'present')
        self.assertEqual(days[8]['status'], 'pending')
        self.assertFalse(days[8]['present'])
        detail = self.client.get(reverse('attendance_day_detail', args=[self.emp.pk, '2026-10-07'])).json()
        self.assertEqual(len(detail['logs']), 2)
        self.assertEqual(detail['logs'][1]['date'], '08-10-2026')
        self.assertEqual(Decimal(detail['total_hours']), Decimal('8'))

    def test_orphan_checkout_is_visible_and_blocks_payroll(self):
        self.mark(date(2026, 10, 7), 17, 'check_out')
        self.client.force_login(self.user)
        detail = self.client.get(reverse('attendance_day_detail', args=[self.emp.pk, '2026-10-07'])).json()
        self.assertTrue(detail['pending'])
        self.assertEqual(len(detail['logs']), 1)
        with self.assertRaisesMessage(ValueError, 'reconcile attendance'):
            self.generate_closed()
        self.assertFalse(Payroll.global_objects.exists())

    def test_rotational_overtime_uses_assigned_rate(self):
        night = WorkRule.global_objects.create(organisation=self.org, name='Premium night',
            shift_start_time=time(22), shift_end_time=time(6), overtime_rate=Decimal('250'))
        EmployeeShiftAssignment.global_objects.create(organisation=self.org, employee=self.emp,
            work_rule=night, start_date=date(2026, 10, 7), end_date=date(2026, 10, 7))
        self.mark(date(2026, 10, 7), 22, 'check_in')
        self.mark(date(2026, 10, 8), 8, 'check_out')
        payroll = self.generate_closed()
        self.assertEqual(payroll.regular_overtime_pay, Decimal('500'))

    def test_full_month_fixed_pay_and_half_day_proration(self):
        import calendar
        for day in range(1, 32):
            current = date(2026, 10, day)
            if current.weekday() < 5:
                self.mark(current, 9, 'check_in')
                self.mark(current, 17, 'check_out')
        payroll = self.generate_closed()
        self.assertEqual(payroll.basic_pay, Decimal('31000.00'))
        self.assertEqual(payroll.allowances, Decimal('4650.00'))
        self.assertEqual(payroll.net_salary, Decimal('35400.00'))

    def test_early_checkin_respects_night_shift_end_after_six(self):
        from core.rotational_shift_fix import get_work_date_for_checkin
        from django.utils import timezone
        from datetime import datetime
        self.rule.shift_start_time = time(22)
        self.rule.shift_end_time = time(8)
        self.rule.save()
        self.emp.work_rule = self.rule
        self.assertEqual(get_work_date_for_checkin(timezone.make_aware(datetime(2026, 10, 8, 7)), self.emp), date(2026, 10, 7))
        self.assertEqual(get_work_date_for_checkin(timezone.make_aware(datetime(2026, 10, 8, 9)), self.emp), date(2026, 10, 8))

    def test_pending_payroll_cannot_be_paid_or_promoted_by_edit_form(self):
        self.role('Admin')
        payroll = self.generate_closed()
        payroll.status = 'pending'; payroll.save()
        self.client.post(reverse('payroll_update', args=[payroll.pk]), {'status': 'paid', 'net_salary': '999999'})
        self.client.post(reverse('pay_salary', args=[payroll.pk]), {'amount_paid': str(payroll.net_salary)})
        payroll.refresh_from_db()
        self.assertEqual(payroll.status, 'pending')

    def test_paid_payroll_blocks_corrections_deletion_and_repeated_payment(self):
        self.role('Admin')
        self.mark(date(2026, 10, 7), 9, 'check_in'); self.mark(date(2026, 10, 7), 17, 'check_out')
        payroll = self.generate_closed()
        with patch('django.utils.timezone.localdate', return_value=date(2026, 11, 2)):
            self.client.post(reverse('pay_salary', args=[payroll.pk]), {'paid_date': '2026-11-01'})
            self.client.post(reverse('pay_salary', args=[payroll.pk]), {'paid_date': '2026-11-02'})
        payroll.refresh_from_db()
        self.assertEqual(payroll.status, 'paid'); self.assertEqual(payroll.paid_date, date(2026, 11, 1))
        self.assertEqual(self.client.post(reverse('attendance_create'), self.correction()).status_code, 409)
        self.client.post(reverse('payroll_delete', args=[payroll.pk]))
        self.assertTrue(Payroll.global_objects.filter(pk=payroll.pk).exists())

    def test_source_change_invalidates_previously_processed_payroll(self):
        payroll = self.generate_closed()
        structure = SalaryStructure.global_objects.get(employee=self.emp)
        structure.base_salary += 100; structure.save()
        payroll.refresh_from_db()
        self.assertEqual(payroll.status, 'pending')

    def test_deductions_never_break_net_reconciliation(self):
        structure = SalaryStructure.global_objects.get(employee=self.emp)
        structure.deductions = Decimal('99999'); structure.save()
        payroll = self.generate_closed()
        self.assertEqual(payroll.net_salary, payroll.gross_earnings - payroll.deductions)
        self.assertEqual(payroll.net_salary, 0)

    def test_hr_cannot_reset_admin_or_own_official_profile(self):
        self.role('HR')
        old = self.user.password
        self.assertEqual(self.client.post(reverse('employee_update', args=[self.emp.pk]), {'password': 'Replacement12345!'}).status_code, 403)
        self.user.refresh_from_db(); self.assertEqual(self.user.password, old)

    def test_salary_setup_and_dashboard_do_not_leak_salary_to_hr(self):
        self.role('HR')
        self.assertEqual(self.client.get(reverse('salary_structure_list')).status_code, 403)
        self.assertEqual(self.client.get(reverse('salary_structure_create')).status_code, 403)

    def test_salary_history_template_renders_after_recalculation(self):
        self.role('Admin')
        self.generate_closed()
        response = self.client.get(reverse('employee_salary_history', args=[self.emp.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Recalculate')

    def test_employee_deactivation_preserves_paid_history(self):
        payroll = self.generate_closed()
        payroll.status = 'paid'; payroll.save()
        admin_user = User.objects.create_user('history-admin')
        Employee.global_objects.create(user=admin_user, organisation=self.org, employee_id='ADMIN', role='Admin')
        self.client.force_login(admin_user)
        self.client.post(reverse('employee_delete', args=[self.emp.pk]))
        self.emp.refresh_from_db()
        self.assertFalse(self.emp.is_active)
        self.assertTrue(Payroll.global_objects.filter(pk=payroll.pk, status='paid').exists())
        with self.assertRaises(ValidationError):
            with transaction.atomic():
                self.user.delete()
        self.client.get(reverse('reactivate_employee', args=[self.emp.pk]))
        self.emp.refresh_from_db(); self.assertFalse(self.emp.is_active)
        self.client.post(reverse('reactivate_employee', args=[self.emp.pk]))
        self.emp.refresh_from_db(); self.assertTrue(self.emp.is_active)
        self.assertEqual(self.emp.employment_status, 'active')

    def test_gallery_invalidation_reaches_separate_matchers(self):
        self.emp.face_encoding = json.dumps([1.] + [0.] * 511); self.emp.save()
        a, b = FaceRecognitionSystem(), FaceRecognitionSystem()
        vector = np.array([1.] + [0.] * 511)
        self.assertEqual(a.identify_face(vector, self.org.pk)[0], self.emp.pk)
        self.assertEqual(b.identify_face(vector, self.org.pk)[0], self.emp.pk)
        with self.captureOnCommitCallbacks(execute=True):
            self.emp.is_active = False; self.emp.save()
        self.assertIsNone(a.identify_face(vector, self.org.pk)[0])
        self.assertIsNone(b.identify_face(vector, self.org.pk)[0])


@override_settings(FACE_SCAN_INFERENCE_SLOTS=2, FACE_INFERENCE_QUEUE_TIMEOUT=0.01, REDIS_URL='')
class ScannerPoolTests(SimpleTestCase):
    @override_settings(FACE_LIVENESS_VERIFIER='test.provider')
    def test_pad_accepts_only_explicit_server_verification(self):
        from core.security.liveness import verify_liveness, LivenessUnavailable
        request = SimpleNamespace(organisation=SimpleNamespace(pk=1))
        with patch('core.security.liveness.import_string', return_value=lambda **kwargs: 'true'):
            self.assertFalse(verify_liveness(None, request))
        with patch('core.security.liveness.import_string', return_value=lambda **kwargs: True):
            self.assertTrue(verify_liveness(None, request))
        with patch('core.security.liveness.import_string', side_effect=RuntimeError('offline')):
            with self.assertRaises(LivenessUnavailable):
                verify_liveness(None, request)

    def test_independent_slots_execute_concurrently(self):
        matcher = FaceRecognitionSystem()
        barrier = Barrier(2)
        def infer(frame):
            barrier.wait(timeout=3)
            return frame
        with patch.object(matcher, '_build_model', side_effect=lambda: SimpleNamespace(get=infer)):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(matcher._infer, ['organisation A', 'organisation B']))
        self.assertEqual(results, ['organisation A', 'organisation B'])

    def test_exhausted_pool_returns_bounded_busy_error(self):
        matcher = FaceRecognitionSystem()
        matcher._model_pool.get(); matcher._model_pool.get()
        with self.assertRaises(BusyInference):
            matcher._infer(None)

    def test_failed_frame_resets_positive_votes(self):
        matcher = FaceRecognitionSystem()
        face = SimpleNamespace(bbox=[20, 20, 200, 200], det_score=.99, embedding=np.ones(512))
        frame = np.zeros((320, 320, 3))
        with patch.object(matcher, '_infer', side_effect=[[face], [face], [], [face]]), patch.object(matcher, 'identify_face', return_value=(1, .9)):
            for _ in range(2):
                self.assertEqual(matcher.recognize_from_frame(frame, 1, 'device')[2]['reason'], 'ACCUMULATING')
            self.assertEqual(matcher.recognize_from_frame(frame, 1, 'device')[2]['reason'], 'NO_FACE')
            self.assertEqual(matcher.recognize_from_frame(frame, 1, 'device')[2]['reason'], 'ACCUMULATING')
