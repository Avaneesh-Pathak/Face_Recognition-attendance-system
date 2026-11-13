"""Regression tests for payroll integrity, calendar consistency and org isolation."""
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from core.models import (Organisation, Employee, WorkRule, SalaryStructure,
    Payroll, Attendance, OfficeLocation)
from core.services.attendance_engine import workday_sessions


class PayrollCalendarTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.org = Organisation.objects.create(name='Company January')
        cls.other = Organisation.objects.create(name='Company February')
        cls.user = User.objects.create_user('testpayuser', password='Qwerty1234!abc')
        cls.user_other = User.objects.create_user('testpayother', password='Qwerty1234!abc')
        cls.rule = WorkRule.global_objects.create(organisation=cls.org, name='Day rule',
            working_days='5_day', full_day_hours=8, half_day_hours=4,
            overtime_rate=Decimal('100'))
        cls.emp = Employee.global_objects.create(organisation=cls.org, user=cls.user,
            employee_id='A1', position='Technician', phone_number='1234567890',
            work_rule=cls.rule, date_of_joining=date(2026,10,1))
        cls.other_emp = Employee.global_objects.create(organisation=cls.other,user=cls.user_other,
            employee_id='A1', position='Technician', phone_number='1234567890',
            date_of_joining=date(2026,10,1))
        SalaryStructure.global_objects.create(organisation=cls.org, employee=cls.emp,
             base_salary=Decimal('31000.00'), hra=Decimal('3100.00'),
             allowances=Decimal('1550.00'), deductions=Decimal('250.00'),
             effective_from=date(2026,1,1))

    def mark(self, day, hour, kind, minute=0):
        stamp = timezone.make_aware(datetime.combine(day,time(hour,minute)))
        return Attendance.global_objects.create(organisation=self.org,employee=self.emp,
            attendance_type=kind,timestamp=stamp)

    def test_payroll_reconcile_idempotent_and_protect_paid(self):
        self.mark(date(2026,10,1),9,'check_in')
        self.mark(date(2026,10,1),19,'check_out')
        results = Payroll.objects.generate_salary(2026,10,organisation=self.org)
        self.assertEqual(results['created'], 1)
        payroll = Payroll.global_objects.get(employee=self.emp,month=date(2026,10,1))
        self.assertGreater(payroll.overtime_hours,0)
        self.assertEqual(payroll.regular_overtime_pay,Decimal('200.00'))
        self.assertEqual(payroll.net_salary, payroll.basic_pay+payroll.allowances+
                         payroll.regular_overtime_pay+payroll.holiday_overtime_pay-payroll.deductions)
        results = Payroll.objects.generate_salary(2026,10,organisation=self.org)
        self.assertEqual(results['updated'],1)
        self.assertEqual(Payroll.global_objects.filter(employee=self.emp).count(),1)
        payroll.status = 'paid'; payroll.save(update_fields=['status'])
        results = Payroll.objects.generate_salary(2026,10,organisation=self.org)
        self.assertEqual(results['skipped_paid'],1)
        self.assertEqual(Payroll.global_objects.get(pk=payroll.pk).status,'paid')
        self.assertFalse(Payroll.global_objects.filter(organisation=self.other).exists())

    def test_night_shift_work_hours_use_check_in_date(self):
        self.mark(date(2026,10,7),22,'check_in')
        self.mark(date(2026,10,8),6,'check_out')
        recs = Attendance.global_objects.filter(employee=self.emp)
        summary = workday_sessions(recs,self.emp)
        self.assertEqual(summary[date(2026,10,7)]['hours'],8.0)
        self.assertNotIn(date(2026,10,8),summary)

    def test_incomplete_checkin_not_paid_hours(self):
        self.mark(date(2026,10,7),9,'check_in')
        summary = workday_sessions(Attendance.global_objects.filter(employee=self.emp),self.emp)
        self.assertTrue(summary[date(2026,10,7)]['pending'])
        self.assertEqual(summary[date(2026,10,7)]['hours'],0)

    def test_calendar_tenant_access_and_detail_requires_login(self):
        path=reverse('attendance_calendar',args=[self.other_emp.pk,2026,10])
        self.client.force_login(self.user)
        self.assertEqual(self.client.get(path).status_code,404)
        own=reverse('attendance_calendar',args=[self.emp.pk,2026,10])
        response=self.client.get(own)
        self.assertEqual(response.status_code,200)
        self.assertContains(response,'October 2026')
        other_data=reverse('attendance_calendar_data',args=[self.other_emp.pk,2026,10])
        self.assertEqual(self.client.get(other_data).status_code,404)

    def test_pdf_render_contains_pdf_header_and_tenant_guard(self):
        Payroll.objects.generate_salary(2026,10,organisation=self.org)
        payroll=Payroll.global_objects.get(employee=self.emp)
        self.client.force_login(self.user)
        url=reverse('payroll_slip_pdf',args=[payroll.pk])
        response=self.client.get(url)
        self.assertEqual(response.status_code,200)
        self.assertEqual(b''.join(response.streaming_content)[:4],b'%PDF')
        self.client.force_login(self.user_other)
        self.assertEqual(self.client.get(url).status_code,404)

class MultiScannerVotingTests(TestCase):
    def test_different_organisations_and_devices_do_not_share_face_votes(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        from core.face_system import FaceRecognitionSystem
        from django.conf import settings
        fake = SimpleNamespace(bbox=[20,20,200,200], det_score=0.99,
            embedding=np.array([1.0] + [0.0]*511))
        matcher = FaceRecognitionSystem()
        frame = np.zeros((320,320,3), dtype=np.uint8)
        with patch.object(matcher,'_infer',return_value=[fake]), \
             patch.object(matcher,'identify_face',side_effect=lambda embedding,organisation_id:(organisation_id+100,0.9)):
            # 2 votes on company A's scanner must not confirm company B's scanner.
            for _ in range(2):
                self.assertEqual(matcher.recognize_from_frame(frame,1,'scanner_a')[2]['reason'],'ACCUMULATING')
            self.assertEqual(matcher.recognize_from_frame(frame,2,'scanner_b')[2]['reason'],'ACCUMULATING')
            self.assertEqual(matcher.recognize_from_frame(frame,1,'scanner_c')[2]['reason'],'ACCUMULATING')
            self.assertEqual(matcher.recognize_from_frame(frame,1,'scanner_a')[0],101)
            self.assertEqual(matcher.recognize_from_frame(frame,2,'scanner_b')[2]['reason'],'ACCUMULATING')
            self.assertEqual(matcher.recognize_from_frame(frame,2,'scanner_b')[0],102)
            self.assertEqual(matcher.recognize_from_frame(frame,1,'scanner_c')[2]['reason'],'ACCUMULATING')
