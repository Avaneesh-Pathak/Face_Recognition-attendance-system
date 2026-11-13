from datetime import date
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth.models import User
from django.db import IntegrityError
from django.test import TestCase

from core.models import Employee, JoiningDetail, OfficeLocation, Organisation, SalaryStructure


class ProductionRegressionTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.organisation = Organisation.objects.create(name="Production Regression Org")
        cls.location = OfficeLocation.global_objects.create(
            organisation=cls.organisation,
            name="Main office",
            latitude=Decimal("12.971600"),
            longitude=Decimal("77.594600"),
        )
        cls.hr_user = User.objects.create_user("regression_hr", password="Safe-test-password-123!")
        Employee.global_objects.create(
            organisation=cls.organisation,
            user=cls.hr_user,
            employee_id="HR-REG-1",
            role="HR",
            phone_number="1234567890",
        )

    def registration_payload(self, username):
        return {
            "username": username,
            "first_name": "Test",
            "last_name": "Employee",
            "email": f"{username}@example.test",
            "password1": "Safe-test-password-123!",
            "password2": "Safe-test-password-123!",
            "role": "Employee",
            "position": "Analyst",
            "phone_number": "1234567890",
            "date_of_joining": "2026-10-01",
            "probation_period_months": "3",
            "confirmation_date": "2027-01-01",
            "location_type": "INDOOR",
            "assigned_location": str(self.location.pk),
            "base_salary": "50000.00",
            "hra": "0.00",
            "allowances": "0.00",
            "deductions": "0.00",
        }

    def test_login_page_uses_password_authentication_without_fake_face_scan(self):
        response = self.client.get("/login/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Sign in")
        self.assertNotContains(response, "Scanning Face")
        self.assertNotContains(response, "Face Detected!")

    def test_employee_registration_rolls_back_related_rows_on_failure(self):
        self.client.force_login(self.hr_user)

        with patch.object(SalaryStructure, "save", side_effect=IntegrityError("simulated failure")):
            response = self.client.post("/register/", self.registration_payload("rollback_employee"))

        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username="rollback_employee").exists())
        self.assertFalse(
            Employee.global_objects.filter(
                organisation=self.organisation,
                user__username="rollback_employee",
            ).exists()
        )
        self.assertFalse(JoiningDetail.global_objects.filter(employee__user__username="rollback_employee").exists())

    def test_employee_registration_persists_joining_and_salary_setup(self):
        self.client.force_login(self.hr_user)
        response = self.client.post("/register/", self.registration_payload("new_employee"))

        self.assertEqual(response.status_code, 302)
        employee = Employee.global_objects.get(user__username="new_employee")
        joining = JoiningDetail.global_objects.get(employee=employee)
        self.assertEqual(employee.date_of_joining, date(2026, 10, 1))
        self.assertEqual(joining.probation_period_months, 3)
        self.assertEqual(joining.confirmation_date, date(2027, 1, 1))
        self.assertEqual(employee.salary_structure.base_salary, Decimal("50000.00"))
