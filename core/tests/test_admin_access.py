from django.contrib import admin
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from core.models import Employee, Organisation


class DjangoAdminTenantAccessTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.org_a = Organisation.objects.create(name='Admin Test Organisation A')
        cls.org_b = Organisation.objects.create(name='Admin Test Organisation B')
        cls.admin_a = User.objects.create_user(
            username='tenant_admin_a', password='Safe-test-password-12345', is_staff=True)
        cls.admin_b = User.objects.create_user(
            username='tenant_admin_b', password='Safe-test-password-12345', is_staff=True)
        cls.global_admin = User.objects.create_superuser(
            username='global_admin', email='global@example.test',
            password='Safe-test-password-12345')
        cls.employee_a = Employee.global_objects.create(
            organisation=cls.org_a, user=cls.admin_a, employee_id='ADM-A',
            position='Administrator', role='Admin', phone_number='9000000001')
        cls.employee_b = Employee.global_objects.create(
            organisation=cls.org_b, user=cls.admin_b, employee_id='ADM-B',
            position='Administrator', role='Admin', phone_number='9000000002')

    def test_tenant_admin_can_open_admin_but_only_sees_tenant_models(self):
        self.client.force_login(self.admin_a)
        response = self.client.get(reverse('admin:index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Employees')
        self.assertNotContains(response, 'Organisations')
        self.assertNotContains(response, 'Users')
        self.assertNotContains(response, 'Audit events')

    def test_tenant_admin_cannot_open_another_organisations_employee(self):
        self.client.force_login(self.admin_a)
        response = self.client.get(reverse(
            'admin:core_employee_change', args=[self.employee_b.pk]))
        self.assertEqual(response.status_code, 404)

    def test_tenant_admin_employee_form_does_not_expose_organisation_field(self):
        self.client.force_login(self.admin_a)
        response = self.client.get(reverse('admin:core_employee_add'))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'name="organisation"')

    def test_global_superuser_sees_global_models(self):
        self.client.force_login(self.global_admin)
        response = self.client.get(reverse('admin:index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Organisations')
        self.assertContains(response, 'Users')
        self.assertContains(response, 'Audit events')

    def test_audit_events_are_not_mutable_even_for_superuser(self):
        model_admin = admin.site._registry[next(
            model for model in admin.site._registry if model._meta.label_lower == 'core.auditevent')]
        request = type('Request', (), {'user': self.global_admin})()
        self.assertFalse(model_admin.has_add_permission(request))
        self.assertFalse(model_admin.has_change_permission(request))
        self.assertFalse(model_admin.has_delete_permission(request))
