# core/test_tenant.py
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from core.models import Organisation, Employee, Department, Attendance
from core.tenant_utils import set_current_organisation, get_current_organisation

class MultiTenancySecurityTestCase(TestCase):
    
    def setUp(self):
        # 1. Create separate Organisations
        self.org_nelson = Organisation.objects.create(name="Nelson Hospital", is_active=True)
        self.org_apex = Organisation.objects.create(name="Apex Clinic", is_active=True)

        # 2. Create separate Users
        self.user_nelson = User.objects.create_user(username="hr_nelson", password="password123")
        self.user_apex = User.objects.create_user(username="hr_apex", password="password123")

        # 3. Create HR Employees
        # We use global_objects because the thread-local context is not set yet
        self.emp_nelson = Employee.global_objects.create(
            organisation=self.org_nelson, user=self.user_nelson, employee_id="HR-NEL001", role="HR"
        )
        self.emp_apex = Employee.global_objects.create(
            organisation=self.org_apex, user=self.user_apex, employee_id="HR-APX001", role="HR"
        )

        # 4. Create isolated Attendance logs
        Attendance.global_objects.create(
            organisation=self.org_nelson, employee=self.emp_nelson, attendance_type="check_in"
        )
        Attendance.global_objects.create(
            organisation=self.org_apex, employee=self.emp_apex, attendance_type="check_in"
        )

    def tearDown(self):
        # Reset current thread local state after every test to prevent cross-leakage
        set_current_organisation(None)

    def test_query_filtering_isolation(self):
        """Verify that Employee.objects.all() restricts records on a per-tenant basis."""
        set_current_organisation(self.org_nelson)
        self.assertEqual(Employee.objects.all().count(), 1)
        self.assertEqual(Employee.objects.first().employee_id, "HR-NEL001")

        set_current_organisation(self.org_apex)
        self.assertEqual(Employee.objects.all().count(), 1)
        self.assertEqual(Employee.objects.first().employee_id, "HR-APX001")

    def test_automatic_creation_stamping(self):
        """Verify that newly created items are automatically stamped with the thread's active tenant."""
        set_current_organisation(self.org_nelson)
        dept = Department.objects.create(name="Nursing")
        self.assertEqual(dept.organisation, self.org_nelson)

    def test_cross_tenant_leakage_prevented(self):
        """Verify that queries looking up items belong to another tenant return empty results."""
        set_current_organisation(self.org_nelson)
        # Attempt to pull Apex Clinic's employee using Nelson Hospital's context
        apex_lookup = Employee.objects.filter(employee_id="HR-APX001")
        self.assertEqual(apex_lookup.count(), 0)

    def test_middleware_request_level_isolation(self):
        """Verify that active requests made by Nelson HR do not return Apex Clinic data."""
        client = Client()
        client.login(username="hr_nelson", password="password123")
        
        # Test standard employee listing endpoint
        response = client.get(reverse('employee_list'))
        employees_in_view = response.context['employees']
        
        self.assertGreater(employees_in_view.count(), 0)
        for emp in employees_in_view:
            self.assertEqual(emp.organisation, self.org_nelson)
            self.assertNotEqual(emp.organisation, self.org_apex)