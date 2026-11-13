from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth import get_user_model
from core.models import Employee
from core.tenant_utils import get_current_organisation

User = get_user_model()


# -------------------------------------------------
# RECURSIVE SUBORDINATE FETCH (SAFE)
# -------------------------------------------------
def get_all_subordinates(employee):
    """
    Recursively fetch all employees under this employee.
    Safe against circular hierarchy.
    """
    visited = set()

    def dfs(emp):
        for sub in emp.subordinates.filter(organisation=emp.organisation):
            if sub.id not in visited:
                visited.add(sub.id)
                dfs(sub)

    if employee:
        dfs(employee)

    if employee and getattr(employee, 'organisation_id', None) is not None:
        return Employee.global_objects.filter(organisation=employee.organisation, id__in=visited)
    return Employee.global_objects.filter(id__in=visited)


# -------------------------------------------------
# MAIN ACCESS CONTROL FUNCTION (BULLETPROOF)
# -------------------------------------------------
def get_visible_employees(user):
    """
    Returns queryset of employees user is allowed to see.
    Safe against:
    - Anonymous user
    - User without employee profile
    - Superuser without employee
    - Broken relations
    """

    # 1️⃣ Not authenticated
    if not user or not user.is_authenticated:
        return Employee.objects.none()

    # 2️⃣ Superuser → see all active employees
    if user.is_superuser:
        return Employee.global_objects.filter(is_active=True)

    # 3️⃣ User without employee profile
    if not hasattr(user, "employee"):
        return Employee.objects.none()

    try:
        emp = user.employee
    except ObjectDoesNotExist:
        return Employee.objects.none()

    # 4️⃣ Role-based visibility
    role = getattr(emp, "role", None)

    # Admin / HR / Finance → see all active employees in the same tenant
    if role in ["Admin", "HR", "Finance"]:
        return Employee.global_objects.filter(organisation=emp.organisation, is_active=True)

    # Manager → self + subordinates
    if role == "Manager":
        subordinates = get_all_subordinates(emp)
        return Employee.global_objects.filter(
            organisation=emp.organisation,
            id__in=[emp.id] + list(subordinates.values_list("id", flat=True))
        )

    # Default → only self
    return Employee.global_objects.filter(organisation=emp.organisation, id=emp.id)


def get_visible_employees_all(user):
    if not user or not user.is_authenticated:
        return Employee.objects.none()

    if user.is_superuser:
        return Employee.global_objects.all()

    if not hasattr(user, "employee"):
        return Employee.objects.none()

    emp = user.employee

    if emp.role in ["Admin", "HR", "Finance"]:
        return Employee.global_objects.filter(organisation=emp.organisation)

    if emp.role == "Manager":
        subordinates = get_all_subordinates(emp)
        return Employee.global_objects.filter(
            organisation=emp.organisation,
            id__in=[emp.id] + list(subordinates.values_list("id", flat=True))
        )

    return Employee.global_objects.filter(organisation=emp.organisation, id=emp.id)



