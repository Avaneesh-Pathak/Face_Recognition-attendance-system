from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth import get_user_model
from core.models import Employee

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
        for sub in emp.subordinates.all():
            if sub.id not in visited:
                visited.add(sub.id)
                dfs(sub)

    if employee:
        dfs(employee)

    return Employee.objects.filter(id__in=visited)


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
        return Employee.objects.filter(is_active=True)

    # 3️⃣ User without employee profile
    if not hasattr(user, "employee"):
        return Employee.objects.none()

    try:
        emp = user.employee
    except ObjectDoesNotExist:
        return Employee.objects.none()

    # 4️⃣ Role-based visibility
    role = getattr(emp, "role", None)

    # Admin / HR / Finance → see all active
    if role in ["Admin", "HR", "Finance"]:
        return Employee.objects.filter(is_active=True)

    # Manager → self + subordinates
    if role == "Manager":
        subordinates = get_all_subordinates(emp)
        return Employee.objects.filter(
            id__in=[emp.id] + list(subordinates.values_list("id", flat=True))
        )

    # Default → only self
    return Employee.objects.filter(id=emp.id)