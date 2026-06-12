from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth import get_user_model
from core.models import Employee

User = get_user_model()


# -------------------------------------------------
# RECURSIVE SUBORDINATE FETCH (SAFE + ORG-AWARE)
# -------------------------------------------------
def get_all_subordinates(employee):
    """
    Recursively fetch all employees under this employee.
    Filters by organization for multi-tenant safety.
    Safe against circular hierarchy.
    """
    visited = set()
    organization = employee.organization  # ✅ Get org context

    def dfs(emp):
        for sub in emp.subordinates.filter(organization=organization):  # ✅ ORG FILTER
            if sub.id not in visited:
                visited.add(sub.id)
                dfs(sub)

    if employee:
        dfs(employee)

    return Employee.objects.filter(id__in=visited, organization=organization)  # ✅ ORG FILTER


# -------------------------------------------------
# MAIN ACCESS CONTROL FUNCTION (BULLETPROOF + ORG-AWARE)
# -------------------------------------------------
def get_visible_employees(user, organization=None):
    """
    Returns queryset of employees user is allowed to see.
    FILTERS BY ORGANIZATION for multi-tenant isolation.
    Safe against:
    - Anonymous user
    - User without employee profile
    - Superuser without employee
    - Broken relations
    """

    # 1️⃣ Not authenticated
    if not user or not user.is_authenticated:
        return Employee.objects.none()

    # 2️⃣ Superuser → see all active employees (with org filter if provided)
    if user.is_superuser:
        qs = Employee.objects.filter(is_active=True)
        if organization:
            qs = qs.filter(organization=organization)
        return qs

    # 3️⃣ User without employee profile
    if not hasattr(user, "employee"):
        return Employee.objects.none()

    try:
        emp = user.employee
    except ObjectDoesNotExist:
        return Employee.objects.none()

    # ✅ CRITICAL: Get user's organization if not provided
    if not organization:
        organization = emp.organization
    
    # ✅ SECURITY: User can only see employees from their organization
    if emp.organization != organization:
        return Employee.objects.none()

    # 4️⃣ Role-based visibility (WITHIN ORGANIZATION)
    role = getattr(emp, "role", None)

    # Admin / HR / Finance → see all active in their organization
    if role in ["Admin", "HR", "Finance"]:
        return Employee.objects.filter(
            is_active=True,
            organization=organization  # ✅ ORG FILTER
        )

    # Manager → self + subordinates (within organization)
    if role == "Manager":
        subordinates = get_all_subordinates(emp)
        return Employee.objects.filter(
            id__in=[emp.id] + list(subordinates.values_list("id", flat=True)),
            organization=organization  # ✅ ORG FILTER
        )

    # Default → only self
    return Employee.objects.filter(id=emp.id, organization=organization)