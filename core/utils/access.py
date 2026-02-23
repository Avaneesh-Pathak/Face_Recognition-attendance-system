from core.models import Employee

def get_all_subordinates(employee):
    """
    Recursively fetch all employees under this employee.
    """
    result = set()

    def dfs(emp):
        subs = emp.subordinates.all()
        for s in subs:
            if s not in result:
                result.add(s)
                dfs(s)

    dfs(employee)
    return Employee.objects.filter(id__in=[e.id for e in result])

def get_visible_employees(user):
    """
    Returns queryset of employees user is allowed to see.
    """
    if not user.is_authenticated:
        return Employee.objects.none()

    try:
        emp = user.employee
    except Employee.DoesNotExist:
        return Employee.objects.none()

    # Admin / HR / Finance → see all
    if emp.role in ["Admin", "HR", "Finance"]:
        return Employee.objects.filter(is_active=True)

    # Manager → self + subordinates
    if emp.role == "Manager":
        subordinates = get_all_subordinates(emp)
        return Employee.objects.filter(
            id__in=[emp.id] + list(subordinates.values_list("id", flat=True))
        )

    # Normal employee → only self
    return Employee.objects.filter(id=emp.id)




