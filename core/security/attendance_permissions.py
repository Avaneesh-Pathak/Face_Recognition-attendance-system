def can_correct_attendance(user, employee=None):
    if not user.is_authenticated:
        return False
    profile = getattr(user, 'employee', None)
    # Even platform users need an Admin employee role to correct their own record.
    if employee is not None and employee.user_id == user.pk:
        return bool(profile and profile.role == 'Admin')
    if user.is_superuser:
        return True
    return bool(profile and profile.is_active and profile.role in ('Admin', 'HR')
                and (employee is None or profile.organisation_id == employee.organisation_id))
