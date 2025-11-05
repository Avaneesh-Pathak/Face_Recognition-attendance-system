# core/decorators.py

from django.http import HttpResponseForbidden
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required

def role_required(allowed_roles=[]):
    """
    ✅ General decorator for checking user roles.
    Usage: @role_required(['Admin', 'HR'])
    """
    def decorator(view_func):
        @login_required
        def wrapper(request, *args, **kwargs):
            if not hasattr(request.user, 'employee'):
                return HttpResponseForbidden("You are not assigned to any employee profile.")
            
            user_role = request.user.employee.role

            if user_role in allowed_roles or request.user.is_superuser:
                return view_func(request, *args, **kwargs)
            else:
                return HttpResponseForbidden(f"Access Denied! Only roles {allowed_roles} are allowed.")
        return wrapper
    return decorator

# ✅ Specific decorators for clean usage:

def admin_required(view_func):
    return role_required(['Admin'])(view_func)

def hr_required(view_func):
    return role_required(['HR', 'Admin'])(view_func)

def finance_required(view_func):
    return role_required(['Finance', 'Admin'])(view_func)

def manager_required(view_func):
    return role_required(['Manager', 'Admin'])(view_func)

def employee_required(view_func):
    return role_required(['Employee', 'Manager', 'HR', 'Finance', 'Admin'])(view_func)
