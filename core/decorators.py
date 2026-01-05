# core/decorators.py

from django.http import HttpResponseForbidden
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages


def role_required(allowed_roles=None):
    """
    Production-grade role-based access control.

    Rules:
    - Superuser: FULL ACCESS
    - Staff (is_staff=True): FULL ACCESS
    - Employee users: role-based access
    - Others: denied
    """
    if allowed_roles is None:
        allowed_roles = []

    def decorator(view_func):
        @login_required
        def wrapper(request, *args, **kwargs):

            # -------------------------------------------------
            # SUPERUSER → ALWAYS ALLOWED
            # -------------------------------------------------
            if request.user.is_superuser:
                return view_func(request, *args, **kwargs)

            # -------------------------------------------------
            # STAFF / ADMIN USER (NO EMPLOYEE PROFILE REQUIRED)
            # -------------------------------------------------
            if request.user.is_staff and not hasattr(request.user, 'employee'):
                return view_func(request, *args, **kwargs)

            # -------------------------------------------------
            # USER WITH EMPLOYEE PROFILE
            # -------------------------------------------------
            if hasattr(request.user, 'employee'):
                user_role = request.user.employee.role
                if user_role in allowed_roles:
                    return view_func(request, *args, **kwargs)

                messages.error(
                    request,
                    f"Access denied. Required role(s): {', '.join(allowed_roles)}."
                )
                return redirect("dashboard")

            # -------------------------------------------------
            # FALLBACK → BLOCK
            # -------------------------------------------------
            messages.error(
                request,
                "You are not authorized to access this page."
            )
            return redirect("dashboard")

        return wrapper
    return decorator


# =====================================================
# ROLE-SPECIFIC DECORATORS
# =====================================================

def admin_required(view_func):
    return role_required(['Admin'])(view_func)


def hr_required(view_func):
    return role_required(['HR', 'Admin'])(view_func)


def finance_required(view_func):
    return role_required(['Finance', 'Admin'])(view_func)


def manager_required(view_func):
    return role_required(['Manager', 'Admin'])(view_func)


def employee_required(view_func):
    return role_required(
        ['Employee', 'Manager', 'HR', 'Finance', 'Admin']
    )(view_func)
