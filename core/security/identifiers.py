"""Signed references must resolve through tenant-bound permission checks."""
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404
from django.http import Http404
from core.models import Employee
from core import views

@login_required
def employee_profile_secure(request, pk):
    org = getattr(request, 'organisation', None)
    if not request.user.is_superuser and org is None:
        raise Http404()
    qs = Employee.global_objects.filter(pk=pk)
    if not request.user.is_superuser:
        qs = qs.filter(organisation=org)
        if request.user.employee.role not in ('Admin','HR','Finance'):
            qs = qs.filter(user=request.user)
    employee = get_object_or_404(qs)
    return views.employee_profile(request, employee.pk)
