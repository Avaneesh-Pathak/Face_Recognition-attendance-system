from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_GET

from .models import Employee, Organisation


@login_required
def org_chart_page(request):
    organizations = Organisation.objects.none()
    selected_organisation_id = ''
    if request.user.is_superuser:
        organizations = Organisation.objects.filter(is_active=True).order_by('name')
        candidate = request.GET.get('organisation_id', '').strip()
        try:
            candidate_id = int(candidate) if candidate else None
        except ValueError:
            candidate_id = None
        if candidate_id and organizations.filter(pk=candidate_id).exists():
            selected_organisation_id = str(candidate_id)

    return render(request, 'ems/org_chart.html', {
        'organizations': organizations,
        'selected_organisation_id': selected_organisation_id,
        'is_superuser_chart_view': request.user.is_superuser,
        'has_employee_profile': hasattr(request.user, 'employee'),
    })


@login_required
@require_GET
def org_tree_api_me(request):
    employee = getattr(request.user, 'employee', None)
    return _org_tree_response(request, employee.pk if employee else None)


@login_required
@require_GET
def org_tree_api(request, employee_id):
    return _org_tree_response(request, employee_id, require_highlight=True)


def _org_tree_response(request, highlighted_employee_id=None, require_highlight=False):
    if request.user.is_superuser:
        selected_id = request.GET.get('organisation_id', '').strip()
        try:
            selected_id = int(selected_id) if selected_id else None
        except ValueError:
            raise Http404('Choose an active organisation.')
        if not selected_id:
            return JsonResponse({'ok': True, 'tree': [], 'needs_organisation': True})
        organisation = get_object_or_404(
            Organisation.objects.filter(is_active=True),
            pk=selected_id,
        )
    else:
        # Ignore any organisation_id supplied by a tenant user. Their chart
        # scope always comes from the server-established membership context.
        organisation = getattr(request, 'organisation', None)
        if organisation is None or not organisation.is_active:
            raise PermissionDenied('An active organisation is required to view this chart.')

    employees = list(Employee.global_objects.filter(
        organisation_id=organisation.pk,
        is_active=True,
        employment_status='active',
    ).select_related('user', 'department').order_by(
        'user__first_name', 'user__last_name', 'employee_id', 'pk'
    ))
    employee_map = {}
    for employee in employees:
        employee_map[employee.pk] = {
            'id': employee.pk,
            'name': employee.user.get_full_name() or employee.user.get_username(),
            'employee_code': employee.employee_id,
            'department': employee.department.name if employee.department else '',
            'position': employee.position or employee.get_role_display(),
            'manager_id': employee.manager_id,
            'children': [],
            'is_current': False,
        }

    if highlighted_employee_id not in employee_map:
        if require_highlight:
            raise Http404('Employee is not part of this organisation chart.')
        highlighted_employee_id = None
    if highlighted_employee_id:
        employee_map[highlighted_employee_id]['is_current'] = True

    # Break invalid manager cycles into visible roots so one bad record cannot
    # make the organisation disappear from the chart.
    cyclic_nodes = set()
    for employee in employees:
        seen = {employee.pk}
        manager_id = employee.manager_id
        while manager_id in employee_map:
            if manager_id in seen:
                cyclic_nodes.add(employee.pk)
                break
            seen.add(manager_id)
            manager_id = employee_map[manager_id]['manager_id']

    roots = []
    for employee in employees:
        node = employee_map[employee.pk]
        manager_id = employee.manager_id
        if manager_id in employee_map and manager_id != employee.pk and employee.pk not in cyclic_nodes:
            employee_map[manager_id]['children'].append(node)
        else:
            roots.append(node)

    return JsonResponse({
        'ok': True,
        'tree': roots,
        'organisation_name': organisation.name,
        'employee_count': len(employees),
    })
