from django.utils.deprecation import MiddlewareMixin
from django.core.exceptions import PermissionDenied
from .tenant_utils import set_current_organisation, set_global_access

class TenantMiddleware(MiddlewareMixin):
    def process_request(self, request):
        set_global_access(False)
        set_current_organisation(None)
        request.organisation = None
        if not request.user.is_authenticated:
            return
        if request.user.is_superuser:
            set_global_access(True)
            return
        try:
            employee = request.user.employee
            org = employee.organisation
        except (AttributeError, Exception) as exc:
            # Users without a valid employee profile must never receive all-tenant access.
            raise PermissionDenied('No active organisation membership') from exc
        if org is None or not org.is_active or not employee.is_active:
            raise PermissionDenied('Organisation membership is not active')
        request.organisation = org
        set_current_organisation(org)
    def process_response(self, request, response):
        set_current_organisation(None)
        set_global_access(False)
        return response
    def process_exception(self, request, exception):
        set_current_organisation(None)
        set_global_access(False)
