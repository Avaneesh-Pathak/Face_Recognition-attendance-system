from django.utils.deprecation import MiddlewareMixin
from .tenant_utils import set_current_organisation

class TenantMiddleware(MiddlewareMixin):
    def process_request(self, request):
        if request.user.is_authenticated:
            # Check if the authenticated user is linked to an Employee profile
            if hasattr(request.user, 'employee') and request.user.employee:
                org = request.user.employee.organisation
                request.organisation = org
                set_current_organisation(org)
            else:
                request.organisation = None
                set_current_organisation(None)
        else:
            request.organisation = None
            set_current_organisation(None)

    def process_response(self, request, response):
        # Clear thread-local variable after response to prevent leakage to subsequent threads
        set_current_organisation(None)
        return response