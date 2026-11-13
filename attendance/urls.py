from django.contrib import admin
from django.urls import path, include, register_converter
from django.contrib.auth import views as auth_views
from core.security.converter import SecureIDConverter
from core.security.health import healthz, readyz
from core.security.private_media import private_media
from core.security import errors
from core.security import identifiers

register_converter(SecureIDConverter, 'secure')
def _admin_site_permission(request):
    """Allow global superusers and active organisation Admin staff only."""
    user = request.user
    if not user.is_active or not user.is_staff:
        return False
    if user.is_superuser:
        return True
    try:
        return bool(user.employee.is_active and user.employee.organisation_id
                    and user.employee.organisation.is_active and user.employee.role == 'Admin')
    except (AttributeError, Exception):
        return False

admin.site.has_permission = _admin_site_permission
urlpatterns = [
    path('healthz/', healthz, name='healthz'),
    path('readyz/', readyz, name='readyz'),
    path('private-media/<path:file_path>', private_media, name='private_media'),
    path('objects/employee/<secure:pk>/profile/', identifiers.employee_profile_secure, name='employee_profile_secure'),
    path('admin/', admin.site.urls),
    path('', include('core.urls')),
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/login/'), name='logout'),
]
handler400 = 'core.security.errors.bad_request'
handler403 = 'core.security.errors.permission_denied'
handler404 = 'core.security.errors.not_found'
handler500 = 'core.security.errors.server_error'
