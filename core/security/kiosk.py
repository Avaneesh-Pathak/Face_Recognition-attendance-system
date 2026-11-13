"""Attendance-only browser sessions with server-side code verification."""
import hashlib
import uuid

from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import check_password, make_password
from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import redirect
from django.utils.cache import patch_cache_control
from django.utils.deprecation import MiddlewareMixin
from django.views.decorators.debug import sensitive_post_parameters
from django.views.decorators.http import require_GET, require_POST

from core.models import AuditEvent

SESSION_KEY = 'attendance_kiosk'


def can_manage_kiosk(request):
    employee = getattr(request.user, 'employee', None)
    return bool(request.user.is_authenticated and employee and employee.is_active
                and employee.role in ('Admin', 'HR') and request.organisation
                and employee.organisation_id == request.organisation.pk)


def is_locked(request):
    return bool(request.session.get(SESSION_KEY))


def audit(request, action, kiosk_id):
    AuditEvent.global_objects.create(
        organisation=request.organisation, actor_id=request.user.pk,
        actor_name=request.user.get_username(), entity='AttendanceKiosk',
        object_id=kiosk_id, action=action,
        after={'locked': action != 'unlock'},
        request_id=getattr(request, 'request_id', ''),
    )


class KioskMiddleware(MiddlewareMixin):
    allowed_views = frozenset({
        'attendance_page', 'mark_attendance', 'attendance_log_api',
        'attendance_heartbeat', 'kiosk_lock', 'kiosk_unlock', 'kiosk_status',
    })

    def process_view(self, request, view_func, view_args, view_kwargs):
        if not is_locked(request):
            return
        match = request.resolver_match
        if match and match.view_name in self.allowed_views:
            return
        if request.method in ('GET', 'HEAD') and 'application/json' not in request.headers.get('Accept', ''):
            return redirect('attendance_page')
        return JsonResponse({'success': False, 'error': 'kiosk_locked',
                             'message': 'Unlock the attendance kiosk to access this page.'}, status=423)

    def process_response(self, request, response):
        if request.user.is_authenticated:
            patch_cache_control(response, private=True, no_store=True)
        return response


@login_required
@require_GET
def kiosk_status(request):
    return JsonResponse({'locked': is_locked(request)})


@sensitive_post_parameters('code', 'confirm_code')
@login_required
@require_POST
def kiosk_lock(request):
    if is_locked(request):
        return JsonResponse({'success': False, 'message': 'Unlock the current kiosk before changing its code.'}, status=409)
    if not can_manage_kiosk(request):
        return JsonResponse({'success': False, 'message': 'Only Admin or HR can lock a terminal.'}, status=403)
    code = request.POST.get('code', '')
    if not 6 <= len(code) <= 128 or not code.strip() or code != request.POST.get('confirm_code'):
        return JsonResponse({'success': False, 'message': 'Enter matching codes of 6–128 characters.'}, status=400)
    kiosk = {'id': uuid.uuid4().hex, 'code_hash': make_password(code)}
    audit(request, 'lock', kiosk['id'])
    request.session[SESSION_KEY] = kiosk
    return JsonResponse({'success': True, 'locked': True})


@sensitive_post_parameters('code')
@login_required
@require_POST
def kiosk_unlock(request):
    kiosk = request.session.get(SESSION_KEY)
    if not kiosk:
        return JsonResponse({'success': True, 'locked': False})
    # One shared counter per terminal, unaffected by IP changes or page reloads.
    key = 'kiosk-unlock:' + hashlib.sha256(kiosk['id'].encode()).hexdigest()
    try:
        attempts = 1 if cache.add(key, 1, timeout=300) else cache.incr(key)
    except Exception:
        return JsonResponse({'success': False, 'message': 'Unlock service unavailable. Try again shortly.'}, status=503)
    if attempts > 5:
        response = JsonResponse({'success': False, 'message': 'Too many attempts. Wait 5 minutes before trying again.'}, status=429)
        response['Retry-After'] = '300'
        return response
    code = request.POST.get('code', '')
    if not 6 <= len(code) <= 128 or not check_password(code, kiosk['code_hash']):
        audit(request, 'unlock_failed', kiosk['id'])
        return JsonResponse({'success': False, 'message': 'Incorrect secret code. The kiosk is still locked.'}, status=403)
    audit(request, 'unlock', kiosk['id'])
    del request.session[SESSION_KEY]
    # The expired counter has a unique kiosk id; no cache deletion can fail after unlocking.
    return JsonResponse({'success': True, 'locked': False})
