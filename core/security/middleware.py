"""Common API protection; Redis cache.add/incr is atomic across workers."""
import hashlib
import logging
import time
import uuid
from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse, HttpResponse
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger('security')

class RequestSecurityMiddleware(MiddlewareMixin):
    def process_request(self, request):
        request.request_id = uuid.uuid4().hex
        request._perf_start = time.monotonic()
        path = request.path
        if path.startswith('/static/') or path in ('/healthz/', '/readyz/'):
            return
        if path == '/login/' or path.startswith('/admin/login/'):
            limit, period = 10, 300
        elif path in ('/mark_attendance/', '/api/v2/attendance/scan/'):
            limit, period = 600, 60
        elif path.startswith('/api/') or path.startswith('/attendance/'):
            limit, period = 120, 60
        elif path == '/register/':
            limit, period = 15, 300
        else:
            return
        org_id = getattr(getattr(request, 'organisation', None), 'pk', 'none')
        # REMOTE_ADDR is supplied by the server, not user-controlled X-Forwarded-For.
        client_id = f"{request.META.get('REMOTE_ADDR', '')}:{request.user.pk if request.user.is_authenticated else 'anon'}:{org_id}:{path}"
        hashed = hashlib.sha256(client_id.encode()).hexdigest()
        bucket = int(time.time() // period)
        key = f'ratelimit:{hashed}:{bucket}'
        try:
            if cache.add(key, 1, timeout=period * 2):
                count = 1
            else:
                count = cache.incr(key)
        except Exception:
            logger.error('rate_limit_store_unavailable')
            return JsonResponse({'ok': False, 'error': 'temporarily_unavailable'}, status=503)
        if path == '/mark_attendance/' and request.method == 'POST':
            scan_session = request.POST.get('scan_session', '')
            if len(scan_session) == 36:
                per_device = hashlib.sha256(f'{hashed}:{scan_session}'.encode()).hexdigest()
                device_key = f'ratelimit:scanner:{per_device}:{bucket}'
                try:
                    device_count = 1 if cache.add(device_key, 1, timeout=period * 2) else cache.incr(device_key)
                except Exception:
                    return JsonResponse({'success':False,'error':'temporarily_unavailable'},status=503)
                if device_count > 60:
                    response = JsonResponse({'success':False,'error':'scan_rate_limit',
                        'message':'Scanner request limit reached. Try again shortly.'}, status=429)
                    response['Retry-After'] = str(period - int(time.time() % period))
                    return response
        if count > limit:
            response = JsonResponse({'ok': False, 'error': 'rate_limit_exceeded'}, status=429)
            response['Retry-After'] = str(period - int(time.time() % period))
            return response
    def process_response(self, request, response):
        if hasattr(request, 'request_id'):
            response['X-Request-ID'] = request.request_id
            duration = round((time.monotonic() - request._perf_start) * 1000, 2)
            if not request.path.startswith('/static/'):
                logger.info('http_request', extra={'request_id': request.request_id,
                       'status_code': response.status_code,
                       'organisation_id': getattr(getattr(request, 'organisation', None), 'pk', None),
                       'duration_ms': duration})
        response['Content-Security-Policy'] = "default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none'; form-action 'self'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://fonts.googleapis.com; font-src 'self' data: https://cdnjs.cloudflare.com https://fonts.gstatic.com"
        response['Permissions-Policy'] = 'camera=(self), microphone=(), geolocation=(self)'
        return response
