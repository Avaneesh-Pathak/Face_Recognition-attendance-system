import logging
from django.http import JsonResponse
from django.db import connection
from django.core.cache import cache
from django.views.decorators.http import require_GET

@require_GET
def healthz(request):
    return JsonResponse({'ok': True})

@require_GET
def readyz(request):
    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT 1')
        cache.set('readiness_probe', 'ok', 5)
        if cache.get('readiness_probe') != 'ok':
            raise RuntimeError('cache not ready')
    except Exception:
        logging.getLogger('security').error('readiness_probe_failed')
        return JsonResponse({'ok': False}, status=503)
    return JsonResponse({'ok': True})
