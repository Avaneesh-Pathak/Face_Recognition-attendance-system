from django.shortcuts import render
from django.http import JsonResponse

def _handle(request, status):
    if request.path.startswith('/api/') or request.headers.get('Accept', '').startswith('application/json'):
        return JsonResponse({'ok': False, 'error': {'code': f'http_{status}', 'message': 'Request could not be completed'}}, status=status)
    return render(request, f'{status}.html', status=status)

def bad_request(request, exception): return _handle(request, 400)
def permission_denied(request, exception): return _handle(request, 403)
def not_found(request, exception): return _handle(request, 404)
def server_error(request): return _handle(request, 500)
