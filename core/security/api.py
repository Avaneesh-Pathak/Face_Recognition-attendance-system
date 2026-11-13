from rest_framework.views import exception_handler

def api_exception_handler(exc, context):
    response = exception_handler(exc, context)
    if response is not None:
        response.data = {'ok': False, 'error': {'code': getattr(exc, 'default_code', 'request_error'), 'detail': response.data}}
    return response
