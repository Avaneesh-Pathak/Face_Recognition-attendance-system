from django.urls.converters import StringConverter
from core.security.crypto import decrypt_token

class SecureIDConverter(StringConverter):
    regex = r'[A-Za-z0-9_:\-=.]+'
    def to_python(self, value):
        return decrypt_token(value)['i']
    def to_url(self, value):
        return str(value)
