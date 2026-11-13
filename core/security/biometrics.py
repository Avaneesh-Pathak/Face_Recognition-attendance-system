"""Opaque biometric templates; no plaintext fallback after migration."""
import os
from decouple import config
from functools import lru_cache
from cryptography.fernet import Fernet, InvalidToken
from django.db import models
from django.core.exceptions import ImproperlyConfigured

@lru_cache(maxsize=1)
def _cipher():
    key = config('FACE_ENCRYPTION_KEY', default='')
    if not key:
        raise ImproperlyConfigured('FACE_ENCRYPTION_KEY required to store/retrieve biometric templates')
    try:
        return Fernet(key.encode('ascii'))
    except Exception as exc:
        raise ImproperlyConfigured('FACE_ENCRYPTION_KEY must be a valid Fernet key') from exc

class EncryptedFaceEncodingField(models.TextField):
    prefix = 'enc:v1:'
    def from_db_value(self, value, expression, connection):
        if not value:
            return value
        if not value.startswith(self.prefix):
            # Compatible only during original-data migration; disallow once encryption completed.
            if os.environ.get('ALLOW_LEGACY_FACE_TEMPLATES') == '1':
                return value
            raise ValueError('Unencrypted biometric record: run encrypt_face_templates')
        try:
            return _cipher().decrypt(value[len(self.prefix):].encode('ascii')).decode('utf-8')
        except (InvalidToken, UnicodeError) as exc:
            raise ValueError('Unable to decrypt biometric data (key mismatch)') from exc
    def get_prep_value(self, value):
        if value is None or value == '':
            return value
        # Values are cleartext *in Python* and encrypted on each DB write.
        if not isinstance(value, str):
            raise TypeError('face_encoding must be a JSON string')
        return self.prefix + _cipher().encrypt(value.encode('utf-8')).decode('ascii')
