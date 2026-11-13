"""Signed, expiring URLs; signing isn't an authorization substitute."""
from django.core import signing
from django.core.signing import BadSignature, SignatureExpired

def encrypt_id(model, pk, organisation):
    return signing.dumps({'m': str(model), 'i': int(pk), 'o': organisation}, salt='attendance:object-link:v2', compress=True)

def decrypt_token(token, max_age=3600):
    try:
        value = signing.loads(token, salt='attendance:object-link:v2', max_age=max_age)
        if not isinstance(value, dict) or not isinstance(value.get('i'), int):
            raise ValueError('Invalid secure URL')
        return value
    except (BadSignature, SignatureExpired, TypeError, ValueError) as exc:
        raise ValueError('Invalid or expired secure URL') from exc
