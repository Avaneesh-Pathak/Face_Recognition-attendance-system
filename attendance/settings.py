"""Production settings. Never commit .env; deployment requires PostgreSQL + Redis."""
import os
from pathlib import Path
from decouple import config, Csv
from django.core.exceptions import ImproperlyConfigured

# NumPy/InsightFace should not start a BLAS thread pool per Django worker.
# Leave explicit deployment overrides intact; otherwise use one thread per process.
for _thread_setting in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_thread_setting, '1')

BASE_DIR = Path(__file__).resolve().parent.parent
DEBUG = config('DEBUG', default=False, cast=bool)
SECRET_KEY = config('SECRET_KEY', default='')
if len(SECRET_KEY) < 50 or SECRET_KEY in {'your-secret-key-here', 'your-secret-key-here-change-in-production'}:
    raise ImproperlyConfigured('Set a strong random SECRET_KEY (at least 50 characters)')
ALLOWED_HOSTS = config('ALLOWED_HOSTS', default='127.0.0.1,localhost' if DEBUG else '', cast=Csv())
if not ALLOWED_HOSTS:
    raise ImproperlyConfigured('ALLOWED_HOSTS is required')
CSRF_TRUSTED_ORIGINS = config('CSRF_TRUSTED_ORIGINS', default='', cast=Csv())
CORS_ALLOWED_ORIGINS = config('CORS_ALLOWED_ORIGINS', default='', cast=Csv())
CORS_ALLOW_ALL_ORIGINS = False
CORS_ALLOW_CREDENTIALS = False
CORS_URLS_REGEX = r'^/api/'
CORS_ALLOWED_ORIGIN_REGEXES = []

INSTALLED_APPS = [
    'django.contrib.admin', 'django.contrib.auth', 'django.contrib.contenttypes',
    'django.contrib.sessions', 'django.contrib.messages', 'django.contrib.staticfiles',
    'rest_framework', 'corsheaders', 'crispy_forms', 'crispy_bootstrap5',
    'django_htmx', 'django_extensions', 'core.apps.CoreConfig',
]
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'core.middleware.TenantMiddleware',
    'core.security.middleware.RequestSecurityMiddleware',
    'core.security.audit.AuditContextMiddleware',
    'core.security.kiosk.KioskMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django_htmx.middleware.HtmxMiddleware',
]
ROOT_URLCONF = 'attendance.urls'
TEMPLATES = [{'BACKEND': 'django.template.backends.django.DjangoTemplates',
              'DIRS': [BASE_DIR / 'templates'], 'APP_DIRS': True,
              'OPTIONS': {'context_processors': [
                  'django.template.context_processors.debug',
                  'django.template.context_processors.request',
                  'django.contrib.auth.context_processors.auth',
                  'django.contrib.messages.context_processors.messages',
                  'core.context_processors.notification_context']}}]
WSGI_APPLICATION = 'attendance.wsgi.application'
ASGI_APPLICATION = 'attendance.asgi.application'

# Never silently fall back to SQLite in production.
DB_ENGINE = config('DB_ENGINE', default='postgresql')
if not DEBUG and DB_ENGINE != 'postgresql':
    raise ImproperlyConfigured('PostgreSQL required in production')
if DB_ENGINE == 'sqlite' and DEBUG:
    DATABASES = {'default': {'ENGINE':'django.db.backends.sqlite3', 'NAME': BASE_DIR / 'dev.sqlite3'}}
else:
    DB_NAME = config('DB_NAME', default='')
    DB_USER = config('DB_USER', default='')
    DB_PASSWORD = config('DB_PASSWORD', default='')
    if not (DB_NAME and DB_USER and DB_PASSWORD):
        raise ImproperlyConfigured('DB_NAME, DB_USER and DB_PASSWORD are required')
    DATABASES = {'default': {
        'ENGINE': 'django.db.backends.postgresql', 'NAME': DB_NAME, 'USER': DB_USER,
        'PASSWORD': DB_PASSWORD, 'HOST': config('DB_HOST', default='127.0.0.1'),
        'PORT': config('DB_PORT', default='5432'), 'CONN_MAX_AGE': 60,
        'CONN_HEALTH_CHECKS': True,
        'OPTIONS': {'connect_timeout': 5, 'sslmode': config('DB_SSLMODE', default='prefer')},
    }}

REDIS_URL = config('REDIS_URL', default='')
if not REDIS_URL and not DEBUG:
    raise ImproperlyConfigured('REDIS_URL required for shared rate limiting and sessions')
if REDIS_URL:
    CACHES = {'default': {'BACKEND': 'django_redis.cache.RedisCache',
                           'LOCATION': REDIS_URL,
                           'OPTIONS': {'CLIENT_CLASS': 'django_redis.client.DefaultClient',
                                       'SOCKET_CONNECT_TIMEOUT': 2, 'SOCKET_TIMEOUT': 2,
                                       'IGNORE_EXCEPTIONS': False},
                           'KEY_PREFIX': 'attendance-v2'}}
    SESSION_ENGINE = 'django.contrib.sessions.backends.cached_db'
else:
    CACHES = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}
SESSION_COOKIE_NAME = 'fa_session'
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_HTTPONLY = False  # JS client obtains CSRF token from cookie/template
CSRF_COOKIE_SAMESITE = 'Lax'
SESSION_COOKIE_AGE = 8 * 60 * 60
SESSION_SAVE_EVERY_REQUEST = False
SECURE_SSL_REDIRECT = not DEBUG
SECURE_HSTS_SECONDS = config('SECURE_HSTS_SECONDS', default=31536000 if not DEBUG else 0, cast=int)
SECURE_HSTS_INCLUDE_SUBDOMAINS = config('SECURE_HSTS_INCLUDE_SUBDOMAINS', default=False, cast=bool)
SECURE_HSTS_PRELOAD = False
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = 'strict-origin-when-cross-origin'
X_FRAME_OPTIONS = 'DENY'
# Set only behind a trusted proxy that overwrites client-supplied X-Forwarded-Proto.
if config('TRUST_PROXY_SSL_HEADER', default=False, cast=bool):
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', 'OPTIONS': {'min_length': 12}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]
PASSWORD_HASHERS = ['django.contrib.auth.hashers.PBKDF2PasswordHasher',
                    'django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher']
LANGUAGE_CODE = 'en-us'
TIME_ZONE = config('TIME_ZONE', default='Asia/Kolkata')
USE_I18N = True
USE_TZ = True
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'
STORAGES = {'staticfiles': {'BACKEND': (
                'django.contrib.staticfiles.storage.StaticFilesStorage'
                if DEBUG else 'whitenoise.storage.CompressedManifestStaticFilesStorage'
            )},
            'default': {'BACKEND': 'django.core.files.storage.FileSystemStorage'}}
WHITENOISE_MAX_AGE = 31536000
MEDIA_ROOT = BASE_DIR / 'private_media'
MEDIA_URL = '/private-media/'
DATA_UPLOAD_MAX_MEMORY_SIZE = 7 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024
MAX_FACE_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_FACE_IMAGE_PIXELS = 4_000_000

CRISPY_ALLOWED_TEMPLATE_PACKS = 'bootstrap5'
CRISPY_TEMPLATE_PACK = 'bootstrap5'
LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = 'dashboard'
LOGOUT_REDIRECT_URL = '/login/'

REST_FRAMEWORK = {
 'DEFAULT_PERMISSION_CLASSES': ['rest_framework.permissions.IsAuthenticated'],
 'DEFAULT_AUTHENTICATION_CLASSES': ['rest_framework.authentication.SessionAuthentication'],
 'DEFAULT_THROTTLE_CLASSES': ['rest_framework.throttling.UserRateThrottle', 'rest_framework.throttling.AnonRateThrottle'],
 'DEFAULT_THROTTLE_RATES': {'user': '120/min', 'anon': '30/min'},
 'EXCEPTION_HANDLER': 'core.security.api.api_exception_handler',
}

INSIGHTFACE_MODEL = config('INSIGHTFACE_MODEL', default='buffalo_l')
INSIGHTFACE_MODEL_ROOT = config('INSIGHTFACE_MODEL_ROOT', default=str(BASE_DIR / 'models'))
INSIGHTFACE_DET_SIZE = (640, 640)
FACE_RECOGNITION_THRESHOLD = config('FACE_RECOGNITION_THRESHOLD', default=0.55, cast=float)
FACE_AMBIGUITY_MARGIN = config('FACE_AMBIGUITY_MARGIN', default=0.08, cast=float)
FACE_DETECTION_MIN_SCORE = 0.65
FACE_STABLE_VOTES = 3
FACE_VOTE_WINDOW = 5
FACE_MATCH_CACHE_TTL = 30
FACE_MATCH_CACHE_MAX_ORGS = 16
FACE_SCAN_INFERENCE_SLOTS = config('FACE_SCAN_INFERENCE_SLOTS', default=1, cast=int)
if not 1 <= FACE_SCAN_INFERENCE_SLOTS <= 8:
    raise ImproperlyConfigured('FACE_SCAN_INFERENCE_SLOTS must be between 1 and 8')
FACE_EXECUTION_PROVIDERS = config('FACE_EXECUTION_PROVIDERS', default='CPUExecutionProvider', cast=Csv())
FACE_INFERENCE_QUEUE_TIMEOUT = config('FACE_INFERENCE_QUEUE_TIMEOUT', default=2.0, cast=float)
COOLDOWN_SECONDS = config('COOLDOWN_SECONDS', default=600, cast=int)
# Fail closed unless a separately tested PAD/liveness integration is enabled.
REQUIRE_VERIFIED_LIVENESS = config('REQUIRE_VERIFIED_LIVENESS', default=True, cast=bool)
FACE_LIVENESS_VERIFIER = config('FACE_LIVENESS_VERIFIER', default='')
LIVENESS = {'BLINK_THRESHOLD': 0.22, 'CONSEC_FRAMES': 3,
            'BLINK_REQUIRED': 1, 'MOTION_THRESHOLD': 0.002, 'HISTORY': 6,
            'REQUIRE_BOTH': True, 'MIN_FACE_SIZE': 120, 'COOLDOWN_SECONDS': 10, 'DEBUG': False}

EMAIL_BACKEND = config('EMAIL_BACKEND', default='django.core.mail.backends.smtp.EmailBackend')
EMAIL_HOST = config('EMAIL_HOST', default='localhost')
EMAIL_PORT = config('EMAIL_PORT', default=587, cast=int)
EMAIL_USE_TLS = config('EMAIL_USE_TLS', default=True, cast=bool)
EMAIL_HOST_USER = config('EMAIL_HOST_USER', default='')
EMAIL_HOST_PASSWORD = config('EMAIL_HOST_PASSWORD', default='')
DEFAULT_FROM_EMAIL = config('DEFAULT_FROM_EMAIL', default='no-reply@localhost')
EMAIL_SUBJECT_PREFIX = '[Attendance] '

LOGGING = {'version': 1, 'disable_existing_loggers': False,
    'formatters': {'json': {'()': 'core.security.logging.JsonFormatter'}},
    'handlers': {'console': {'class': 'logging.StreamHandler', 'formatter': 'json'}},
    'root': {'handlers': ['console'], 'level': config('LOG_LEVEL', default='INFO')},
    'loggers': {'django.request': {'handlers': ['console'], 'level':'WARNING','propagate': False}}}

SENTRY_DSN = config('SENTRY_DSN', default='')
if SENTRY_DSN:
    import sentry_sdk
    sentry_sdk.init(dsn=SENTRY_DSN, send_default_pii=False,
                    traces_sample_rate=config('SENTRY_TRACES_SAMPLE_RATE', default=0.02, cast=float),
                    before_send=lambda event, hint: {**event, 'request': {
                       'url': event.get('request', {}).get('url', '').split('?')[0]}} if event.get('request') else event)
