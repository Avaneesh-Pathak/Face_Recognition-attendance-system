# core/apps.py
from django.apps import AppConfig

class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        # Initialize the pool; model instances are loaded lazily per inference slot.
        try:
            from .face_system import get_face_system
            get_face_system()
        except Exception:
            # Never crash Django startup
            pass

        # Audit initialization errors must not silently disable business history.
        import core.signals
