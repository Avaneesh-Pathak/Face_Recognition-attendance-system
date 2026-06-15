# core/apps.py

import os
from django.apps import AppConfig
from django.conf import settings

class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):
        # Load signals only
        try:
            import core.signals
        except Exception:
            pass

        # Only preload in production
        if not settings.DEBUG:
            try:
                from .face_system import get_face_system
                get_face_system()
            except Exception:
                pass