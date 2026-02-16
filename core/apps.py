# core/apps.py
from django.apps import AppConfig

class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        # 🔥 PRELOAD FaceRecognitionSystem SINGLETON
        try:
            from .face_system import get_face_system
            get_face_system()   # this warms InsightFace model
        except Exception:
            # Never crash Django startup
            pass

        # ✅ Load signals
        try:
            import core.signals
        except Exception:
            pass
