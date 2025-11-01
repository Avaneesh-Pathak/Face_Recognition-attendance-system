# in core/apps.py
from django.apps import AppConfig

class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        from .face_recognition import FaceRecognitionSystem
        self.face_system = FaceRecognitionSystem()
