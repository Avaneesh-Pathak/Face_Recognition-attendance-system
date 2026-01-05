# core/apps.py
from django.apps import AppConfig

class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        # ✅ Load Face Recognition System (already present in your code)
        from .face_recognition import FaceRecognitionSystem
        self.face_system = FaceRecognitionSystem()

        # ✅ Load Signals (SalaryStructure → Auto Payroll)
        import core.signals  # Ensure this file exists
