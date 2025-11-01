import face_recognition
import cv2
import numpy as np
import pickle
import os
from django.conf import settings
from .models import Employee
import json

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_employee_ids = []
        self.confidence_threshold = 0.6
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load all known face encodings from database"""
        try:
            employees = Employee.objects.filter(is_active=True).exclude(face_encoding__isnull=True)
            self.known_face_encodings.clear()
            self.known_face_employee_ids.clear()

            for employee in employees:
                encoding = employee.get_face_encoding()
                if encoding is not None:
                    self.known_face_encodings.append(encoding)
                    self.known_face_employee_ids.append(employee.employee_id)
        except Exception as e:
            # Prevent crash if database or table doesn't exist yet
            print(f"[WARN] Skipping face load (DB not ready): {e}")
    
    def recognize_face(self, image_path):
        """Recognize face from image file"""
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image, model=settings.FACE_RECOGNITION_MODEL)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if not face_encodings:
                return None, 0.0

            face_encoding = face_encodings[0]
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            confidence = 1 - face_distances[best_match_index]

            if matches[best_match_index] and confidence >= self.confidence_threshold:
                return self.known_face_employee_ids[best_match_index], confidence

            return None, confidence
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return None, 0.0
    
    def recognize_face_from_frame(self, frame):
        """Recognize face from video frame"""
        try:
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame, model=settings.FACE_RECOGNITION_MODEL)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if not face_encodings:
                return None, 0.0, None

            face_encoding = face_encodings[0]
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            confidence = 1 - face_distances[best_match_index]

            if matches[best_match_index] and confidence >= self.confidence_threshold:
                return self.known_face_employee_ids[best_match_index], confidence, face_locations[0]

            return None, confidence, face_locations[0]
        except Exception as e:
            print(f"Error in face recognition from frame: {e}")
            return None, 0.0, None
    
    def register_face(self, image_path, employee):
        """Register a new face for an employee"""
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if not face_encodings:
                return False, "No face detected in the image"
            if len(face_encodings) > 1:
                return False, "Multiple faces detected. Please provide an image with only one face."
            
            employee.save_face_encoding(face_encodings[0])
            self.load_known_faces()
            return True, "Face registered successfully"
        except Exception as e:
            return False, f"Error registering face: {str(e)}"


# ✅ Lazy load instance — safe for migrations
_face_system_instance = None

def get_face_system():
    global _face_system_instance
    if _face_system_instance is None:
        _face_system_instance = FaceRecognitionSystem()
    return _face_system_instance
