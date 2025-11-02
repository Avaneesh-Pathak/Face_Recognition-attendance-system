from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import os
import json

class Employee(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    employee_id = models.CharField(max_length=20, unique=True)
    department = models.CharField(max_length=100)
    position = models.CharField(max_length=100)
    phone_number = models.CharField(max_length=15)
    face_encoding = models.TextField(blank=True, null=True)  # Store face encoding as JSON
    face_image = models.ImageField(upload_to='faces/', blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.get_full_name()} ({self.employee_id})"
    
    def save_face_encoding(self, encoding):
        """Save face encoding as JSON string"""
        if encoding is not None:
            self.face_encoding = json.dumps(encoding.tolist())
            self.save()
    
    def get_face_encoding(self):
        """Retrieve face encoding from JSON string"""
        if self.face_encoding:
            import numpy as np
            return np.array(json.loads(self.face_encoding))
        return None

class Attendance(models.Model):
    ATTENDANCE_TYPES = [
        ('check_in', 'Check In'),
        ('check_out', 'Check Out'),
    ]
    
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    attendance_type = models.CharField(max_length=10, choices=ATTENDANCE_TYPES)
    timestamp = models.DateTimeField(default=timezone.now)
    location = models.CharField(max_length=100, blank=True, null=True)
    confidence_score = models.FloatField(blank=True, null=True)
    image_capture = models.ImageField(upload_to='attendance_captures/', blank=True, null=True)
    notes = models.TextField(blank=True, null=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['employee', 'timestamp']),
        ]
    
    def __str__(self):
        return f"{self.employee} - {self.attendance_type} at {self.timestamp}"

class AttendanceSettings(models.Model):
    check_in_start = models.TimeField(default='09:00:00')
    check_in_end = models.TimeField(default='10:00:00')
    check_out_start = models.TimeField(default='17:00:00')
    check_out_end = models.TimeField(default='18:00:00')
    late_threshold = models.TimeField(default='09:15:00')
    confidence_threshold = models.FloatField(default=0.6)
    max_daily_hours = models.FloatField(default=8.0)

    # 🕒 New field: minimum hours before checkout allowed
    min_hours_before_checkout = models.FloatField(default=3.0, help_text="Minimum hours required between check-in and check-out")

    def __str__(self):
        return "Attendance Settings"

    def save(self, *args, **kwargs):
        # Ensure only one instance exists
        if not self.pk and AttendanceSettings.objects.exists():
            raise ValidationError("There can be only one AttendanceSettings instance")
        super().save(*args, **kwargs)


class DailyReport(models.Model):
    date = models.DateField(unique=True)
    total_employees = models.IntegerField(default=0)
    present_count = models.IntegerField(default=0)
    absent_count = models.IntegerField(default=0)
    late_count = models.IntegerField(default=0)
    average_hours = models.FloatField(default=0.0)
    
    class Meta:
        ordering = ['-date']
    
    def __str__(self):
        return f"Report for {self.date}"
    

    