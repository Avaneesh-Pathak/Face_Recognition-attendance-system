from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError

from .models import Employee, AttendanceSettings

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    
    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2']

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if email and User.objects.filter(email__iexact=email).exclude(pk=getattr(self.instance, 'pk', None)).exists():
            raise ValidationError("A user with this email already exists.")
        return email


class EmployeeRegistrationForm(forms.ModelForm):
    face_image = forms.ImageField(required=True, help_text="Upload a clear front-facing photo for face recognition")
    
    class Meta:
        model = Employee
        fields = ['employee_id', 'department', 'position', 'phone_number', 'face_image']

    def clean_phone_number(self):
        phone = self.cleaned_data.get('phone_number', '').strip()
        # allow digits, spaces, plus and hyphen
        cleaned = ''.join(ch for ch in phone if ch.isdigit())
        if not (7 <= len(cleaned) <= 15):
            raise ValidationError("Enter a valid phone number (7-15 digits).")
        return phone

    def clean_employee_id(self):
        emp_id = self.cleaned_data.get('employee_id')
        if emp_id and Employee.objects.filter(employee_id=emp_id).exclude(pk=getattr(self.instance, 'pk', None)).exists():
            raise ValidationError("Employee ID must be unique.")
        return emp_id


class AttendanceSettingsForm(forms.ModelForm):
    class Meta:
        model = AttendanceSettings
        fields = '__all__'
        widgets = {
            'check_in_start': forms.TimeInput(attrs={'type': 'time'}),
            'check_in_end': forms.TimeInput(attrs={'type': 'time'}),
            'check_out_start': forms.TimeInput(attrs={'type': 'time'}),
            'check_out_end': forms.TimeInput(attrs={'type': 'time'}),
            'late_threshold': forms.TimeInput(attrs={'type': 'time'}),
            # min_hours_before_checkout is a FloatField -> use number input
            'min_hours_before_checkout': forms.NumberInput(attrs={'step': '0.1', 'min': '0', 'max': '24'}),
        }

    def clean_min_hours_before_checkout(self):
        val = self.cleaned_data.get('min_hours_before_checkout')
        if val is None:
            return val
        if val < 0 or val > 24:
            raise ValidationError("Minimum hours must be between 0 and 24.")
        max_daily = self.cleaned_data.get('max_daily_hours')
        if max_daily is not None and val > max_daily:
            raise ValidationError("Minimum hours before checkout cannot exceed maximum daily hours.")
        return val