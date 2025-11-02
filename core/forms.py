from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Employee, AttendanceSettings

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    
    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2']

class EmployeeRegistrationForm(forms.ModelForm):
    face_image = forms.ImageField(required=True, help_text="Upload a clear front-facing photo for face recognition")
    
    class Meta:
        model = Employee
        fields = ['employee_id', 'department', 'position', 'phone_number', 'face_image']

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
            'min_hours_before_checkout': forms.TimeInput(attrs={'type': 'time'}),
        }