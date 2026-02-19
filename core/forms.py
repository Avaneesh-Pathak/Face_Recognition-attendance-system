from django import forms
import calendar

from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from datetime import datetime, date
from .models import (Employee, AttendanceSettings,Department, SalaryStructure, Payroll, LeaveType, 
    LeaveApplication, LeaveWorkflowStage, LeaveApproval,
    JoiningDetail, Resignation, Notification,WorkRule)
from django.forms.widgets import ClearableFileInput

# ✅ Custom Multi-File Input
class MultiFileInput(ClearableFileInput):
    allow_multiple_selected = True


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




class EmployeeRegistrationForm(forms.Form):
    # --- 👤 User Info ---
    username = forms.CharField(max_length=150, label="Username")
    first_name = forms.CharField(max_length=50, label="First Name")
    last_name = forms.CharField(max_length=50, label="Last Name")
    email = forms.EmailField(label="Email Address")
    password1 = forms.CharField(widget=forms.PasswordInput, label="Password")
    password2 = forms.CharField(widget=forms.PasswordInput, label="Confirm Password")
    role = forms.ChoiceField(choices=Employee.ROLE_CHOICES, label="Role")
    # --- 🧾 Employee Info ---
    # employee_id = forms.CharField(max_length=20, label="Employee ID")
    department = forms.ModelChoiceField(queryset=Department.objects.all(), required=False)
    position = forms.CharField(max_length=100)
    manager = forms.ModelChoiceField(queryset=Employee.objects.all(), required=False, help_text="Reporting Manager")
    work_rule = forms.ModelChoiceField(   # ✅ ADD THIS
        queryset=WorkRule.objects.all(),
        required=False,
        label="Work Rule",
        help_text="Defines shift, working days, overtime & night rules"
    )
    phone_number = forms.CharField(max_length=15)
    face_image = forms.ImageField(required=False, label="Face Image (Upload or Webcam)")

    # --- 💰 Salary Info ---
    base_salary = forms.DecimalField(max_digits=10, decimal_places=2, required=False)
    hra = forms.DecimalField(max_digits=10, decimal_places=2, required=False, initial=0)
    allowances = forms.DecimalField(max_digits=10, decimal_places=2, required=False, initial=0)
    deductions = forms.DecimalField(max_digits=10, decimal_places=2, required=False, initial=0)

    # # ------------- Joining Info ----------------
    date_of_joining = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}), label="Date of Joining")
    probation_period_months = forms.IntegerField(min_value=0, max_value=24, initial=3, label="Probation Period (Months)")
    confirmation_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date', 'readonly': 'readonly'}), required=False,label="Confirmation Date")
    
    def clean(self):
        cleaned = super().clean()
        if cleaned.get('password1') != cleaned.get('password2'):
            raise ValidationError("Passwords do not match")
        return cleaned

    

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
    

# ============================================================
# 🏢 ORGANISATION STRUCTURE FORMS
# ============================================================

class DepartmentForm(forms.ModelForm):
    class Meta:
        model = Department
        fields = ['name', 'head', 'parent_department']
        widgets = {
            'name': forms.Select(attrs={
                'class': 'form-control select2',
            }),
            'head': forms.Select(attrs={
                'class': 'form-control select2'
            }),
            'parent_department': forms.Select(attrs={
                'class': 'form-control select2'
            }),
        }
        labels = {
            'name': 'Department Name',
            'head': 'Department Head',
            'parent_department': 'Parent Department'
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Filter only active employees as possible heads
        self.fields['head'].queryset = Employee.objects.filter(
            employment_status='active'
        ).select_related('user')

        # Avoid showing self as parent department
        if self.instance.pk:
            self.fields['parent_department'].queryset = Department.objects.exclude(pk=self.instance.pk)
        else:
            self.fields['parent_department'].queryset = Department.objects.all()

    def clean(self):
        cleaned_data = super().clean()
        name = cleaned_data.get('name')
        parent = cleaned_data.get('parent_department')

        # ✅ Prevent choosing same department as parent
        if self.instance.pk and parent and parent.pk == self.instance.pk:
            self.add_error('parent_department', "A department cannot be its own parent!")

        return cleaned_data


# ============================================================
# 💰 SALARY & PAYROLL FORMS
# ============================================================

class SalaryStructureForm(forms.ModelForm):
    class Meta:
        model = SalaryStructure
        fields = ['employee', 'base_salary', 'hra', 'allowances', 'deductions', 'effective_from']
        widgets = {
            'employee': forms.Select(attrs={
                'class': 'form-control select2',
                'data-placeholder': 'Select employee'
            }),
            'base_salary': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.01',
                'min': '0',
                'placeholder': '0.00'
            }),
            'hra': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.01',
                'min': '0',
                'placeholder': '0.00'
            }),
            'allowances': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.01',
                'min': '0',
                'placeholder': '0.00'
            }),
            'deductions': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.01',
                'min': '0',
                'placeholder': '0.00'
            }),
            'effective_from': forms.DateInput(attrs={
                'class': 'form-control',
                'type': 'date'
            }),
        }
        labels = {
            'base_salary': 'Basic Salary',
            'hra': 'House Rent Allowance (HRA)',
            'allowances': 'Other Allowances',
            'deductions': 'Deductions',
            'effective_from': 'Effective From Date'
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only show active employees without existing salary structure
        existing_employees = SalaryStructure.objects.values_list('employee_id', flat=True)
        if self.instance.pk:
            # When editing, include the current employee
            existing_employees = existing_employees.exclude(employee_id=self.instance.employee_id)
        
        self.fields['employee'].queryset = Employee.objects.filter(
            employment_status='active'
        ).exclude(id__in=existing_employees).select_related('user')

    def clean_base_salary(self):
        base_salary = self.cleaned_data.get('base_salary')
        if base_salary and base_salary < 0:
            raise ValidationError('Basic salary cannot be negative.')
        return base_salary

    def clean_effective_from(self):
        effective_from = self.cleaned_data.get('effective_from')
        if effective_from and effective_from > date.today():
            raise ValidationError('Effective date cannot be in the future.')
        return effective_from

class PayrollFilterForm(forms.Form):
    MONTH_CHOICES = [
        ('', 'All Months'),
        (1, 'January'), (2, 'February'), (3, 'March'), 
        (4, 'April'), (5, 'May'), (6, 'June'),
        (7, 'July'), (8, 'August'), (9, 'September'),
        (10, 'October'), (11, 'November'), (12, 'December')
    ]
    
    YEAR_CHOICES = [
        ('', 'All Years')
    ] + [(year, str(year)) for year in range(2020, datetime.now().year + 2)]
    
    month = forms.ChoiceField(
        choices=MONTH_CHOICES,
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-control',
            'onchange': 'this.form.submit()'
        })
    )
    year = forms.ChoiceField(
        choices=YEAR_CHOICES,
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-control',
            'onchange': 'this.form.submit()'
        })
    )
    status = forms.ChoiceField(
        choices=[('', 'All Status')] + Payroll._meta.get_field('status').choices,
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-control',
            'onchange': 'this.form.submit()'
        })
    )
    employee = forms.ModelChoiceField(
        queryset=Employee.objects.filter(employment_status='active'),
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-control select2',
            'data-placeholder': 'Select employee'
        })
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # Set initial values
        self.fields['month'].initial = str(current_month)
        self.fields['year'].initial = str(current_year)

class PayrollGenerationForm(forms.Form):
    month = forms.ChoiceField(
        choices=[(i, calendar.month_name[i]) for i in range(1, 13)],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    year = forms.IntegerField(
        min_value=2020,
        max_value=2030,
        initial=datetime.now().year,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        current_month = datetime.now().month
        self.fields['month'].initial = current_month

# ============================================================
# 📝 LEAVE MANAGEMENT FORMS
# ============================================================

class LeaveTypeForm(forms.ModelForm):
    class Meta:
        model = LeaveType
        fields = ['name', 'max_days_per_year']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter leave type name'
            }),
            'max_days_per_year': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '1',
                'max': '365'
            }),
        }
        labels = {
            'max_days_per_year': 'Maximum Days Per Year'
        }
        help_texts = {
            'max_days_per_year': 'Maximum number of leaves allowed per year for this type'
        }

    def clean_name(self):
        name = self.cleaned_data.get('name')
        if name:
            # Check for duplicate leave types (case-insensitive)
            leave_types = LeaveType.objects.filter(name__iexact=name)
            if self.instance.pk:
                leave_types = leave_types.exclude(pk=self.instance.pk)
            if leave_types.exists():
                raise ValidationError('A leave type with this name already exists.')
        return name

    def clean_max_days_per_year(self):
        max_days = self.cleaned_data.get('max_days_per_year')
        if max_days and max_days > 365:
            raise ValidationError('Maximum days cannot exceed 365.')
        if max_days and max_days < 1:
            raise ValidationError('Maximum days must be at least 1.')
        return max_days

class LeaveApplicationForm(forms.ModelForm):
    class Meta:
        model = LeaveApplication
        fields = ['leave_type', 'start_date', 'end_date', 'reason']
        widgets = {
            'leave_type': forms.Select(attrs={
                'class': 'form-control select2'
            }),
            'start_date': forms.DateInput(attrs={
                'class': 'form-control',
                'type': 'date',
                'min': str(date.today())
            }),
            'end_date': forms.DateInput(attrs={
                'class': 'form-control',
                'type': 'date',
                'min': str(date.today())
            }),
            'reason': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Please provide a reason for your leave application...'
            }),
        }
        labels = {
            'leave_type': 'Type of Leave',
            'start_date': 'Start Date',
            'end_date': 'End Date',
            'reason': 'Reason for Leave'
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only show active leave types
        self.fields['leave_type'].queryset = LeaveType.objects.all()

    def clean(self):
        cleaned_data = super().clean()
        start_date = cleaned_data.get('start_date')
        end_date = cleaned_data.get('end_date')
        leave_type = cleaned_data.get('leave_type')
        
        if start_date and end_date:
            if start_date > end_date:
                raise ValidationError("End date cannot be before start date.")
            
            if start_date < date.today():
                raise ValidationError("Cannot apply for leave in the past.")
            
            # Check if leave duration exceeds maximum allowed days
            if leave_type:
                total_days = (end_date - start_date).days + 1
                if total_days > leave_type.max_days_per_year:
                    raise ValidationError(
                        f"Leave duration ({total_days} days) exceeds maximum allowed "
                        f"({leave_type.max_days_per_year} days) for {leave_type.name}."
                    )
        
        return cleaned_data

class LeaveApprovalForm(forms.ModelForm):
    class Meta:
        model = LeaveApproval
        fields = ['status', 'remarks']
        widgets = {
            'status': forms.Select(attrs={
                'class': 'form-control',
                'onchange': 'toggleRemarks(this.value)'
            }),
            'remarks': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Enter your remarks or comments...',
                'id': 'remarks-field'
            }),
        }
        labels = {
            'status': 'Decision',
            'remarks': 'Remarks (Optional)'
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove 'pending' from status choices for approval form
        self.fields['status'].choices = [
            ('approved', 'Approve'),
            ('rejected', 'Reject')
        ]

class LeaveWorkflowStageForm(forms.ModelForm):
    class Meta:
        model = LeaveWorkflowStage
        fields = ['level', 'role_name', 'next_level']
        widgets = {
            'level': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '1',
                'placeholder': '1'
            }),
            'role_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., Team Lead, Manager, HR'
            }),
            'next_level': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '2',
                'placeholder': 'Leave blank for final approval'
            }),
        }
        labels = {
            'level': 'Approval Level',
            'role_name': 'Approver Role',
            'next_level': 'Next Level (Optional)'
        }
        help_texts = {
            'next_level': 'Set the next approval level. Leave blank if this is the final approval stage.'
        }

    def clean_level(self):
        level = self.cleaned_data.get('level')
        if level and level < 1:
            raise ValidationError('Level must be at least 1.')
        
        # Check for duplicate levels
        stages = LeaveWorkflowStage.objects.filter(level=level)
        if self.instance.pk:
            stages = stages.exclude(pk=self.instance.pk)
        if stages.exists():
            raise ValidationError('A workflow stage with this level already exists.')
        
        return level

    def clean_next_level(self):
        level = self.cleaned_data.get('level')
        next_level = self.cleaned_data.get('next_level')
        
        if next_level and next_level <= level:
            raise ValidationError('Next level must be greater than current level.')
        
        return next_level

# ============================================================
# 📄 JOINING & RESIGNATION FORMS
# ============================================================
# ✅ Custom widget to support multiple file uploads


class JoiningDetailForm(forms.ModelForm):
    class Meta:
        model = JoiningDetail
        fields = [
            'employee',
            'date_of_joining',
            'probation_period_months',
            'confirmation_date',
        ]
        widgets = {
            'employee': forms.Select(attrs={
                'class': 'form-control select2',
                'data-placeholder': 'Select employee'
            }),
            'date_of_joining': forms.DateInput(attrs={
                'class': 'form-control',
                'type': 'date',
                'max': str(date.today())
            }),
            'probation_period_months': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '0',
                'max': '24',
                'placeholder': '3'
            }),
            'confirmation_date': forms.DateInput(attrs={
                'class': 'form-control',
                'type': 'date',
                'readonly': 'readonly'
            }),
        }
        labels = {
            'probation_period_months': 'Probation Period (Months)',
        }
        help_texts = {
            'confirmation_date': 'Automatically calculated based on joining date + probation period'
        }

    def clean_probation_period_months(self):
        probation_months = self.cleaned_data.get('probation_period_months')
        if probation_months and probation_months > 24:
            raise ValidationError('Probation period cannot exceed 24 months.')
        return probation_months

    def clean(self):
        cleaned_data = super().clean()
        date_of_joining = cleaned_data.get('date_of_joining')
        probation_months = cleaned_data.get('probation_period_months')

        if date_of_joining and probation_months:
            from dateutil.relativedelta import relativedelta
            cleaned_data['confirmation_date'] = (
                date_of_joining + relativedelta(months=probation_months)
            )
        return cleaned_data



class ResignationForm(forms.ModelForm):
    class Meta:
        model = Resignation
        fields = ['employee', 'resignation_date', 'last_working_day', 'reason']
        widgets = {
            'employee': forms.Select(attrs={
                'class': 'form-control select2',
                'data-placeholder': 'Select employee'
            }),
            'resignation_date': forms.DateInput(attrs={
                'class': 'form-control',
                'type': 'date',
                'min': str(date.today())
            }),
            'last_working_day': forms.DateInput(attrs={
                'class': 'form-control',
                'type': 'date',
                'min': str(date.today())
            }),
            'reason': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Please provide your reason for resignation...'
            }),
        }
        labels = {
            'resignation_date': 'Resignation Date',
            'last_working_day': 'Last Working Day',
            'reason': 'Reason for Resignation'
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # For staff users, show all active employees
        # For regular employees, show only themselves
        if not kwargs.get('instance') and hasattr(self, 'initial') and 'employee' in self.initial:
            self.fields['employee'].queryset = Employee.objects.filter(
                id=self.initial['employee']
            )
        else:
            self.fields['employee'].queryset = Employee.objects.filter(
                employment_status='active'
            ).select_related('user')

    def clean(self):
        cleaned_data = super().clean()
        resignation_date = cleaned_data.get('resignation_date')
        last_working_day = cleaned_data.get('last_working_day')
        
        if resignation_date and last_working_day:
            if last_working_day <= resignation_date:
                raise ValidationError("Last working day must be after resignation date.")
            
            # Check if notice period is reasonable (not more than 3 months)
            notice_period = (last_working_day - resignation_date).days
            if notice_period > 90:
                raise ValidationError("Notice period cannot exceed 90 days.")
        
        return cleaned_data

# ============================================================
# 🔔 NOTIFICATION FORMS
# ============================================================

class NotificationForm(forms.ModelForm):
    class Meta:
        model = Notification
        fields = ['recipient', 'message', 'link']
        widgets = {
            'recipient': forms.Select(attrs={
                'class': 'form-control select2',
                'data-placeholder': 'Select recipient'
            }),
            'message': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Enter notification message...',
                'maxlength': '255'
            }),
            'link': forms.URLInput(attrs={
                'class': 'form-control',
                'placeholder': 'https://example.com (optional)'
            }),
        }
        labels = {
            'recipient': 'Send To',
            'message': 'Notification Message',
            'link': 'Related Link (Optional)'
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['recipient'].queryset = User.objects.filter(
            is_active=True
        ).select_related('employee')

    def clean_message(self):
        message = self.cleaned_data.get('message')
        if message and len(message.strip()) < 5:
            raise ValidationError('Message must be at least 5 characters long.')
        return message

# ============================================================
# 🔧 EMPLOYEE MANAGEMENT FORMS
# ============================================================

class EmployeeSearchForm(forms.Form):
    search = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search by name, employee ID, department...',
            'autocomplete': 'off'
        })
    )
    department = forms.ModelChoiceField(
        queryset=Department.objects.all(),
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-control select2',
            'data-placeholder': 'All Departments'
        })
    )
    status = forms.ChoiceField(
        choices=[('', 'All Status')] + Employee._meta.get_field('employment_status').choices,
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )

# ============================================================
# 📊 REPORT FILTER FORMS
# ============================================================

class DateRangeFilterForm(forms.Form):
    start_date = forms.DateField(
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date'
        })
    )
    end_date = forms.DateField(
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date'
        })
    )

    def clean(self):
        cleaned_data = super().clean()
        start_date = cleaned_data.get('start_date')
        end_date = cleaned_data.get('end_date')
        
        if start_date and end_date:
            if start_date > end_date:
                raise ValidationError("Start date cannot be after end date.")
            
            if (end_date - start_date).days > 365:
                raise ValidationError("Date range cannot exceed 1 year.")
        
        return cleaned_data

class AttendanceReportFilterForm(DateRangeFilterForm):
    department = forms.ModelChoiceField(
        queryset=Department.objects.all(),
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-control select2',
            'data-placeholder': 'All Departments'
        })
    )
    employee = forms.ModelChoiceField(
        queryset=Employee.objects.filter(employment_status='active'),
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-control select2',
            'data-placeholder': 'All Employees'
        })
    )

class LeaveReportFilterForm(DateRangeFilterForm):
    leave_type = forms.ModelChoiceField(
        queryset=LeaveType.objects.all(),
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-control select2',
            'data-placeholder': 'All Leave Types'
        })
    )
    status = forms.ChoiceField(
        choices=[('', 'All Status')] + LeaveApplication._meta.get_field('status').choices,
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )


STATUS_CHOICES = (
    ('', 'All'),
    ('pending', 'Pending'),
    ('processed', 'Processed'),
    ('paid', 'Paid'),
)

class PayrollFilterForm(forms.Form):
    month = forms.IntegerField(required=False, min_value=1, max_value=12,
                               widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '1-12'}))
    year = forms.IntegerField(required=False,
                              widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'YYYY'}))
    status = forms.ChoiceField(required=False, choices=STATUS_CHOICES,
                               widget=forms.Select(attrs={'class': 'form-select'}))