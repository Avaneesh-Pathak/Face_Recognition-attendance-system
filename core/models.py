# models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.validators import MinValueValidator
from django.core.exceptions import ValidationError
from decimal import Decimal
from datetime import date, timedelta
import calendar
import json


# ============================================================
# 👤 EMPLOYEE MASTER
# ============================================================

class Employee(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    employee_id = models.CharField(max_length=20, unique=True)
    
    department = models.ForeignKey('Department', on_delete=models.SET_NULL, null=True, blank=True, related_name='employees')
    position = models.CharField(max_length=100)
    manager = models.ForeignKey("self", on_delete=models.SET_NULL, null=True, blank=True, related_name="subordinates")
    phone_number = models.CharField(max_length=15)
    date_of_joining = models.DateField(default=timezone.now)
    date_of_resignation = models.DateField(blank=True, null=True)
    employment_status = models.CharField(
        max_length=20,
        choices=[
            ('active', 'Active'),
            ('on_leave', 'On Leave'),
            ('resigned', 'Resigned'),
            ('terminated', 'Terminated'),
        ],
        default='active'
    )
    face_encoding = models.TextField(blank=True, null=True)
    face_image = models.ImageField(upload_to='faces/', blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.get_full_name()} ({self.employee_id})"

    def save_face_encoding(self, encoding):
        if encoding is not None:
            enc_list = [float(x) for x in (encoding.tolist() if hasattr(encoding, 'tolist') else encoding)]
            self.face_encoding = json.dumps(enc_list)
            self.save(update_fields=['face_encoding'])

    def get_face_encoding(self):
        if self.face_encoding:
            import numpy as np
            arr = np.array(json.loads(self.face_encoding), dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                return arr / norm
            return arr
        return None
    
    def save(self, *args, **kwargs):
        if not self.employee_id:
            last_emp = Employee.objects.order_by('-id').first()
            if last_emp and last_emp.employee_id.startswith('EMP'):
                last_num = int(last_emp.employee_id.replace('EMP', ''))
                new_id = f"EMP{last_num + 1:03d}"
            else:
                new_id = "EMP001"
            self.employee_id = new_id
        super().save(*args, **kwargs)


# ============================================================
# 🏢 ORGANISATION STRUCTURE
# ============================================================

class Department(models.Model):
    name = models.CharField(max_length=100, unique=True)
    head = models.ForeignKey(Employee, on_delete=models.SET_NULL, null=True, blank=True, related_name='headed_departments')
    parent_department = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='sub_departments')

    def __str__(self):
        return self.name


# ============================================================
# 💰 SALARY & PAYROLL
# ============================================================

class SalaryStructure(models.Model):
    employee = models.OneToOneField(Employee, on_delete=models.CASCADE)
    base_salary = models.DecimalField(max_digits=10, decimal_places=2)
    hra = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    allowances = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    deductions = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    effective_from = models.DateField(default=timezone.now)

    def total_salary(self):
        return self.base_salary + self.hra + self.allowances - self.deductions

    def __str__(self):
        return f"Salary Structure for {self.employee}"


class PayrollManager(models.Manager):
    def generate_monthly_salary(self, month: int, year: int):
        employees = Employee.objects.filter(employment_status='active')
        total_days_in_month = calendar.monthrange(year, month)[1]
        first_day = date(year, month, 1)
        last_day = date(year, month, total_days_in_month)

        for emp in employees:
            # Calculate attendance days
            attendance_days = Attendance.objects.filter(
                employee=emp,
                timestamp__date__range=[first_day, last_day],
                attendance_type='check_in'
            ).values('timestamp__date').distinct().count()

            # Calculate approved leaves
            approved_leaves = LeaveApplication.objects.filter(
                employee=emp,
                status='approved',
                start_date__lte=last_day,
                end_date__gte=first_day
            ).count()

            # Calculate working days (Monday-Friday)
            working_days = sum(1 for i in range(total_days_in_month)
                               if (first_day + timedelta(days=i)).weekday() < 5)

            total_present = attendance_days + approved_leaves
            absent_days = max(0, working_days - total_present)
            attendance_ratio = Decimal(total_present) / Decimal(working_days or 1)

            try:
                structure = SalaryStructure.objects.get(employee=emp)
            except SalaryStructure.DoesNotExist:
                continue

            gross_salary = structure.total_salary()
            earned_salary = gross_salary * attendance_ratio

            Payroll.objects.update_or_create(
                employee=emp,
                month=date(year, month, 1),
                defaults={
                    'basic_pay': structure.base_salary,
                    'allowances': structure.allowances,
                    'deductions': structure.deductions + Decimal(absent_days) * (structure.base_salary / Decimal(working_days or 1)),
                    'net_salary': earned_salary,
                    'status': 'processed',
                    'processed_at': timezone.now(),
                }
            )
        return True


class Payroll(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    month = models.DateField(help_text="First day of the salary month")
    basic_pay = models.DecimalField(max_digits=10, decimal_places=2)
    allowances = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    deductions = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    net_salary = models.DecimalField(max_digits=10, decimal_places=2)
    status = models.CharField(
        max_length=20,
        choices=[('pending', 'Pending'), ('processed', 'Processed'), ('paid', 'Paid')],
        default='pending'
    )
    processed_at = models.DateTimeField(blank=True, null=True)

    objects = PayrollManager()

    def __str__(self):
        return f"{self.employee} - {self.month.strftime('%B %Y')}"


# ============================================================
# 📝 LEAVE MANAGEMENT
# ============================================================

class LeaveType(models.Model):
    name = models.CharField(max_length=100)
    max_days_per_year = models.PositiveIntegerField(default=12)

    def __str__(self):
        return self.name


class LeaveApplication(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
        ('cancelled', 'Cancelled'),
    ]

    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    leave_type = models.ForeignKey(LeaveType, on_delete=models.SET_NULL, null=True)
    start_date = models.DateField()
    end_date = models.DateField()
    reason = models.TextField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    applied_on = models.DateTimeField(default=timezone.now)
    approved_by = models.ForeignKey(Employee, on_delete=models.SET_NULL, null=True, blank=True, related_name='approved_leaves')

    def total_days(self):
        return (self.end_date - self.start_date).days + 1

    def __str__(self):
        return f"{self.employee} - {self.leave_type} ({self.status})"


class LeaveWorkflowStage(models.Model):
    level = models.PositiveIntegerField(unique=True)
    role_name = models.CharField(max_length=50)
    next_level = models.PositiveIntegerField(blank=True, null=True)

    def __str__(self):
        return f"Level {self.level} - {self.role_name}"


class LeaveApproval(models.Model):
    leave = models.ForeignKey(LeaveApplication, on_delete=models.CASCADE, related_name='approvals')
    approver = models.ForeignKey(Employee, on_delete=models.SET_NULL, null=True, blank=True)
    level = models.PositiveIntegerField()
    status = models.CharField(
        max_length=20,
        choices=[('pending', 'Pending'), ('approved', 'Approved'), ('rejected', 'Rejected')],
        default='pending'
    )
    remarks = models.TextField(blank=True, null=True)
    acted_at = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return f"Approval L{self.level} - {self.leave.employee}"


# ============================================================
# 📄 JOINING & RESIGNATION
# ============================================================

class JoiningDetail(models.Model):
    employee = models.OneToOneField(Employee, on_delete=models.CASCADE)
    date_of_joining = models.DateField(default=timezone.now)
    documents = models.FileField(upload_to="joining_documents/", blank=True, null=True)
    probation_period_months = models.PositiveIntegerField(default=3)
    confirmation_date = models.DateField(blank=True, null=True)

    def __str__(self):
        return f"Joining - {self.employee}"


class JoiningDocument(models.Model):
    joining = models.ForeignKey(
        JoiningDetail,
        on_delete=models.CASCADE,
        related_name='joining_documents'   # ✅ avoid conflict
    )
    file = models.FileField(upload_to="joining_documents/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Document for {self.joining.employee.employee_id}"
    

class Resignation(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    resignation_date = models.DateField(default=timezone.now)
    last_working_day = models.DateField()
    reason = models.TextField()
    approved_by = models.ForeignKey(Employee, on_delete=models.SET_NULL, null=True, blank=True, related_name='approved_resignations')
    approval_status = models.CharField(
        max_length=20,
        choices=[('pending', 'Pending'), ('approved', 'Approved'), ('rejected', 'Rejected')],
        default='pending'
    )

    def __str__(self):
        return f"Resignation - {self.employee}"


# ============================================================
# 🔔 NOTIFICATIONS
# ============================================================

class Notification(models.Model):
    recipient = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.CharField(max_length=255)
    link = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    def __str__(self):
        return f"Notification for {self.recipient.username}"


# ============================================================
# 🕒 ATTENDANCE SYSTEM (If not already exists)
# ============================================================

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
    min_hours_before_checkout = models.FloatField(default=3.0)

    def __str__(self):
        return "Attendance Settings"

    def save(self, *args, **kwargs):
        if not self.pk and AttendanceSettings.objects.exists():
            raise ValidationError("Only one AttendanceSettings instance allowed.")
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
    


