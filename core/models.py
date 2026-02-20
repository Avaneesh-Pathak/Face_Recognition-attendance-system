# models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from dateutil.relativedelta import relativedelta
from django.core.validators import MinValueValidator
from django.core.exceptions import ValidationError
from decimal import Decimal, ROUND_HALF_UP
from decimal import Decimal
from datetime import date, datetime, timedelta
import calendar
import json

from decimal import Decimal, ROUND_HALF_UP
from datetime import date, datetime, time, timedelta
import calendar
from django.utils import timezone
from django.core.files.base import ContentFile

from core.utils.payslip_pdf import generate_payslip_pdf
from core.utils.payslip_email import email_payslip


# ============================================================
# 👤 EMPLOYEE MASTER
# ============================================================

class Employee(models.Model):
    ROLE_CHOICES = [
        ('Admin', 'Admin'),
        ('HR', 'HR Manager'),
        ('Finance', 'Accounts / Finance'),
        ('Manager', 'Manager'),
        ('Employee', 'Employee'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    employee_id = models.CharField(max_length=20, unique=True)
    
    department = models.ForeignKey('Department', on_delete=models.SET_NULL, null=True, blank=True, related_name='employees')
    position = models.CharField(max_length=100)
    manager = models.ForeignKey("self", on_delete=models.SET_NULL, null=True, blank=True, related_name="subordinates")
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='Employee')  # ✅ NEW
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
    work_rule = models.ForeignKey('WorkRule', on_delete=models.SET_NULL, null=True, blank=True, related_name='employees')
    
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
    
    def get_reporting_chain(self):
        """Return list of employees from CEO to this employee."""
        chain = []
        employee = self
        while employee:
            chain.append(employee)
            employee = employee.manager  # move to senior
        return chain[::-1]  # reverse to show from CEO to employee
        
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
    DEPT_CHOICES = [
        ('Management', 'Management / CEO Office'),
        ('HR', 'Human Resources'),
        ('Finance', 'Finance / Accounts'),
        ('IT', 'IT / Technical'),
        ('Operations', 'Operations'),
        ('Pharmacy', 'Pharmacy'),
        ('House Keeping', 'House Keeping'),
        ('Nursing', 'Nursing'),
        ('Reception', 'Reception'),
        ('Sales', 'Sales & Marketing'),
        ('Other', 'Other'),
    ]
    name = models.CharField(max_length=100, choices=DEPT_CHOICES, unique=True)
    head = models.ForeignKey('Employee', on_delete=models.SET_NULL, null=True, blank=True, related_name='headed_departments')
    parent_department = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='sub_departments')

    def __str__(self):
        return self.get_name_display()



# ============================================================
# 💰 SALARY & PAYROLL
# ============================================================
# ============================================================
# 🔁 SHIFT WINDOW (DAY + NIGHT SAFE)
# ============================================================
def get_shift_window(day: date, work_rule):
    start_time = work_rule.shift_start_time
    end_time = work_rule.shift_end_time

    start_dt = datetime.combine(day, start_time)

    # Night shift (cross-midnight)
    if end_time <= start_time:
        end_dt = datetime.combine(day + timedelta(days=1), end_time)
    else:
        end_dt = datetime.combine(day, end_time)

    return (
        timezone.make_aware(start_dt),
        timezone.make_aware(end_dt),
    )


class SalaryStructure(models.Model):
    employee = models.OneToOneField(
        "Employee",
        on_delete=models.CASCADE,
        related_name="salary_structure"
    )

    base_salary = models.DecimalField(max_digits=10, decimal_places=2)
    hra = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    allowances = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    deductions = models.DecimalField(max_digits=10, decimal_places=2, default=0)

    effective_from = models.DateField(default=timezone.now)

    def gross_monthly(self):
        return self.base_salary + self.hra + self.allowances

    def total_salary(self):
        return (
            self.base_salary +
            self.hra +
            self.allowances -
            self.deductions
        )

    total_salary.short_description = "Total Salary"
    def __str__(self):
        return f"SalaryStructure({self.employee})"

def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

# ============================================================
# 📋 EMPLOYEE RULE (SAFE DICT)
# ============================================================
def get_employee_rule(emp: Employee):
    rule = emp.work_rule
    if not rule:
        raise ValueError("Employee has no WorkRule assigned")

    return {
        "working_days": rule.working_days,
        "saturday_is_working": rule.saturday_is_working,
        "sunday_is_working": rule.sunday_is_working,
        "full_day_hours": Decimal(str(rule.full_day_hours)),
        "half_day_hours": Decimal(str(rule.half_day_hours)),
        "overtime_rate": Decimal(str(rule.overtime_rate)),
        "count_weekend_overtime": rule.count_weekend_overtime,
        "count_holiday_overtime": rule.count_holiday_overtime,
    }

# ============================================================
# 📅 WORKING DAY CHECK
# ============================================================
def is_working_day(day: date, rule: dict) -> bool:
    wd = day.weekday()  # 0=Mon

    if rule["working_days"] == "5_day":
        return wd < 5

    if rule["working_days"] == "6_day":
        return wd < 6

    if wd == 5:
        return rule["saturday_is_working"]

    if wd == 6:
        return rule["sunday_is_working"]

    return True


def get_hours_worked(employee: 'Employee', day: date) -> float:
    """Compute hours from earliest check-in to latest check-out on a given date.
       If no checkout, treat as half-day if >= half_day_hours else absent."""
    qs = Attendance.objects.filter(employee=employee, timestamp__date=day).order_by('timestamp')
    if not qs.exists():
        return 0.0
    first_in = qs.filter(attendance_type='check_in').first()
    last_out = qs.filter(attendance_type='check_out').last()
    if first_in and last_out and last_out.timestamp > first_in.timestamp:
        return (last_out.timestamp - first_in.timestamp).total_seconds() / 3600.0
    # fallback: if only check_in exists, estimate as 0 (you can tweak to min_hours)
    return 0.0

def has_approved_leave(employee: 'Employee', day: date):
    """Return (is_on_leave, is_paid) for the day."""
    leave = LeaveApplication.objects.filter(
        employee=employee,
        status='approved',
        start_date__lte=day,
        end_date__gte=day
    ).select_related('leave_type').first()
    if not leave:
        return (False, False)
    return (True, bool(getattr(leave.leave_type, 'is_paid', True)))

# ============================================================
# 💰 PAYROLL MANAGER (FINAL & CORRECT)
class PayrollManager(models.Manager):

    def generate_monthly_salary(self, year: int, month: int):

        first_day = date(year, month, 1)
        last_day = date(year, month, calendar.monthrange(year, month)[1])
        days_in_month = Decimal(calendar.monthrange(year, month)[1])

        employees = Employee.objects.filter(
            is_active=True,
            employment_status="active"
        ).select_related("work_rule", "user")

        for emp in employees:

            # Prevent duplicate payroll
            if Payroll.objects.filter(employee=emp, month=first_day).exists():
                continue

            if not emp.work_rule:
                continue

            try:
                structure = SalaryStructure.objects.get(employee=emp)
            except SalaryStructure.DoesNotExist:
                continue

            rule = get_employee_rule(emp)

            salary_per_day = (
                structure.base_salary / days_in_month
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            full_days = Decimal("0.0")
            half_days = Decimal("0.0")
            lop_days = Decimal("0.0")
            overtime_hours = Decimal("0.0")

            for day in self._daterange(first_day, last_day):

                # ---------- WORKING DAY ----------
                if not is_working_day(day, rule):
                    continue

                # ---------- HOLIDAY ----------
                if Holiday.objects.filter(date=day).exists():
                    full_days += 1
                    continue

                # ---------- APPROVED LEAVE ----------
                if LeaveApplication.objects.filter(
                    employee=emp,
                    status="approved",
                    start_date__lte=day,
                    end_date__gte=day
                ).exists():
                    full_days += 1
                    continue

                # ---------- SHIFT WINDOW ----------
                shift_start, shift_end = get_shift_window(day, emp.work_rule)

                logs = Attendance.objects.filter(
                    employee=emp,
                    timestamp__range=(shift_start, shift_end)
                ).order_by("timestamp")

                if not logs.exists():
                    lop_days += 1
                    continue

                cin = logs.filter(attendance_type="check_in").first()
                cout = logs.filter(attendance_type="check_out").last()

                if not cin or not cout or cout.timestamp <= cin.timestamp:
                    lop_days += 1
                    continue

                work_hours = Decimal(
                    (cout.timestamp - cin.timestamp).total_seconds() / 3600
                ).quantize(Decimal("0.01"))

                # ---------- DAY CLASSIFICATION ----------
                if work_hours >= rule["full_day_hours"]:
                    full_days += 1
                elif work_hours >= rule["half_day_hours"]:
                    half_days += Decimal("0.5")
                else:
                    lop_days += 1
                    continue

                # ---------- OVERTIME ----------
                if work_hours > rule["full_day_hours"]:
                    overtime_hours += (
                        work_hours - rule["full_day_hours"]
                    )

            paid_days = full_days + half_days

            basic_pay = (
                salary_per_day * paid_days
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            ot_pay = (
                overtime_hours * rule["overtime_rate"]
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            net_salary = (
                basic_pay
                + structure.hra
                + structure.allowances
                + ot_pay
                - structure.deductions
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            payroll = Payroll.objects.create(
                employee=emp,
                month=first_day,
                basic_pay=basic_pay,
                allowances=structure.hra + structure.allowances,
                deductions=structure.deductions,
                calculated_salary=net_salary,
                net_salary=net_salary,
                present_days=paid_days,
                half_days=half_days,
                absent_days=lop_days,
                overtime_hours=overtime_hours,
                status="processed",
                processed_at=timezone.now(),
            )

            pdf, filename = generate_payslip_pdf(payroll)
            payroll.payslip_pdf.save(filename, ContentFile(pdf.read()))
            email_payslip(payroll)

    def _daterange(self, start: date, end: date):
        d = start
        while d <= end:
            yield d
            d += timedelta(days=1)

class Payroll(models.Model):
    employee = models.ForeignKey("Employee", on_delete=models.CASCADE)
    month = models.DateField(help_text="First day of month")

    basic_pay = models.DecimalField(max_digits=10, decimal_places=2)
    allowances = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    deductions = models.DecimalField(max_digits=10, decimal_places=2, default=0)

    present_days = models.DecimalField(max_digits=6, decimal_places=2, default=0)
    half_days = models.DecimalField(max_digits=6, decimal_places=2, default=0)
    absent_days = models.DecimalField(max_digits=6, decimal_places=2, default=0)
    overtime_hours = models.DecimalField(max_digits=7, decimal_places=2, default=0)
    payslip_pdf = models.FileField(
        upload_to="payslips/",
        blank=True,
        null=True
    )
    # 🔴 CRITICAL FIX: defaults added
    calculated_salary = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=0
    )
    net_salary = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=0
    )

    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("processed", "Processed"),
            ("paid", "Paid"),
        ],
        default="pending",
    )

    processed_at = models.DateTimeField(null=True, blank=True)
    paid_date = models.DateField(null=True, blank=True)

    objects = PayrollManager()

    @property
    def paid_leave_days(self):
        return Decimal("0.00")

    @property
    def unpaid_leave_days(self):
        return self.absent_days

    def save(self, *args, **kwargs):
        # Do NOT recompute salary here.
        # PayrollManager is the single source of truth.
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.employee} | {self.month.strftime('%b %Y')}"


class PayrollSettings(models.Model):
    professional_tax = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('200.00'))
    esi_limit = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('21000'))

    # ✅ NEW defaults if Employee has no WorkRule
    default_working_days = models.CharField(
        max_length=10,
        choices=[('5_day', 'Mon–Fri'), ('6_day', 'Mon–Sat')],
        default='5_day'
    )
    default_full_day_hours = models.FloatField(default=8.0)
    default_half_day_hours = models.FloatField(default=4.0)
    default_overtime_rate = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('100.00'))
    default_saturday_is_working = models.BooleanField(default=False)
    default_sunday_is_working = models.BooleanField(default=False)
    default_count_weekend_overtime = models.BooleanField(default=True)
    default_count_holiday_overtime = models.BooleanField(default=True)

    def __str__(self):
        return "Payroll Settings"

    class Meta:
        verbose_name = "Payroll Setting"
        verbose_name_plural = "Payroll Settings"

class WorkRule(models.Model):
    WORKING_DAYS_CHOICES = [
        ('5_day', 'Mon–Fri'),
        ('6_day', 'Mon–Sat'),
        ('custom', 'Custom'),
    ]

    name = models.CharField(max_length=100, unique=True)

    working_days = models.CharField(
        max_length=10,
        choices=WORKING_DAYS_CHOICES,
        default='5_day'
    )

    saturday_is_working = models.BooleanField(default=False)
    sunday_is_working = models.BooleanField(default=False)

    shift_start_time = models.TimeField(default="09:00")
    shift_end_time = models.TimeField(default="18:00")

    full_day_hours = models.FloatField(default=8.0)
    half_day_hours = models.FloatField(default=4.0)

    overtime_rate = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=Decimal("100.00")
    )

    count_weekend_overtime = models.BooleanField(default=True)
    count_holiday_overtime = models.BooleanField(default=True)

    def __str__(self):
        return self.name


# ============================================================
# 📝 LEAVE MANAGEMENT
# ============================================================

class LeaveType(models.Model):
    name = models.CharField(max_length=100)
    max_days_per_year = models.PositiveIntegerField(default=12)
    is_paid = models.BooleanField(default=True)  # ✅ new

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

class Holiday(models.Model):
    name = models.CharField(max_length=100)
    date = models.DateField(unique=True)

    def __str__(self):
        return f"{self.name} ({self.date})"

# ============================================================
# 📄 JOINING & RESIGNATION
# ============================================================

class JoiningDetail(models.Model):
    employee = models.OneToOneField(Employee, on_delete=models.CASCADE)
    date_of_joining = models.DateField(default=timezone.now)
    
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
    timestamp = models.DateTimeField(default=timezone.now,editable=True)
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
    

class FaceMatchStats(models.Model):
    employee_id = models.CharField(max_length=20, unique=True)
    false_rejects = models.IntegerField(default=0)
    false_accepts = models.IntegerField(default=0)
    total_attempts = models.IntegerField(default=0)

class AttendanceMatchLog(models.Model):
    employee_id = models.CharField(max_length=20)
    cosine = models.FloatField()
    euclidean = models.FloatField()
    confidence = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)


class EmployeeThreshold(models.Model):
    employee_id = models.CharField(max_length=20, unique=True)
    cos_max = models.FloatField(default=0.85)
    euc_max = models.FloatField(default=1.40)


