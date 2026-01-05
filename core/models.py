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

class SalaryStructure(models.Model):
    employee = models.OneToOneField(Employee, on_delete=models.CASCADE)
    base_salary = models.DecimalField(max_digits=10, decimal_places=2)
    hra = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    allowances = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    deductions = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    effective_from = models.DateField(default=timezone.now)
    hourly_override = models.DecimalField(
        max_digits=10, decimal_places=2,
        null=True, blank=True,
        help_text="Optional custom hourly rate"
    )
    def total_salary(self):
        return self.base_salary + self.hra + self.allowances - self.deductions

    def __str__(self):
        return f"Salary Structure for {self.employee}"

def daterange(start_date: date, end_date: date):
    d = start_date
    while d <= end_date:
        yield d
        d += timedelta(days=1)

def get_employee_rule(emp):
    rule = emp.work_rule
    if not rule:
        raise ValueError("Employee has no WorkRule assigned")

    return {
        "working_days": rule.working_days,
        "saturday_is_working": rule.saturday_is_working,
        "sunday_is_working": rule.sunday_is_working,

        "shift_start": rule.shift_start_time,
        "shift_end": rule.shift_end_time,

        "full_day_hours": Decimal(str(rule.full_day_hours)),
        "half_day_hours": Decimal(str(rule.half_day_hours)),

        "night_start": rule.night_shift_start,
        "night_end": rule.night_shift_end,
        "night_multiplier": Decimal(rule.night_bonus_multiplier),

        "overtime_rate": Decimal(rule.overtime_rate),
        "count_weekend_overtime": rule.count_weekend_overtime,
        "count_holiday_overtime": rule.count_holiday_overtime,
    }


def is_working_day(day, rule):
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


def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)



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


from decimal import Decimal, ROUND_HALF_UP
from datetime import date, datetime, time, timedelta
import calendar
from django.utils import timezone
from django.core.files.base import ContentFile

from core.utils.payslip_pdf import generate_payslip_pdf
from core.utils.payslip_email import email_payslip


NIGHT_START = time(22, 0)
NIGHT_END = time(6, 0)
NIGHT_BONUS_MULTIPLIER = Decimal("0.20")  # 20%


class PayrollManager(models.Manager):

    def generate_monthly_salary(self, year: int, month: int):
        first_day = date(year, month, 1)
        last_day = date(year, month, calendar.monthrange(year, month)[1])

        holidays = set(
            Holiday.objects.filter(date__range=(first_day, last_day))
            .values_list("date", flat=True)
        )

        employees = Employee.objects.filter(
            is_active=True,
            employment_status="active"
        ).select_related("work_rule")

        for emp in employees:

            if Payroll.objects.filter(employee=emp, month=first_day).exists():
                continue  # 🔒 prevent duplicate payroll

            try:
                structure = SalaryStructure.objects.get(employee=emp)
            except SalaryStructure.DoesNotExist:
                continue

            rule = get_employee_rule(emp)

            working_days = sum(
                1 for d in daterange(first_day, last_day)
                if is_working_day(d, rule) and d not in holidays
            )

            expected_hours = max(
                Decimal(working_days) * rule["full_day_hours"],
                Decimal("1.0")
            )

            gross_salary = structure.total_salary()

            hourly_rate = (
                structure.hourly_override
                if structure.hourly_override
                else (gross_salary / expected_hours).quantize(Decimal("0.01"))
            )

            total_hours = Decimal("0.00")
            overtime_hours = Decimal("0.00")
            night_hours = Decimal("0.00")
            present_days = Decimal("0.00")
            absent_days = Decimal("0.00")

            for day in daterange(first_day, last_day):
                is_holiday = day in holidays or not is_working_day(day, rule)

                logs = Attendance.objects.filter(
                    employee=emp,
                    timestamp__date=day
                ).order_by("timestamp")

                if not logs.exists():
                    if not is_holiday:
                        absent_days += 1
                    continue

                cin = logs.filter(attendance_type="check_in").first()
                cout = logs.filter(attendance_type="check_out").last()

                if not cin or not cout:
                    absent_days += 1
                    continue

                start = timezone.localtime(cin.timestamp)
                end = timezone.localtime(cout.timestamp)

                hours = Decimal(
                    (end - start).total_seconds() / 3600
                ).quantize(Decimal("0.01"))

                if hours <= 0:
                    absent_days += 1
                    continue

                present_days += 1

                if is_holiday:
                    overtime_hours += hours
                    continue

                total_hours += hours

                if hours > rule["full_day_hours"]:
                    overtime_hours += (hours - rule["full_day_hours"])

                # Night hours
                cur = start
                while cur < end:
                    t = cur.time()
                    ns, ne = rule["night_start"], rule["night_end"]
                    if (ns <= ne and ns <= t <= ne) or (ns > ne and (t >= ns or t <= ne)):
                        night_hours += Decimal("0.5")
                    cur += timedelta(minutes=30)

            normal_pay = (total_hours * hourly_rate).quantize(Decimal("0.01"))
            overtime_pay = (overtime_hours * rule["overtime_rate"]).quantize(Decimal("0.01"))
            night_bonus = (
                night_hours * hourly_rate * rule["night_multiplier"]
            ).quantize(Decimal("0.01"))

            net_salary = (
                normal_pay
                + overtime_pay
                + night_bonus
                + structure.allowances
                + structure.hra
                - structure.deductions
            )

            net_salary = max(net_salary, Decimal("0.00"))

            payroll = Payroll.objects.create(
                employee=emp,
                month=first_day,
                basic_pay=structure.base_salary,
                allowances=structure.allowances + structure.hra,
                deductions=structure.deductions,
                calculated_salary=net_salary,
                net_salary=net_salary,
                present_days=present_days,
                absent_days=absent_days,
                overtime_hours=overtime_hours,
                status="processed",
                processed_at=timezone.now(),
            )

            pdf, name = generate_payslip_pdf(payroll)
            payroll.payslip_pdf.save(name, ContentFile(pdf.read()))
            email_payslip(payroll)

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
    calculated_salary = models.DecimalField(
        max_digits=10, decimal_places=2,
        help_text="System calculated salary before override",
        null=True, blank=True
    )
    actual_paid_salary = models.DecimalField(
        max_digits=10, decimal_places=2,
        null=True, blank=True,
        help_text="Actual salary paid after manual override (if any)",
    )
    # ✅ NEW FIELDS:
    paid_date = models.DateField(null=True, blank=True)
    next_pay_date = models.DateField(null=True, blank=True)

    # ✅ Breakdown fields:
    present_days = models.DecimalField(max_digits=6, decimal_places=2, default=Decimal('0.00'))  # includes half-days as 0.5 if you prefer
    half_days = models.DecimalField(max_digits=6, decimal_places=2, default=Decimal('0.00'))
    paid_leave_days = models.DecimalField(max_digits=6, decimal_places=2, default=Decimal('0.00'))
    unpaid_leave_days = models.DecimalField(max_digits=6, decimal_places=2, default=Decimal('0.00'))
    absent_days = models.DecimalField(max_digits=6, decimal_places=2, default=Decimal('0.00'))
    overtime_hours = models.DecimalField(max_digits=7, decimal_places=2, default=Decimal('0.00'))
    remarks = models.TextField(blank=True, null=True)
    payslip_pdf = models.FileField(
        upload_to="payslips/",
        null=True,
        blank=True
    )
    objects = PayrollManager()

    def save(self, *args, **kwargs):
        if isinstance(self.paid_date, str):
            self.paid_date = datetime.strptime(self.paid_date, "%Y-%m-%d").date()

        if self.status == 'paid' and self.paid_date and not self.next_pay_date:
            next_month = self.paid_date + relativedelta(months=1)
            self.next_pay_date = next_month.replace(day=1)

        super().save(*args, **kwargs)



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
    """
    Office / Shift based working rules
    """
    name = models.CharField(max_length=100, unique=True)

    WORKING_DAYS_CHOICES = [
        ('5_day', 'Mon–Fri'),
        ('6_day', 'Mon–Sat'),
        ('custom', 'Custom'),
    ]
    working_days = models.CharField(
        max_length=10,
        choices=WORKING_DAYS_CHOICES,
        default='5_day'
    )

    saturday_is_working = models.BooleanField(default=False)
    sunday_is_working = models.BooleanField(default=False)

    # Shift timing
    shift_start_time = models.TimeField(default=time(9, 0))
    shift_end_time = models.TimeField(default=time(18, 0))

    # Hours
    full_day_hours = models.FloatField(default=8.0)
    half_day_hours = models.FloatField(default=4.0)

    # Night shift
    night_shift_start = models.TimeField(default=time(22, 0))
    night_shift_end = models.TimeField(default=time(6, 0))
    night_bonus_multiplier = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=Decimal("0.20")
    )

    # Overtime
    overtime_rate = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=Decimal('100.00')
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
    


