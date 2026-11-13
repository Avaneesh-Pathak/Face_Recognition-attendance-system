# models.py
import json
import calendar
from django.db import models
from django.db.models import Q
from django.utils import timezone
from decimal import Decimal, ROUND_HALF_UP
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from dateutil.relativedelta import relativedelta
from core.utils.payslip_email import email_payslip
from django.core.exceptions import ValidationError
from django.core.validators import MinValueValidator
from datetime import date, datetime, time, timedelta
from core.utils.payslip_pdf import generate_payslip_pdf
from collections import defaultdict
from django.db import models, transaction
from .rotational_shift_fix import get_work_date_for_checkin
from core.tenant_utils import TenantManager, get_current_organisation
from core.security.biometrics import EncryptedFaceEncodingField

# ============================================================
# 🏢 ORGANISATION (TENANT)
# ============================================================
class Organisation(models.Model):
    name = models.CharField(max_length=150, unique=True)

    logo = models.ImageField(
        upload_to="organisation_logos/",
        blank=True,
        null=True
    )

    address = models.TextField(blank=True)

    city = models.CharField(
        max_length=100,
        blank=True
    )

    state = models.CharField(
        max_length=100,
        blank=True
    )

    pincode = models.CharField(
        max_length=10,
        blank=True
    )

    phone = models.CharField(
        max_length=20,
        blank=True
    )

    alternate_phone = models.CharField(
        max_length=20,
        blank=True
    )

    email = models.EmailField(
        blank=True
    )

    website = models.URLField(
        blank=True
    )

    created_at = models.DateTimeField(auto_now_add=True)

    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name


class AuditQuerySet(models.QuerySet):
    def update(self, **kwargs):
        raise ValidationError('Audit events cannot be edited')

    def delete(self):
        raise ValidationError('Audit events cannot be deleted')


class AuditManager(TenantManager):
    def get_queryset(self):
        from core.tenant_utils import get_current_organisation, is_global_access
        qs = AuditQuerySet(self.model, using=self._db)
        if is_global_access():
            return qs
        org = get_current_organisation()
        return qs.filter(organisation_id=getattr(org, 'pk', org)) if org else qs.none()


class AuditEvent(models.Model):
    """Append-only business history. Actor label survives account deletion."""
    organisation = models.ForeignKey(Organisation, on_delete=models.PROTECT)
    actor_id = models.BigIntegerField(null=True, blank=True)
    actor_name = models.CharField(max_length=150, default='System')
    action = models.CharField(max_length=16)
    entity = models.CharField(max_length=64)
    object_id = models.CharField(max_length=64)
    before = models.JSONField(default=dict)
    after = models.JSONField(default=dict)
    request_id = models.CharField(max_length=32, blank=True)
    reason = models.CharField(max_length=1000, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    objects = AuditManager()
    global_objects = AuditQuerySet.as_manager()

    class Meta:
        ordering = ['-created_at', '-pk']
        indexes = [models.Index(fields=['organisation', '-created_at'], name='audit_org_created_idx')]

    def save(self, *args, **kwargs):
        if not self._state.adding:
            raise ValidationError('Audit events cannot be edited')
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        raise ValidationError('Audit events cannot be deleted')

# ============================================================
# 🛡 ABSTRACT TENANT BASE MODEL
# ============================================================
class TenantModel(models.Model):
    organisation = models.ForeignKey(
        Organisation,
        on_delete=models.CASCADE,
        related_name="%(class)s_records",
        null=True,
        blank=True
    )

    # Overwrite the default manager with our TenantManager
    objects = TenantManager()
    # Provide an unrestricted manager for global background tasks/superadmins if needed
    global_objects = models.Manager()

    class Meta:
        abstract = True

    @property
    def secure_id(self):
        from core.security.crypto import encrypt_id

        return encrypt_id(
            self.__class__.__name__,
            self.pk,
            self.organisation_id
        )
    def save(self, *args, **kwargs):
        if getattr(self, 'organisation_id', None) is None:
            current_org = get_current_organisation()
            if current_org is not None:
                self.organisation = current_org
        # For new/updated records, reject mixed-tenant references before writing.
        if self.organisation_id is not None:
            for field_name in ('employee', 'department', 'manager', 'work_rule',
                               'assigned_location', 'leave_type', 'approver',
                               'leave', 'joining', 'head', 'parent_department'):
                try:
                    field = self._meta.get_field(field_name)
                except Exception:
                    continue
                # A relation name can resolve to a reverse OneToMany/OneToOne
                # descriptor. Only forward concrete foreign keys have an attname
                # that belongs to this model and can be checked for tenant scope.
                if not field.concrete or not field.is_relation:
                    continue
                related_id = getattr(self, field.attname, None)
                if related_id is None or not hasattr(field.related_model, 'global_objects'):
                    continue
                if not field.related_model.global_objects.filter(
                        pk=related_id, organisation_id=self.organisation_id).exists():
                    raise ValidationError(f'{field_name}: cross-organisation reference is forbidden')
        # Audit signals run in the same transaction as the business write.
        with transaction.atomic():
            # Payroll generation, scans and employee-source writes use the same
            # lock order. This also protects employee/leave/salary edits.
            employee_id = self.pk if type(self).__name__ == 'Employee' else getattr(self, 'employee_id', None)
            if employee_id:
                Employee.global_objects.select_for_update().filter(pk=employee_id).exists()
            elif type(self).__name__ in ('WorkRule', 'Holiday'):
                list(Employee.global_objects.filter(organisation_id=self.organisation_id)
                     .order_by('pk').select_for_update().values_list('pk', flat=True))
            super().save(*args, **kwargs)


class OfficeLocation(TenantModel):
    name = models.CharField(max_length=100)
    latitude = models.DecimalField(max_digits=9, decimal_places=6)
    longitude = models.DecimalField(max_digits=9, decimal_places=6)
    radius_meters = models.PositiveIntegerField(default=100)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name

# ============================================================
# 👤 EMPLOYEE MASTER
# ============================================================

class Employee(TenantModel):
    ROLE_CHOICES = [
        ('Admin', 'Admin'),
        ('HR', 'HR Manager'),
        ('Finance', 'Accounts / Finance'),
        ('Manager', 'Manager'),
        ('Employee', 'Employee'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    employee_id = models.CharField(max_length=20)
    
    department = models.ForeignKey('Department', on_delete=models.SET_NULL, null=True, blank=True, related_name='employees')
    position = models.CharField(max_length=100)
    manager = models.ForeignKey("self", on_delete=models.SET_NULL, null=True, blank=True, related_name="subordinates")
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='Employee')  #NEW
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
    face_encoding = EncryptedFaceEncodingField(blank=True, null=True)
    face_image = models.ImageField(upload_to='faces/', blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    work_rule = models.ForeignKey('WorkRule', on_delete=models.SET_NULL, null=True, blank=True, related_name='employees')
    LOCATION_TYPE_CHOICES = [
        ("INDOOR", "Indoor Only"),
        ("OUTDOOR", "Outdoor / Multi Office"),
    ]

    location_type = models.CharField(
        max_length=10,
        choices=LOCATION_TYPE_CHOICES,
        default="INDOOR"
    )
 
    assigned_location = models.ForeignKey(
        OfficeLocation,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    class Meta:
        indexes = [models.Index(fields=['organisation', 'is_active', 'employment_status'], name='emp_org_active_idx')]
        # FIX: Ensure employee_id is unique strictly within each organisation
        unique_together = ('organisation', 'employee_id')

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

class Department(TenantModel):
    DEPT_CHOICES = [
        # Management & Administration
        ('Management', 'Management / CEO Office'),
        ('HR', 'Human Resources'),
        ('Finance', 'Finance / Accounts'),
        ('IT', 'IT / Technical'),
        ('Operations', 'Operations'),

        # Clinical Services
        ('Doctor', 'Doctor'),
        ('Nursing', 'Nursing'),
        ('OPD', 'Out Patient Department (OPD)'),
        ('IPD', 'In Patient Department (IPD)'),
        ('Laboratory', 'Laboratory / Pathology'),
        ('Pharmacy', 'Pharmacy'),
        ('ICU', 'ICU / Critical Care'),
        ('OT', 'Operation Theatre'),
        ('Radiology', 'Radiology / Imaging'),
        ('Security', 'Security'),
        ('Maintenance', 'Maintenance & Engineering'),
        ('Ambulance', 'Ambulance Services'),
        ('Emergency', 'Emergency / Casualty'),
        
        # Patient Support Services
        ('Reception', 'Reception / Front Desk'),
        ('Medical Records', 'Medical Records Department'),

        # Facility & Support Services
        ('Store', 'Store & Inventory'),
        ('House Keeping', 'House Keeping'),

        # Business Development
        ('Sales', 'Sales & Marketing'),

        # Miscellaneous
        ('Other', 'Other'),
    ]
    name = models.CharField(max_length=100, choices=DEPT_CHOICES)
    head = models.ForeignKey('Employee', on_delete=models.SET_NULL, null=True, blank=True, related_name='headed_departments')
    parent_department = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='sub_departments')

    def __str__(self):
        return self.get_name_display()
    
    class Meta:
        # FIX: Ensure department names are unique strictly within each organisation
        unique_together = ('organisation', 'name')

# ============================================================
# 💰 SALARY & PAYROLL
# ============================================================
def get_shift_window(day: date, work_rule):
    start_time = work_rule.shift_start_time
    end_time = work_rule.shift_end_time

    start_dt = datetime.combine(day, start_time)

    if end_time <= start_time:
        end_dt = datetime.combine(day + timedelta(days=1), end_time)
    else:
        end_dt = datetime.combine(day, end_time)

    return (
        timezone.make_aware(start_dt),
        timezone.make_aware(end_dt),
    )


def get_attendance_pair(logs):
    checkin = None
    checkout = None

    for log in logs:
        if log.attendance_type == "check_in" and not checkin:
            checkin = log.timestamp

        if log.attendance_type == "check_out":
            checkout = log.timestamp

    return checkin, checkout


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

# ============================================================
# 📋 EMPLOYEE RULE
# ============================================================

def get_employee_rule(emp):

    rule = emp.work_rule

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

def get_shift_for_day(employee, day):
    if hasattr(employee, '_calendar_shifts'):
        return next((s.work_rule for s in employee._calendar_shifts if s.start_date <= day
                     and (s.end_date is None or s.end_date >= day)), employee.work_rule)
    assignment = EmployeeShiftAssignment.global_objects.filter(
        employee=employee, organisation_id=employee.organisation_id,
        start_date__lte=day
    ).filter(
        Q(end_date__gte=day) | Q(end_date__isnull=True)
    ).select_related("work_rule").order_by('-start_date', '-pk').first()

    if assignment:
        return assignment.work_rule

    return employee.work_rule

# ============================================================
# 📅 WORKING DAY CHECK
# ============================================================

def is_working_day(day: date, rule: dict):

    wd = day.weekday()

    if rule["working_days"] == "5_day":
        return wd < 5

    if rule["working_days"] == "6_day":
        return wd < 6

    if wd == 5:
        return rule["saturday_is_working"]

    if wd == 6:
        return rule["sunday_is_working"]

    return True


class SalaryStructure(TenantModel):
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


class EmployeeShiftAssignment(TenantModel):

    employee = models.ForeignKey(
        "Employee",
        on_delete=models.CASCADE,
        related_name="shift_assignments"
    )

    work_rule = models.ForeignKey(
        "WorkRule",
        on_delete=models.CASCADE
    )

    start_date = models.DateField()
    end_date = models.DateField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["employee", "start_date", "end_date"]),
        ]
        ordering = ["-start_date"]

    def __str__(self):
        return f"{self.employee} → {self.work_rule} ({self.start_date} - {self.end_date or 'Present'})"

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

def get_employee_rule_from_rule(rule):
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
# 🔥 ADVANCED ATTENDANCE ENGINE (FINAL)
# ============================================================

def build_sessions(logs):
    sessions = []
    open_in = None

    for log in sorted(logs, key=lambda x: x.timestamp):
        if log.attendance_type == "check_in":
            open_in = log.timestamp

        elif log.attendance_type == "check_out" and open_in:
            if log.timestamp > open_in:
                sessions.append((open_in, log.timestamp))
            open_in = None

    return sessions

def split_session_by_day(start, end):
    from datetime import datetime, time, timedelta

    result = {}
    current = start

    while current.date() <= end.date():
        day_start = datetime.combine(current.date(), time.min, tzinfo=start.tzinfo)
        day_end = datetime.combine(current.date(), time.max, tzinfo=start.tzinfo)

        s = max(start, day_start)
        e = min(end, day_end)

        if s < e:
            hours = (e - s).total_seconds() / 3600
            result[current.date()] = result.get(current.date(), 0) + hours

        current += timedelta(days=1)

    return result


def calculate_shift_blocks(work_hours, full_day_hours, max_ot_hours):
    work_hours = float(work_hours)
    fsh = float(full_day_hours)
    max_ot = float(max_ot_hours)

    normal = 0
    overtime = 0
    extra = 0

    if work_hours <= fsh:
        normal = work_hours

    elif work_hours <= fsh + max_ot:
        normal = fsh
        overtime = work_hours - fsh

    else:
        normal = fsh
        overtime = max_ot
        extra = work_hours - (fsh + max_ot)

    return normal, overtime, extra


def split_extra_shifts(extra_hours, full_day_hours):
    fsh = float(full_day_hours)

    full_shifts = int(extra_hours // fsh)
    remaining = extra_hours % fsh

    return full_shifts, remaining


def get_attendance_status(worked_hours, rule):
    worked_hours = float(worked_hours)

    if worked_hours == 0:
        return "Absent"

    if worked_hours < float(rule.half_day_hours):
        return "Partial"

    if worked_hours <= float(rule.full_day_hours):
        return "Present"

    if worked_hours <= float(rule.full_day_hours + rule.max_overtime_hours):
        return "Present + OT"

    return "Extra Shift"

class EnterprisePayrollManager(TenantManager):
    """Tenant-scoped, idempotent payroll generation.

    Calendar-day proration preserves the application's existing salary policy:
    scheduled rest days and holidays are paid, while unexcused absences and
    unpaid leave reduce monthly fixed pay. Paid records are immutable.
    """
    def generate_salary(self, year, month, employee=None, organisation=None):
        organisation = organisation or getattr(employee, 'organisation', None) or get_current_organisation()
        org_id = getattr(organisation, 'pk', organisation)
        if not org_id:
            raise ValueError('Choose an organisation before generating payroll')
        with transaction.atomic():
            Organisation.objects.select_for_update().get(pk=org_id, is_active=True)
            # Take locks BEFORE reading source attendance. Scans/corrections/payment
            # lock the same employee rows; independent tenants remain concurrent.
            locked = Employee.global_objects.filter(organisation_id=org_id)
            if employee is not None:
                locked = locked.filter(pk=employee.pk)
            list(locked.order_by('pk').select_for_update().values_list('pk', flat=True))
            return self._generate_salary(year, month, employee, organisation)

    def _generate_salary(self, year, month, employee=None, organisation=None):
        from core.services.attendance_engine import workday_sessions
        if not 1 <= month <= 12 or not 2000 <= year <= 2100:
            raise ValueError('Invalid payroll period')
        if organisation is None:
            organisation = getattr(employee, 'organisation', None) or get_current_organisation()
        org_id = getattr(organisation, 'pk', organisation)
        if not org_id:
            raise ValueError('Choose an organisation before generating payroll')
        first_day = date(year, month, 1)
        last_day = date(year, month, calendar.monthrange(year, month)[1])
        days_in_month = Decimal(last_day.day)
        employees = Employee.global_objects.filter(
            organisation_id=org_id,
            date_of_joining__lte=last_day,
        ).filter(Q(date_of_resignation__isnull=True) | Q(date_of_resignation__gte=first_day)).filter(
            Q(is_active=True, employment_status__in=['active', 'on_leave']) |
            Q(date_of_resignation__gte=first_day)
        ).select_related('salary_structure', 'work_rule', 'user')
        if employee is not None:
            if employee.organisation_id != org_id:
                raise ValueError('Employee belongs to a different organisation')
            employees = employees.filter(pk=employee.pk)
        employees = list(employees)
        ids = [e.pk for e in employees]
        if not ids:
            return {'created': 0, 'updated': 0, 'skipped_paid': 0, 'missing_setup': 0}
        # Include adjacent days for shifts spanning midnight at month boundaries.
        from django.utils import timezone as tz
        start_bound = tz.make_aware(datetime.combine(first_day - timedelta(days=2), time.min))
        end_bound = tz.make_aware(datetime.combine(last_day + timedelta(days=2), time.max))
        attendance = defaultdict(list)
        for log in Attendance.global_objects.filter(
            organisation_id=org_id, employee_id__in=ids,
            timestamp__gte=start_bound, timestamp__lte=end_bound
        ).order_by('timestamp').iterator():
            attendance[log.employee_id].append(log)
        leave_days = {}
        for leave in LeaveApplication.global_objects.filter(
            organisation_id=org_id, employee_id__in=ids, status='approved',
            start_date__lte=last_day, end_date__gte=first_day
        ).select_related('leave_type'):
            for day in daterange(max(first_day, leave.start_date), min(last_day, leave.end_date)):
                # Unpaid leave wins if conflicting approved leave records exist.
                key = (leave.employee_id, day)
                leave_days[key] = leave_days.get(key, True) and bool(getattr(leave.leave_type, 'is_paid', True))
        holidays = set(Holiday.global_objects.filter(
            organisation_id=org_id, date__range=(first_day, last_day)
        ).values_list('date', flat=True))
        assignments = defaultdict(dict)
        shift_lists = defaultdict(list)
        for shift in EmployeeShiftAssignment.global_objects.filter(
            organisation_id=org_id, employee_id__in=ids, start_date__lte=last_day
        ).filter(Q(end_date__gte=first_day - timedelta(days=2)) | Q(end_date__isnull=True)).select_related('work_rule').order_by('start_date', 'pk'):
            shift_lists[shift.employee_id].append(shift)
            for day in daterange(max(first_day, shift.start_date), min(last_day, shift.end_date or last_day)):
                assignments[shift.employee_id][day] = shift.work_rule
        counts = {'created': 0, 'updated': 0, 'skipped_paid': 0, 'missing_setup': 0}
        # Transaction + row locks stop simultaneous payroll jobs modifying the same month.
        # Organisation row is an effective tenant-wide lock on PostgreSQL.
        with transaction.atomic():
            Organisation.objects.select_for_update().get(pk=org_id)
            for emp in employees:
                try:
                    structure = emp.salary_structure
                except SalaryStructure.DoesNotExist:
                    counts['missing_setup'] += 1
                    continue
                if not emp.work_rule or structure.effective_from > last_day:
                    counts['missing_setup'] += 1
                    continue
                if structure.effective_from > max(first_day, emp.date_of_joining):
                    raise ValueError(f'Employee {emp.employee_id}: salary starts mid-period; historical salary is required.')
                if any(value < 0 for value in (structure.base_salary, structure.hra,
                                               structure.allowances, structure.deductions)):
                    raise ValueError(f'Employee {emp.employee_id}: salary amounts cannot be negative.')
                existing = Payroll.global_objects.select_for_update().filter(
                    organisation_id=org_id, employee=emp, month=first_day).first()
                if existing and existing.status == 'paid':
                    counts['skipped_paid'] += 1
                    continue
                emp._calendar_shifts = list(reversed(shift_lists[emp.pk]))
                days = workday_sessions(attendance.get(emp.pk, []), emp)
                unresolved = [str(day) for day, entry in days.items()
                              if first_day <= day <= last_day and entry['pending']]
                if unresolved:
                    raise ValueError(f'Employee {emp.employee_id}: reconcile attendance on {", ".join(unresolved)} before payroll.')
                present = half = absent = paid_rest = Decimal('0')
                regular_ot = holiday_ot = Decimal('0')
                regular_ot_pay = holiday_ot_pay = Decimal('0')
                for day in daterange(first_day, last_day):
                    if day < emp.date_of_joining or (emp.date_of_resignation and day > emp.date_of_resignation):
                        continue
                    rule = assignments[emp.pk].get(day) or emp.work_rule
                    if not (0 < rule.half_day_hours <= rule.full_day_hours) or rule.overtime_rate < 0:
                        raise ValueError(f'Invalid work rule for employee {emp.employee_id}')
                    rate = Decimal(str(rule.overtime_rate))
                    scheduled = is_working_day(day, get_employee_rule_from_rule(rule))
                    holiday = day in holidays
                    leave_status = leave_days.get((emp.pk, day))
                    if leave_status is False:
                        absent += 1
                        continue
                    if leave_status is True:
                        paid_rest += 1
                        continue
                    hours = Decimal(str(days.get(day, {}).get('hours', 0)))
                    if not scheduled or holiday:
                        paid_rest += 1
                        if hours and ((holiday and rule.count_holiday_overtime) or
                                      (not scheduled and rule.count_weekend_overtime)):
                            holiday_ot += hours
                            holiday_ot_pay += hours * rate * Decimal('1.5')
                        continue
                    if not hours:
                        absent += 1
                        continue
                    full = Decimal(str(rule.full_day_hours))
                    half_req = Decimal(str(rule.half_day_hours))
                    if full <= 0 or half_req <= 0:
                        raise ValueError(f'Invalid work rule for employee {emp.pk}')
                    if hours >= full * Decimal('0.90'):
                        present += 1
                    elif hours >= half_req:
                        half += 1
                    else:
                        absent += 1
                    regular_ot += max(Decimal('0'), hours - full)
                    regular_ot_pay += max(Decimal('0'), hours - full) * rate
                payable = present + paid_rest + (half * Decimal('0.5'))
                fraction = payable / days_in_month
                money = lambda v: Decimal(v).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                basic = money(structure.base_salary * fraction)
                allowances = money((structure.hra + structure.allowances) * fraction)
                deductions = money(structure.deductions)
                # Follow the applicable base work rule rate; store overtime hours on the record.
                # Premium is 1.5x for configured weekend/holiday overtime.
                regular_overtime_pay = money(regular_ot_pay)
                holiday_overtime_pay = money(holiday_ot_pay)
                overtime_payment = regular_overtime_pay + holiday_overtime_pay
                deductions = min(deductions, basic + allowances + overtime_payment)
                net = max(Decimal('0'), money(basic + allowances + overtime_payment - deductions))
                attrs = dict(basic_pay=basic, allowances=allowances, deductions=deductions,
                    calculated_salary=net, net_salary=net,
                    present_days=present, half_days=half, absent_days=absent,
                    paid_leave_days_count=paid_rest, overtime_hours=money(regular_ot),
                    holiday_overtime_hours=money(holiday_ot),
                    regular_overtime_pay=regular_overtime_pay,
                    holiday_overtime_pay=holiday_overtime_pay,
                    status='pending' if last_day >= timezone.localdate() else 'processed',
                    processed_at=timezone.now(), paid_date=None)
                if existing:
                    for field, value in attrs.items():
                        setattr(existing, field, value)
                    # Regeneration invalidates prior generated copies.
                    if existing.payslip_pdf:
                        existing.payslip_pdf = None
                    existing.save(update_fields=list(attrs.keys()) + ['payslip_pdf'])
                    counts['updated'] += 1
                else:
                    Payroll.global_objects.create(organisation_id=org_id, employee=emp,
                                                  month=first_day, **attrs)
                    counts['created'] += 1
        return counts

class Payroll(TenantModel):
    employee = models.ForeignKey("Employee", on_delete=models.CASCADE)
    month = models.DateField(help_text="First day of month")

    # Financials
    basic_pay = models.DecimalField(max_digits=10, decimal_places=2, help_text="Earned Basic (After pro-rata/LOP)")
    allowances = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    deductions = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    
    # Days Accounting
    present_days = models.DecimalField(max_digits=6, decimal_places=2, default=0)
    half_days = models.DecimalField(max_digits=6, decimal_places=2, default=0)
    absent_days = models.DecimalField(max_digits=6, decimal_places=2, default=0)
    paid_leave_days_count = models.DecimalField(max_digits=6, decimal_places=2, default=0)
    
    # Advanced HR Metrics
    late_days_count = models.PositiveIntegerField(default=0, help_text="Total days employee arrived late")
    late_penalty_deduction = models.DecimalField(max_digits=6, decimal_places=2, default=0, help_text="Days deducted as late penalty")
    
    # Overtime
    overtime_hours = models.DecimalField(max_digits=7, decimal_places=2, default=0, help_text="Regular Overtime")
    holiday_overtime_hours = models.DecimalField(max_digits=7, decimal_places=2, default=0, help_text="Weekend/Holiday Extra Hours")
    regular_overtime_pay = models.DecimalField(max_digits=10, decimal_places=2, default=0,
        help_text="Snapshot: paid regular overtime amount")
    holiday_overtime_pay = models.DecimalField(max_digits=10, decimal_places=2, default=0,
        help_text="Snapshot: holiday/weekend overtime premium amount")
    
    # Final Pay
    calculated_salary = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    net_salary = models.DecimalField(max_digits=10, decimal_places=2, default=0)

    payslip_pdf = models.FileField(upload_to="payslips/", blank=True, null=True)
    status = models.CharField(
        max_length=20,
        choices=[("pending", "Pending"), ("processed", "Processed"), ("paid", "Paid")],
        default="pending",
    )
    processed_at = models.DateTimeField(null=True, blank=True)
    paid_date = models.DateField(null=True, blank=True)

    objects = EnterprisePayrollManager()

    @property
    def gross_earnings(self):
        return (self.basic_pay + self.allowances + self.regular_overtime_pay +
                self.holiday_overtime_pay)

    @property
    def paid_leave_days(self):
        return self.paid_leave_days_count

    @property
    def unpaid_leave_days(self):
        return self.absent_days

    class Meta:
        unique_together = ('employee', 'month')
        indexes = [
            models.Index(fields=['month', 'status']),
            models.Index(fields=['organisation', '-month', 'status'], name='payroll_org_month_idx'),
        ]

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.employee.employee_id} | {self.month.strftime('%b %Y')} | ₹{self.net_salary}"
        
class PayrollSettings(TenantModel):
    professional_tax = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('200.00'))
    esi_limit = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('21000'))

    #NEW defaults if Employee has no WorkRule
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

class WorkRule(TenantModel):
    WORKING_DAYS_CHOICES = [
        ('5_day', 'Mon–Fri'),
        ('6_day', 'Mon–Sat'),
        ('custom', 'Custom'),
    ]

    name = models.CharField(max_length=100)

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
    max_overtime_hours = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=4,
        help_text="Max overtime allowed before counting extra shift"
    )
    overtime_rate = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=Decimal("100.00")
    )
    flexible_attendance = models.BooleanField(
        default=False,
        help_text="Allow check-in/check-out at any time (for gardener/cleaning staff)"
    )
    count_weekend_overtime = models.BooleanField(default=True)
    count_holiday_overtime = models.BooleanField(default=True)

    def __str__(self):
        return self.name

    class Meta:
        # FIX: Ensure work rule names are unique strictly within each organisation
        unique_together = ('organisation', 'name')

    # ⭐ SHIFT TYPE DETECTOR
    @property
    def shift_type(self):
        start = self.shift_start_time
        end = self.shift_end_time

        # Night shift (cross midnight)
        if end <= start:
            return "Night"

        # Day shift
        if start >= time(6, 0) and start < time(14, 0):
            return "Day"

        # Evening shift
        if start >= time(14, 0) and start < time(20, 0):
            return "Evening"

        return "General"

# ============================================================
# 📝 LEAVE MANAGEMENT
# ============================================================

class LeaveType(TenantModel):
    name = models.CharField(max_length=100)
    max_days_per_year = models.PositiveIntegerField(default=12)
    is_paid = models.BooleanField(default=True)  #new

    def __str__(self):
        return self.name

class LeaveApplication(TenantModel):
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
    class Meta:
        indexes = [models.Index(fields=['organisation', 'status', '-applied_on'], name='leave_org_status_idx')]
    approved_by = models.ForeignKey(Employee, on_delete=models.SET_NULL, null=True, blank=True, related_name='approved_leaves')

    def total_days(self):
        return (self.end_date - self.start_date).days + 1

    def __str__(self):
        return f"{self.employee} - {self.leave_type} ({self.status})"

class LeaveWorkflowStage(TenantModel):
    level = models.PositiveIntegerField(null=True)
    role_name = models.CharField(max_length=50)
    next_level = models.PositiveIntegerField(blank=True, null=True)

    class Meta:
        # FIX: Ensure holidays are unique strictly within each organisation
        unique_together = ('organisation', 'level')
    def __str__(self):
        return f"Level {self.level} - {self.role_name}"

class LeaveApproval(TenantModel):
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

class Holiday(TenantModel):
    name = models.CharField(max_length=100)
    date = models.DateField()

    def __str__(self):
        return f"{self.name} ({self.date})"

    class Meta:
        # FIX: Ensure holidays are unique strictly within each organisation
        unique_together = ('organisation', 'date')

# ============================================================
# 📄 JOINING & RESIGNATION
# ============================================================

class JoiningDetail(TenantModel):
    employee = models.OneToOneField(Employee, on_delete=models.CASCADE)
    date_of_joining = models.DateField(default=timezone.now)
    
    probation_period_months = models.PositiveIntegerField(default=3)
    confirmation_date = models.DateField(blank=True, null=True)

    def __str__(self):
        return f"Joining - {self.employee}"

class JoiningDocument(TenantModel):
    joining = models.ForeignKey(
        JoiningDetail,
        on_delete=models.CASCADE,
        related_name='joining_documents'   #avoid conflict
    )
    file = models.FileField(upload_to="joining_documents/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Document for {self.joining.employee.employee_id}"
    
class Resignation(TenantModel):
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

class Notification(TenantModel):
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

class Attendance(TenantModel):
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
            models.Index(fields=['organisation', '-timestamp'], name='att_org_ts_idx'),
            models.Index(fields=['organisation', 'employee', '-timestamp'], name='att_org_emp_ts_idx'),
        ]
    
    def __str__(self):
        return f"{self.employee} - {self.attendance_type} at {self.timestamp}"

class AttendanceSettings(TenantModel):
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

class DailyReport(TenantModel):
    date = models.DateField()
    total_employees = models.IntegerField(default=0)
    present_count = models.IntegerField(default=0)
    absent_count = models.IntegerField(default=0)
    late_count = models.IntegerField(default=0)
    average_hours = models.FloatField(default=0.0)
    
    class Meta:
        # FIX: Scoped uniqueness to the organisation level
        unique_together = ('organisation', 'date')
        ordering = ['-date']
    
    def __str__(self):
        return f"Report for {self.date}"
    
# ============================================================
# 🧬 BIOMETRICS STATS & THRESHOLDS
# ============================================================
class FaceMatchStats(TenantModel):
    # FIX: Remove unique=True to allow matching employee_ids in different organisations
    employee_id = models.CharField(max_length=20)
    
    false_rejects = models.IntegerField(default=0)
    false_accepts = models.IntegerField(default=0)
    total_attempts = models.IntegerField(default=0)

    class Meta:
        # FIX: Scoped uniqueness to the organisation level
        unique_together = ('organisation', 'employee_id')


class AttendanceMatchLog(TenantModel):
    employee_id = models.CharField(max_length=20)
    cosine = models.FloatField()
    euclidean = models.FloatField()
    confidence = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=['organisation', '-created_at'], name='matchlog_org_ts_idx')]


class EmployeeThreshold(TenantModel):
    # FIX: Remove unique=True to allow matching employee_ids in different organisations
    employee_id = models.CharField(max_length=20)
    
    cos_max = models.FloatField(default=0.85)
    euc_max = models.FloatField(default=1.40)

    class Meta:
        # FIX: Scoped uniqueness to the organisation level
        unique_together = ('organisation', 'employee_id')
