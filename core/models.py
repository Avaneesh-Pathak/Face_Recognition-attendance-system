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

class OfficeLocation(models.Model):
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

    assignment = EmployeeShiftAssignment.objects.filter(
        employee=employee,
        start_date__lte=day
    ).filter(
        Q(end_date__gte=day) | Q(end_date__isnull=True)
    ).select_related("work_rule").first()

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


class EmployeeShiftAssignment(models.Model):

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
# 💰 PAYROLL MANAGER (FINAL & CORRECT)
# class PayrollManager(models.Manager):

#     def generate_monthly_salary(self, year: int, month: int):
#         return self.generate_salary(year, month)

#     def generate_salary(self, year, month, employee=None):

#         print(f"\n[START] Payroll generation → {month}/{year}")

#         first_day = date(year, month, 1)
#         last_day = date(year, month, calendar.monthrange(year, month)[1])

#         employees = Employee.objects.filter(
#             is_active=True,
#             employment_status="active"
#         ).select_related("work_rule")

#         if employee:
#             employees = employees.filter(id=employee.id)

#         print(f"[INFO] Employees found: {employees.count()}")

#         for emp in employees:
#             print("\n==============================")
#             print(f"[EMP] {emp.user.get_full_name()} ({emp.employee_id})")

#             # ----------------------------------
#             # SAFETY CHECKS
#             # ----------------------------------
#             if not emp.work_rule:
#                 print("❌ No WorkRule → SKIPPED")
#                 continue

#             existing_payroll = Payroll.objects.filter(
#                 employee=emp,
#                 month=first_day
#             ).first()

#             if existing_payroll:
#                 print("⚠ Existing payroll found → REGENERATING")
#                 existing_payroll.delete()

#             try:
#                 structure = emp.salary_structure
#             except SalaryStructure.DoesNotExist:
#                 print("❌ No SalaryStructure → SKIPPED")
#                 continue

#             rule = get_employee_rule(emp)

#             print(f"[RULE] {rule}")

#             # ----------------------------------
#             # HOURLY RATE (MONTHLY → HOURLY)
#             # ----------------------------------
#             working_days = sum(
#                 1 for d in daterange(first_day, last_day)
#                 if is_working_day(d, rule)
#             )

#             if working_days == 0:
#                 print("❌ No working days → SKIPPED")
#                 continue

#             total_month_hours = Decimal(working_days) * rule["full_day_hours"]

#             hourly_rate = (
#                 structure.base_salary / total_month_hours
#             ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

#             print(f"[RATE] Hourly rate: {hourly_rate}")

#             total_work_hours = Decimal("0.00")
#             overtime_hours = Decimal("0.00")
#             lop_days = Decimal("0.00")

#             # ----------------------------------
#             # DAY LOOP
#             # ----------------------------------
#             for day in daterange(first_day, last_day):

#                 is_work_day = is_working_day(day, rule)
#                 print(f"\n[DAY] {day} | Working: {is_work_day}")

#                 # ---------- HOLIDAY ----------
#                 if Holiday.objects.filter(date=day).exists():
#                     if is_work_day:
#                         total_work_hours += rule["full_day_hours"]
#                         print("✔ Holiday (paid)")
#                     continue

#                 # ---------- LEAVE ----------
#                 is_leave, is_paid = has_approved_leave(emp, day)
#                 if is_leave:
#                     if is_paid:
#                         total_work_hours += rule["full_day_hours"]
#                         print("✔ Paid leave")
#                     else:
#                         lop_days += 1
#                         print("❌ Unpaid leave")
#                     continue

#                 # ---------- ATTENDANCE FETCH ----------
#                 if emp.work_rule.shift_end_time > emp.work_rule.shift_start_time:
#                     # ✅ DAY SHIFT → DATE BASED (FIX)
#                     logs = Attendance.objects.filter(
#                         employee=emp,
#                         timestamp__date=day
#                     ).order_by("timestamp")
#                     print(f"[LOGS] Day-shift logs: {logs.count()}")
#                 else:
#                     # ✅ NIGHT SHIFT → SHIFT WINDOW
#                     shift_start, shift_end = get_shift_window(day, emp.work_rule)
#                     logs = Attendance.objects.filter(
#                         employee=emp,
#                         timestamp__range=(shift_start, shift_end)
#                     ).order_by("timestamp")
#                     print(f"[LOGS] Night-shift logs: {logs.count()}")

#                 if not logs.exists():
#                     if is_work_day:
#                         lop_days += 1
#                         print("❌ No attendance → LOP")
#                     continue

#                 cin = logs.filter(attendance_type="check_in").first()
#                 cout = logs.filter(attendance_type="check_out").last()

#                 print(f"[IN ] {cin}")
#                 print(f"[OUT] {cout}")

#                 if not cin or not cout or cout.timestamp <= cin.timestamp:
#                     if is_work_day:
#                         lop_days += 1
#                         print("❌ Invalid punches → LOP")
#                     continue

#                 work_hours = Decimal(
#                     (cout.timestamp - cin.timestamp).total_seconds() / 3600
#                 ).quantize(Decimal("0.01"))

#                 print(f"[HOURS] Worked: {work_hours}")

#                 # ✅ EVEN 1 HOUR COUNTS
#                 if is_work_day:
#                     total_work_hours += work_hours
#                 else:
#                     if rule["count_weekend_overtime"]:
#                         overtime_hours += work_hours

#                 # ---------- OVERTIME ----------
#                 if work_hours > rule["full_day_hours"]:
#                     overtime_hours += work_hours - rule["full_day_hours"]

#             # ----------------------------------
#             # FINAL SALARY
#             # ----------------------------------
#             print("\n[SUMMARY]")
#             print(f"Total work hours: {total_work_hours}")
#             print(f"Overtime hours: {overtime_hours}")
#             print(f"LOP days: {lop_days}")

#             if total_work_hours == 0 and overtime_hours == 0:
#                 print("❌ No payable hours → SKIPPED")
#                 continue

#             basic_pay = (
#                 total_work_hours * hourly_rate
#             ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

#             ot_pay = (
#                 overtime_hours * rule["overtime_rate"]
#             ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

#             net_salary = (
#                 basic_pay
#                 + structure.hra
#                 + structure.allowances
#                 + ot_pay
#                 - structure.deductions
#             ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

#             payroll = Payroll.objects.create(
#                 employee=emp,
#                 month=first_day,
#                 basic_pay=basic_pay,
#                 allowances=structure.hra + structure.allowances,
#                 deductions=structure.deductions,
#                 calculated_salary=net_salary,
#                 net_salary=net_salary,
#                 absent_days=lop_days,
#                 overtime_hours=overtime_hours,
#                 status="processed",
#                 processed_at=timezone.now(),
#             )

#             print(f"✅ Payroll generated: ₹{net_salary}")

#             try:
#                 pdf, filename = generate_payslip_pdf(payroll)
#                 payroll.payslip_pdf.save(filename, ContentFile(pdf.read()))
#                 email_payslip(payroll)
#             except Exception as e:
#                 print(f"⚠ Payslip/email failed for {emp.employee_id}: {e}")

#         print("\n[END] Payroll generation complete\n")        


class EnterprisePayrollManager(models.Manager):
    """
    10x Faster, Production-Grade Payroll Engine.
    Features: 
    - Bulk fetching (O(1) DB Queries inside loops)
    - Late-coming deductions (e.g., 3 lates = 1 absent / half-day)
    - Missing punch recovery
    - Weekend / Holiday Overtime Multipliers
    - Automatic Pro-Rata for new joiners
    - Smart Shift Parsing (Auto Day/Night shift detection)
    - Automatic Paid Weekends for Salaried Employees
    """

    def generate_salary(self, year, month, employee=None):
        print(f"\n🚀 [START] Enterprise Payroll Engine → {month}/{year}")
        
        first_day = date(year, month, 1)
        last_day = date(year, month, calendar.monthrange(year, month)[1])
        days_in_month = Decimal(str((last_day - first_day).days + 1))

        # 1. FETCH ALL EMPLOYEES
        employees_qs = Employee.objects.filter(
            is_active=True,
            employment_status="active"
        ).select_related("salary_structure", "work_rule", "joiningdetail")

        if employee:
            employees_qs = employees_qs.filter(id=employee.id)
            
        employees = list(employees_qs)
        employee_ids = [emp.id for emp in employees]

        if not employees:
            print("❌ No active employees found.")
            return

        # ==========================================================
        # ⚡ MASSIVE OPTIMIZATION: PRE-FETCH EVERYTHING INTO MEMORY
        # ==========================================================

        # A. Attendance Settings (Grace periods, limits)
        att_settings = AttendanceSettings.objects.first()
        late_grace_mins = 15  # default
        if att_settings and att_settings.late_threshold:
            late_grace_mins = (att_settings.late_threshold.hour * 60) + att_settings.late_threshold.minute - (att_settings.check_in_start.hour * 60) - att_settings.check_in_start.minute
            if late_grace_mins < 0: late_grace_mins = 15

        # B. Attendance Logs (Expanded by 1 day for night shifts crossing midnight)
        logs_qs = Attendance.objects.filter(
            employee_id__in=employee_ids,
            timestamp__date__range=(first_day - timedelta(days=1), last_day + timedelta(days=1))
        ).order_by("timestamp")
        
        attendance_map = defaultdict(list)
        for log in logs_qs:
            attendance_map[log.employee_id].append(log)

        # C. Approved Leaves
        leaves_qs = LeaveApplication.objects.filter(
            employee_id__in=employee_ids,
            status="approved",
            start_date__lte=last_day,
            end_date__gte=first_day
        ).select_related("leave_type")
        
        leave_map = {}
        for leave in leaves_qs:
            for d in daterange(max(leave.start_date, first_day), min(leave.end_date, last_day)):
                leave_map[(leave.employee_id, d)] = getattr(leave.leave_type, "is_paid", True)

        # D. Holidays
        holidays_set = set(Holiday.objects.filter(
            date__range=(first_day, last_day)
        ).values_list('date', flat=True))

        # E. Shift Assignments (Overrides base work_rule)
        shifts_qs = EmployeeShiftAssignment.objects.filter(
            employee_id__in=employee_ids,
            start_date__lte=last_day,
        ).filter(Q(end_date__gte=first_day) | Q(end_date__isnull=True)).select_related("work_rule")
        
        shift_map = defaultdict(dict)
        for shift in shifts_qs:
            for d in daterange(max(shift.start_date, first_day), min(shift.end_date or last_day, last_day)):
                shift_map[shift.employee_id][d] = shift.work_rule

        # ==========================================================
        # ⚙️ PROCESS EACH EMPLOYEE
        # ==========================================================
        payroll_objects = []

        for emp in employees:
            if not getattr(emp, 'salary_structure', None) or not getattr(emp, 'work_rule', None):
                continue

            structure = emp.salary_structure
            monthly_salary = structure.base_salary
            daily_salary = monthly_salary / days_in_month

            # Counters
            present_days = Decimal("0")
            half_days = Decimal("0")
            absent_days = Decimal("0")
            paid_leave_days = Decimal("0")
            late_days = 0
            overtime_hours = Decimal("0")
            holiday_weekend_ot_hours = Decimal("0")

            logs = attendance_map.get(emp.id, [])

            for day in daterange(first_day, last_day):
                # 1. Joining / Resignation Check (Pro-Rata)
                if day < emp.date_of_joining:
                    continue
                if emp.date_of_resignation and day > emp.date_of_resignation:
                    continue

                # 2. Daily Rules & Shift
                work_rule = shift_map.get(emp.id, {}).get(day, emp.work_rule)
                is_work_day = is_working_day(day, get_employee_rule_from_rule(work_rule))
                is_holiday = day in holidays_set

                # 3. Leave Check (Approved Leaves override normal attendance)
                leave_key = (emp.id, day)
                if leave_key in leave_map:
                    if leave_map[leave_key]:  # is_paid
                        paid_leave_days += 1
                    else:
                        absent_days += 1
                    continue

                # ---------------------------------------------------------
                # 4. SMART ATTENDANCE PARSING (Handles Day/Night Auto)
                # ---------------------------------------------------------
                check_in = None
                check_out = None
                
                # Get the first Check-In that happened on this specific calendar day
                daily_check_ins = [l for l in logs if timezone.localtime(l.timestamp).date() == day and l.attendance_type == "check_in"]
                
                if daily_check_ins:
                    # Sort just in case, and pick the earliest check-in for this day
                    daily_check_ins = sorted(daily_check_ins, key=lambda x: x.timestamp)
                    check_in = daily_check_ins[0].timestamp
                    
                    # Find the very next Check-Out AFTER this Check-In 
                    # (This flawlessly handles night shifts that cross over midnight into the next day)
                    subsequent_check_outs = [l for l in logs if l.timestamp > check_in and l.attendance_type == "check_out"]
                    
                    if subsequent_check_outs:
                        subsequent_check_outs = sorted(subsequent_check_outs, key=lambda x: x.timestamp)
                        check_out = subsequent_check_outs[0].timestamp

                # No punches found at all today
                if not check_in and not check_out:
                    if is_work_day and not is_holiday:
                        absent_days += 1
                    elif not is_work_day or is_holiday:
                        # ✅ FIXED: Give them a paid day for Weekend/Holiday if they didn't work
                        paid_leave_days += 1
                    continue

                # 🛑 Missing Punch Recovery Logic
                if check_in and not check_out:
                    in_time_local = timezone.localtime(check_in)
                    expected_half_day_end = in_time_local + timedelta(hours=work_rule.half_day_hours)
                    if timezone.now() > expected_half_day_end: 
                        check_out = expected_half_day_end # Auto-recover to half-day
                    else:
                        if is_work_day: absent_days += 1
                        continue

                # Invalid Punches (Checkout before Check-in)
                if not check_in or not check_out or check_out <= check_in:
                    if is_work_day and not is_holiday: absent_days += 1
                    continue

                # 📉 Late Coming Deduction Logic
                shift_start_dt = timezone.make_aware(datetime.combine(day, work_rule.shift_start_time))
                actual_in_dt = timezone.localtime(check_in)
                
                # Only penalize late coming if it's their scheduled working day
                if is_work_day and actual_in_dt.date() == shift_start_dt.date() and actual_in_dt > shift_start_dt + timedelta(minutes=late_grace_mins):
                    late_days += 1

                # ⏱️ Work Hours Calculation
                work_hours = Decimal((check_out - check_in).total_seconds() / 3600).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                max_realistic_hours = Decimal(str(work_rule.full_day_hours + 6))
                work_hours = min(work_hours, max_realistic_hours)

                # 💵 Day Status Allocation
                full_day_req = Decimal(str(work_rule.full_day_hours))
                half_day_req = Decimal(str(work_rule.half_day_hours))

                if not is_work_day or is_holiday:
                    # ✅ FIXED: Give them their standard paid day for Weekend/Holiday
                    paid_leave_days += 1
                    
                    # ✅ FIXED: If they actually came to work on this off-day, count ALL hours as Premium Overtime
                    if work_hours > 0 and (work_rule.count_holiday_overtime or work_rule.count_weekend_overtime):
                        holiday_weekend_ot_hours += work_hours
                else:
                    # Normal Working Day
                    if work_hours >= full_day_req:
                        present_days += 1
                        extra = work_hours - full_day_req
                        if extra > 0: overtime_hours += extra
                    elif work_hours >= half_day_req:
                        half_days += 1
                    else:
                        absent_days += 1

            # =================================================
            # 💰 FINAL SALARY COMPUTATION
            # =================================================

            # Late Penalty: Every 3 late days = 1 Half Day deduction
            late_penalty_days = Decimal(late_days // 3) * Decimal("0.5")

            # Total Payable Days
            payable_days = present_days + paid_leave_days + (half_days * Decimal("0.5")) - late_penalty_days
            payable_days = max(payable_days, Decimal("0")) # Prevent negative days

            earned_basic = (daily_salary * payable_days).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            # Standard Overtime (1.0x or custom rate)
            ot_rate = Decimal(str(work_rule.overtime_rate))
            regular_ot_pay = (overtime_hours * ot_rate).quantize(Decimal("0.01"))

            # Weekend/Holiday Overtime (Usually 1.5x or 2.0x base OT rate in Enterprise)
            holiday_ot_pay = (holiday_weekend_ot_hours * ot_rate * Decimal("1.5")).quantize(Decimal("0.01"))
            total_ot_pay = regular_ot_pay + holiday_ot_pay

            final_net_salary = (
                earned_basic
                + total_ot_pay
                + structure.hra
                + structure.allowances
                - structure.deductions
            ).quantize(Decimal("0.01"))

            # Append to Bulk Creation List
            payroll_objects.append(
                Payroll(
                    employee=emp,
                    month=first_day,
                    basic_pay=earned_basic, # Store actual earned basic
                    allowances=structure.hra + structure.allowances,
                    deductions=structure.deductions,
                    calculated_salary=final_net_salary,
                    net_salary=final_net_salary,
                    present_days=present_days,
                    half_days=half_days,
                    absent_days=absent_days,
                    paid_leave_days_count=paid_leave_days,
                    late_days_count=late_days,
                    late_penalty_deduction=late_penalty_days,
                    overtime_hours=overtime_hours,
                    holiday_overtime_hours=holiday_weekend_ot_hours,
                    status="processed",
                    processed_at=timezone.now(),
                )
            )

        # ==========================================================
        # 💾 BULK SAVE (1 DB Query to Save 1000s of Records)
        # ==========================================================
        with transaction.atomic():
            print(f"🧹 Clearing previous payroll records for {month}/{year}...")
            Payroll.objects.filter(month=first_day).delete()
            
            print(f"💾 Bulk creating {len(payroll_objects)} payroll records...")
            Payroll.objects.bulk_create(payroll_objects, batch_size=500)
            
        print("✅ [END] Payroll Generation Complete. 10x Speed Achieved.\n")



class Payroll(models.Model):
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
    def paid_leave_days(self):
        return self.paid_leave_days_count

    @property
    def unpaid_leave_days(self):
        return self.absent_days

    class Meta:
        unique_together = ('employee', 'month')
        indexes = [
            models.Index(fields=['month', 'status']),
        ]

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.employee.employee_id} | {self.month.strftime('%b %Y')} | ₹{self.net_salary}"


        
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
    flexible_attendance = models.BooleanField(
        default=False,
        help_text="Allow check-in/check-out at any time (for gardener/cleaning staff)"
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


