import os
from urllib import request
import cv2
import ast
import json
import io
import zipfile
import base64
from core.models import get_employee_rule

import json 
import logging
import tempfile
import calendar
import openpyxl
from django import forms
import numpy as np
from django.forms import ModelForm
from io import BytesIO
from datetime import date, datetime, timedelta
from calendar import monthrange, month_name
from PIL import Image
from django.utils import timezone
from datetime import datetime, date, time
import calendar
from django.db.models import Prefetch
from django.core.paginator import Paginator
from django.http import JsonResponse, Http404
from django.urls import reverse
from decimal import Decimal
from django.db.models.functions import TruncDate
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.views.decorators.http import require_GET
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse, HttpResponseServerError
from django.utils import timezone
from django.db.models import Count, Q, Avg
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.utils.datastructures import MultiValueDict
from django.db.models import Q, Sum, Count, Avg
from django.db.models import Sum, F, ExpressionWrapper, fields
from datetime import date
from django.core.mail import send_mail
from core.utils.salary_slip import generate_salary_slip
from django.conf import settings
from django.urls import reverse
from dateutil.relativedelta import relativedelta 
from django.contrib.auth.models import User
from .decorators import finance_required, hr_required, admin_required
from .utils_pdf import build_salary_slip_pdf
from django.core.mail import EmailMessage
from django.http import FileResponse
from io import BytesIO
from django.db.utils import ProgrammingError, OperationalError
from .models import (Employee, Attendance, AttendanceSettings, DailyReport,Department, SalaryStructure, Payroll, LeaveType, 
    LeaveApplication, LeaveWorkflowStage, LeaveApproval,
    JoiningDetail, Resignation, Notification,JoiningDocument,WorkRule)

from .forms import ( UserRegistrationForm, EmployeeRegistrationForm, AttendanceSettingsForm,DepartmentForm, SalaryStructureForm, PayrollFilterForm,
    LeaveTypeForm, LeaveApplicationForm, LeaveApprovalForm, LeaveWorkflowStageForm,
    JoiningDetailForm, ResignationForm, NotificationForm,PayrollFilterForm
    )
from .face_system import get_face_system
from core.liveness import LivenessDetector
from core.face_recognition_utils import get_face_embedding
# ============================================================
# 💰 SALARY & PAYROLL VIEWS (UPDATED)
# ============================================================

from datetime import date
import calendar

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import FileResponse, JsonResponse
from django.db.models import Sum
from django.utils import timezone
from django.core.files.base import ContentFile
from django.core.mail import EmailMessage

from .models import (
    SalaryStructure, Payroll, Employee
)
from .forms import SalaryStructureForm, PayrollFilterForm
from .decorators import finance_required

from core.utils.payslip_pdf import generate_payslip_pdf
from core.utils.payslip_email import email_payslip

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler('registration.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

logger.setLevel(logging.INFO)

# Initialize face system (lazy init inside register also)
face_system = get_face_system()


try:
    from core.face_system import get_face_system as get_if_system
except Exception:
    get_if_system = lambda: None

try:
    from core.face_recognition import get_face_system as get_fr_system
except Exception:
    get_fr_system = lambda: None

# Liveness (your module is named livenss.py)
try:
    from core.liveness import LivenessDetector
except Exception:
    LivenessDetector = None


def home(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return redirect('login')



def register(request):
    face_sys = get_face_system()
    reg_success = request.session.pop('reg_success', None)

    if request.method == "POST":
        form = EmployeeRegistrationForm(request.POST)

        if form.is_valid():
            # 1. Create User
            try:
                user = User.objects.create_user(
                    username=form.cleaned_data['username'],
                    email=form.cleaned_data['email'],
                    password=form.cleaned_data['password1'],
                    first_name=form.cleaned_data['first_name'],
                    last_name=form.cleaned_data['last_name']
                )
            except Exception as e:
                messages.error(request, f"❌ User creation failed: {str(e)}")
                return render(request, 'register.html', {
                    'form': form,
                    'reg_success': reg_success,
                })


            # 2. Create Employee
            try:
                employee = Employee.objects.create(
                    user=user,
                    department=form.cleaned_data['department'],
                    position=form.cleaned_data['position'],
                    manager=form.cleaned_data['manager'],
                    phone_number=form.cleaned_data['phone_number'],
                    role=form.cleaned_data['role'],
                    work_rule=form.cleaned_data.get('work_rule'),
                )
            except Exception as e:
                user.delete()  # rollback user
                messages.error(request, f"❌ Employee creation failed: {str(e)}")
                return render(request, 'register.html', {
                    'form': form,
                    'reg_success': reg_success,
                })


            # 3. Handle Uploaded Face Image
            face_image = request.FILES.get('face_image')
            if face_image:
                employee.face_image = face_image
                employee.save(update_fields=['face_image'])

            # 4. Handle Webcam Image (Base64)
            if request.POST.get('captured_image'):
                try:
                    fmt, imgstr = request.POST['captured_image'].split(';base64,')
                    ext = fmt.split('/')[-1]
                    employee.face_image = ContentFile(
                        base64.b64decode(imgstr),
                        name=f"face_{employee.employee_id}.{ext}"
                    )
                    employee.save(update_fields=['face_image'])
                except Exception as e:
                    messages.warning(request, f"⚠ Webcam image could not be saved: {e}")

            # 5. Salary
            if form.cleaned_data.get('base_salary'):
                SalaryStructure.objects.create(
                    employee=employee,
                    base_salary=form.cleaned_data['base_salary'],
                    hra=form.cleaned_data.get('hra') or 0,
                    allowances=form.cleaned_data.get('allowances') or 0,
                    deductions=form.cleaned_data.get('deductions') or 0,
                )

            # 6. Joining Details
            doj = form.cleaned_data['date_of_joining']
            probation = form.cleaned_data['probation_period_months']
            confirmation_date = doj + relativedelta(months=probation)
            joining = JoiningDetail.objects.create(
                employee=employee,
                date_of_joining=doj,
                probation_period_months=probation,
                confirmation_date=confirmation_date
            )

            # 7. Save Documents
            docs = []
            for file in request.FILES.getlist('documents'):
                doc = JoiningDocument.objects.create(joining=joining, file=file)
                docs.append({'name': file.name, 'url': doc.file.url})

            # 8. Extract & Save Face Embedding using our new system
            if employee.face_image:
                try:
                    img_path = employee.face_image.path
                    emb = face_sys.get_embedding(img_path)

                    if emb is not None:
                        # Save as list of vectors (for future multi-embedding support)
                        employee.face_encoding = json.dumps([emb.tolist()])
                        employee.save(update_fields=['face_encoding'])
                    else:
                        messages.warning(request, "⚠ No recognizable face found in the image.")
                except Exception as e:
                    messages.warning(request, f"⚠ Face encoding failed: {e}")

            request.session['reg_success'] = {
                "employee_id": employee.employee_id,
                "name": user.get_full_name() or user.username,
                "docs": docs,
            }
            messages.success(
                request,
                f"Employee {employee.user.get_full_name()} ({employee.employee_id}) registered successfully."
            )
            return redirect('employee_list')

        else:
            # 🔍 Show exact form errors
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")

    else:
        form = EmployeeRegistrationForm()

    return render(request, 'register.html', {
        'form': form,
        'reg_success': reg_success,
    })


# --- Real-time AJAX validators ---
def check_username(request):
    username = request.GET.get('username', '').strip()
    exists = User.objects.filter(username__iexact=username).exists()
    return JsonResponse({"ok": bool(username and not exists), "exists": exists})

def check_email(request):
    email = request.GET.get('email', '').strip()
    exists = User.objects.filter(email__iexact=email).exists()
    return JsonResponse({"ok": bool(email and not exists), "exists": exists})

def get_managers_by_department(request, dept_id):
    managers = Employee.objects.filter(department_id=dept_id, employment_status='active')
    data = [{'id': m.id, 'name': m.user.get_full_name()} for m in managers]
    return JsonResponse({'managers': data})

@login_required
@hr_required
def employee_list(request):
    employees = (
        Employee.objects
        .select_related('user', 'department', 'work_rule', 'manager')
        .order_by('employee_id')
    )

    return render(request, 'ems/employee_list.html', {
        'employees': employees,
        'title': 'Employee List'
    })

class EmployeeUpdateForm(forms.ModelForm):
    class Meta:
        model = Employee
        fields = [
            'department',
            'position',
            'manager',
            'role',
            'phone_number',
            'employment_status',
            'work_rule',
            'is_active',
        ]

@login_required
@hr_required
def employee_update(request, pk):
    employee = get_object_or_404(Employee, pk=pk)
    user = employee.user

    if request.method == "POST":
        form = EmployeeUpdateForm(request.POST, instance=employee)

        if form.is_valid():
            # Update Employee
            form.save()

            # Update User basic info
            user.first_name = request.POST.get('first_name', user.first_name)
            user.last_name = request.POST.get('last_name', user.last_name)
            user.email = request.POST.get('email', user.email)
            user.save()

            messages.success(request, "Employee updated successfully.")
            return redirect('employee_list')
    else:
        form = EmployeeUpdateForm(instance=employee)

    return render(request, 'ems/employee_form.html', {
        'form': form,
        'employee': employee,
        'user_obj': user,
        'title': 'Update Employee'
    })

@login_required
@admin_required
def employee_delete(request, pk):
    employee = get_object_or_404(Employee, pk=pk)
    user = employee.user

    if request.method == "POST":
        user.delete()  # ✅ cascades to Employee
        messages.success(request, "Employee and user account deleted successfully.")
        return redirect('employee_list')

    return render(request, 'ems/employee_confirm_delete.html', {
        'employee': employee,
        'title': 'Delete Employee'
    })


@login_required
def dashboard(request):
    user = request.user
    is_admin = user.is_staff or user.is_superuser

    # Common date values
    today = timezone.localdate()
    start_week = today - timedelta(days=today.weekday())
    start_month = today.replace(day=1)
    current_month = today.month
    current_year = today.year

    # Detect employee object for the logged-in user (if exists)
    try:
        employee = request.user.employee
        employee_exists = True
    except Exception:
        employee = None
        employee_exists = False

    # Admin selection (per-employee selector)
    employees = Employee.objects.all().select_related('user') if is_admin else None
    emp_id = request.GET.get('emp_id')

    # Attendance queryset scope
    if is_admin:
        if emp_id:
            selected_employee = Employee.objects.filter(id=emp_id).first()
            # For charts & lists, use selected employee's attendance
            attendance_qs = Attendance.objects.filter(employee=selected_employee)
            # Also set "employee" for EMS sections if selected (optional)
            employee = selected_employee
        else:
            selected_employee = None
            attendance_qs = Attendance.objects.all()
    else:
        selected_employee = None
        attendance_qs = Attendance.objects.filter(employee=employee)

    # Attendance lists
    today_records = attendance_qs.filter(timestamp__date=today).order_by('timestamp')
    week_records = attendance_qs.filter(timestamp__date__gte=start_week)
    month_records = attendance_qs.filter(timestamp__date__gte=start_month)

    # Aggregates
    avg_confidence = attendance_qs.aggregate(avg=Avg('confidence_score'))['avg'] or 0
    total_employees = Employee.objects.count()

    # Employee status (if we have a concrete employee context)
    current_status = None
    total_hours = 0
    last_checkin = None
    if (is_admin and selected_employee) or (not is_admin and employee):
        concrete_emp = selected_employee if selected_employee else employee
        # Filter for today for that employee
        _emp_today = Attendance.objects.filter(
            employee=concrete_emp, timestamp__date=today
        ).order_by('timestamp')

        last_checkin = _emp_today.filter(attendance_type='check_in').last()
        last_checkout = _emp_today.filter(attendance_type='check_out').last()

        if last_checkin and (not last_checkout or last_checkout.timestamp < last_checkin.timestamp):
            current_status = 'Checked In'
        elif last_checkout:
            current_status = 'Checked Out'
        else:
            current_status = 'Not Checked In'

        # Calculate hours
        checkins = _emp_today.filter(attendance_type='check_in')
        checkouts = _emp_today.filter(attendance_type='check_out')
        for ci in checkins:
            co = checkouts.filter(timestamp__gt=ci.timestamp).first()
            if co:
                total_hours += (co.timestamp - ci.timestamp).total_seconds() / 3600

    # =========================
    # EMS blocks (per-employee)
    # =========================
    leave_stats = None
    pending_approvals = 0
    recent_notifications = None
    recent_payroll = None

    # Decide which employee to show EMS panels for:
    ems_emp = selected_employee if selected_employee else employee

    if ems_emp:
        # Leave stats for current year (status buckets)
        leave_stats = (
            LeaveApplication.objects
            .filter(employee=ems_emp, start_date__year=current_year)
            .values('status').annotate(count=Count('id'))
        )

        # Pending approvals for this employee (if they are approver/manager)
        pending_approvals = (
            LeaveApplication.objects
            .filter(approvals__approver=ems_emp, approvals__status='pending')
            .count()
        )

        # Recent notifications to the logged-in user
        recent_notifications = (
            Notification.objects
            .filter(recipient=user)
            .order_by('-created_at')[:5]
        )

        # Latest payroll for this employee
        recent_payroll = (
            Payroll.objects
            .filter(employee=ems_emp)
            .order_by('-month').first()
        )

    # =========================
    # Department stats (admin)
    # =========================
    department_stats = Department.objects.annotate(
        # If you don't have related_name="employees", use Count('employee') or Count('employee_set')
        employee_count=Count('employees')
    ).values('name', 'employee_count')

    # Build department chart arrays
    department_names = []
    department_employee_counts = []
    if is_admin and department_stats:
        for d in department_stats:
            department_names.append(d['name'])
            department_employee_counts.append(d['employee_count'])

    # =========================
    # Charts (using available data)
    # =========================

    # 30-day attendance (count of check-ins per day) using the current attendance_qs scope
    start_30 = today - timedelta(days=29)
    att_30 = (
        attendance_qs
        .filter(timestamp__date__gte=start_30, attendance_type='check_in')
        .annotate(day=TruncDate('timestamp'))
        .values('day')
        .annotate(count=Count('id'))
    )
    att_counts = {item['day']: item['count'] for item in att_30}
    date_range = [start_30 + timedelta(days=i) for i in range(30)]
    attendance_labels = [d.strftime('%d %b') for d in date_range]
    attendance_values = [att_counts.get(d, 0) for d in date_range]

    # Leave donut chart arrays
    leave_labels, leave_values = [], []
    if leave_stats:
        status_order = ['approved', 'pending', 'rejected', 'cancelled']
        by_status = {row['status'].lower(): row['count'] for row in leave_stats}
        for st in status_order:
            if st in by_status:
                leave_labels.append(st.capitalize())
                leave_values.append(by_status[st])
        for row in leave_stats:
            k = row['status'].capitalize()
            if k not in leave_labels:
                leave_labels.append(k)
                leave_values.append(row['count'])

    payroll_history = None
    leave_applications = None
    if ems_emp:
        payroll_history = (
        Payroll.objects.filter(employee=ems_emp).order_by('-month')[1:7]
        )
        leave_applications = (
        LeaveApplication.objects.filter(employee=ems_emp).order_by('-start_date')[:10]
        )
    
    # JSON dumps for template
    context = {
        'is_admin': is_admin,
        'employees': employees,
        'employee': ems_emp,
        'employee_exists': ems_emp is not None,

        'today_attendance': today_records,
        'week_attendance': week_records,
        'month_attendance': month_records,
        'avg_confidence': round(avg_confidence, 2),
        'current_status': current_status,
        'last_checkin': last_checkin,
        'total_hours': round(total_hours, 2),
        'total_employees': total_employees,
        'selected_emp_id': int(emp_id) if emp_id else None,

        'leave_stats': leave_stats,
        'pending_approvals': pending_approvals,
        'recent_notifications': recent_notifications,
        'department_stats': department_stats,
        'recent_payroll': recent_payroll,

        'year': today.year,
        'month': today.month,
        'title': 'Dashboard',

        # charts
        'attendance_labels_json': json.dumps(attendance_labels),
        'attendance_values_json': json.dumps(attendance_values),
        'leave_labels_json': json.dumps(leave_labels),
        'leave_values_json': json.dumps(leave_values),
        'department_names_json': json.dumps(department_names),
        'department_employee_counts_json': json.dumps(department_employee_counts),
        'payroll_history': payroll_history,
        'leave_applications': leave_applications,
    }

    return render(request, 'dashboard.html', context)


@login_required
def my_profile_view(request):
    # -----------------------------------
    # USER HAS EMPLOYEE PROFILE
    # -----------------------------------
    if hasattr(request.user, 'employee'):
        employee = request.user.employee
        return render(request, 'ems/my_profile.html', {
            'employee': employee
        })

    # -----------------------------------
    # ADMIN / SYSTEM USER (NO EMPLOYEE)
    # -----------------------------------
    return render(request, 'ems/system_profile.html', {
        'user': request.user
    })

@login_required
def attendance_summary_api(request):
    """
    Returns live attendance aggregated for the last N days.
    """
    try:
        range_days = int(request.GET.get("range", 7))
    except Exception:
        range_days = 7

    emp_id = request.GET.get("emp_id")

    today = timezone.localdate()
    start_date = today - timedelta(days=range_days - 1)

    if emp_id:
        employee = Employee.objects.filter(id=emp_id).first()
        attendance_qs = Attendance.objects.filter(employee=employee)
    else:
        attendance_qs = Attendance.objects.all()

    labels = []
    present = []
    absent = []

    total_employees = Employee.objects.count() if not emp_id else 1

    for i in range(range_days):
        day = start_date + timedelta(days=i)
        labels.append(day.strftime('%a'))

        if emp_id:
            day_present = attendance_qs.filter(timestamp__date=day, attendance_type="check_in").exists()
            present.append(1 if day_present else 0)
            absent.append(0 if day_present else 1)
        else:
            day_checkins = attendance_qs.filter(timestamp__date=day, attendance_type="check_in").values_list("employee", flat=True).distinct().count()
            day_absent = max(total_employees - day_checkins, 0)
            present.append(day_checkins)
            absent.append(day_absent)

    yesterday = present[-2] if len(present) > 1 else 0

    return JsonResponse({
        "labels": labels,
        "present": present,
        "absent": absent,
        "yesterday": yesterday,
    })


# ------------------ Utility to Fix NumPy JSON Error ------------------
def to_native(obj):
    """Convert non-JSON serializable numpy and boolean types."""
    import numpy as _np
    if isinstance(obj, (_np.bool_, _np.bool8)):
        return bool(obj)
    if isinstance(obj, (_np.float32, _np.float64)):
        return float(obj)
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    return obj


# ------------------ Cache for Reusing Liveness Detector ---------------
_LIVE_CACHE = {}   # {session_key: (detector_instance, last_access_time)}
_LIVE_TTL = 60     # Cleanup after 60 seconds of inactivity

def _get_liveness(session_key):
    """Fetch cached LivenessDetector OR create new if expired/missing."""
    import time
    now = time.time()

    det, last_time = _LIVE_CACHE.get(session_key, (None, 0))
    if det is None:
        if LivenessDetector is None:
            return None
        det = LivenessDetector(adaptive=True)

    _LIVE_CACHE[session_key] = (det, now)

    # Clean stale detectors
    for k, (_, t) in list(_LIVE_CACHE.items()):
        if now - t > _LIVE_TTL:
            try:
                _LIVE_CACHE[k][0].close()
            except Exception:
                pass
            _LIVE_CACHE.pop(k, None)
    return det


# ------------------ Face recognition helper (InsightFace → fallback) ---------------
def _recognize_employee(frame_bgr):
    """
    Returns (employee_instance_or_None, score_float, bbox_or_None, backend_used)
    """
    # 1) Try InsightFace pipeline
    fs_if = get_if_system()
    if fs_if is not None:
        try:
            emp_id, score, bbox = fs_if.recognize_from_frame(frame_bgr)
            if emp_id:
                emp = Employee.objects.filter(employee_id=emp_id, is_active=True).first()
                if emp:
                    return emp, float(score), bbox, "insightface"
        except (ProgrammingError, OperationalError):
            # DB not ready during migrations
            return None, 0.0, None, "db-not-ready"
        except Exception:
            # continue to fallback
            pass

    # 2) Fallback to dlib/face_recognition pipeline
    fs_fr = get_fr_system()
    if fs_fr is not None:
        try:
            emp_id, score = fs_fr.recognize_face_from_frame(frame_bgr)[:2] \
                if hasattr(fs_fr, "recognize_face_from_frame") else (None, 0.0)
            if not emp_id and hasattr(fs_fr, "recognize_face"):
                # if code path is different in your file
                emp_id, score = fs_fr.recognize_face_from_frame(frame_bgr)[:2]
            if emp_id:
                emp = Employee.objects.filter(employee_id=emp_id, is_active=True).first()
                if emp:
                    return emp, float(score), None, "face_recognition"
        except (ProgrammingError, OperationalError):
            return None, 0.0, None, "db-not-ready"
        except Exception:
            pass

    return None, 0.0, None, "none"


# ------------------ ✅ Final mark_attendance Endpoint ------------------
@login_required
def mark_attendance(request):
    if request.method == "POST" and request.FILES.get("capture"):
        try:
            # ---- 1) Decode frame ----
            file_bytes = np.frombuffer(request.FILES["capture"].read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame is None:
                return JsonResponse({"success": False, "message": "Invalid image data."})

            # ---- 2) Face Recognition (with fallback + migration-safe) ----
            try:
                emp, score, bbox, backend = _recognize_employee(frame)
            except (ProgrammingError, OperationalError):
                return JsonResponse({
                    "success": False,
                    "message": "Database is not ready yet. Please finish migrations and retry.",
                    "color": "red"
                })

            if backend == "db-not-ready":
                return JsonResponse({
                    "success": False,
                    "message": "Database is not ready yet. Please finish migrations and retry.",
                    "color": "red"
                })

            if not emp:
                return JsonResponse({
                    "success": False,
                    "message": "Face not recognized",
                    "color": "red",
                    "backend": backend
                })

            # ---- 3) Liveness Detection ----
            detector = _get_liveness(request.session.session_key or str(emp.employee_id))
            if detector is None:
                # If mediapipe is unavailable, allow attendance but tag as not-checked
                liveness_data = {"ear": 0.0, "blink": False, "motion": False, "live": True, "face_detected": True}
            else:
                lres = detector.detect_detail(frame)
                # bring to native types and include dynamic threshold if available
                dynamic_thr = getattr(detector, "dynamic_threshold", None)
                liveness_data = {
                    "ear": float(lres.get("ear", 0.0)),
                    "blink": bool(lres.get("blink", False)),
                    "motion": bool(lres.get("motion", False)),
                    "live": bool(lres.get("live", False)),
                    "face_detected": bool(lres.get("face_detected", False)),
                    "threshold": float(dynamic_thr) if dynamic_thr is not None else None
                }

                # If motion present, treat as live (your original rule)
                if liveness_data["motion"]:
                    liveness_data["blink"] = True
                    liveness_data["live"] = True

                if not liveness_data["live"]:
                    return JsonResponse({
                        "success": False,
                        "message": (
                            f"Liveness failed — Blink:{liveness_data['blink']} | "
                            f"Motion:{liveness_data['motion']} | EAR:{round(liveness_data['ear'],3)}"
                        ),
                        "liveness": to_native(liveness_data),
                        "color": "yellow"
                    })

            # ---- 4) Attendance Logic (pairs map to hours → payroll later) ----
            now = timezone.now()
            settings_obj = AttendanceSettings.objects.first()
            min_hours = float(getattr(settings_obj, "min_hours_before_checkout", 3.0))

            last_att = Attendance.objects.filter(employee=emp).order_by("-timestamp").first()

            # Already has a record today?
            if last_att and timezone.localtime(last_att.timestamp).date() == now.date():
                if last_att.attendance_type == "check_in":
                    elapsed = (now - last_att.timestamp).total_seconds() / 3600.0
                    if elapsed < min_hours:
                        remain_min = int((min_hours - elapsed) * 60)
                        return JsonResponse({
                            "success": False,
                            "message": f"Wait {remain_min} minutes before checkout.",
                            "liveness": to_native(liveness_data),
                            "color": "yellow"
                        })

                    Attendance.objects.create(
                        employee=emp,
                        attendance_type="check_out",
                        confidence_score=score
                    )
                    return JsonResponse({
                        "success": True,
                        "attendance_type": "Check-Out",
                        "name": emp.user.get_full_name(),
                        "confidence": round(score * 100, 2),
                        "liveness": to_native(liveness_data),
                        "backend": backend,
                        "color": "green",
                        "message": f"✅ Checked-Out at {now.strftime('%H:%M:%S')}"
                    })
                else:
                    # already checked out today
                    return JsonResponse({
                        "success": False,
                        "message": "✅ Already checked out today",
                        "liveness": to_native(liveness_data),
                        "backend": backend,
                        "color": "yellow"
                    })

            # First check-in today
            Attendance.objects.create(
                employee=emp,
                attendance_type="check_in",
                confidence_score=score
            )
            return JsonResponse({
                "success": True,
                "attendance_type": "Check-In",
                "name": emp.user.get_full_name(),
                "confidence": round(score * 100, 2),
                "liveness": to_native(liveness_data),
                "backend": backend,
                "color": "green",
                "message": f"✅ Checked-In at {now.strftime('%H:%M:%S')}"
            })

        except Exception as e:
            import traceback, logging
            logging.getLogger(__name__).error("Error in mark_attendance: %s", traceback.format_exc())
            return JsonResponse({"success": False, "message": f"Internal Error: {e}", "color": "red"})

    return JsonResponse({"success": False, "message": "Invalid Request Method"})


@login_required
def attendance_log_api(request):
    today = timezone.now().date()
    logs = (
        Attendance.objects.filter(timestamp__date=today)
        .select_related('employee__user')
        .order_by('-timestamp')[:10]
    )
    data = [
        {
            "name": log.employee.user.get_full_name() or log.employee.user.username,
            "type": log.attendance_type.replace('_', ' ').title(),
            "time": log.timestamp.strftime("%I:%M %p"),
            "confidence": round(log.confidence_score * 100, 2) if log.confidence_score is not None else None,
        }
        for log in logs
    ]
    return JsonResponse(data, safe=False)


def _video_frame_generator(camera_index=0):
    """Internal generator used by video_feed view."""
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        logger.error("Camera not accessible")
        return  # generator will end

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                logger.warning("Failed to read frame from camera")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                face_region = frame[y:y+h, x:x+w]
                try:
                    emb = get_face_embedding(face_region)
                except Exception:
                    logger.exception("get_face_embedding failed on region")
                    emb = None

                color = (0, 0, 255)
                label = "Unknown"

                if emb is not None:
                    emb = np.asarray(emb, dtype=np.float32).ravel()
                    n = np.linalg.norm(emb)
                    if n > 0:
                        emb = emb / n

                    best_match = None
                    settings_obj = AttendanceSettings.objects.first()
                    threshold = settings_obj.confidence_threshold if settings_obj else 0.60
                    best_score = float(threshold)

                    employees = Employee.objects.exclude(Q(face_encoding__isnull=True) | Q(face_encoding__exact=''))

                    for emp in employees:
                        try:
                            db_emb = None
                            if hasattr(emp, 'get_face_encoding'):
                                db_emb = emp.get_face_encoding()
                            if db_emb is None:
                                raw = emp.face_encoding
                                if isinstance(raw, str):
                                    try:
                                        db_emb = json.loads(raw)
                                    except Exception:
                                        db_emb = None
                                elif isinstance(raw, (list, tuple, np.ndarray)):
                                    db_emb = raw

                            if db_emb is None:
                                continue
                            db_emb = np.asarray(db_emb, dtype=np.float32).ravel()
                            dnorm = np.linalg.norm(db_emb)
                            if dnorm > 0:
                                db_emb = db_emb / dnorm

                            sim = float(np.dot(emb, db_emb))
                            if sim > best_score:
                                best_match = emp
                                best_score = sim
                        except Exception:
                            logger.exception("Error comparing embeddings")
                            continue

                    if best_match:
                        name = best_match.user.get_full_name() or best_match.user.username
                        confidence = best_score * 100
                        label = f"{name} ({confidence:.1f}%)"
                        color = (0, 255, 0)

                # draw overlays
                try:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame, (x, y - 25), (x + max(80, len(label) * 10), y), color, -1)
                    cv2.putText(frame, label, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception:
                    pass

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_jpg = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_jpg + b'\r\n')
    finally:
        try:
            camera.release()
        except Exception:
            logger.exception("Error releasing camera in video generator")


def video_feed():
    """Live video stream generator used by StreamingHttpResponse."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logger.error("Camera not accessible")
        return  # generator ends; caller should handle this case

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    try:
        while True:
            success, frame = camera.read()
            if not success:
                logger.warning("Failed to read frame from camera")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                face_region = frame[y:y+h, x:x+w]
                emb = get_face_embedding(face_region)
                color = (0, 0, 255)
                label = "Unknown"

                if emb is not None:
                    emb = np.array(emb, dtype=np.float32)
                    n = np.linalg.norm(emb)
                    if n > 0:
                        emb = emb / n

                    best_match = None
                    settings_obj = AttendanceSettings.objects.first()
                    threshold = settings_obj.confidence_threshold if settings_obj else 0.60
                    best_score = float(threshold)
                    employees = Employee.objects.exclude(Q(face_encoding__isnull=True) | Q(face_encoding__exact=''))

                    for emp in employees:
                        try:
                            db_emb = emp.get_face_encoding() if hasattr(emp, 'get_face_encoding') else None
                            if db_emb is None:
                                raw = emp.face_encoding
                                if isinstance(raw, str):
                                    try:
                                        db_emb = json.loads(raw)
                                    except Exception:
                                        db_emb = None
                                elif isinstance(raw, (list, tuple, np.ndarray)):
                                    db_emb = raw
                            if db_emb is None:
                                continue
                            db_emb = np.asarray(db_emb, dtype=np.float32).ravel()
                            dnorm = np.linalg.norm(db_emb)
                            if dnorm > 0:
                                db_emb = db_emb / dnorm

                            sim = float(np.dot(emb, db_emb))
                            if sim > best_score:
                                best_match = emp
                                best_score = sim
                        except Exception:
                            continue

                    if best_match:
                        name = best_match.user.get_full_name() or best_match.user.username
                        confidence = best_score * 100
                        label = f"{name} ({confidence:.1f}%)"
                        color = (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y - 25), (x + len(label) * 10, y), color, -1)
                cv2.putText(frame, label, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_jpg = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_jpg + b'\r\n')
    finally:
        try:
            camera.release()
        except Exception:
            logger.exception("Error releasing camera in video_feed")


@login_required
def attendance_page(request):
    return render(request, 'attendance.html')


@login_required
def video_feed_view(request):
    gen = video_feed()
    if gen is None:
        return HttpResponseServerError("Camera not accessible")
    return StreamingHttpResponse(gen, content_type='multipart/x-mixed-replace; boundary=frame')



@staff_member_required
def attendance_reports(request):
    # 1. Get Parameters
    start_date_str = request.GET.get("start_date")
    end_date_str = request.GET.get("end_date")
    employee_id = request.GET.get("employee")
    date_range = request.GET.get("range") # Handle the "Period" dropdown
    download = request.GET.get("download")

    # 2. Date Parsing Logic
    today = timezone.now().date()
    
    # Handle Shortcuts (Today, Week, Month)
    if date_range == 'today':
        start_date = end_date = today
    elif date_range == 'week':
        start_date = today - timedelta(days=today.weekday()) # Start of week (Monday)
        end_date = today
    elif date_range == 'month':
        start_date = today.replace(day=1)
        end_date = today
    elif start_date_str and end_date_str:
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        except ValueError:
            start_date = end_date = today
    else:
        # Default view (e.g., Today)
        start_date = end_date = today

    # 3. Base QuerySet
    # Filter by date range
    attendance_qs = Attendance.objects.select_related(
        "employee", "employee__user"
    ).filter(
        timestamp__date__range=(start_date, end_date)
    ).order_by("-timestamp") # Latest first for the table

    # Filter by specific employee if selected
    if employee_id:
        attendance_qs = attendance_qs.filter(employee_id=employee_id)

    # 4. EXCEL EXPORT (Logic preserved and cleaned up)
    if download == "excel":
        return generate_excel_report(attendance_qs, start_date, end_date)

    # 5. Dashboard Statistics Calculation
    total_employees_count = Employee.objects.filter(is_active=True).count()
    
    # Count unique employees who have at least one 'check_in' record in the filtered range
    present_employees_count = attendance_qs.filter(
        attendance_type='check_in'
    ).values('employee').distinct().count()
    
    absent_employees_count = max(0, total_employees_count - present_employees_count)

    # 6. Pagination
    paginator = Paginator(attendance_qs, 20) # Show 20 records per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # ---------------------------------------------------------
    # 7. CHART DATA PREPARATION (NEW CODE)
    # ---------------------------------------------------------
    # Get daily counts of unique employees present
    daily_attendance = attendance_qs.filter(attendance_type='check_in') \
        .annotate(date=TruncDate('timestamp')) \
        .values('date') \
        .annotate(count=Count('employee', distinct=True)) \
        .order_by('date')
    
    # Convert QuerySet to a dictionary {date: count} for fast lookup
    daily_stats = {item['date']: item['count'] for item in daily_attendance}
    
    chart_labels = []
    chart_data_present = []
    
    # Loop through every day in range (handles gaps where count is 0)
    current_d = start_date
    while current_d <= end_date:
        label = current_d.strftime("%b %d") # e.g., "Jan 05"
        chart_labels.append(label)
        chart_data_present.append(daily_stats.get(current_d, 0))
        current_d += timedelta(days=1)

    # 7. Context Preparation
    context = {
        "attendance_data": page_obj, # This acts as the list in the template
        "start_date": start_date,
        "end_date": end_date,
        "employees": Employee.objects.filter(is_active=True).order_by('user__first_name'),
        "selected_employee": int(employee_id) if employee_id else None,
        
        # Dashboard Stats
        "total_employees": total_employees_count,
        "present_employees": present_employees_count,
        "absent_employees": absent_employees_count,
        
        # For UI logic
        "date_filter": start_date if start_date == end_date else None,
        "chart_labels": json.dumps(chart_labels),
        "chart_data": json.dumps(chart_data_present),
    }

    return render(request, "reports.html", context)


def generate_excel_report(queryset, start_date, end_date):
    import calendar
    from decimal import Decimal
    from django.utils import timezone
    import openpyxl

    wb = openpyxl.Workbook()

    ws_daily = wb.active
    ws_daily.title = "Daily Attendance"
    ws_salary = wb.create_sheet("Salary Calculation")

    ws_daily.append([
        "Employee", "Date", "Check In", "Check Out",
        "Work Hours", "Day Type", "Overtime (Hrs)"
    ])

    ws_salary.append([
        "Employee", "Month",
        "Monthly Salary",
        "Paid Days", "Half Days", "LOP Days",
        "Salary / Day",
        "Gross Pay",
        "OT Hours", "OT Amount",
        "Net Salary"
    ])

    queryset = queryset.order_by("employee", "timestamp")

    daily = {}
    for rec in queryset:
        key = (rec.employee.id, rec.timestamp.date())
        daily.setdefault(key, {
            "employee": rec.employee,
            "date": rec.timestamp.date(),
            "check_in": None,
            "check_out": None,
        })

        ts = timezone.localtime(rec.timestamp)
        if rec.attendance_type == "check_in" and not daily[key]["check_in"]:
            daily[key]["check_in"] = ts
        elif rec.attendance_type == "check_out":
            daily[key]["check_out"] = ts

    monthly = {}

    for row in daily.values():
        emp = row["employee"]
        ci, co = row["check_in"], row["check_out"]
        if not ci or not co or co <= ci:
            continue

        work_hours = Decimal(
            (co - ci).total_seconds() / 3600
        ).quantize(Decimal("0.01"))

        rule = get_employee_rule(emp)

        if work_hours >= rule["full_day_hours"]:
            day_value = Decimal("1.0")
            day_type = "Full"
        elif work_hours >= rule["half_day_hours"]:
            day_value = Decimal("0.5")
            day_type = "Half"
        else:
            day_value = Decimal("0.0")
            day_type = "LOP"

        ot_hours = max(Decimal("0.0"), work_hours - rule["full_day_hours"])

        ws_daily.append([
            emp.user.get_full_name(),
            row["date"].strftime("%d-%m-%Y"),
            ci.strftime("%H:%M"),
            co.strftime("%H:%M"),
            float(work_hours),
            day_type,
            float(ot_hours),
        ])

        key = (emp.id, row["date"].year, row["date"].month)
        monthly.setdefault(key, {
            "employee": emp,
            "paid_days": Decimal("0.0"),
            "half_days": Decimal("0.0"),
            "lop_days": Decimal("0.0"),
            "ot_hours": Decimal("0.0"),
        })

        m = monthly[key]
        if day_value == 1:
            m["paid_days"] += 1
        elif day_value == Decimal("0.5"):
            m["paid_days"] += Decimal("0.5")
            m["half_days"] += Decimal("0.5")
        else:
            m["lop_days"] += 1

        m["ot_hours"] += ot_hours

    for (emp_id, y, mth), m in monthly.items():
        emp = m["employee"]
        structure = SalaryStructure.objects.filter(employee=emp).first()
        if not structure:
            continue

        days_in_month = Decimal(calendar.monthrange(y, mth)[1])
        salary_per_day = (structure.base_salary / days_in_month).quantize(Decimal("0.01"))

        gross = (salary_per_day * m["paid_days"]).quantize(Decimal("0.01"))
        rate = emp.work_rule.overtime_rate if emp.work_rule else Decimal("0.00")
        ot_amount = (m["ot_hours"] * rate).quantize(Decimal("0.01"))


        net = (
            gross +
            structure.hra +
            structure.allowances +
            ot_amount -
            structure.deductions
        )

        ws_salary.append([
            emp.user.get_full_name(),
            f"{y}-{mth:02d}",
            float(structure.base_salary),
            float(m["paid_days"]),
            float(m["half_days"]),
            float(m["lop_days"]),
            float(salary_per_day),
            float(gross),
            float(m["ot_hours"]),
            float(ot_amount),
            float(net),
        ])

    response = HttpResponse(
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response["Content-Disposition"] = (
        f'attachment; filename="attendance_salary_{start_date}_to_{end_date}.xlsx"'
    )
    wb.save(response)
    return response



def attendance_settings(request):
    settings_obj, created = AttendanceSettings.objects.get_or_create(pk=1)

    if request.method == 'POST':
        form = AttendanceSettingsForm(request.POST, instance=settings_obj)
        if form.is_valid():
            form.save()
            return redirect('attendance_settings')
    else:
        form = AttendanceSettingsForm(instance=settings_obj)

    return render(request, 'settings.html', {'form': form})


@login_required
def get_attendance_history(request):
    days = int(request.GET.get('days', 30))
    start_date = timezone.now().date() - timedelta(days=days)

    if request.user.is_staff:
        attendance_qs = Attendance.objects.filter(timestamp__date__gte=start_date)
    else:
        try:
            employee = request.user.employee
            attendance_qs = Attendance.objects.filter(employee=employee, timestamp__date__gte=start_date)
        except Employee.DoesNotExist:
            attendance_qs = Attendance.objects.none()

    data = [
        {
            'employee': str(record.employee),
            'date': record.timestamp.strftime('%Y-%m-%d'),
            'time': record.timestamp.strftime('%H:%M:%S'),
            'type': record.get_attendance_type_display(),
            'confidence': f"{record.confidence_score * 100:.1f}%" if record.confidence_score else 'N/A'
        }
        for record in attendance_qs.order_by('-timestamp')
    ]

    return JsonResponse({'data': data})


@login_required
def attendance_history_page(request):
    days = int(request.GET.get('days', 30))
    start_date = timezone.now().date() - timedelta(days=days)

    if request.user.is_staff:
        attendance_qs = Attendance.objects.filter(timestamp__date__gte=start_date).order_by('-timestamp')
    else:
        try:
            employee = request.user.employee
            attendance_qs = Attendance.objects.filter(employee=employee, timestamp__date__gte=start_date).order_by('-timestamp')
        except Employee.DoesNotExist:
            attendance_qs = Attendance.objects.none()

    context = {
        'attendance_records': attendance_qs,
        'days': days,
    }
    return render(request, 'attendance_history.html', context)


@login_required
@require_GET
def attendance_calendar_data(request, employee_id, year, month):
    """Get attendance data for calendar display"""
    # ensure ints
    try:
        year = int(year)
        month = int(month)
    except Exception:
        return JsonResponse({'error': 'Invalid year/month'}, status=400)

    emp = get_object_or_404(Employee, id=employee_id)

    start_date = date(year, month, 1)
    end_day = calendar.monthrange(year, month)[1]
    end_date = date(year, month, end_day)

    present_days = Attendance.objects.filter(
        employee=emp,
        timestamp__date__range=(start_date, end_date),
        attendance_type='check_in'
    ).values_list('timestamp__date', flat=True)

    present_day_set = {d.day for d in present_days}

    month_data = []
    for day in range(1, end_day + 1):
        current_date = date(year, month, day)
        month_data.append({
            'day': day,
            'date': current_date.isoformat(),
            'present': day in present_day_set,
            'is_weekend': current_date.weekday() >= 5,
            'is_future': current_date > timezone.now().date()
        })

    return JsonResponse({
        'employee': emp.user.get_full_name(),
        'year': year,
        'month': month,
        'month_name': calendar.month_name[month],
        'data': month_data
    })


@login_required
def attendance_calendar(request, employee_id, year, month):
    employee = get_object_or_404(Employee, id=employee_id)
    year = int(year)
    month = int(month)

    start_date = date(year, month, 1)
    end_day = monthrange(year, month)[1]
    end_date = date(year, month, end_day)

    attendance_records = Attendance.objects.filter(
        employee=employee,
        timestamp__date__range=(start_date, end_date)
    ).order_by('timestamp')

    settings = AttendanceSettings.objects.first()
    max_daily_hours = settings.max_daily_hours if settings else 8.0

    daily_attendance = {}
    for record in attendance_records:
        day = record.timestamp.date()
        if day not in daily_attendance:
            daily_attendance[day] = {'check_in': None, 'check_out': None}
        if record.attendance_type == 'check_in' and not daily_attendance[day]['check_in']:
            daily_attendance[day]['check_in'] = record.timestamp
        elif record.attendance_type == 'check_out':
            daily_attendance[day]['check_out'] = record.timestamp

    month_data = []
    total_present = total_absent = total_holiday = 0
    total_working_hours = timedelta()

    for d in range(1, end_day + 1):
        day_date = date(year, month, d)
        weekday = calendar.day_name[day_date.weekday()]
        is_holiday = weekday == "Sunday"

        check_in = daily_attendance.get(day_date, {}).get('check_in')
        check_out = daily_attendance.get(day_date, {}).get('check_out')
        daily_hours = None

        if check_in and check_out:
            daily_hours = check_out - check_in
            total_working_hours += daily_hours
            total_present += 1
        elif is_holiday:
            total_holiday += 1
        else:
            total_absent += 1

        month_data.append({
            'day': d,
            'day_name': weekday[:3],
            'present': bool(check_in and check_out),
            'holiday': is_holiday,
            'daily_hours': str(daily_hours).split('.')[0] if daily_hours else None,
        })

    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    previous_month = month - 1 if month > 1 else 12
    previous_year = year if month > 1 else year - 1

    first_day_offset = (calendar.weekday(year, month, 1) + 1) % 7

    total_hours = total_working_hours.total_seconds() / 3600
    total_expected_hours = (total_present * max_daily_hours)

    context = {
        'employee': employee,
        'year': year,
        'month': month,
        'month_name': month_name[month],
        'data': month_data,
        'next_month': next_month,
        'next_year': next_year,
        'previous_month': previous_month,
        'previous_year': previous_year,
        'day_names': list(calendar.day_name),
        'first_day_offset': first_day_offset,
        'total_present': total_present,
        'total_absent': total_absent,
        'total_holiday': total_holiday,
        'all_employees': Employee.objects.all().order_by('user__first_name'),
        'total_working_hours': round(total_hours, 2),
        'total_expected_hours': round(total_expected_hours, 2),
    }

    return render(request, 'attendance_calendar.html', context)


def attendance_day_detail(request, emp_id, date):
    """
    Returns JSON with all attendance logs for a specific employee and date.
    """
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        return JsonResponse({'error': 'Invalid date format'}, status=400)

    logs = Attendance.objects.filter(
        employee_id=emp_id,
        timestamp__date=date_obj
    ).order_by('timestamp')

    data = {
        'logs': [
            {
                'type': log.get_attendance_type_display(),
                'timestamp': log.timestamp.strftime("%I:%M %p"),
                'location': log.location or '—',
                'confidence': f"{log.confidence_score * 100:.1f}%" if log.confidence_score else '—',
                'image': log.image_capture.url if getattr(log, 'image_capture', None) else '',
                'notes': log.notes or ''
            }
            for log in logs
        ]
    }

    return JsonResponse(data)



# ============================================================
# 🏢 ORGANISATION STRUCTURE VIEWS
# ============================================================

@login_required
def org_chart_view(request, employee_id=None):
    # Use logged-in employee if no ID passed
    if employee_id:
        employee = get_object_or_404(Employee, employee_id=employee_id)
    else:
        employee = request.user.employee

    reporting_chain = employee.get_reporting_chain()

    return render(request, 'ems/org_chart.html', {'reporting_chain': reporting_chain})

# -------- Helpers --------
def build_path_to_root(emp):
    """CEO -> ... -> emp (list of dicts)."""
    path = []
    visited = set()
    cur = emp
    while cur and cur.id not in visited:
        visited.add(cur.id)
        path.append(cur)
        cur = cur.manager
    # reverse for CEO->...->emp
    return list(reversed(path))

def employee_to_dict(emp):
    return {
        "id": emp.id,
        "employee_id": emp.employee_id,
        "name": emp.user.get_full_name() if emp and emp.user else "",
        "position": emp.position or "",
        "department": emp.department.name if emp and emp.department else "",
        "image": (emp.face_image.url if emp.face_image else None),
        "profile_url": reverse('employee_detail', args=[emp.employee_id]),  # <- add this if available
    }

def build_subtree(emp, visited=None, max_depth=10, depth=0):
    """
    Build full subordinate tree for 'emp' up to max_depth to avoid infinite loops.
    """
    if visited is None:
        visited = set()
    if not emp or emp.id in visited or depth > max_depth:
        return None
    visited.add(emp.id)
    node = employee_to_dict(emp)
    # children
    children = []
    # Prefetch is usually handled outside; keeping simple here
    for child in emp.subordinates.all().order_by('position', 'user__first_name'):
        if child.id not in visited:
            sub = build_subtree(child, visited, max_depth, depth+1)
            if sub:
                children.append(sub)
    node["children"] = children
    return node

# -------- Page view --------
@login_required
def org_chart_page(request):
    """
    Renders the page; data comes via AJAX from /api/org-tree/
    """
    return render(request, 'ems/org_chart.html')

# -------- APIs --------
@login_required
def org_tree_api_me(request):
    """
    Returns JSON with:
      - path: CEO -> ... -> you
      - team: you -> all subordinates
    """
    try:
        emp = request.user.employee
    except Employee.DoesNotExist:
        raise Http404("No employee bound to the current user.")

    # Prefetch children to reduce queries
    emp = Employee.objects.select_related('user', 'department', 'manager').prefetch_related(
        Prefetch('subordinates', queryset=Employee.objects.select_related('user', 'department'))
    ).get(pk=emp.pk)

    # Ensure a similar prefetch chain for all nodes in path
    path_emps = build_path_to_root(emp)
    # Build a set of IDs we’ll need to prefetch for subtree root (emp)
    # (already prefetched on emp via subordinates)
    path = [employee_to_dict(e) for e in path_emps]

    team_tree = build_subtree(emp)

    return JsonResponse({"ok": True, "path": path, "team": team_tree})

@login_required
def org_tree_api(request, employee_id):
    """
    Same as above but for any employee_id (admin/HR usage).
    """
    employee = get_object_or_404(Employee.objects.select_related('user', 'department', 'manager'), employee_id=employee_id)
    employee = Employee.objects.select_related('user', 'department', 'manager').prefetch_related(
        Prefetch('subordinates', queryset=Employee.objects.select_related('user', 'department'))
    ).get(pk=employee.pk)

    path_emps = build_path_to_root(employee)
    path = [employee_to_dict(e) for e in path_emps]
    team_tree = build_subtree(employee)

    return JsonResponse({"ok": True, "path": path, "team": team_tree})

def employee_detail(request, employee_id):
    employee = get_object_or_404(Employee, employee_id=employee_id)
    return render(request, 'ems/my_profile.html', {'employee': employee})

@login_required
def department_list(request):
    departments = Department.objects.select_related('head__user', 'parent_department').all()
    
    # Calculate stats for the template
    departments_with_heads = departments.filter(head__isnull=False).count()
    parent_departments = departments.filter(parent_department__isnull=True).count()
    sub_departments = departments.filter(parent_department__isnull=False).count()
    
    return render(request, 'ems/department_list.html', {
        'departments': departments,
        'departments_with_heads': departments_with_heads,
        'parent_departments': parent_departments,
        'sub_departments': sub_departments,
        'title': 'Departments'
    })

@login_required
def department_create(request):
    if request.method == 'POST':
        form = DepartmentForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Department created successfully!')
            return redirect('department_list')
    else:
        form = DepartmentForm()
    
    return render(request, 'ems/department_form.html', {
        'form': form,
        'title': 'Create Department'
    })

@login_required
def department_edit(request, pk):
    department = get_object_or_404(Department, pk=pk)
    if request.method == 'POST':
        form = DepartmentForm(request.POST, instance=department)
        if form.is_valid():
            form.save()
            messages.success(request, 'Department updated successfully!')
            return redirect('department_list')
    else:
        form = DepartmentForm(instance=department)
    
    return render(request, 'ems/department_form.html', {
        'form': form,
        'title': 'Edit Department',
        'department': department
    })

@login_required
def department_delete(request, pk):
    department = get_object_or_404(Department, pk=pk)
    if request.method == 'POST':
        department.delete()
        messages.success(request, 'Department deleted successfully!')
        return redirect('department_list')
    
    return render(request, 'ems/confirm_delete.html', {
        'object': department,
        'title': 'Delete Department'
    })

# ============================================================
# 💰 SALARY & PAYROLL VIEWS
# ============================================================


@login_required
def salary_structure_list(request):
    structures = SalaryStructure.objects.select_related('employee__user')
    return render(request, 'ems/salary_structure_list.html', {
        'structures': structures,
        'title': 'Salary Structures'
    })


@login_required
def salary_structure_create(request):
    if request.method == 'POST':
        form = SalaryStructureForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Salary structure saved successfully.')
            return redirect('salary_structure_list')
    else:
        form = SalaryStructureForm()

    return render(request, 'ems/salary_structure_form.html', {
        'form': form,
        'title': 'Create Salary Structure'
    })



@login_required
@finance_required
def payroll_list(request):
    form = PayrollFilterForm(request.GET or None)
    payrolls = Payroll.objects.select_related(
        'employee__user'
    ).order_by('-month', 'employee__user__first_name')

    if form.is_valid():
        month = form.cleaned_data.get('month')
        year = form.cleaned_data.get('year')
        status = form.cleaned_data.get('status')

        if month:
            payrolls = payrolls.filter(month__month=month)
        if year:
            payrolls = payrolls.filter(month__year=year)
        if status:
            payrolls = payrolls.filter(status=status)

    totals = payrolls.aggregate(
        total_basic=Sum('basic_pay') or 0,
        total_allowances=Sum('allowances') or 0,
        total_deductions=Sum('deductions') or 0,
        total_net=Sum('net_salary') or 0,
    )

    return render(request, 'ems/payroll_list.html', {
        'payrolls': payrolls,
        'form': form,
        'totals': totals,
        'title': 'Payroll'
    })



@login_required
# @finance_required
def generate_payroll(request):
    if request.method == 'POST':
        month = int(request.POST.get('month'))
        year = int(request.POST.get('year'))

        try:
            Payroll.objects.generate_monthly_salary(
                year=year,
                month=month
            )
            messages.success(
                request,
                f"Payroll generated for {calendar.month_name[month]} {year}"
            )
        except Exception as e:
            messages.error(request, f"Error: {e}")

        return redirect('payroll_list')

    active_employees = Employee.objects.filter(
        employment_status='active',
        is_active=True
    ).count()

    return render(request, 'ems/generate_payroll.html', {
        'title': 'Generate Payroll',
        'active_employees_count': active_employees,
    })



@login_required
def my_salary(request):
    try:
        employee = request.user.employee
        payrolls = Payroll.objects.filter(
            employee=employee
        ).order_by('-month')
        structure = SalaryStructure.objects.filter(
            employee=employee
        ).first()
    except Employee.DoesNotExist:
        payrolls = []
        structure = None

    return render(request, 'ems/my_salary.html', {
        'payrolls': payrolls,
        'salary_structure': structure,
        'title': 'My Salary'
    })




@login_required
# @finance_required
def pay_salary(request, pk):
    payroll = get_object_or_404(Payroll, pk=pk)

    if request.method == "POST":
        paid_date = request.POST.get("paid_date")
        amount_paid = request.POST.get("amount_paid")

        payroll.status = "paid"
        payroll.paid_date = paid_date or date.today()

        # -----------------------------
        # Override logic (IMPORTANT)
        # -----------------------------
        if amount_paid:
            amount_paid = Decimal(amount_paid)

            if amount_paid != payroll.calculated_salary:
                payroll.actual_paid_salary = amount_paid
                payroll.net_salary = amount_paid
            else:
                payroll.actual_paid_salary = None
                payroll.net_salary = payroll.calculated_salary

        payroll.save()

        # Re-send payslip if needed
        if payroll.payslip_pdf:
            email_payslip(payroll)

        messages.success(
            request,
            "✅ Salary marked as PAID. Override recorded if applied."
        )
        return redirect("payroll_list")

    return render(request, "ems/pay_salary.html", {
        "payroll": payroll,
        "title": "Pay Salary"
    })




@login_required
def download_salary_slip(request, pk):
    payroll = get_object_or_404(
        Payroll.objects.select_related('employee__user'),
        pk=pk
    )

    # Permission check
    if hasattr(request.user, 'employee'):
        role = request.user.employee.role
        if payroll.employee != request.user.employee and role not in ['Finance', 'Admin']:
            messages.error(request, "Access denied.")
            return redirect('my_salary')
    elif not request.user.is_superuser:
        messages.error(request, "Access denied.")
        return redirect('payroll_list')

    return FileResponse(
        open(payroll.payslip_pdf.path, 'rb'),
        content_type='application/pdf'
    )



@login_required
def payroll_slip_pdf(request, pk):
    payroll = get_object_or_404(
        Payroll.objects.select_related('employee__user'),
        pk=pk
    )

    # -------------------------------
    # Security check
    # -------------------------------
    allow = False
    if hasattr(request.user, 'employee'):
        role = request.user.employee.role
        if role in ['Finance', 'Admin']:
            allow = True
        if payroll.employee_id == request.user.employee.id:
            allow = True

    if not allow and not request.user.is_superuser:
        messages.error(request, "You are not allowed to download this slip.")
        return redirect('payroll_list')

    # -------------------------------
    # Generate PDF using utils/payslip_pdf.py
    # -------------------------------
    pdf_buffer, filename = generate_payslip_pdf(payroll)

    return FileResponse(
        pdf_buffer,
        as_attachment=True,
        filename=filename,
        content_type="application/pdf"
    )


@login_required
# @finance_required
def payroll_expense_chart(request):
    """
    Renders page with Chart.js that fetches data from payroll_expense_api.
    """
    return render(request, 'ems/payroll_expense_chart.html', {'title': 'Payroll Expense'})

@login_required
@finance_required
def payroll_expense_api(request):
    today = date.today()
    labels, totals = [], []

    for i in range(11, -1, -1):
        y = today.year if today.month - i > 0 else today.year - 1
        m = ((today.month - i - 1) % 12) + 1
        labels.append(f"{y}-{str(m).zfill(2)}")

        total = Payroll.objects.filter(
            month__year=y,
            month__month=m
        ).aggregate(
            s=Sum('net_salary')
        )['s'] or 0

        totals.append(float(total))

    return JsonResponse({
        'labels': labels,
        'totals': totals
    })



@login_required
# @finance_required
def employee_salary_history(request, employee_id):
    emp = get_object_or_404(
        Employee.objects.select_related('user'),
        id=employee_id
    )

    qs = Payroll.objects.filter(
        employee=emp
    ).order_by('-month')

    totals = qs.aggregate(
        total_basic=Sum('basic_pay') or 0,
        total_allowances=Sum('allowances') or 0,
        total_deductions=Sum('deductions') or 0,
        total_net=Sum('net_salary') or 0,
    )

    return render(request, 'ems/employee_salary_history.html', {
        'employee': emp,
        'payrolls': qs,
        'totals': totals,
        'title': f"Salary History – {emp.user.get_full_name()}"
    })


# ============================================================
# 📝 LEAVE MANAGEMENT VIEWS
# ============================================================

@login_required
def leave_type_list(request):
    leave_types = LeaveType.objects.all()
    return render(request, 'ems/leave_type_list.html', {
        'leave_types': leave_types,
        'title': 'Leave Types'
    })

@login_required
def leave_type_create(request):
    if request.method == 'POST':
        form = LeaveTypeForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Leave type created successfully!')
            return redirect('leave_type_list')
    else:
        form = LeaveTypeForm()
    
    return render(request, 'ems/leave_type_form.html', {
        'form': form,
        'title': 'Create Leave Type'
    })

@login_required
def leave_application_list(request):
    try:
        employee = request.user.employee
        my_leaves = LeaveApplication.objects.filter(employee=employee).select_related('leave_type').order_by('-applied_on')
        
        # Leaves pending approval (for managers)
        pending_approvals = LeaveApplication.objects.filter(
            approvals__approver=employee,
            approvals__status='pending'
        ).select_related('employee__user', 'leave_type').distinct()
    except Employee.DoesNotExist:
        my_leaves = []
        pending_approvals = []
        messages.error(request, 'Employee profile not found!')
    
    return render(request, 'ems/leave_application_list.html', {
        'my_leaves': my_leaves,
        'pending_approvals': pending_approvals,
        'title': 'Leave Applications'
    })

@login_required
def leave_application_create(request):
    try:
        employee = request.user.employee
    except Employee.DoesNotExist:
        messages.error(request, 'Employee profile not found!')
        return redirect('leave_application_list')

    # Stats (unchanged)
    total_allowed = sum(LeaveType.objects.values_list('max_days_per_year', flat=True))
    used_leave_days = sum(l.total_days() for l in LeaveApplication.objects.filter(employee=employee, status='approved'))
    pending_applications = LeaveApplication.objects.filter(employee=employee, status='pending').count()
    available_leave_days = max(total_allowed - used_leave_days, 0)

    if request.method == 'POST':
        form = LeaveApplicationForm(request.POST)
        if form.is_valid():
            leave_app = form.save(commit=False)
            leave_app.employee = employee
            leave_app.save()

            # 1) choose approver: manager → HR → superuser
            approver = employee.manager
            if not approver:
                approver = Employee.objects.filter(position__icontains="HR", employment_status='active').first()
            if not approver:
                approver = Employee.objects.filter(user__is_superuser=True).first()

            if approver:
                LeaveApproval.objects.create(
                    leave=leave_app,
                    approver=approver,
                    level=1,
                    status='pending'
                )
                # Notify approver
                link = reverse('leave_approval_action', args=[leave_app.pk])
                notify_user(approver.user, f"New leave request from {employee.user.get_full_name()} awaiting your approval.", link)

            # Notify applicant
            notify_user(request.user, "Your leave application has been submitted.", reverse('leave_application_list'))

            messages.success(request, 'Leave submitted and sent to your approver.')
            return redirect('leave_application_list')
    else:
        form = LeaveApplicationForm()

    return render(request, 'ems/leave_application_form.html', {
        'form': form,
        'title': 'Apply for Leave',
        'available_leave_days': available_leave_days,
        'used_leave_days': used_leave_days,
        'pending_applications': pending_applications,
    })

@login_required
def leave_approval_action(request, pk):
    leave_app = get_object_or_404(LeaveApplication, pk=pk)

    # ✅ Ensure current user is an approver
    try:
        approver = request.user.employee
    except Employee.DoesNotExist:
        messages.error(request, 'Employee profile not found!')
        return redirect('leave_application_list')

    approval = LeaveApproval.objects.filter(
        leave=leave_app,
        approver=approver,
        status='pending'
    ).first()

    if not approval:
        messages.error(request, 'You are not authorized to approve this leave!')
        return redirect('leave_application_list')

    if request.method == 'POST':
        form = LeaveApprovalForm(request.POST, instance=approval)
        if form.is_valid():
            approval_obj = form.save(commit=False)
            approval_obj.acted_at = timezone.now()
            approval_obj.save()

            # ✅ APPROVED CASE
            if approval_obj.status == 'approved':
                next_stage = LeaveWorkflowStage.objects.filter(level=approval.level + 1).first()

                if next_stage:
                    # Try to find next approver
                    next_approver = Employee.objects.filter(
                        employment_status='active',
                        position__icontains=next_stage.role_name
                    ).first()

                    if next_approver:
                        LeaveApproval.objects.create(
                            leave=leave_app,
                            approver=next_approver,
                            level=next_stage.level,
                            status='pending'
                        )
                        notify_user(next_approver.user,
                                    f"Leave request for {leave_app.employee.user.get_full_name()} moved to you for approval.",
                                    reverse('leave_approval_action', args=[leave_app.pk]))
                    else:
                        # ✅ No approver found for next level → Final approval
                        leave_app.status = 'approved'
                        leave_app.approved_by = approver
                        leave_app.save()
                else:
                    # ✅ No next stage → Final approval
                    leave_app.status = 'approved'
                    leave_app.approved_by = approver
                    leave_app.save()

                notify_user(leave_app.employee.user, "Your leave has been approved.", reverse('leave_application_list'))

            # ✅ REJECTED CASE
            elif approval_obj.status == 'rejected':
                leave_app.status = 'rejected'
                leave_app.save()
                notify_user(leave_app.employee.user, "Your leave has been rejected.", reverse('leave_application_list'))

            messages.success(request, f'Leave {approval_obj.status} successfully!')
            return redirect('leave_application_list')

    else:
        form = LeaveApprovalForm(instance=approval)

    return render(request, 'ems/leave_approval_form.html', {
        'form': form,
        'leave_app': leave_app,
        'title': 'Approve Leave'
    })


@login_required
def leave_workflow_list(request):
    stages = LeaveWorkflowStage.objects.all().order_by('level')
    return render(request, 'ems/leave_workflow_list.html', {
        'stages': stages,
        'title': 'Leave Workflow Stages'
    })

@login_required
def leave_workflow_create(request):
    if request.method == 'POST':
        form = LeaveWorkflowStageForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Workflow stage created successfully!')
            return redirect('leave_workflow_list')
    else:
        form = LeaveWorkflowStageForm()
    
    return render(request, 'ems/leave_workflow_form.html', {
        'form': form,
        'title': 'Create Workflow Stage'
    })

@login_required
def leave_application_update(request, pk):
    leave_app = get_object_or_404(LeaveApplication, pk=pk, employee=request.user.employee)

    if request.method == 'POST':
        form = LeaveApplicationForm(request.POST, instance=leave_app)
        if form.is_valid():
            leave = form.save(commit=False)

            # ✅ Reset status to pending on update
            leave.status = 'pending'
            leave.approved_by = None
            leave.save()

            # ✅ Remove old workflow approvals
            LeaveApproval.objects.filter(leave=leave).delete()

            # ✅ Reassign to manager/HR
            approver = leave.employee.manager or \
                       Employee.objects.filter(position__icontains="HR").first() or \
                       Employee.objects.filter(user__is_superuser=True).first()

            if approver:
                LeaveApproval.objects.create(
                    leave=leave,
                    approver=approver,
                    level=1,
                    status='pending'
                )

            messages.success(request, "Leave updated and sent for approval again.")
            return redirect('leave_application_list')

    else:
        form = LeaveApplicationForm(instance=leave_app)

    return render(request, 'ems/leave_application_form.html', {
        'form': form,
        'title': 'Update Leave Application'
    })


@login_required
def my_pending_approvals(request):
    try:
        me = request.user.employee
    except Employee.DoesNotExist:
        messages.error(request, "Employee profile not found.")
        return redirect('leave_application_list')

    approvals = LeaveApproval.objects.filter(
        approver=me,
        status='pending'
    ).select_related('leave__employee__user', 'leave__leave_type').order_by('-leave__applied_on')

    return render(request, 'ems/my_pending_approvals.html', {
        'approvals': approvals,
        'title': 'My Approvals'
    })

@login_required
def leave_application_detail(request, pk):
    leave_app = get_object_or_404(
        LeaveApplication.objects.select_related('employee__user', 'leave_type'),
        pk=pk
    )
    history = leave_app.approvals.select_related('approver__user').order_by('level', 'acted_at', 'id')
    return render(request, 'ems/leave_application_detail.html', {
        'leave': leave_app,
        'history': history,
        'title': 'Leave Details'
    })
# ============================================================
# 📄 JOINING & RESIGNATION VIEWS
# ============================================================

@login_required
def joining_detail_list(request):
    joining_details = JoiningDetail.objects.select_related('employee__user').prefetch_related('joining_documents')
    return render(request, 'ems/joining_detail_list.html', {
        'joining_details': joining_details,
        'title': 'Joining Details'
    })


# ✅ Download Individual Document
@login_required
def download_document(request, doc_id):
    doc = get_object_or_404(JoiningDocument, id=doc_id)
    file_path = doc.file.path

    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=os.path.basename(file_path))
    else:
        messages.error(request, "File not found.")
        return redirect('joining_detail_list')

# ✅ Delete a Document
@login_required
def delete_document(request, doc_id):
    doc = get_object_or_404(JoiningDocument, id=doc_id)

    if request.user.is_staff:  # Only admin or HR can delete
        file_path = doc.file.path
        doc.delete()
        if os.path.exists(file_path):
            os.remove(file_path)
        messages.success(request, "Document deleted successfully.")
    else:
        messages.error(request, "Permission denied.")

    return redirect('joining_detail_list')

# ✅ Download all documents as a ZIP
@login_required
def download_all_documents(request, detail_id):
    detail = get_object_or_404(JoiningDetail, id=detail_id)
    docs = detail.joining_documents.all()

    if not docs:
        messages.warning(request, "No documents to download.")
        return redirect('joining_detail_list')

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        for doc in docs:
            file_path = doc.file.path
            zip_file.write(file_path, os.path.basename(file_path))

    buffer.seek(0)
    response = HttpResponse(buffer, content_type='application/zip')
    response['Content-Disposition'] = f'attachment; filename=documents_{detail.employee.employee_id}.zip'
    return response


@login_required
def joining_detail_create(request):
    if request.method == 'POST':
        form = JoiningDetailForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Joining details added successfully!')
            return redirect('joining_detail_list')
    else:
        form = JoiningDetailForm()
    
    return render(request, 'ems/joining_detail_form.html', {
        'form': form,
        'title': 'Add Joining Details'
    })

@login_required
def resignation_list(request):
    try:
        employee = request.user.employee
        my_resignations = Resignation.objects.filter(employee=employee)
        
        # Resignations pending approval (for managers)
        if request.user.is_staff:
            pending_resignations = Resignation.objects.filter(
                approval_status='pending'
            ).select_related('employee__user')
        else:
            pending_resignations = []
    except Employee.DoesNotExist:
        my_resignations = []
        pending_resignations = []
        messages.error(request, 'Employee profile not found!')
    
    return render(request, 'ems/resignation_list.html', {
        'my_resignations': my_resignations,
        'pending_resignations': pending_resignations,
        'title': 'Resignations'
    })

@login_required
def resignation_create(request):
    try:
        employee = request.user.employee
    except Employee.DoesNotExist:
        messages.error(request, 'Employee profile not found!')
        return redirect('resignation_list')
    
    if request.method == 'POST':
        form = ResignationForm(request.POST)
        if form.is_valid():
            resignation = form.save(commit=False)
            resignation.employee = employee
            resignation.save()
            messages.success(request, 'Resignation submitted successfully!')
            return redirect('resignation_list')
    else:
        form = ResignationForm(initial={'employee': employee})
    
    return render(request, 'ems/resignation_form.html', {
        'form': form,
        'title': 'Submit Resignation'
    })

@login_required
def resignation_approve(request, pk):
    resignation = get_object_or_404(Resignation, pk=pk)
    if request.method == 'POST':
        action = request.POST.get('action')
        if action in ['approved', 'rejected']:
            resignation.approval_status = action
            resignation.approved_by = request.user.employee
            resignation.save()
            
            # Update employee status if approved
            if action == 'approved':
                employee = resignation.employee
                employee.employment_status = 'resigned'
                employee.date_of_resignation = resignation.resignation_date
                employee.is_active = False
                employee.save()
            
            messages.success(request, f'Resignation {action} successfully!')
    
    return redirect('resignation_list')

# ============================================================
# 🔔 NOTIFICATION VIEWS
# ============================================================

@login_required
def notification_list(request):
    notifications = Notification.objects.filter(recipient=request.user).order_by('-created_at')
    unread_count = notifications.filter(is_read=False).count()
    
    return render(request, 'ems/notification_list.html', {
        'notifications': notifications,
        'unread_count': unread_count,
        'title': 'Notifications'
    })

@login_required
def mark_notification_read(request, pk):
    notification = get_object_or_404(Notification, pk=pk, recipient=request.user)
    notification.is_read = True
    notification.save()
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'success': True})
    
    return redirect('notification_list')

@login_required
def mark_all_notifications_read(request):
    Notification.objects.filter(recipient=request.user, is_read=False).update(is_read=True)
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'success': True})
    
    return redirect('notification_list')



def notify_user(user: User, message: str, link_path: str = ""):
    # In-app notification
    Notification.objects.create(
        recipient=user,
        message=message,
        link=link_path or None
    )
    # Email (best-effort)
    if getattr(settings, "EMAIL_HOST", None) and user.email:
        try:
            send_mail(
                subject="EMS Notification",
                message=message + (f"\n\nOpen: {link_path}" if link_path else ""),
                from_email=getattr(settings,"avaneeshpathak900@gmail.com"),
                recipient_list=[user.email],
                fail_silently=True,
            )
        except Exception:
            pass


# =========================================================
# FORM (kept in views.py as requested)
# =========================================================
class WorkRuleForm(ModelForm):
    class Meta:
        model = WorkRule
        fields = "__all__"


# =========================================================
# LIST WORK RULES
# =========================================================
@login_required
@finance_required
def workrule_list(request):
    rules = WorkRule.objects.all().order_by("name")
    return render(request, "ems/workrule_list.html", {
        "rules": rules,
        "title": "Work Rules"
    })


# =========================================================
# CREATE WORK RULE
# =========================================================
@login_required
@finance_required
def workrule_create(request):
    if request.method == "POST":
        form = WorkRuleForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Work rule created successfully.")
            return redirect("workrule_list")
    else:
        form = WorkRuleForm()

    return render(request, "ems/workrule_form.html", {
        "form": form,
        "title": "Create Work Rule"
    })


# =========================================================
# UPDATE WORK RULE
# =========================================================
@login_required
@finance_required
def workrule_update(request, pk):
    rule = get_object_or_404(WorkRule, pk=pk)

    if request.method == "POST":
        form = WorkRuleForm(request.POST, instance=rule)
        if form.is_valid():
            form.save()
            messages.success(request, "Work rule updated successfully.")
            return redirect("workrule_list")
    else:
        form = WorkRuleForm(instance=rule)

    return render(request, "ems/workrule_form.html", {
        "form": form,
        "rule": rule,
        "title": "Update Work Rule"
    })

@login_required
@finance_required
def workrule_delete(request, pk):
    rule = get_object_or_404(WorkRule, pk=pk)

    # ----------------------------------
    # BLOCK DELETE IF RULE IS ASSIGNED
    # ----------------------------------
    if rule.employees.exists():
        messages.error(
            request,
            "Cannot delete this work rule because it is assigned to one or more employees."
        )
        return redirect("workrule_list")

    # ----------------------------------
    # CONFIRMATION + DELETE
    # ----------------------------------
    if request.method == "POST":
        rule.delete()
        messages.success(request, "Work rule deleted successfully.")
        return redirect("workrule_list")

    return render(request, "ems/workrule_confirm_delete.html", {
        "rule": rule,
        "title": "Delete Work Rule"
    })




