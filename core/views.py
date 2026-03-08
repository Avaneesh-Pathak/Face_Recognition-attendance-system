import os
from urllib import request
import cv2
import ast
import json
import io
import zipfile
import base64
from core.models import get_employee_rule
from django.db import transaction
import json 
import logging
import tempfile
import calendar
import openpyxl
from django import forms
import numpy as np
from django.forms import ModelForm
from django.utils import timezone
from io import BytesIO
from datetime import date, datetime, timedelta
from calendar import monthrange, month_name
from PIL import Image
from django.utils import timezone
from django.contrib.auth import update_session_auth_hash
from django.db import IntegrityError
from datetime import datetime, date, time
import calendar
from django.db.models import Prefetch
from django.core.paginator import Paginator
from django.http import JsonResponse, Http404
from django.urls import reverse
from decimal import Decimal, InvalidOperation
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
from django.views.decorators.csrf import csrf_exempt
from django.db.utils import ProgrammingError, OperationalError
from .models import (Employee, Attendance, AttendanceSettings, DailyReport,Department, SalaryStructure, Payroll, LeaveType, 
    LeaveApplication, LeaveWorkflowStage, LeaveApproval,
    JoiningDetail, Resignation, Notification,JoiningDocument,WorkRule,OfficeLocation)

from .forms import ( UserRegistrationForm, EmployeeRegistrationForm, AttendanceSettingsForm,DepartmentForm, SalaryStructureForm, PayrollFilterForm,
    LeaveTypeForm, LeaveApplicationForm, LeaveApprovalForm, LeaveWorkflowStageForm,
    JoiningDetailForm, ResignationForm, NotificationForm,PayrollFilterForm
    )
from decimal import Decimal, ROUND_HALF_UP
import calendar
from django.db.models import Sum
from django.views.decorators.http import require_POST, require_GET
from django.shortcuts import get_object_or_404, render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from datetime import timedelta
from decimal import Decimal, InvalidOperation
import json

from core.models import (
    Employee,
    JoiningDetail,
    JoiningDocument,
    SalaryStructure
)
from core.face_system import get_face_system
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
from django.core.exceptions import ObjectDoesNotExist
from .models import (
    SalaryStructure, Payroll, Employee
)
from .forms import SalaryStructureForm, PayrollFilterForm
from .decorators import finance_required
from django.db.models import Sum, Q

from django.db.models import Sum
from django.utils.timezone import now
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden
from core.utils.payslip_pdf import generate_payslip_pdf
from core.utils.payslip_email import email_payslip
from core.face_system import get_face_system
from core.models import OfficeLocation
from core.utils.location import is_inside_office

from django.template.loader import render_to_string
from django.contrib.auth.decorators import login_required
from weasyprint import HTML
from django.http import HttpResponse
from django.template.loader import get_template
from xhtml2pdf import pisa
from django.shortcuts import get_object_or_404
from core.utils.access import get_visible_employees
# Setup logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler('registration.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

logger.setLevel(logging.INFO)

# Initialize face system (lazy init inside register also)



try:
    from core.face_system import get_face_system as get_if_system
except Exception:
    get_if_system = lambda: None


# Liveness (your module is named livenss.py)



def home(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return redirect('login')



# =====================================================
# 📝 REGISTRATION VIEW (FINAL FIXED)
# =====================================================

def register(request):
    if request.method == "POST":
        form = EmployeeRegistrationForm(request.POST)

        if form.is_valid():
            try:
                user = User.objects.create_user(
                    username=form.cleaned_data["username"],
                    email=form.cleaned_data["email"],
                    password=form.cleaned_data["password1"],
                    first_name=form.cleaned_data["first_name"],
                    last_name=form.cleaned_data["last_name"],
                )

                employee = Employee.objects.create(
                    user=user,
                    department=form.cleaned_data["department"],
                    position=form.cleaned_data["position"],
                    manager=form.cleaned_data["manager"],
                    phone_number=form.cleaned_data["phone_number"],
                    role=form.cleaned_data["role"],
                    work_rule=form.cleaned_data.get("work_rule"),
                    location_type=form.cleaned_data["location_type"],
                    assigned_location=form.cleaned_data.get("assigned_location"),
                )

                SalaryStructure.objects.create(
                    employee=employee,
                    base_salary=form.cleaned_data.get("base_salary") or Decimal("0.00"),
                    hra=form.cleaned_data.get("hra") or Decimal("0.00"),
                    allowances=form.cleaned_data.get("allowances") or Decimal("0.00"),
                    deductions=form.cleaned_data.get("deductions") or Decimal("0.00"),
                )


                face_image = request.FILES.get("face_image")

                if not face_image and request.POST.get("captured_image"):
                    fmt, imgstr = request.POST["captured_image"].split(";base64,")
                    ext = fmt.split("/")[-1]
                    face_image = ContentFile(
                        base64.b64decode(imgstr),
                        name=f"face_{employee.employee_id}.{ext}",
                    )

                if face_image:
                    employee.face_image = face_image
                    employee.save(update_fields=["face_image"])

                if employee.face_image:
                    face_sys = get_face_system()
                    emb = face_sys.get_embedding(employee.face_image.path)

                    if emb is None:
                        employee.face_encoding = json.dumps({"status": "FACE_PENDING"})
                        employee.save(update_fields=["face_encoding"])
                        messages.warning(
                            request,
                            "Face not registered. You can add face ID later."
                        )
                    else:
                        employee.face_encoding = json.dumps([emb])
                        employee.save(update_fields=["face_encoding"])
                        face_sys.load_from_db()

                messages.success(request, "Employee registered successfully")
                return redirect("employee_list")

            except Exception as e:
                logger.exception("Registration failed")
                messages.error(request, str(e))

    else:
        form = EmployeeRegistrationForm()

    return render(request, "register.html", {"form": form})


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
def employee_list(request):
    # 🔒 BASE QUERYSET (SECURITY LAYER)
    employees = get_visible_employees(request.user).select_related(
        "user", "department", "salary_structure"
    )

    # ==========================
    # READ FILTER PARAMS
    # ==========================
    department_id = request.GET.get("department")
    role = request.GET.get("role")
    sort_salary = request.GET.get("sort_salary")

    # ==========================
    # APPLY FILTERS
    # ==========================
    if department_id:
        employees = employees.filter(department_id=department_id)

    if role:
        employees = employees.filter(role=role)

    # ==========================
    # SALARY SORT
    # ==========================
    if sort_salary == "highest":
        employees = employees.order_by("-salary_structure__base_salary")
    elif sort_salary == "lowest":
        employees = employees.order_by("salary_structure__base_salary")

    # ==========================
    # KPIs (AFTER FILTERING)
    # ==========================
    total_employees = employees.count()

    active_employees = employees.filter(
        employment_status="Active"
    ).count()

    inactive_employees = employees.exclude(
        employment_status="Active"
    ).count()

    total_payroll = employees.aggregate(
        total=Sum("salary_structure__base_salary")
    )["total"] or 0

    # ==========================
    # CONTEXT REQUIRED BY TEMPLATE
    # ==========================
    context = {
        "employees": employees,
        "departments": Department.objects.all(),
        "roles": Employee.ROLE_CHOICES,

        # keep dropdown selections
        "selected_dept": department_id,
        "selected_role": role,
        "selected_sort": sort_salary,

        "kpi": {
            "total": total_employees,
            "active": active_employees,
            "inactive": inactive_employees,
            "payroll": total_payroll,
        }
    }

    return render(request, "ems/employee_list.html", context)



class EmployeeUpdateForm(forms.ModelForm):
    face_image = forms.ImageField(required=False)
    remove_face = forms.BooleanField(required=False)
     # 🔐 USER FIELDS (NEW)
    username = forms.CharField(required=True)
    first_name = forms.CharField(required=False)
    last_name = forms.CharField(required=False)
    email = forms.EmailField(required=False)

    password = forms.CharField(
        required=False,
        widget=forms.PasswordInput,
        help_text="Leave blank to keep current password"
    )
    class Meta:
        model = Employee
        fields = [
            'department',
            'position',
            'manager',
            'phone_number',
            'employment_status',
            'work_rule',
            'is_active',
        ]
#=====================
# SAFE DECIMAL HELPER
# =====================
def safe_decimal(value, default=Decimal("0.00")):
    try:
        if value in (None, "", " "):
            return default
        return Decimal(value)
    except (InvalidOperation, TypeError):
        return default


@login_required
def employee_update(request, pk):
    employee = get_object_or_404(Employee, pk=pk)
    user = employee.user

    joining = getattr(employee, "joiningdetail", None)
    salary = getattr(employee, "salary_structure", None)

    if request.method == "POST":
        form = EmployeeUpdateForm(
            request.POST,
            request.FILES,
            instance=employee
        )

        if form.is_valid():

            # 🔐 ONE TRANSACTION = SAFE SYSTEM
            with transaction.atomic():

                # =====================
                # UPDATE USER
                # =====================
                new_username = request.POST.get("username", "").strip()
                new_password = request.POST.get("password", "")
                print("USERNAME:", new_username)
                print("PASSWORD PROVIDED:", bool(new_password))
                # 🔐 Username update (safe)
                if new_username and new_username != user.username:
                    if User.objects.exclude(pk=user.pk).filter(username=new_username).exists():
                        messages.error(request, "Username already exists.")
                        return redirect("employee_update", pk=employee.pk)
                    user.username = new_username

                user.first_name = request.POST.get("first_name", user.first_name)
                user.last_name = request.POST.get("last_name", user.last_name)
                user.email = request.POST.get("email", user.email)

                # 🔐 Password update (secure)
                if new_password:
                    user.set_password(new_password)
                    update_session_auth_hash(request, user)  # 🔥 THIS IS REQUIRED

                user.save()

                # =====================
                # UPDATE EMPLOYEE
                # =====================
                employee = form.save(commit=False)

                # =====================
                # FACE IMAGE HANDLING
                # =====================
                if form.cleaned_data.get("remove_face"):
                    if employee.face_image:
                        employee.face_image.delete(save=False)
                    employee.face_image = None
                    employee.face_encoding = None

                if request.FILES.get("face_image"):
                    if employee.face_image:
                        employee.face_image.delete(save=False)

                    employee.face_image = request.FILES["face_image"]
                    employee.save(update_fields=["face_image"])

                    face_sys = get_face_system()
                    emb = face_sys.get_embedding(employee.face_image.path)

                    if emb is None:
                        employee.face_encoding = json.dumps({"status": "FACE_PENDING"})
                    else:
                        employee.face_encoding = json.dumps([emb])
                        face_sys.load_from_db()
                    
                    employee.save(update_fields=["face_encoding"])

                employee.save()

                # =====================
                # SALARY UPDATE
                # =====================
                salary, _ = SalaryStructure.objects.get_or_create(
                    employee=employee,
                    defaults={
                        "base_salary": Decimal("0.00"),
                        "hra": Decimal("0.00"),
                        "allowances": Decimal("0.00"),
                        "deductions": Decimal("0.00"),
                    }
                )

                salary.base_salary = safe_decimal(
                    request.POST.get("base_salary"),
                    salary.base_salary
                )
                salary.hra = safe_decimal(request.POST.get("hra"))
                salary.allowances = safe_decimal(request.POST.get("allowances"))
                salary.deductions = safe_decimal(request.POST.get("deductions"))
                salary.save()

                # =====================
                # JOINING DETAIL
                # =====================
                joining, _ = JoiningDetail.objects.get_or_create(
                    employee=employee
                )

                # =====================
                # MULTIPLE DOCUMENT UPLOAD
                # =====================
                files = request.FILES.getlist("documents")
                for file in files:
                    JoiningDocument.objects.create(
                        joining=joining,
                        file=file
                    )

                # =====================
                # DOCUMENT DELETE
                # =====================
                delete_ids = request.POST.getlist("delete_docs")
                if delete_ids:
                    JoiningDocument.objects.filter(
                        id__in=delete_ids,
                        joining=joining
                    ).delete()

            messages.success(request, "Employee updated successfully.")
            if request.user == employee.user:
                # Employee updating his own profile
                return redirect("my_profile")
            else:
                # HR / Admin / Manager updating others
                return redirect("employee_list")

    else:
        form = EmployeeUpdateForm(instance=employee)

    documents = joining.joining_documents.all() if joining else []

    return render(request, "ems/employee_form.html", {
        "form": form,
        "employee": employee,
        "user_obj": user,
        "joining": joining,
        "documents": documents,
        "salary": salary,
    })



@login_required
@admin_required
def employee_delete(request, pk):
    employee = get_object_or_404(Employee, pk=pk)
    user = employee.user

    if request.method == "POST":
        # delete all files explicitly
        if employee.face_image:
            employee.face_image.delete(save=False)

        for doc in JoiningDocument.objects.filter(joining__employee=employee):
            doc.delete()

        user.delete()  # ✅ cascades Employee

        messages.success(request, "Employee deleted permanently")
        return redirect("employee_list")

    return render(request, "ems/employee_confirm_delete.html", {
        "employee": employee
    })

@login_required
def dashboard(request):
    user = request.user
    today = timezone.localdate()

    # ----------------------------------
    # 🔐 ROLE-BASED EMPLOYEE VISIBILITY
    # ----------------------------------
    visible_employees = get_visible_employees(user)

    is_admin = user.is_superuser or (
        hasattr(user, "employee") and user.employee.role in ["Admin", "HR", "Finance"]
    )

    # ----------------------------------
    # EMPLOYEE CONTEXT
    # ----------------------------------
    emp_id = request.GET.get("emp_id")

    if emp_id:
        if not visible_employees.filter(id=emp_id).exists():
            return HttpResponseForbidden("Not allowed to view this employee")

        ems_emp = visible_employees.get(id=emp_id)
    else:
        ems_emp = user.employee if hasattr(user, "employee") else None

    # ----------------------------------
    # ATTENDANCE SCOPE
    # ----------------------------------
    attendance_qs = Attendance.objects.filter(
        employee__in=[ems_emp] if ems_emp else visible_employees
    )

    # ----------------------------------
    # DATE RANGES
    # ----------------------------------
    start_week = today - timedelta(days=today.weekday())
    start_month = today.replace(day=1)

    today_records = attendance_qs.filter(timestamp__date=today)
    week_records = attendance_qs.filter(timestamp__date__gte=start_week)
    month_records = attendance_qs.filter(timestamp__date__gte=start_month)

    # ----------------------------------
    # BASIC METRICS
    # ----------------------------------
    avg_confidence = attendance_qs.aggregate(
        avg=Avg("confidence_score")
    )["avg"] or 0

    total_employees = visible_employees.count()

    # ----------------------------------
    # SINGLE EMPLOYEE INSIGHTS
    # ----------------------------------
    current_status = None
    total_hours_today = 0
    total_hours_week = 0
    total_hours_month = 0
    last_checkin = None
    late_days = 0
    attendance_days = 0

    if ems_emp:
        rule = ems_emp.work_rule

        emp_attendance = Attendance.objects.filter(employee=ems_emp)

        # ---- Today status
        today_logs = emp_attendance.filter(timestamp__date=today).order_by("timestamp")
        last_checkin = today_logs.filter(attendance_type="check_in").last()
        last_checkout = today_logs.filter(attendance_type="check_out").last()

        if last_checkin and (not last_checkout or last_checkout.timestamp < last_checkin.timestamp):
            current_status = "Checked In"
        elif last_checkout:
            current_status = "Checked Out"
        else:
            current_status = "Not Checked In"

        # ---- Hours calculation helper
        def calc_hours(qs):
            seconds = 0
            checkins = qs.filter(attendance_type="check_in")
            checkouts = qs.filter(attendance_type="check_out")
            for ci in checkins:
                co = checkouts.filter(timestamp__gt=ci.timestamp).first()
                if co:
                    seconds += (co.timestamp - ci.timestamp).total_seconds()
            return round(seconds / 3600, 2)

        total_hours_today = calc_hours(today_logs)
        total_hours_week = calc_hours(emp_attendance.filter(timestamp__date__gte=start_week))
        total_hours_month = calc_hours(emp_attendance.filter(timestamp__date__gte=start_month))

        # ---- Attendance percentage (month)
        days_passed = today.day
        attendance_days = (
            emp_attendance
            .filter(timestamp__date__gte=start_month, attendance_type="check_in")
            .values("timestamp__date")
            .distinct()
            .count()
        )

        attendance_percentage = round(
            (attendance_days / days_passed) * 100, 1
        ) if days_passed else 0

        # ---- Leave stats
        leave_stats = (
            LeaveApplication.objects
            .filter(employee=ems_emp, start_date__year=today.year)
            .values("status")
            .annotate(count=Count("id"))
        )

        # ---- Payroll snapshot
        recent_payroll = (
            Payroll.objects
            .filter(employee=ems_emp)
            .order_by("-month")[:6]
        )

    else:
        attendance_percentage = 0
        leave_stats = None
        recent_payroll = None

    # ----------------------------------
    # 30-DAY ATTENDANCE CHART
    # ----------------------------------
    start_30 = today - timedelta(days=29)
    att_30 = (
        attendance_qs
        .filter(timestamp__date__gte=start_30, attendance_type="check_in")
        .annotate(day=TruncDate("timestamp"))
        .values("day")
        .annotate(count=Count("id"))
    )

    att_map = {a["day"]: a["count"] for a in att_30}
    chart_labels = [(start_30 + timedelta(days=i)).strftime("%d %b") for i in range(30)]
    chart_values = [att_map.get(start_30 + timedelta(days=i), 0) for i in range(30)]

    # ----------------------------------
    # CONTEXT
    # ----------------------------------
    context = {
        "is_admin": is_admin,
        "employees": visible_employees.order_by("user__first_name"),
        "employee": ems_emp,
        "selected_emp_id": ems_emp.id if ems_emp else None,

        # attendance
        "today_attendance": today_records,
        "week_attendance": week_records,
        "month_attendance": month_records,

        # metrics
        "avg_confidence": round(avg_confidence, 2),
        "current_status": current_status,
        "last_checkin": last_checkin,
        "total_hours": total_hours_today,
        "total_hours_week": total_hours_week,
        "total_hours_month": total_hours_month,
        "attendance_percentage": attendance_percentage,
        "total_employees": total_employees,

        # employee panels
        "leave_stats": leave_stats,
        "recent_payroll": recent_payroll,

        # charts
        "attendance_labels_json": json.dumps(chart_labels),
        "attendance_values_json": json.dumps(chart_values),

        "year": today.year,
        "month": today.month,
        "title": "Dashboard",
    }

    return render(request, "dashboard.html", context)

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
@hr_required
def employee_profile(request, pk):
    employee = get_object_or_404(
        Employee.objects.select_related(
            "user",
            "department",
            "manager",
            "work_rule"
        ),
        pk=pk
    )
    today = date.today()
    salary = getattr(employee, "salary_structure", None)
    joining = getattr(employee, "joiningdetail", None)
    documents = joining.joining_documents.all() if joining else []

    return render(request, "ems/employee_profile.html", {
        "employee": employee,
        "salary": salary,
        "joining": joining,
        "documents": documents,
        "year": today.year,   
        "month": today.month, 
    })



@login_required
@hr_required # Assuming this is your custom decorator
def employee_id_card(request, pk):
    # Retrieve data
    employee = get_object_or_404(
        Employee.objects.select_related(
            "user", "department", "manager", "work_rule"
        ),
        pk=pk
    )

    # Render context into HTML string
    html_string = render_to_string(
        "ems/employee_id_card.html",
        {"employee": employee}
    )

    # Prepare response headers
    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = f'inline; filename="ID_{employee.employee_id}.pdf"'

    # Generate the PDF
    # base_url is critical here; it ensures images properly map to their static/media routes!
    HTML(
        string=html_string,
        base_url=request.build_absolute_uri("/")
    ).write_pdf(response)

    return response

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

import time

def attendance_heartbeat(request):
    return JsonResponse({
        "status": "ok",
        "ts": time.time()
    })

# ------------------ ✅ Final mark_attendance Endpoint ------------------
# ============================================================
# 🔁 SHIFT WINDOW (DAY + NIGHT SAFE + BUFFER FIX)
# ============================================================
def get_shift_window(time_ref, rule):
    """
    time_ref: Can be a `datetime` (from mark_attendance) or `date` (from Payroll).
    Includes a buffer so early check-ins and late check-outs are caught!
    """
    start_time = rule.shift_start_time
    end_time = rule.shift_end_time
    
    # Safely extract the date whether time_ref is datetime or date
    today = time_ref.date() if isinstance(time_ref, datetime) else time_ref

    if end_time <= start_time:
        # Night shift (crosses midnight)
        shift_start = datetime.combine(today, start_time)
        shift_end = datetime.combine(today + timedelta(days=1), end_time)

        # If we are checking during the early AMs of a night shift
        if isinstance(time_ref, datetime) and time_ref.time() < end_time:
            shift_start -= timedelta(days=1)
            shift_end -= timedelta(days=1)
    else:
        # Day shift
        shift_start = datetime.combine(today, start_time)
        shift_end = datetime.combine(today, end_time)

    # 🚨 CRITICAL FIX: Add Buffer Time 🚨
    # Look for records 3 hours early and up to 8 hours late (for overtime)
    shift_start -= timedelta(hours=3)
    shift_end += timedelta(hours=8)

    # Ensure datetimes are timezone aware
    if timezone.is_naive(shift_start):
        shift_start = timezone.make_aware(shift_start)
    if timezone.is_naive(shift_end):
        shift_end = timezone.make_aware(shift_end)

    return shift_start, shift_end

def get_open_checkin(employee):
    """
    Returns the latest check-in that does not yet have a checkout.
    Works perfectly for night shifts and day shifts.
    """
    last_checkin = Attendance.objects.filter(
        employee=employee,
        attendance_type="check_in"
    ).order_by("-timestamp").first()

    if not last_checkin:
        return None

    has_checkout = Attendance.objects.filter(
        employee=employee,
        attendance_type="check_out",
        timestamp__gt=last_checkin.timestamp
    ).exists()

    return None if has_checkout else last_checkin

# ============================================================
# 📸 MARK ATTENDANCE
@csrf_exempt
@login_required
def mark_attendance(request):
    if request.method != "POST":
        return JsonResponse({"success": False, "message": "Method not allowed"}, status=405)

    if "capture" not in request.FILES:
        return JsonResponse({"success": False, "message": "No image data"}, status=400)

    try:
        # ----------------------------------------------------
        # 1️⃣ Decode Image
        # ----------------------------------------------------
        file_bytes = np.frombuffer(request.FILES["capture"].read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            return JsonResponse({"success": False, "message": "Invalid image"}, status=400)

        # ----------------------------------------------------
        # 2️⃣ Face Recognition
        # ----------------------------------------------------
        fs = get_face_system()
        fs.load_from_db()

        emp_id, score, meta = fs.recognize_from_frame(frame)

        if not emp_id:
            reason = meta.get("reason", "")
            msg = "Face not recognized"
            color = "red"

            if reason == "TOO_FAR":
                msg = "Come Closer"
                color = "yellow"
            elif reason == "ACCUMULATING":
                msg = "Hold Still..."
                color = "blue"
            elif reason == "NO_FACE":
                msg = "No Face Detected"

            return JsonResponse({
                "success": False,
                "message": msg,
                "color": color
            })

        # ----------------------------------------------------
        # 3️⃣ Employee Validation
        # ----------------------------------------------------
        employee = Employee.objects.filter(
            employee_id=emp_id,
            is_active=True
        ).select_related("work_rule", "user", "assigned_location").first()

        if not employee:
            return JsonResponse({
                "success": False,
                "message": "Employee not found",
                "color": "red"
            })
        
        is_flexible = bool(
            employee.work_rule and employee.work_rule.flexible_attendance
        )
        # ----------------------------------------------------
        # 📍 Location Validation
        # ----------------------------------------------------
        lat = request.POST.get("latitude")
        lng = request.POST.get("longitude")

        if not lat or not lng:
            return JsonResponse({
                "success": False,
                "message": "Location permission required",
                "color": "red"
            })

        lat = float(lat)
        lng = float(lng)

        # 🔒 Employee must have assigned location (INDOOR + OUTDOOR)
        if not employee.assigned_location:
            return JsonResponse({
                "success": False,
                "message": "No office location assigned to your profile",
                "color": "red"
            })

        active_offices = OfficeLocation.objects.filter(is_active=True)

        matched_office = next(
            (o for o in active_offices if is_inside_office(lat, lng, o)),
            None
        )

        if not matched_office:
            return JsonResponse({
                "success": False,
                "message": "You are not inside any authorized office location",
                "color": "red"
            })

        # 🔐 INDOOR → ONLY assigned office
        if employee.location_type == "INDOOR":
            if matched_office.id != employee.assigned_location_id:
                return JsonResponse({
                    "success": False,
                    "message": "Attendance allowed only at your assigned office",
                    "color": "red"
                })

        # 🌍 OUTDOOR → ANY authorized office (NO restriction)
        # (nothing needed here)

        # ----------------------------------------------------
        # 🔒 ATTENDANCE LOGIC (PERMANENT FIX)
        # ----------------------------------------------------
        with transaction.atomic():
            now = timezone.now()
            user_name = employee.user.get_full_name()

            settings_obj = AttendanceSettings.objects.first()
            min_hours = settings_obj.min_hours_before_checkout if settings_obj else 0
            REST_HOURS_AFTER_CHECKOUT = 8  # 🔒 HARD LOCK

            # --------------------------------------------
            # 1️⃣ Check for open shift
            # --------------------------------------------
            open_checkin = get_open_checkin(employee)

            # =====================================================
            # 🟢 FLEXIBLE STAFF (Gardener / Cleaning)
            # =====================================================
            if is_flexible:
                att_type = "check_out" if open_checkin else "check_in"
            # --------------------------------------------
            # 2️⃣ If open shift → CHECK-OUT
            # --------------------------------------------
            else:
                if open_checkin:
                    worked_hours = (now - open_checkin.timestamp).total_seconds() / 3600

                    if worked_hours < min_hours:
                        mins_left = int((min_hours - worked_hours) * 60)
                        
                        return JsonResponse({
                            "success": False,
                            "message": f"Wait {mins_left} minutes to checkout",
                            "color": "yellow"
                        })

                    att_type = "check_out"

                # --------------------------------------------
                # 3️⃣ No open shift → CHECK-IN (8h lock)
                # --------------------------------------------
                else:
                    last_checkout = Attendance.objects.filter(
                        employee=employee,
                        attendance_type="check_out"
                    ).order_by("-timestamp").first()

                    if last_checkout:
                        remaining_minutes = int(
                            (REST_HOURS_AFTER_CHECKOUT * 60) -
                            ((now - last_checkout.timestamp).total_seconds() / 60)
                        )

                        if remaining_minutes > 0:
                            hours_left = remaining_minutes // 60
                            minutes_left = remaining_minutes % 60

                            if hours_left > 0:
                                message = f"Next check-in allowed after {hours_left}h {minutes_left}m"
                            else:
                                message = f"Next check-in allowed after {minutes_left} minutes"

                            return JsonResponse({
                                "success": False,
                                "message": message,
                                "color": "yellow"
                            })

                    att_type = "check_in"

            # --------------------------------------------
            # 4️⃣ Create Attendance Record
            # --------------------------------------------
            Attendance.objects.create(
                employee=employee,
                attendance_type=att_type,
                confidence_score=score,
                location=matched_office.name
            )

        # ----------------------------------------------------
        # ✅ Success Response
        # ----------------------------------------------------
        local_time = timezone.localtime(now)
        return JsonResponse({
            "success": True,
            "status": "new",
            "message": f"Success: {user_name}",
            "name": user_name,
            "attendance_type": att_type.replace("_", " ").title(),
            "confidence": int(score * 100),
            "time": local_time.strftime("%I:%M %p"),
            "color": "green"
        })

    except Exception:
        logger.exception("Attendance Error")
        return JsonResponse({
            "success": False,
            "message": "Server Error"
        }, status=500)

        

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

def can_manage_attendance(user):
    # 🔑 Superuser always allowed
    if user.is_authenticated and user.is_superuser:
        return True

    if not user.is_authenticated:
        return False

    try:
        emp = user.employee
    except:
        return False

    return emp.role in ['Admin', 'HR', 'Finance']

@login_required
def attendance_page(request):
    return render(request, 'attendance.html')


@login_required
def video_feed_view(request):
    gen = video_feed()
    if gen is None:
        return HttpResponseServerError("Camera not accessible")
    return StreamingHttpResponse(gen, content_type='multipart/x-mixed-replace; boundary=frame')


@login_required
def attendance_reports(request):
    # =====================================================
    # 1. GET PARAMETERS
    # =====================================================
    start_date_str = request.GET.get("start_date")
    end_date_str = request.GET.get("end_date")
    employee_id = request.GET.get("employee")
    date_range = request.GET.get("range")
    download = request.GET.get("download")

    # =====================================================
    # 2. DATE PARSING
    # =====================================================
    today = timezone.now().date()

    if date_range == 'today':
        start_date = end_date = today
    elif date_range == 'week':
        start_date = today - timedelta(days=today.weekday())
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
        start_date = end_date = today

    # =====================================================
    # 3. 🔒 EMPLOYEE VISIBILITY (CRITICAL)
    # =====================================================
    visible_employees = get_visible_employees(request.user)

    # =====================================================
    # 4. BASE ATTENDANCE QUERYSET (SECURE)
    # =====================================================
    attendance_qs = Attendance.objects.select_related(
        "employee", "employee__user"
    ).filter(
        employee__in=visible_employees,
        timestamp__date__range=(start_date, end_date)
    ).order_by("-timestamp")

    # Filter by specific employee (SAFE)
    if employee_id:
        attendance_qs = attendance_qs.filter(employee_id=employee_id)

    # =====================================================
    # 5. EXCEL EXPORT
    # =====================================================
    if download == "excel":
        return generate_excel_report(attendance_qs, start_date, end_date)

    # =====================================================
    # 6. DASHBOARD STATISTICS (VISIBLE EMPLOYEES ONLY)
    # =====================================================
    active_employees = visible_employees.filter(is_active=True).select_related("user")
    total_employees_count = active_employees.count()

    present_emp_ids = attendance_qs.filter(
        attendance_type='check_in'
    ).values_list('employee_id', flat=True).distinct()

    present_employees_list = active_employees.filter(id__in=present_emp_ids)
    absent_employees_list = active_employees.exclude(id__in=present_emp_ids)

    present_employees_count = present_employees_list.count()
    absent_employees_count = max(0, total_employees_count - present_employees_count)

    # =====================================================
    # 7. PAGINATION
    # =====================================================
    paginator = Paginator(attendance_qs, 20)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    # =====================================================
    # 8. CHART DATA
    # =====================================================
    daily_attendance = attendance_qs.filter(
        attendance_type="check_in"
    ).annotate(
        date=TruncDate("timestamp")
    ).values("date").annotate(
        count=Count("employee", distinct=True)
    ).order_by("date")

    daily_stats = {item["date"]: item["count"] for item in daily_attendance}

    chart_labels = []
    chart_data_present = []

    current_d = start_date
    while current_d <= end_date:
        chart_labels.append(current_d.strftime("%b %d"))
        chart_data_present.append(daily_stats.get(current_d, 0))
        current_d += timedelta(days=1)

    # =====================================================
    # 9. CONTEXT
    # =====================================================
    context = {
        "attendance_data": page_obj,
        "start_date": start_date,
        "end_date": end_date,

        # 🔒 dropdown only shows visible employees
        "employees": active_employees.order_by("user__first_name"),
        "selected_employee": int(employee_id) if employee_id else None,

        # KPIs
        "total_employees": total_employees_count,
        "present_employees": present_employees_count,
        "absent_employees": absent_employees_count,
        "present_employees_list": present_employees_list,
        "absent_employees_list": absent_employees_list,

        # Charts
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


def get_work_date(attendance, rule):
    ts = timezone.localtime(attendance.timestamp)

    if not rule:
        return ts.date()

    start = rule.shift_start_time
    end = rule.shift_end_time

    # Night shift (crosses midnight)
    if end <= start and ts.time() < end:
        return ts.date() - timedelta(days=1)

    return ts.date()

@login_required
def attendance_calendar(request, employee_id, year, month):
    employee = get_object_or_404(Employee, id=employee_id)
    rule = employee.work_rule

    year = int(year)
    month = int(month)

    # Use Calendar to get a perfect grid (Lists of weeks containing date objects)
    # firstweekday=0 means Monday is the first day of the week
    cal = calendar.Calendar(firstweekday=0) 
    month_days = cal.monthdatescalendar(year, month) 

    # Determine exact query range (From the first visible grid day to the last)
    start_date = month_days[0][0]
    end_date = month_days[-1][-1]

    records = Attendance.objects.filter(
        employee=employee,
        timestamp__gte=start_date - timedelta(days=1),
        timestamp__lte=end_date + timedelta(days=1),
    ).order_by("timestamp")

    # ------------------------------------
    # SEQUENTIAL PAIRING (PERMANENT FIX)
    # ------------------------------------
    daily_attendance = {}
    open_checkin = None
    open_work_date = None

    for rec in records:
        ts = timezone.localtime(rec.timestamp)
        work_date = get_work_date(rec, rule)

        if rec.attendance_type == "check_in":
            open_checkin = ts
            open_work_date = work_date

            daily_attendance.setdefault(work_date, {
                "check_in": None,
                "check_out": None,
            })

            # earliest check-in wins
            if not daily_attendance[work_date]["check_in"] or ts < daily_attendance[work_date]["check_in"]:
                daily_attendance[work_date]["check_in"] = ts

        elif rec.attendance_type == "check_out" and open_checkin:
            daily_attendance.setdefault(open_work_date, {
                "check_in": open_checkin,
                "check_out": None,
            })

            # latest checkout wins
            if not daily_attendance[open_work_date]["check_out"] or ts > daily_attendance[open_work_date]["check_out"]:
                daily_attendance[open_work_date]["check_out"] = ts

            open_checkin = None
            open_work_date = None

    # ------------------------------------
    # BUILD PERFECT CALENDAR DATA
    # ------------------------------------
    month_data = []
    total_present = total_absent = total_holiday = 0
    total_work_seconds = 0
    today = timezone.localdate()

    for week in month_days:
        for day_date in week:
            is_current_month = day_date.month == month
            is_future = day_date > today
            is_sunday = day_date.weekday() == 6 # 6 is Sunday
            is_today = day_date == today

            record = daily_attendance.get(day_date)
            check_in = record["check_in"] if record else None
            check_out = record["check_out"] if record else None

            present = False
            daily_hours = None

            if check_in:
                present = True
                if is_current_month:
                    total_present += 1

                if check_out and check_out > check_in:
                    delta = check_out - check_in
                    if is_current_month:
                        total_work_seconds += delta.total_seconds()
                    
                    # Format as Xh Ym
                    hours, remainder = divmod(delta.seconds, 3600)
                    minutes, _ = divmod(remainder, 60)
                    daily_hours = f"{int(hours)}h {int(minutes)}m"
            else:
                # Don't count future dates as absent
                if is_current_month and not is_future:
                    if is_sunday:
                        total_holiday += 1
                    else:
                        total_absent += 1

            month_data.append({
                "date_str": day_date.strftime("%Y-%m-%d"),
                "day": day_date.day,
                "day_name": calendar.day_name[day_date.weekday()][:3],
                "is_current_month": is_current_month,
                "is_future": is_future,
                "is_today": is_today,
                "present": present,
                "holiday": is_sunday and not present,
                "daily_hours": daily_hours,
            })

    total_working_hours = round(total_work_seconds / 3600, 2)
    settings = AttendanceSettings.objects.first()
    expected_hours = total_present * (settings.max_daily_hours if settings else 8)

    context = {
        "employee": employee,
        "year": year,
        "month": month,
        "month_name": calendar.month_name[month],
        "data": month_data,
        "total_present": total_present,
        "total_absent": total_absent,
        "total_holiday": total_holiday,
        "total_working_hours": total_working_hours,
        "total_expected_hours": round(expected_hours, 2),
        "next_month": month + 1 if month < 12 else 1,
        "next_year": year if month < 12 else year + 1,
        "previous_month": month - 1 if month > 1 else 12,
        "previous_year": year if month > 1 else year - 1,
        "day_names": list(calendar.day_name),
        "all_employees": Employee.objects.all().order_by("user__first_name"),
    }

    return render(request, "attendance_calendar.html", context)


def attendance_day_detail(request, emp_id, date):
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        return JsonResponse({"error": "Invalid date"}, status=400)

    emp = get_object_or_404(Employee, id=emp_id)
    rule = emp.work_rule

    logs = Attendance.objects.filter(employee=emp).order_by("timestamp")

    day_logs = [
        log for log in logs
        if get_work_date(log, rule) == date_obj
    ]

    return JsonResponse({
        "date": date_obj.strftime("%d %B %Y"),
        "logs": [
            {
                "type": log.get_attendance_type_display(),
                "timestamp": timezone.localtime(log.timestamp).strftime("%I:%M %p"),
                "date": timezone.localtime(log.timestamp).strftime("%d-%m-%Y"),
                "location": log.location or "—",
                "confidence": f"{log.confidence_score * 100:.1f}%" if log.confidence_score else "—",
            }
            for log in day_logs
        ]
    })

@login_required
@require_GET
def attendance_calendar_data(request, employee_id, year, month):
    emp = get_object_or_404(Employee, id=employee_id)
    rule = emp.work_rule

    year = int(year)
    month = int(month)

    start_date = date(year, month, 1)
    end_day = calendar.monthrange(year, month)[1]
    end_date = date(year, month, end_day)

    records = Attendance.objects.filter(
        employee=emp,
        timestamp__gte=start_date - timedelta(days=1),
        timestamp__lte=end_date + timedelta(days=1),
    ).order_by("timestamp")

    present_days = set()
    open_checkin = None
    open_work_date = None

    for rec in records:
        wd = get_work_date(rec, rule)

        if rec.attendance_type == "check_in":
            open_checkin = rec
            open_work_date = wd
            if start_date <= wd <= end_date:
                present_days.add(wd.day)

        elif rec.attendance_type == "check_out" and open_checkin:
            open_checkin = None
            open_work_date = None

    data = []
    for d in range(1, end_day + 1):
        current = date(year, month, d)
        data.append({
            "day": d,
            "present": d in present_days,
            "is_weekend": current.weekday() >= 5,
            "is_future": current > timezone.localdate(),
        })

    return JsonResponse({
        "employee": emp.user.get_full_name(),
        "year": year,
        "month": month,
        "month_name": calendar.month_name[month],
        "data": data,
    })

@login_required
@require_POST
def update_attendance(request, att_id):
    if not can_manage_attendance(request.user):
        return JsonResponse({'error': 'Permission denied'}, status=403)

    attendance = get_object_or_404(Attendance, id=att_id)

    try:
        attendance.attendance_type = request.POST.get('attendance_type')
        attendance.timestamp = datetime.strptime(
            request.POST.get('timestamp'),
            "%Y-%m-%d %H:%M"
        )
        attendance.location = request.POST.get('location', '')
        attendance.notes = request.POST.get('notes', '')
        attendance.save()

        return JsonResponse({'success': True})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@login_required
@require_POST
def delete_attendance(request, att_id):
    if not can_manage_attendance(request.user):
        return JsonResponse({'error': 'Permission denied'}, status=403)

    attendance = get_object_or_404(Attendance, id=att_id)
    attendance.delete()

    return JsonResponse({'success': True})

@login_required
@require_POST
def create_attendance(request):
    if not can_manage_attendance(request.user):
        return JsonResponse({'error': 'Permission denied'}, status=403)

    employee = get_object_or_404(Employee, id=request.POST['employee'])

    Attendance.objects.create(
        employee=employee,
        attendance_type=request.POST['attendance_type'],
        timestamp=datetime.strptime(
            request.POST['timestamp'], "%Y-%m-%d %H:%M"
        ),
        location=request.POST.get('location', ''),
        notes=request.POST.get('notes', ''),
        confidence_score=None  # manual entry
    )

    return JsonResponse({'success': True})

class OfficeLocationForm(forms.ModelForm):
    class Meta:
        model = OfficeLocation
        fields = [
            "name",
            "latitude",
            "longitude",
            "radius_meters",
            "is_active",
        ]
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control"}),
            "latitude": forms.NumberInput(attrs={"class": "form-control", "step": "any"}),
            "longitude": forms.NumberInput(attrs={"class": "form-control", "step": "any"}),
            "radius_meters": forms.NumberInput(attrs={"class": "form-control"}),
            "is_active": forms.CheckboxInput(attrs={"class": "form-check-input"}),
        }


@login_required
def office_location_list(request):
    locations = OfficeLocation.objects.all().order_by("name")
    # We pass an empty form so the "Add Modal" can render the fields
    form = OfficeLocationForm()
    
    return render(request, "office_location/list.html", {
        "locations": locations,
        "form": form
    })

@login_required
def office_location_create(request):
    if request.method == "POST":
        form = OfficeLocationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Office location added successfully.")
            return redirect("office_location_list")
        else:
            # In a real modal implementation, handling validation errors 
            # usually requires AJAX. For simplicity here, if error, 
            # we redirect to the standalone form or reload list with errors.
            messages.error(request, "Error adding location. Please check inputs.")
            return redirect("office_location_list")
    return redirect("office_location_list")

@login_required
def office_location_update(request, pk):
    location = get_object_or_404(OfficeLocation, pk=pk)
    if request.method == "POST":
        form = OfficeLocationForm(request.POST, instance=location)
        if form.is_valid():
            form.save()
            messages.success(request, "Office location updated successfully.")
            return redirect("office_location_list")
        else:
            messages.error(request, "Error updating location.")
    
    return redirect("office_location_list")

@login_required
def office_location_delete(request, pk):
    location = get_object_or_404(OfficeLocation, pk=pk)
    if request.method == "POST":
        location.delete()
        messages.success(request, "Office location deleted successfully.")
    return redirect("office_location_list")


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
def employee_to_dict(emp):
    return {
        "id": emp.id,
        "name": emp.user.get_full_name(),
        "department": emp.department.name if emp.department else "",
        "position": emp.position,
        "profile_url": f"/employees/{emp.id}/profile/",
        "children": [],
        "is_current": False,
    }

def build_path_to_root(emp):
    path = []
    while emp:
        path.append(emp)
        emp = emp.manager
    return list(reversed(path))

def build_subtree(emp, visited=None):
    if visited is None:
        visited = set()
    if emp.id in visited:
        return None
    visited.add(emp.id)

    node = employee_to_dict(emp)
    for child in emp.subordinates.all():
        sub = build_subtree(child, visited)
        if sub:
            node["children"].append(sub)
    return node

def collect_ids(node, ids):
    if not node:
        return
    ids.add(node["id"])
    for c in node.get("children", []):
        collect_ids(c, ids)

def filter_tree(nodes, allowed_ids):
    result = []
    for n in nodes:
        if n["id"] in allowed_ids:
            n["children"] = filter_tree(n["children"], allowed_ids)
            result.append(n)
    return result

def mark_current_employee(tree, emp_id):
    for n in tree:
        if n["id"] == emp_id:
            n["is_current"] = True
        mark_current_employee(n.get("children", []), emp_id)

# -------- APIs --------
@login_required
def org_tree_api_me(request):
    """
    Always returns a valid org tree.
    If Employee is not linked → show only User card.
    """

    user = request.user

    # -------------------------------------------------
    # CASE 1: USER HAS NO EMPLOYEE PROFILE
    # -------------------------------------------------
    if not hasattr(user, "employee"):
        single_node = {
            "id": f"user-{user.id}",
            "name": user.get_full_name() or user.username,
            "department": "—",
            "position": "User",
            "image": "",
            "profile_url": "",
            "children": [],
            "is_current": True,
        }
        return JsonResponse({"ok": True, "tree": [single_node]})

    # -------------------------------------------------
    # CASE 2: USER HAS EMPLOYEE PROFILE
    # -------------------------------------------------
    current_emp = user.employee
    is_hr = user.is_superuser or current_emp.role in ["HR", "Admin"]

    employees = (
        current_emp.__class__.objects
        .select_related("user", "department", "manager")
        .prefetch_related("subordinates")
    )

    # Build employee map
    emp_map = {}
    for emp in employees:
        emp_map[emp.id] = {
            "id": emp.id,
            "name": emp.user.get_full_name(),
            "department": emp.department.name if emp.department else "",
            "position": emp.position,
            "manager_id": emp.manager_id,
            "image": emp.profile_image.url if getattr(emp, "profile_image", None) else "",
            "profile_url": reverse("employee_detail", args=[emp.employee_id]),
            "children": [],
            "is_current": emp.id == current_emp.id,
        }

    # Build hierarchy
    roots = []
    for emp in employees:
        node = emp_map[emp.id]
        if emp.manager_id and emp.manager_id in emp_map:
            emp_map[emp.manager_id]["children"].append(node)
        else:
            roots.append(node)

    # -------------------------------------------------
    # HR / ADMIN → FULL TREE
    # -------------------------------------------------
    if is_hr:
        return JsonResponse({"ok": True, "tree": roots})

    # -------------------------------------------------
    # EMPLOYEE → LIMITED TREE
    # -------------------------------------------------
    allowed_ids = {current_emp.id}

    # Managers
    mgr = current_emp.manager
    while mgr:
        allowed_ids.add(mgr.id)
        mgr = mgr.manager

    # Subordinates
    def collect_children(eid):
        for c in emp_map[eid]["children"]:
            allowed_ids.add(c["id"])
            collect_children(c["id"])

    collect_children(current_emp.id)

    def filter_tree(nodes):
        result = []
        for n in nodes:
            if n["id"] in allowed_ids:
                n_copy = n.copy()
                n_copy["children"] = filter_tree(n["children"])
                result.append(n_copy)
        return result

    restricted_tree = filter_tree(roots)

    # -------------------------------------------------
    # FALLBACK → ONLY SELF
    # -------------------------------------------------
    if not restricted_tree:
        return JsonResponse({
            "ok": True,
            "tree": [{
                **emp_map[current_emp.id],
                "children": []
            }]
        })

    return JsonResponse({"ok": True, "tree": restricted_tree})


@login_required
def org_chart_page(request):
    """
    Renders the HTML page. Data is fetched via AJAX.
    """
    return render(request, 'ems/org_chart.html')

@login_required
def org_tree_api(request):
    """
    Returns the JSON Org Tree based on user role:
    - HR/Admin: Full organization
    - Employee: Only their managers (path to CEO) -> them -> their subordinates
    If they have no manager/subordinates, returns just their card.
    """
    try:
        current_emp = request.user.employee
    except getattr(request.user, 'employee', None).DoesNotExist if hasattr(request.user, 'employee') else Exception:
        return JsonResponse({"ok": False, "error": "Employee not linked"}, status=403)

    # Determine role access
    emp_role = getattr(current_emp, 'role', '')
    is_hr = request.user.is_superuser or emp_role in ["HR", "Admin"]

    # 1. Fetch all employees to build map efficiently (1 DB query)
    # Adjust related fields based on your actual model architecture
    employees = current_emp.__class__.objects.select_related("user", "department", "manager").all()

    emp_map = {}
    for emp in employees:
        image_url = emp.profile_image.url if getattr(emp, 'profile_image', None) else ""
        
        # Safely get profile URL
        try:
            profile_url = reverse('employee_detail', args=[emp.employee_id])
        except:
            profile_url = f"/employees/{emp.id}/profile/"

        emp_map[emp.id] = {
            "id": emp.id,
            "name": emp.user.get_full_name() if hasattr(emp, 'user') else "Unknown",
            "department": emp.department.name if getattr(emp, 'department', None) else "",
            "position": getattr(emp, 'role', getattr(emp, 'designation', 'Employee')),
            "manager_id": emp.manager_id,
            "image": image_url,
            "profile_url": profile_url,
            "children": [],
            "is_current": (emp.id == current_emp.id)
        }

    # 2. Build Full Tree
    roots = []
    for emp in employees:
        node = emp_map[emp.id]
        if emp.manager_id and emp.manager_id in emp_map:
            emp_map[emp.manager_id]["children"].append(node)
        else:
            roots.append(node)

    # If HR, return full org chart
    if is_hr:
        return JsonResponse({"ok": True, "tree": roots})

    # 3. If Normal Employee -> Restrict View Logic
    allowed_ids = set()

    # Add self
    allowed_ids.add(current_emp.id)

    # Add managers up to CEO
    curr_mgr_id = current_emp.manager_id
    while curr_mgr_id and curr_mgr_id in emp_map:
        allowed_ids.add(curr_mgr_id)
        curr_mgr_id = emp_map[curr_mgr_id]["manager_id"]

    # Add all subordinates downward
    def collect_subordinates(node_id):
        for child in emp_map[node_id]["children"]:
            allowed_ids.add(child["id"])
            collect_subordinates(child["id"])

    collect_subordinates(current_emp.id)

    # 4. Filter the tree to only allowed nodes
    def filter_tree(nodes):
        result = []
        for n in nodes:
            if n["id"] in allowed_ids:
                # Copy dict so we don't mutate original during recursion
                n_copy = n.copy()
                n_copy["children"] = filter_tree(n["children"])
                result.append(n_copy)
        return result

    restricted_tree = filter_tree(roots)

    # Note: Even if 'restricted_tree' only has 1 node (the user), it perfectly returns just their card!
    return JsonResponse({"ok": True, "tree": restricted_tree})


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


class PayrollForm(forms.ModelForm):
    class Meta:
        model = Payroll
        fields = [
            # Pay structure
            'basic_pay',
            'allowances',
            'deductions',

            # Attendance
            'present_days',
            'half_days',
            'absent_days',
            'overtime_hours',

            # Document
            'payslip_pdf',

            # Status
            'status',
            'paid_date',
        ]

        widgets = {
            'paid_date': forms.DateInput(attrs={'type': 'date'}),
        }

    # -------------------------------
    # FIELD-LEVEL VALIDATIONS
    # -------------------------------

    def clean_basic_pay(self):
        val = self.cleaned_data.get('basic_pay', Decimal('0'))
        if val < 0:
            raise forms.ValidationError("Basic pay cannot be negative.")
        return val

    def clean_allowances(self):
        val = self.cleaned_data.get('allowances', Decimal('0'))
        if val < 0:
            raise forms.ValidationError("Allowances cannot be negative.")
        return val

    def clean_deductions(self):
        val = self.cleaned_data.get('deductions', Decimal('0'))
        if val < 0:
            raise forms.ValidationError("Deductions cannot be negative.")
        return val

    def clean_overtime_hours(self):
        val = self.cleaned_data.get('overtime_hours', Decimal('0'))
        if val < 0:
            raise forms.ValidationError("Overtime hours cannot be negative.")
        return val

    def clean_present_days(self):
        val = self.cleaned_data.get('present_days', Decimal('0'))
        if val < 0:
            raise forms.ValidationError("Present days cannot be negative.")
        return val

    def clean_half_days(self):
        val = self.cleaned_data.get('half_days', Decimal('0'))
        if val < 0:
            raise forms.ValidationError("Half days cannot be negative.")
        return val

    def clean_absent_days(self):
        val = self.cleaned_data.get('absent_days', Decimal('0'))
        if val < 0:
            raise forms.ValidationError("Absent days cannot be negative.")
        return val

    # -------------------------------
    # FORM-LEVEL VALIDATION
    # -------------------------------

    def clean(self):
        cleaned = super().clean()

        status = cleaned.get('status')
        paid_date = cleaned.get('paid_date')

        # Rule: Paid payroll MUST have paid_date
        if status == 'paid' and not paid_date:
            raise forms.ValidationError(
                "Paid date is required when payroll status is PAID."
            )

        return cleaned

@login_required
def payroll_list(request):
    form = PayrollFilterForm(request.GET or None)

    # 🔒 visibility layer
    visible_employees = get_visible_employees(request.user)

    payrolls = Payroll.objects.select_related(
        'employee__user'
    ).filter(
        employee__in=visible_employees
    ).only(
        'employee__face_image',
        'employee__employee_id',
        'employee__user__first_name',
        'employee__user__last_name',
        'basic_pay',
        'allowances',
        'deductions',
        'net_salary',
        'status',
        'month'
    )

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
@finance_required
def generate_payroll(request):
    if request.method == 'POST':
        try:
            month = int(request.POST.get('month'))
            year = int(request.POST.get('year'))
            employee_id = request.POST.get('employee', '').strip()

            if not (1 <= month <= 12):
                raise ValueError("Invalid month selected.")

            if year < 2000 or year > 2100:
                raise ValueError("Invalid year selected.")

            if employee_id == "":
                Payroll.objects.generate_salary(year=year, month=month)
                messages.success(
                    request,
                    f"Payroll generated for all employees "
                    f"({calendar.month_name[month]} {year})"
                )
            else:
                emp = Employee.objects.get(id=employee_id)
                Payroll.objects.generate_salary(year=year, month=month, employee=emp)
                messages.success(
                    request,
                    f"Payroll generated for {emp.user.get_full_name()} "
                    f"({calendar.month_name[month]} {year})"
                )

            return redirect('payroll_list')

        except Employee.DoesNotExist:
            messages.error(request, "Selected employee does not exist.")

        except ValueError as e:
            messages.error(request, str(e))

        except Exception:
            messages.error(
                request,
                "Payroll generation failed. Please check salary structure, work rule, or attendance."
            )

    employees = Employee.objects.filter(
        employment_status='active',
        is_active=True
    ).select_related('user')

    return render(request, 'ems/generate_payroll.html', {
        'title': 'Generate Payroll',
        'employees': employees,
    })

@login_required
@finance_required
def payroll_update(request, pk):
    payroll = get_object_or_404(Payroll, pk=pk)

    if payroll.status == 'paid':
        messages.error(request, "Paid payrolls cannot be edited.")
        return redirect('employee_salary_history', payroll.employee_id)

    if request.method == 'POST':
        form = PayrollForm(
            request.POST,
            request.FILES,
            instance=payroll
        )

        if form.is_valid():
            payroll = form.save(commit=False)

            # Salary recalculation
            payroll.calculated_salary = (
                payroll.basic_pay +
                payroll.allowances -
                payroll.deductions
            )
            payroll.net_salary = payroll.calculated_salary

            # Convert hourly → paid days
            try:
                rule = payroll.employee.work_rule
                structure = payroll.employee.salary_structure

                full_day_hours = Decimal(rule.full_day_hours or 0)
                if full_day_hours > 0 and structure.base_salary > 0:
                    total_days = calendar.monthrange(
                        payroll.month.year,
                        payroll.month.month
                    )[1]

                    hourly_rate = (
                        structure.base_salary /
                        (Decimal(total_days) * full_day_hours)
                    )

                    worked_hours = Decimal(payroll.basic_pay) / hourly_rate

                    payroll.present_days = (
                        worked_hours / full_day_hours
                    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

                    payroll.absent_days = Decimal(total_days) - payroll.present_days

            except Exception as e:
                messages.warning(
                    request,
                    f"Paid days auto-calculation skipped: {str(e)}"
                )

            payroll.save()
            messages.success(request, "Payroll updated successfully.")
            return redirect('employee_salary_history', payroll.employee_id)

        # ❌ Show all errors directly on frontend
        for field, errors in form.errors.items():
            for error in errors:
                messages.error(
                    request,
                    f"{field.replace('_', ' ').title()}: {error}"
                )

        for error in form.non_field_errors():
            messages.error(request, error)

        return redirect('employee_salary_history', payroll.employee_id)

    return redirect('employee_salary_history', payroll.employee_id)

@login_required
@finance_required
def payroll_delete(request, pk):
    payroll = get_object_or_404(Payroll, pk=pk)

    if payroll.status == 'paid' and not request.user.is_superuser:
        messages.error(request, "Only Admin can delete paid payrolls.")
        return redirect('employee_salary_history', payroll.employee_id)

    if request.method == 'POST':
        emp_id = payroll.employee_id
        payroll.delete()
        messages.success(request, "Payroll deleted successfully.")
        return redirect('employee_salary_history', emp_id)

    messages.error(request, "Invalid request.")
    return redirect('employee_salary_history', payroll.employee_id)


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
@finance_required
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
@finance_required
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
def employee_salary_history(request, employee_id):
    visible_employees = get_visible_employees(request.user)

    # 🔒 HARD SECURITY CHECK
    if not visible_employees.filter(id=employee_id).exists():
        return HttpResponseForbidden("You are not allowed to view this employee's salary.")

    emp = get_object_or_404(
        Employee.objects.select_related('user', 'work_rule'),
        id=employee_id
    )

    payrolls = Payroll.objects.filter(
        employee=emp
    ).order_by('-month')

    # -------------------------------------------------
    # ✅ AUTO-FIX PAID DAYS (UNCHANGED LOGIC)
    # -------------------------------------------------
    for p in payrolls:
        try:
            if not p.present_days or p.present_days == 0:
                rule = emp.work_rule
                structure = emp.salary_structure

                full_day_hours = Decimal(rule.full_day_hours or 0)
                if full_day_hours <= 0 or structure.base_salary <= 0:
                    continue

                total_days = calendar.monthrange(
                    p.month.year,
                    p.month.month
                )[1]

                hourly_rate = (
                    structure.base_salary /
                    (Decimal(total_days) * full_day_hours)
                )

                if hourly_rate <= 0:
                    continue

                worked_hours = Decimal(p.basic_pay) / hourly_rate

                p.present_days = (
                    worked_hours / full_day_hours
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

                p.absent_days = (
                    Decimal(total_days) - p.present_days
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

                p.save(update_fields=["present_days", "absent_days"])

        except Exception:
            continue

    totals = payrolls.aggregate(
        total_basic=Sum('basic_pay') or 0,
        total_allowances=Sum('allowances') or 0,
        total_deductions=Sum('deductions') or 0,
        total_net=Sum('net_salary') or 0,
    )

    return render(request, 'ems/employee_salary_history.html', {
        'employee': emp,
        'payrolls': payrolls,
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
        me = request.user.employee
    except Employee.DoesNotExist:
        messages.error(request, 'Employee profile not found!')
        return redirect('dashboard')

    visible_employees = get_visible_employees(request.user)

    # My leaves (always)
    my_leaves = LeaveApplication.objects.filter(
        employee=me
    ).select_related('leave_type').order_by('-applied_on')

    # Subordinates / team leaves (for managers / HR)
    team_leaves = LeaveApplication.objects.filter(
        employee__in=visible_employees.exclude(id=me.id)
    ).select_related('employee__user', 'leave_type')

    # Pending approvals assigned to me
    pending_approvals = LeaveApplication.objects.filter(
        approvals__approver=me,
        approvals__status='pending'
    ).select_related('employee__user', 'leave_type').distinct()

    return render(request, 'ems/leave_application_list.html', {
        'my_leaves': my_leaves,
        'team_leaves': team_leaves,
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

    total_allowed = sum(
        LeaveType.objects.values_list('max_days_per_year', flat=True)
    )
    used_leave_days = sum(
        l.total_days()
        for l in LeaveApplication.objects.filter(
            employee=employee, status='approved'
        )
    )
    pending_applications = LeaveApplication.objects.filter(
        employee=employee, status='pending'
    ).count()

    available_leave_days = max(total_allowed - used_leave_days, 0)

    if request.method == 'POST':
        form = LeaveApplicationForm(request.POST)
        if form.is_valid():
            leave_app = form.save(commit=False)
            leave_app.employee = employee
            leave_app.save()

            approver = (
                employee.manager
                or Employee.objects.filter(position__icontains="HR", employment_status='active').first()
                or Employee.objects.filter(user__is_superuser=True).first()
            )

            if approver:
                LeaveApproval.objects.create(
                    leave=leave_app,
                    approver=approver,
                    level=1,
                    status='pending'
                )

            messages.success(request, 'Leave submitted successfully.')
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

    try:
        approver = request.user.employee
    except Employee.DoesNotExist:
        return HttpResponseForbidden()

    approval = LeaveApproval.objects.filter(
        leave=leave_app,
        approver=approver,
        status='pending'
    ).first()

    if not approval:
        return HttpResponseForbidden("Not authorized to approve this leave")

    # (rest of your approval logic stays exactly the same)



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
    leave_app = get_object_or_404(
        LeaveApplication,
        pk=pk,
        employee=request.user.employee
    )

    if request.method == 'POST':
        form = LeaveApplicationForm(request.POST, instance=leave_app)
        if form.is_valid():
            leave = form.save(commit=False)
            leave.status = 'pending'
            leave.approved_by = None
            leave.save()

            LeaveApproval.objects.filter(leave=leave).delete()

            approver = (
                leave.employee.manager
                or Employee.objects.filter(position__icontains="HR").first()
                or Employee.objects.filter(user__is_superuser=True).first()
            )

            if approver:
                LeaveApproval.objects.create(
                    leave=leave,
                    approver=approver,
                    level=1,
                    status='pending'
                )

            messages.success(request, "Leave updated successfully.")
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
    visible_employees = get_visible_employees(request.user)

    leave_app = get_object_or_404(
        LeaveApplication.objects.select_related('employee__user', 'leave_type'),
        pk=pk
    )

    if leave_app.employee not in visible_employees:
        return HttpResponseForbidden()

    history = leave_app.approvals.select_related(
        'approver__user'
    ).order_by('level', 'acted_at', 'id')

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
    joining_details = (
        JoiningDetail.objects
        .select_related('employee__user')
        .prefetch_related('joining_documents')  # correct related_name
    )

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
            joining = form.save()

            # 📎 Handle document upload
            for file in request.FILES.getlist('documents'):
                JoiningDocument.objects.create(
                    joining=joining,
                    file=file
                )

            messages.success(request, 'Joining details added successfully!')
            return redirect('joining_detail_list')
    else:
        form = JoiningDetailForm()

    return render(request, 'ems/joining_detail_form.html', {
        'form': form,
        'title': 'Add Joining Details'
    })

@login_required
def upload_joining_documents(request, detail_id):
    joining = get_object_or_404(JoiningDetail, id=detail_id)

    if request.method == 'POST':
        files = request.FILES.getlist('documents')

        if not files:
            messages.warning(request, "No documents selected.")
            return redirect('joining_detail_list')

        for file in files:
            JoiningDocument.objects.create(
                joining=joining,
                file=file
            )

        messages.success(request, "Documents uploaded successfully.")

    return redirect('joining_detail_list')


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


    