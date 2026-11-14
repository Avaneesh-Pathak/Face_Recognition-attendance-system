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
from .decorators import finance_required, hr_required, admin_required, hr_finance_required
from .utils_pdf import build_salary_slip_pdf
from django.core.mail import EmailMessage
from django.http import FileResponse
from io import BytesIO
from django.core.exceptions import PermissionDenied
from django.views.decorators.csrf import csrf_protect
from django.db.utils import ProgrammingError, OperationalError
from .models import (Employee, Attendance, AttendanceSettings, DailyReport,Department, SalaryStructure, Payroll, LeaveType,
    LeaveApplication, LeaveWorkflowStage, LeaveApproval,
    JoiningDetail, Resignation, Notification,JoiningDocument,WorkRule,OfficeLocation,Holiday)

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
from django.utils.http import url_has_allowed_host_and_scheme
from datetime import timedelta
from .notifications import create_user_notification
from decimal import Decimal, InvalidOperation
import json

from core.models import (
    Employee,
    JoiningDetail,
    JoiningDocument,
    SalaryStructure
)
from core.face_system import get_face_system, BusyInference, ModelUnavailable
from .tenant_utils import get_current_organisation
# ============================================================
# 💰 SALARY & PAYROLL VIEWS (UPDATED)
# ============================================================

from datetime import date
import calendar
from collections import defaultdict
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
from core.face_system import get_face_system, BusyInference, ModelUnavailable
from core.models import OfficeLocation
from core.utils.location import distance_to_office_m, is_inside_office
from django.db.models import Max    
from django.template.loader import render_to_string
from django.contrib.auth.decorators import login_required
from weasyprint import HTML
from django.http import HttpResponse
from django.template.loader import get_template
from xhtml2pdf import pisa
from django.shortcuts import get_object_or_404
from core.utils.access import get_visible_employees,get_visible_employees_all
from django.contrib.auth.decorators import user_passes_test



# Setup logging
logger = logging.getLogger(__name__)

# Initialize face system (lazy init inside register also)



try:
    from core.face_system import get_face_system, BusyInference, ModelUnavailable as get_if_system
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

@login_required
@hr_required
def register(request):
    # Retrieve organisation from logged-in HR/Admin user
    active_org = request.organisation
    if active_org is None:
        return HttpResponseForbidden('An active organisation account is required')
    if request.method == "POST":
        form = EmployeeRegistrationForm(request.POST, request.FILES, organisation=active_org)

        if form.is_valid():
            if request.user.employee.role == 'HR' and form.cleaned_data['role'] in ('Admin', 'Finance'):
                form.add_error('role', 'Only an organisation administrator can assign this role.')
                return render(request, 'register.html', {'form': form})
            face_image = request.FILES.get("face_image")
            captured_image = request.POST.get("captured_image", "")
            if not face_image and captured_image:
                header, separator, encoded_image = captured_image.partition(",")
                mime_type = header.removeprefix("data:").partition(";")[0].lower()
                if (not separator or ";base64" not in header.lower()
                        or mime_type not in {"image/jpeg", "image/png", "image/webp"}
                        or len(encoded_image) > 7 * 1024 * 1024):
                    form.add_error("face_image", "Upload a valid JPEG, PNG, or WebP image up to 5 MB.")
                    return render(request, "register.html", {"form": form})
                try:
                    face_image = ContentFile(
                        base64.b64decode(encoded_image, validate=True),
                        name="captured-face.upload",
                    )
                except (ValueError, base64.binascii.Error):
                    form.add_error("face_image", "The captured image could not be decoded.")
                    return render(request, "register.html", {"form": form})

            if face_image:
                try:
                    from core.security.images import optimize_face_upload
                    face_image = optimize_face_upload(face_image)
                except ValueError as exc:
                    form.add_error("face_image", str(exc))
                    return render(request, "register.html", {"form": form})

            employee = None
            try:
                # Keep account, employee, salary, and image fields transactionally aligned.
                with transaction.atomic():
                    user = User.objects.create_user(
                        username=form.cleaned_data["username"],
                        email=form.cleaned_data["email"],
                        password=form.cleaned_data["password1"],
                        first_name=form.cleaned_data["first_name"],
                        last_name=form.cleaned_data["last_name"],
                    )
                    employee = Employee.objects.create(
                        organisation=active_org,
                        user=user,
                        department=form.cleaned_data["department"],
                        position=form.cleaned_data["position"],
                        manager=form.cleaned_data["manager"],
                        phone_number=form.cleaned_data["phone_number"],
                        role=form.cleaned_data["role"],
                        work_rule=form.cleaned_data.get("work_rule"),
                        date_of_joining=form.cleaned_data["date_of_joining"],
                        location_type=form.cleaned_data["location_type"],
                        assigned_location=form.cleaned_data.get("assigned_location"),
                    )
                    JoiningDetail.objects.create(
                        organisation=active_org,
                        employee=employee,
                        date_of_joining=form.cleaned_data["date_of_joining"],
                        probation_period_months=form.cleaned_data["probation_period_months"],
                        confirmation_date=form.cleaned_data.get("confirmation_date"),
                    )
                    SalaryStructure.objects.create(
                        organisation=active_org,
                        employee=employee,
                        base_salary=form.cleaned_data.get("base_salary") or Decimal("0.00"),
                        hra=form.cleaned_data.get("hra") or Decimal("0.00"),
                        allowances=form.cleaned_data.get("allowances") or Decimal("0.00"),
                        deductions=form.cleaned_data.get("deductions") or Decimal("0.00"),
                    )
                    if face_image:
                        employee.face_image = face_image
                        employee.save(update_fields=["face_image"])
            except Exception:
                if employee and employee.face_image:
                    employee.face_image.delete(save=False)
                logger.exception("Employee registration failed")
                messages.error(request, "Employee registration failed. Review the form and try again.")
                return render(request, "register.html", {"form": form})

            # Model inference may be slow, so do it after the database transaction commits.
            if employee.face_image:
                try:
                    face_sys = get_face_system()
                    embedding = face_sys.get_embedding(employee.face_image.path)
                    if embedding is None:
                        employee.face_encoding = json.dumps({"status": "FACE_PENDING"})
                        messages.warning(request, "Employee registered. Face enrollment is pending.")
                    else:
                        employee.face_encoding = json.dumps(embedding)
                        face_sys.invalidate_tenant(active_org.id)
                    employee.save(update_fields=["face_encoding"])
                except Exception:
                    logger.exception(
                        "Employee face enrollment failed",
                        extra={"employee_pk": employee.pk, "organisation_id": active_org.pk},
                    )
                    employee.face_encoding = json.dumps({"status": "FACE_PENDING"})
                    employee.save(update_fields=["face_encoding"])
                    messages.warning(request, "Employee registered. Face enrollment is pending.")

            messages.success(request, "Employee registered successfully")
            return redirect("employee_list")

    else:
        form = EmployeeRegistrationForm(organisation=active_org)

    return render(request, "register.html", {"form": form})

# --- Real-time AJAX validators ---
@login_required
def check_username(request):
    username = request.GET.get('username', '').strip()
    exists = User.objects.filter(username__iexact=username).exists()
    return JsonResponse({"ok": bool(username and not exists), "exists": exists})

@login_required
def check_email(request):
    email = request.GET.get('email', '').strip()
    exists = User.objects.filter(email__iexact=email).exists()
    return JsonResponse({"ok": bool(email and not exists), "exists": exists})

@login_required
def get_managers_by_department(request, dept_id):
    managers = Employee.objects.filter(department_id=dept_id, employment_status='active')
    if getattr(request, 'organisation', None) is not None:
        managers = managers.filter(organisation=request.organisation)
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
    finance_access = request.user.is_superuser or getattr(getattr(request.user, 'employee', None), 'role', '') in ('Admin', 'Finance')
    if not finance_access:
        sort_salary = None
    shift_filter = request.GET.get("shift")

    if shift_filter in ["day", "night"]:

        employee_ids = []

        latest_checkins = (
            Attendance.objects
            .filter(
                employee_id__in=employees.values_list("id", flat=True),
                attendance_type="check_in"
            )
            .values("employee_id")
            .annotate(last_checkin=Max("timestamp"))
        )

        for row in latest_checkins:
            checkin_time = timezone.localtime(row["last_checkin"])
            hour = checkin_time.hour

            # classify ONLY from check-in time
            if 6 <= hour < 19:
                shift = "day"
            else:
                shift = "night"

            if shift == shift_filter:
                employee_ids.append(row["employee_id"])

        employees = employees.filter(id__in=employee_ids)
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
    # KPI queryset (independent)
    if request.user.is_superuser:
        kpi_qs = Employee.global_objects.all()

    elif hasattr(request.user, "employee") and request.user.employee.role in [
        "Admin", "HR", "Finance"
    ]:
        if getattr(request, "organisation", None) is not None:
            kpi_qs = Employee.global_objects.filter(organisation=request.organisation)
        else:
            kpi_qs = Employee.global_objects.none()

    else:
        # Managers/employees see KPIs only for visible staff
        kpi_qs = employees

    total_employees = kpi_qs.count()

    active_employees = kpi_qs.filter(
        is_active=True,
        employment_status="active"
    ).count()

    inactive_employees = kpi_qs.exclude(
        is_active=True,
        employment_status="active"
    ).count()

    total_payroll = (
        kpi_qs.aggregate(
            total=Sum("salary_structure__base_salary")
        )["total"] or 0
    )
    if not finance_access:
        total_payroll = None
    
    # ==========================
    # CONTEXT REQUIRED BY TEMPLATE
    # ==========================
    context = {
        "employees": employees,
        "departments": Department.objects.filter(
            organisation=request.organisation
        ) if getattr(request, "organisation", None) is not None else Department.objects.none(),
        "roles": Employee.ROLE_CHOICES,

        # keep dropdown selections
        "selected_dept": department_id,
        "selected_role": role,
        "selected_sort": sort_salary,
        "selected_shift": shift_filter,

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

    def __init__(self, *args, **kwargs):
        # 🟢 Safely pop the organisation argument from the view
        organisation = kwargs.pop('organisation', None)
        super().__init__(*args, **kwargs)

        # 🟢 Filter dropdown querysets to the current tenant to load correct options
        if organisation:
            self.fields['department'].queryset = Department.global_objects.filter(organisation=organisation)
            self.fields['manager'].queryset = Employee.global_objects.filter(
                organisation=organisation,
                is_active=True,
                employment_status='active'
            ).exclude(id=self.instance.id if self.instance else None).select_related('user')
            self.fields['work_rule'].queryset = WorkRule.global_objects.filter(organisation=organisation)
        else:
            # Empty querysets if no organization context is passed
            self.fields['department'].queryset = Department.global_objects.none()
            self.fields['manager'].queryset = Employee.global_objects.none()
            self.fields['work_rule'].queryset = WorkRule.global_objects.none()

        # 🟢 Apply correct Bootstrap 5 classes to prevent rendering issues in template pages
        for field_name, field in self.fields.items():
            if isinstance(field.widget, forms.Select):
                field.widget.attrs.update({'class': 'form-select'})
            elif isinstance(field.widget, (forms.TextInput, forms.EmailInput, forms.PasswordInput, forms.ClearableFileInput)):
                field.widget.attrs.update({'class': 'form-control'})


# =====================
# SAFE DECIMAL HELPER
# =====================
def safe_decimal(value, default=Decimal("0.00")):
    try:
        if value in (None, "", " "):
            return default
        return Decimal(value)
    except (InvalidOperation, TypeError):
        return default

from django.shortcuts import get_object_or_404
@login_required
@hr_required
def employee_update(request, pk):
    # Retrieve employee safely and perform automated tenant validation
    if request.user.is_superuser:
        employee = get_object_or_404(
            Employee.global_objects.select_related(
                "user",
                "department",
                "manager",
                "work_rule"
            ),
            pk=pk
        )
    else:
        employee = get_object_or_404(
            Employee.objects.select_related(
                "user",
                "department",
                "manager",
                "work_rule"
            ),
            pk=pk
        )
    actor = getattr(request.user, 'employee', None)
    is_org_admin = request.user.is_superuser or bool(actor and actor.role == 'Admin')
    if not is_org_admin and (employee.user_id == request.user.pk or employee.role in ('Admin', 'HR', 'Finance')):
        return HttpResponseForbidden('Only Admin may edit privileged accounts or their own official employee record')
    user = employee.user
    account_before = {'username': user.username, 'first_name': user.first_name, 'last_name': user.last_name}
    active_org = employee.organisation if request.user.is_superuser else request.organisation  # Get current organisation [2]

    joining = getattr(employee, "joiningdetail", None)
    salary = getattr(employee, "salary_structure", None)

    if request.method == "POST":
        form = EmployeeUpdateForm(
            request.POST,
            request.FILES,
            instance=employee,
            organisation=active_org  # 👈 Pass organisation context [2]
        )

        # =====================
        # ROLE CHECK
        # =====================
        is_self = request.user == employee.user
        user_role = getattr(getattr(request.user, "employee", None), "role", "").lower()
        is_admin_or_finance = user_role in ["admin", "finance"] or request.user.is_superuser

        if form.is_valid():

            with transaction.atomic():

                # =====================
                # UPDATE USER
                # =====================
                new_username = request.POST.get("username", "").strip()
                new_password = request.POST.get("password", "")

                if new_username and new_username != user.username:
                    if User.objects.exclude(pk=user.pk).filter(username=new_username).exists():
                        messages.error(request, "Username already exists.")
                        return redirect("employee_update", pk=employee.pk)
                    user.username = new_username

                user.first_name = request.POST.get("first_name", user.first_name)
                user.last_name = request.POST.get("last_name", user.last_name)
                user.email = request.POST.get("email", user.email)

                if new_password:
                    from django.contrib.auth.password_validation import validate_password
                    from django.core.exceptions import ValidationError
                    try:
                        validate_password(new_password, user=user)
                    except ValidationError as exc:
                        messages.error(request, 'Password rejected: ' + '; '.join(exc.messages))
                        return redirect('employee_update', pk=employee.pk)
                    user.set_password(new_password)
                    if request.user.pk == user.pk:
                        update_session_auth_hash(request, user)

                user.save()
                account_after = {'username': user.username, 'first_name': user.first_name, 'last_name': user.last_name}
                if account_before != account_after or new_password:
                    from core.models import AuditEvent
                    from core.security.audit import audit_context
                    AuditEvent.global_objects.create(organisation_id=employee.organisation_id,
                        entity='User', object_id=str(user.pk), action='update',
                        before=account_before, after={**account_after, 'credentials_changed': bool(new_password)},
                        **audit_context.get())

                # =====================
                # UPDATE EMPLOYEE
                # =====================
                old_employee = Employee.objects.get(pk=employee.pk)
                employee = form.save(commit=False)

                desired_role = request.POST.get('role', old_employee.role)
                if desired_role not in dict(Employee.ROLE_CHOICES):
                    raise PermissionDenied('Invalid employee role')
                if not is_org_admin and desired_role != old_employee.role:
                    raise PermissionDenied('Only Admin can change employee roles')
                employee.role = desired_role

                # =====================
                # FACE IMAGE
                # =====================
                if form.cleaned_data.get("remove_face"):
                    if employee.face_image:
                        employee.face_image.delete(save=False)
                    employee.face_image = None
                    employee.face_encoding = None

                if request.FILES.get("face_image"):
                    if employee.face_image:
                        employee.face_image.delete(save=False)

                    from core.security.images import optimize_face_upload
                    employee.face_image = optimize_face_upload(request.FILES["face_image"])
                    employee.save(update_fields=["face_image"])

                    face_sys = get_face_system()
                    emb = face_sys.get_embedding(employee.face_image.path)

                    if emb is None:
                        employee.face_encoding = json.dumps({"status": "FACE_PENDING"})
                    else:
                        employee.face_encoding = json.dumps([emb])
                        face_sys.invalidate_tenant(employee.organisation.id)

                    employee.save(update_fields=["face_encoding"])

                employee.save()

                # =====================
                # SALARY
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

                if is_admin_or_finance:
                    salary.base_salary = safe_decimal(
                        request.POST.get("base_salary"),
                        salary.base_salary
                    )
                    salary.hra = safe_decimal(request.POST.get("hra"))
                    salary.allowances = safe_decimal(request.POST.get("allowances"))
                    salary.deductions = safe_decimal(request.POST.get("deductions"))
                    salary.save()
                else:
                    if is_self:
                        messages.warning(request, " Salary changes are not allowed.")

                # =====================
                # JOINING
                # =====================
                joining, _ = JoiningDetail.objects.get_or_create(employee=employee)

                files = request.FILES.getlist("documents")
                for file in files:
                    JoiningDocument.objects.create(joining=joining, file=file)

                delete_ids = request.POST.getlist("delete_docs")
                if delete_ids:
                    JoiningDocument.objects.filter(
                        id__in=delete_ids,
                        joining=joining
                    ).delete()

            messages.success(request, " Employee updated successfully.")

            if is_self:
                return redirect("my_profile")
            return redirect("employee_list")

        else:
            # =====================
            # FORM ERROR HANDLING (IMPORTANT)
            # =====================
            messages.error(request, "Please correct the highlighted form fields.")

    else:
        form = EmployeeUpdateForm(instance=employee, organisation=active_org)  # 👈 Pass organisation here [2]

    documents = joining.joining_documents.all() if joining and (
        request.user.is_superuser or request.user.employee.role in ('HR', 'Admin')
        or request.user.employee.pk == employee.pk) else []

    return render(request, "ems/employee_form.html", {
        "form": form,
        "employee": employee,
        "user_obj": user,
        "joining": joining,
        "documents": documents,
        "salary": salary if is_org_admin else None,
        "role_choices": Employee.ROLE_CHOICES,
    })


@login_required
@admin_required
def employee_delete(request, pk):
    employee = get_object_or_404(Employee, pk=pk)
    user = employee.user

    if request.method == "POST":
        if employee.user_id == request.user.pk:
            return HttpResponseForbidden('An administrator cannot deactivate their own account')
        with transaction.atomic():
            employee.is_active = False
            employee.employment_status = 'terminated'
            employee.date_of_resignation = employee.date_of_resignation or timezone.localdate()
            employee.save(update_fields=['is_active', 'employment_status', 'date_of_resignation'])
        messages.success(request, "Employee deactivated. Attendance, payroll and audit history retained.")
        return redirect("employee_list")

    return render(request, "ems/employee_confirm_delete.html", {
        "employee": employee
    })

@login_required
def inactive_employee_list(request):
    user_role = getattr(getattr(request.user, "employee", None), "role", "").lower()

    #  Only allowed roles
    if user_role in ["admin", "hr", "finance"] or request.user.is_superuser:
        employees = Employee.objects.exclude(
            employment_status__iexact="active"
        ).select_related("user", "department")
    else:
        # ❌ others should not see inactive employees
        employees = Employee.objects.none()

    return render(request, "ems/inactive_employee_list.html", {
        "employees": employees
    })

@login_required
@hr_required
def reactivate_employee(request, pk):
    employee = get_object_or_404(Employee, pk=pk)
    if not request.user.is_superuser and request.user.employee.role != 'Admin' and employee.role in ('Admin', 'HR', 'Finance'):
        return HttpResponseForbidden('Only Admin can reactivate privileged accounts')
    if request.method != 'POST':
        return render(request, 'confirm_mutation.html', {'title': 'Reactivate employee?', 'confirm_label': 'Reactivate'})
    employee.employment_status = "active"
    employee.is_active = True
    employee.date_of_resignation = None
    employee.save()

    messages.success(request, f"{employee.user.get_full_name()} reactivated successfully ")
    return redirect("inactive_employee_list")


@login_required
def dashboard(request):
    user = request.user
    today = timezone.localdate()

    # 🔐 ROLE-BASED EMPLOYEE VISIBILITY
    visible_employees = get_visible_employees(user).filter(
        is_active=True,
        employment_status="active"
    )

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

    # Base attendance queryset for today
    today_records = Attendance.objects.filter(
        employee__in=visible_employees,
        timestamp__date=today
    ).select_related('employee__user', 'employee__department').order_by("-timestamp")

    # Initialize tracking structures for today
    total_staff_count = visible_employees.count()
    present_today_ids = set()
    checked_in_now_ids = set()
    late_today_ids = set()

    # Sort and group logs by employee in memory to optimize database hits
    emp_today_logs = defaultdict(list)
    for log in today_records:
        emp_today_logs[log.employee_id].append(log)

    # Late threshold buffer (default 15 minutes grace period)
    grace_minutes = 15

    for emp_id_key, logs in emp_today_logs.items():
        logs_sorted = sorted(logs, key=lambda x: x.timestamp)
        first_in = next((l for l in logs_sorted if l.attendance_type == 'check_in'), None)
        last_out = next((l for l in reversed(logs_sorted) if l.attendance_type == 'check_out'), None)

        if first_in:
            present_today_ids.add(emp_id_key)

            # Determine if currently checked in (the last log was check_in or is after checkout)
            last_log = logs_sorted[-1]

            if last_log.attendance_type == "check_in":
                checked_in_now_ids.add(emp_id_key)

            # Detect late arrivals based on Assigned Shift Rule
            emp = first_in.employee
            rule = emp.work_rule
            if rule:
                shift_start = rule.shift_start_time
                local_in_time = timezone.localtime(first_in.timestamp).time()

                # Generate target cutoff datetime
                shift_start_dt = datetime.combine(today, shift_start)
                late_cutoff_dt = shift_start_dt + timedelta(minutes=grace_minutes)
                late_cutoff = late_cutoff_dt.time()

                if local_in_time > late_cutoff:
                    late_today_ids.add(emp_id_key)

    # Process Leaves Active Today
    leaves_today_qs = LeaveApplication.objects.filter(
        employee__in=visible_employees,
        status='approved',
        start_date__lte=today,
        end_date__gte=today
    ).select_related('employee__user', 'employee__department')
    leave_today_ids = set(leaves_today_qs.values_list('employee_id', flat=True))

    # staff who have not checked in and are not on approved leave today
    not_marked_qs = visible_employees.exclude(
        id__in=present_today_ids | leave_today_ids
    ).select_related('user', 'department')

    # Fetch complete relational detail sets for administrative modal reviews
    on_duty_details = visible_employees.filter(id__in=checked_in_now_ids).select_related('user', 'department')
    late_details = visible_employees.filter(id__in=late_today_ids).select_related('user', 'department')
    present_details = visible_employees.filter(id__in=present_today_ids).select_related('user', 'department')

    # Departmental staffing rates
    department_stats = []

    departments = (
        visible_employees
        .values(
            "department__id",
            "department__name"
        )
        .annotate(total_count=Count("id"))
    )

    for dept in departments:
        dept_id = dept["department__id"]
        
        # Query list of all active employees in this department
        dept_emps = visible_employees.filter(department_id=dept_id).select_related('user', 'department')
        # Filter down to those present today
        dept_present_emps = dept_emps.filter(id__in=present_today_ids)

        department_stats.append({
            "department_name": dept["department__name"] or "Unassigned",
            "total_count": dept["total_count"],
            "present_count": dept_present_emps.count(),
            "presence_rate": round(
                (dept_present_emps.count() / dept["total_count"]) * 100
            ) if dept["total_count"] else 0,
            # Adding objects for modal viewing
            "managed_employees": dept_emps,
            "present_employees": dept_present_emps,
        })

    # Individual Employee Analytics Dashboard Indicators
    current_status = None
    total_hours_today = 0
    total_hours_week = 0
    total_hours_month = 0
    last_checkin_record = None
    attendance_percentage = 0
    leave_stats = None
    recent_payroll = None

    if ems_emp:
        emp_attendance = Attendance.objects.filter(employee=ems_emp)
        today_logs = emp_attendance.filter(timestamp__date=today).order_by("timestamp")
        last_checkin_record = today_logs.filter(attendance_type="check_in").last()
        last_checkout_record = today_logs.filter(attendance_type="check_out").last()

        if last_checkin_record and (not last_checkout_record or last_checkout_record.timestamp < last_checkin_record.timestamp):
            current_status = "Checked In"
        elif last_checkout_record:
            current_status = "Checked Out"
        else:
            current_status = "Not Checked In"

        from core.services.calendar import calendar_days
        start_week = today - timedelta(days=today.weekday())
        start_month = today.replace(day=1)
        summaries = calendar_days(ems_emp, min(start_week, start_month), today)
        total_hours_today = sum(d['hours'] for d in summaries if d['date'] == today)
        total_hours_week = sum(d['hours'] for d in summaries if d['date'] >= start_week)
        total_hours_month = sum(d['hours'] for d in summaries if d['date'] >= start_month)

        days_passed = today.day
        attendance_days = (
            emp_attendance
            .filter(timestamp__date__gte=start_month, attendance_type="check_in")
            .values("timestamp__date")
            .distinct()
            .count()
        )
        attendance_percentage = round((attendance_days / days_passed) * 100, 1) if days_passed else 0

        leave_qs = LeaveApplication.objects.filter(
            employee=ems_emp,
            start_date__year=today.year
        )

        leave_stats = {
            "total": leave_qs.count(),
            "approved": leave_qs.filter(status="approved").count(),
            "pending": leave_qs.filter(status="pending").count(),
            "rejected": leave_qs.filter(status="rejected").count(),
        }
        if user.is_superuser or (hasattr(user, 'employee') and
                (user.employee.role in ('Admin', 'Finance') or user.employee.pk == ems_emp.pk)):
            recent_payroll = Payroll.objects.filter(employee=ems_emp).order_by("-month")[:6]

    # Global Average trust score calculation
    avg_confidence = Attendance.objects.filter(
        employee__in=visible_employees,
        timestamp__date=today
    ).aggregate(avg=Avg("confidence_score"))["avg"] or 0

    # Verification counts for the rolling 30-day analytics chart
    start_30 = today - timedelta(days=29)
    att_30 = (
        Attendance.objects.filter(
            employee__in=visible_employees,
            timestamp__date__gte=start_30,
            attendance_type="check_in"
        )
        .annotate(day=TruncDate("timestamp"))
        .values("day")
        .annotate(count=Count("employee", distinct=True))
    )
    att_map = {a["day"]: a["count"] for a in att_30}
    chart_labels = [(start_30 + timedelta(days=i)).strftime("%d %b") for i in range(30)]
    chart_values = [att_map.get(start_30 + timedelta(days=i), 0) for i in range(30)]

    context = {
        "is_admin": is_admin,
        "employees": visible_employees.order_by("user__first_name"),
        "employee": ems_emp,
        "selected_emp_id": ems_emp.id if ems_emp else None,
        
        # Biometric Streams & Today's Headcounts
        "today_attendance": today_records,
        "present_count": len(present_today_ids),
        "on_duty_count": len(checked_in_now_ids),
        "absent_count": not_marked_qs.count(),
        "late_count": len(late_today_ids),
        
        # Details sets for management lists
        "on_duty_employees": on_duty_details,
        "not_marked_employees": not_marked_qs,
        "late_employees": late_details,
        "present_details": present_details,
        "leaves_today": leaves_today_qs,
        
        # Department metrics
        "department_stats": department_stats,
        
        # Personal Context Info
        "avg_confidence": round(avg_confidence * 100, 1) if avg_confidence else 0.0,
        "total_employees": total_staff_count,
        "current_status": current_status,
        "last_checkin": last_checkin_record,
        "total_hours": total_hours_today,
        "total_hours_week": total_hours_week,
        "total_hours_month": total_hours_month,
        "attendance_percentage": attendance_percentage,
        "leave_stats": leave_stats,
        "recent_payroll": recent_payroll,
        
        # Line Chart Metrics
        "attendance_labels_json": json.dumps(chart_labels),
        "attendance_values_json": json.dumps(chart_values),
        
        "year": today.year,
        "month": today.month,
        "title": "Hospital Command Center",
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
def employee_profile(request, pk):
    from core.utils.access import get_visible_employees
    employee = get_object_or_404(
        Employee.objects.select_related(
            "user",
            "department",
            "manager",
            "work_rule"
        ),
        pk=pk
    )
    if not get_visible_employees(request.user).filter(pk=employee.pk).exists():
        raise PermissionDenied('Not authorised to view this employee')
    can_view_finance = request.user.is_superuser or (hasattr(request.user, 'employee') and
          (request.user.employee.role in ('Admin', 'Finance') or request.user.employee.pk == employee.pk))
    today = date.today()
    salary = getattr(employee, "salary_structure", None) if can_view_finance else None
    joining = getattr(employee, "joiningdetail", None)
    documents = joining.joining_documents.all() if joining and (
        request.user.is_superuser or request.user.employee.role in ('HR', 'Admin')
        or request.user.employee.pk == employee.pk) else []

    return render(request, "ems/employee_profile.html", {
        "employee": employee,
        "salary": salary,
        "joining": joining,
        "documents": documents,
        "year": today.year,   
        "month": today.month, 
    })



@login_required
@hr_finance_required
def employee_id_card(request, pk):

    employee = get_object_or_404(
        Employee.objects.select_related(
            "user",
            "department",
            "manager",
            "work_rule",
            "organisation"
        ),
        pk=pk
    )

    html_string = render_to_string(
        "ems/employee_id_card.html",
        {
            "employee": employee,
            "organisation": employee.organisation,
        }
    )

    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = (
        f'inline; filename="ID_{employee.employee_id}.pdf"'
    )

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

    if getattr(request, 'organisation', None) is not None:
        attendance_qs = attendance_qs.filter(organisation=request.organisation)
        if emp_id:
            employee = Employee.objects.filter(id=emp_id, organisation=request.organisation).first()
            attendance_qs = Attendance.objects.filter(employee=employee, organisation=request.organisation)

    labels = []
    present = []
    absent = []

    total_employees = Employee.objects.filter(organisation=request.organisation).count() if not emp_id and getattr(request, 'organisation', None) is not None else (Employee.objects.count() if not emp_id else 1)

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
    if isinstance(obj, (_np.bool_,)):
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
    current_org = get_current_organisation()

    # 1) Try InsightFace pipeline
    fs_if = get_if_system()
    if fs_if is not None:
        try:
            emp_id, score, bbox = fs_if.recognize_from_frame(frame_bgr)
            if emp_id:
                emp_qs = Employee.objects.filter(employee_id=emp_id, is_active=True)
                if current_org is not None:
                    emp_qs = emp_qs.filter(organisation=current_org)
                emp = emp_qs.first()
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
                emp_qs = Employee.objects.filter(employee_id=emp_id, is_active=True)
                if current_org is not None:
                    emp_qs = emp_qs.filter(organisation=current_org)
                emp = emp_qs.first()
                if emp:
                    return emp, float(score), None, "face_recognition"
        except (ProgrammingError, OperationalError):
            return None, 0.0, None, "db-not-ready"
        except Exception:
            pass

    return None, 0.0, None, "none"



def attendance_heartbeat(request):
    return JsonResponse({
        "status": "ok",
        "ts": time.time()
    })

# ------------------  Final mark_attendance Endpoint ------------------
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
@csrf_protect
@login_required
@require_POST
def mark_attendance(request):
    if request.method != "POST":
        return JsonResponse({"success": False, "message": "Method not allowed"}, status=405)

    if "capture" not in request.FILES:
        return JsonResponse({"success": False, "message": "No image data"}, status=400)

    active_org = request.organisation
    if active_org is None or not active_org.is_active:
        return JsonResponse({'success': False, 'error': 'tenant_required'}, status=403)
    if settings.REQUIRE_VERIFIED_LIVENESS and not settings.FACE_LIVENESS_VERIFIER:
        return JsonResponse({'success': False, 'error': 'liveness_required',
            'message': 'Biometric attendance is locked until a verified anti-spoofing (PAD) integration is configured; contact HR.'}, status=503)
    upload = request.FILES['capture']
    from core.security.images import decode_face_upload
    try:
        frame = decode_face_upload(upload)
    except ValueError as exc:
        return JsonResponse({'success': False, 'error': 'invalid_capture', 'message': str(exc)}, status=400)
    if settings.REQUIRE_VERIFIED_LIVENESS:
        from core.security.liveness import verify_liveness, LivenessUnavailable
        try:
            if not verify_liveness(frame, request):
                return JsonResponse({'success': False, 'error': 'liveness_failed',
                    'message': 'Live presence could not be verified. Try a new capture.'}, status=403)
        except LivenessUnavailable:
            return JsonResponse({'success': False, 'error': 'liveness_unavailable'}, status=503)
    try:
        # ----------------------------------------------------
        # 1️⃣ Decode Image (Concurrent and memory-isolated)
        # ----------------------------------------------------
        # Image is pre-validated, decoded and downscaled before model inference.

        # ----------------------------------------------------
        # 2️⃣ Face Recognition (Non-blocking execution)
        # ----------------------------------------------------
        fs = get_face_system()
        
        # We removed the global serialized block. The face recognition system
        # now manages multi-tenant isolation internally and safely.
        if not request.session.session_key:
            request.session.save()
        client_scan_id = request.POST.get('scan_session', '')
        import uuid
        try:
            parsed_session = str(uuid.UUID(client_scan_id))
            if client_scan_id.lower() != parsed_session:
                raise ValueError('Non-canonical session ID')
        except (ValueError, TypeError, AttributeError):
            return JsonResponse({'success': False, 'error': 'scan_session_required'}, status=400)
        stable_session = f'{request.session.session_key}:{client_scan_id}'
        try:
            emp_id, score, meta = fs.recognize_from_frame(frame, organisation_id=active_org.id,
                                                          session_id=stable_session)
        except BusyInference:
            response = JsonResponse({'success': False, 'error': 'scanner_busy'}, status=503)
            response['Retry-After'] = '2'
            return response
        except ModelUnavailable:
            return JsonResponse({'success': False, 'error': 'model_unavailable',
                'message': 'Face recognition model could not be loaded. Ask the administrator to check the model files and server log.'}, status=503)

        if not emp_id:
            reason = meta.get("reason", "")
            msg = "Face not recognized"
            color = "red"

            if reason == 'MULTIPLE_FACES':
                msg = 'Only one person may scan at a time'
                color = 'yellow'
            elif reason == "TOO_FAR":
                msg = "Come Closer"
                color = "yellow"
            elif reason == "ACCUMULATING":
                msg = "Hold Still..."
                color = "blue"
            elif reason == "NO_FACE":
                msg = "No Face Detected"
            elif reason == 'NO_ENROLLED_FACES':
                msg = 'No usable face enrolments in this organisation. Ask HR to enrol or repair employee photos.'
                color = 'yellow'

            return JsonResponse({
                "success": False,
                "error": reason.lower() or 'unknown_face',
                "message": msg,
                "color": color
            })

        # ----------------------------------------------------
        # 3️⃣ Employee Validation
        # ----------------------------------------------------
        employee = Employee.objects.filter(
            organisation=active_org,
            pk=emp_id,
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

        try:
            lat, lng = float(lat), float(lng)
            if not (np.isfinite(lat) and np.isfinite(lng) and -90 <= lat <= 90 and -180 <= lng <= 180):
                raise ValueError('Invalid coordinate range')
        except (TypeError, ValueError):
            return JsonResponse({'success': False, 'error': 'invalid_coordinates'}, status=400)

        if not employee.assigned_location:
            return JsonResponse({
                "success": False,
                "message": "No office location assigned to your profile",
                "color": "red"
            })

        active_offices = list(OfficeLocation.objects.filter(organisation=active_org, is_active=True))

        matched_office = next(
            (o for o in active_offices if is_inside_office(lat, lng, o)),
            None
        )

        if not matched_office:
            nearest = min(active_offices, key=lambda office: distance_to_office_m(lat, lng, office), default=None)
            message = "You are not inside any authorized office location"
            if nearest:
                distance = round(distance_to_office_m(lat, lng, nearest))
                accuracy = request.POST.get("accuracy")
                try:
                    accuracy = float(accuracy)
                    if not np.isfinite(accuracy) or accuracy < 0:
                        accuracy = None
                except (TypeError, ValueError):
                    accuracy = None
                message += f". Closest: {nearest.name}, about {distance} m away (radius {nearest.radius_meters} m)"
                if accuracy is not None:
                    message += f"; GPS accuracy ±{round(accuracy)} m"
                message += ". Refresh your location or ask your organisation admin to verify the office pin and radius"
            return JsonResponse({
                "success": False,
                "message": message,
                "color": "red"
            })

        if employee.location_type == "INDOOR":
            if matched_office.id != employee.assigned_location_id:
                return JsonResponse({
                    "success": False,
                    "message": "Attendance allowed only at your assigned office",
                    "color": "red"
                })

        # ----------------------------------------------------
        # 🔒 Attendance database operations
        # ----------------------------------------------------
        with transaction.atomic():
            # Locks the employee row across *all* workers to serialize scan decisions.
            employee = Employee.objects.select_for_update(of=('self',)).select_related('user', 'work_rule').get(
                pk=employee.pk, organisation=active_org, is_active=True, employment_status='active', user__is_active=True)
            now = timezone.now()
            user_name = employee.user.get_full_name()
            last_scan = Attendance.objects.filter(organisation=active_org, employee=employee).order_by('-timestamp').first()
            if last_scan and (now - last_scan.timestamp).total_seconds() < settings.COOLDOWN_SECONDS:
                return JsonResponse({'success': False, 'error': 'scan_cooldown',
                    'message': 'Already scanned recently; please wait'}, status=409)

            settings_obj = AttendanceSettings.objects.first()
            min_hours = settings_obj.min_hours_before_checkout if settings_obj else 0
            REST_HOURS_AFTER_CHECKOUT = 8
            DOUBLE_SCAN_COOLDOWN_MINUTES = 10

            open_checkin = get_open_checkin(employee)
            if open_checkin and (now - open_checkin.timestamp).total_seconds() > 20 * 3600:
                return JsonResponse({'success': False, 'error': 'attendance_review_required',
                    'message': 'An earlier check-in is missing a checkout. Contact HR to reconcile it.'}, status=409)

            if is_flexible:
                today = timezone.localdate(now)
                today_records = Attendance.objects.filter(
                    employee=employee,
                    timestamp__date=today
                ).order_by("timestamp")

                today_checkin_record = today_records.filter(attendance_type="check_in").first()
                today_checkout = today_records.filter(attendance_type="check_out").exists()

                if today_checkout:
                    return JsonResponse({
                        "success": False,
                        "message": "Attendance already completed for today",
                        "color": "yellow"
                    })

                if today_checkin_record:
                    min_checkout_gap = timedelta(minutes=10)
                    if now - today_checkin_record.timestamp < min_checkout_gap:
                        return JsonResponse({
                            "success": False,
                            "message": "Checkout not allowed yet",
                            "color": "orange"
                        })
                    att_type = "check_out"
                else:
                    att_type = "check_in"
            else:
                if open_checkin:
                    worked_hours = (now - open_checkin.timestamp).total_seconds() / 3600

                    if worked_hours < min_hours:
                        mins_left = int((min_hours - worked_hours) * 60)
                        hours_left = mins_left // 60
                        minutes_left = mins_left % 60
                        
                        message = f"Wait {hours_left}h {minutes_left}m to checkout" if hours_left > 0 else f"Wait {minutes_left} minutes to checkout"
                        return JsonResponse({
                            "success": False,
                            "message": message,
                            "color": "yellow"
                        })
                    att_type = "check_out"
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

                            message = f"Next check-in allowed after {hours_left}h {minutes_left}m" if hours_left > 0 else f"Next check-in allowed after {minutes_left} minutes"
                            return JsonResponse({
                                "success": False,
                                "message": message,
                                "color": "yellow"
                            })
                    att_type = "check_in"

            Attendance.objects.create(
                organisation=active_org,
                employee=employee,
                attendance_type=att_type,
                confidence_score=score,
                location=matched_office.name
            )

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
    today = timezone.localdate()   # use local date instead of UTC

    logs = (
        Attendance.objects.filter(organisation=request.organisation, timestamp__date=today)
        .select_related("employee__user")
        .order_by("-timestamp")
    )

    data = [
        {
            "name": log.employee.user.get_full_name() or log.employee.user.username,
            "type": log.attendance_type.replace("_", " ").title(),
            "time": timezone.localtime(log.timestamp).strftime("%I:%M %p"),  # convert to IST
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


@login_required
def attendance_page(request):
    from core.security.kiosk import can_manage_kiosk, is_locked
    if getattr(request, 'organisation', None) is None:
        raise PermissionDenied('Choose an organisation-specific account to use the scanner')
    return render(request, 'attendance.html', {
        'face_only_mode': not settings.REQUIRE_VERIFIED_LIVENESS,
        'liveness_unconfigured': settings.REQUIRE_VERIFIED_LIVENESS and not settings.FACE_LIVENESS_VERIFIER,
        'kiosk_locked': is_locked(request),
        'can_manage_kiosk': can_manage_kiosk(request),
    })


# views.py

@login_required
def video_feed_view(request):
    # Never share a server-side /dev/video camera with remote organisation clients.
    # Browser getUserMedia powers the local kiosk instead.
    return JsonResponse({'ok': False, 'error': 'server_camera_stream_disabled'}, status=410)


def video_feed(organisation_id):  # <-- Update signature
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logger.error("Camera not accessible")
        return

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
                    
                    # FIX: Use global_objects and filter strictly by active organisation
                    employees = Employee.global_objects.filter(
                        organisation_id=organisation_id
                    ).exclude(Q(face_encoding__isnull=True) | Q(face_encoding__exact=''))

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

    for record in attendance_qs:
        work_rule = getattr(record.employee, "work_rule", None)

        if work_rule:
            # Shift type from model property
            record.shift = work_rule.shift_type

            # Format timing
            record.shift_time = (
                f"{work_rule.shift_start_time.strftime('%I:%M %p')} - "
                f"{work_rule.shift_end_time.strftime('%I:%M %p')}"
            )
        else:
            record.shift = "N/A"
            record.shift_time = "Not Assigned"

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


@login_required
@hr_required
def attendance_settings(request):
    active_org = request.organisation
    if not active_org:
        # Fallback security check if a superuser with no employee profile tries to access settings
        return HttpResponseForbidden("You must be logged in as an organization administrator to access settings.")

    # FIX: Query or create settings mapped strictly to the active organisation
    settings_obj, created = AttendanceSettings.objects.get_or_create(
        organisation=active_org
    )

    if request.method == 'POST':
        form = AttendanceSettingsForm(request.POST, instance=settings_obj)
        if form.is_valid():
            form.save()
            messages.success(request, "Attendance settings updated successfully.")
            return redirect('attendance_settings')
    else:
        form = AttendanceSettingsForm(instance=settings_obj)

    return render(request, 'settings.html', {'form': form})

@login_required
def get_attendance_history(request):
    try:
        days = max(1, min(366, int(request.GET.get('days', 30))))
    except (TypeError, ValueError):
        return JsonResponse({'error': 'Invalid day range'}, status=400)
    start_date = timezone.localdate() - timedelta(days=days)
    attendance_qs = Attendance.objects.filter(employee__in=get_visible_employees(request.user),
        timestamp__date__gte=start_date).select_related('employee__user').order_by('-timestamp')

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
    try:
        days = max(1, min(366, int(request.GET.get('days', 30))))
    except (TypeError, ValueError):
        return JsonResponse({'error': 'Invalid day range'}, status=400)
    start_date = timezone.localdate() - timedelta(days=days)
    attendance_qs = Attendance.objects.filter(employee__in=get_visible_employees(request.user),
        timestamp__date__gte=start_date).select_related('employee__user').order_by('-timestamp')

    context = {
        'attendance_records': attendance_qs,
        'days': days,
    }
    return render(request, 'attendance_history.html', context)


from core.attendance_views import (
    attendance_calendar, attendance_calendar_data, attendance_day_detail,
    create_attendance, update_attendance, delete_attendance,
)


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
@hr_required
def office_location_create(request):
    if request.method == "POST":
        form = OfficeLocationForm(request.POST)
        if form.is_valid():
            loc = form.save(commit=False)
            loc.organisation = request.organisation  # <-- Set Organisation
            loc.save()
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
@hr_required
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
@hr_required
def office_location_delete(request, pk):
    location = get_object_or_404(OfficeLocation, pk=pk)
    if request.method == "POST":
        location.delete()
        messages.success(request, "Office location deleted successfully.")
    return redirect("office_location_list")


# ============================================================
# Organisation structure views live in core.org_chart.
# Preserve these import paths for existing callers.
# ============================================================
from .org_chart import org_chart_page, org_tree_api_me, org_tree_api

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
@hr_required
def department_create(request):
    active_org = request.organisation
    if request.method == 'POST':
        # Pass organisation parameter here
        form = DepartmentForm(request.POST, organisation=active_org)
        if form.is_valid():
            dept = form.save(commit=False)
            dept.organisation = active_org
            dept.save()
            messages.success(request, 'Department created successfully!')
            return redirect('department_list')
    else:
        # Pass organisation parameter here
        form = DepartmentForm(organisation=active_org)
    
    return render(request, 'ems/department_form.html', {
        'form': form,
        'title': 'Create Department'
    })

@login_required
@hr_required
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
@hr_required
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
@finance_required
def salary_structure_list(request):
    structures = SalaryStructure.objects.select_related('employee__user')
    return render(request, 'ems/salary_structure_list.html', {
        'structures': structures,
        'title': 'Salary Structures'
    })

@login_required
@finance_required
def salary_structure_create(request):
    active_org = request.organisation
    if request.method == 'POST':
        # Pass organisation parameter here
        form = SalaryStructureForm(request.POST, organisation=active_org)
        if form.is_valid():
            struct = form.save(commit=False)
            struct.organisation = active_org
            struct.save()
            messages.success(request, 'Salary structure saved successfully.')
            return redirect('salary_structure_list')
    else:
        # Pass organisation parameter here
        form = SalaryStructureForm(organisation=active_org)

    return render(request, 'ems/salary_structure_form.html', {
        'form': form,
        'title': 'Create Salary Structure'
    })


@login_required
def payroll_list(request):
    form = PayrollFilterForm(request.GET or None)

    # Salaries are sensitive: managers must not see subordinate salary records.
    # Finance and organisation admins can view the tenant; others see only self.
    if request.organisation is None:
        return HttpResponseForbidden('Organisation membership required')
    my_profile = getattr(request.user, 'employee', None)
    finance_access = my_profile and my_profile.role in ('Finance', 'Admin')
    visible_employees = (Employee.objects.filter(organisation=request.organisation)
                         if finance_access else Employee.objects.filter(
                             organisation=request.organisation, user=request.user))

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
        'regular_overtime_pay',
        'holiday_overtime_pay',
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

    from django.core.paginator import Paginator
    payrolls = Paginator(payrolls.order_by('-month', 'employee_id'), 40).get_page(request.GET.get('page'))

    return render(request, 'ems/payroll_list.html', {
        'payrolls': payrolls,
        'form': form,
        'totals': totals,
        'title': 'Payroll'
    })



@login_required
@finance_required
def generate_payroll(request):
    if request.organisation is None:
        return HttpResponseForbidden('Organisation membership required')
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
                result = Payroll.objects.generate_salary(year=year, month=month, organisation=request.organisation)
                if result['created'] + result['updated']:
                    messages.success(request, f"Payroll recalculated for {calendar.month_name[month]} {year}")
                else:
                    messages.warning(request, 'No unpaid payroll was generated; check missing setup or locked paid records.')
            else:
                emp = Employee.objects.get(id=employee_id, organisation=request.organisation)
                result = Payroll.objects.generate_salary(year=year, month=month, employee=emp, organisation=request.organisation)
                if result['created'] + result['updated']:
                    messages.success(request, f"Payroll generated for {emp.user.get_full_name()} ({calendar.month_name[month]} {year})")
                else:
                    messages.warning(request, 'Payroll not changed: missing setup or a previously paid record.')

            messages.info(request, f"Created: {result['created']}; updated: {result['updated']}; "
                                   f"paid records protected: {result['skipped_paid']}; "
                                   f"missing salary setup: {result['missing_setup']}.")
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

from core.payroll_views import payroll_update, payroll_delete, pay_salary


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
def download_salary_slip(request, pk):
    payroll = get_object_or_404(
        Payroll.objects.select_related('employee__user'),
        pk=pk, organisation=request.organisation
    )

    # 🔒 Unified multi-tenant permission check
    allow = False
    if hasattr(request.user, 'employee'):
        role = request.user.employee.role
        if role in ['Finance', 'Admin']:
            allow = True
        if payroll.employee_id == request.user.employee.id:
            allow = True
    elif request.user.is_superuser:
        allow = True

    if not allow:
        messages.error(request, "Access denied.")
        return redirect('my_salary')

    pdf_buffer, filename = generate_payslip_pdf(payroll)
    return FileResponse(pdf_buffer, as_attachment=True, filename=filename,
                        content_type='application/pdf')


@login_required
def payroll_slip_pdf(request, pk):
    payroll = get_object_or_404(
        Payroll.objects.select_related('employee__user'),
        pk=pk, organisation=request.organisation
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
    # Managers and HR can view employment data, but not payroll by default.
    if not request.user.is_superuser:
        own_employee = getattr(request.user, 'employee', None)
        if own_employee is None:
            return HttpResponseForbidden('Organisation membership required')
        if own_employee.role not in ('Finance', 'Admin') and own_employee.pk != employee_id:
            return HttpResponseForbidden('Not authorised to view this salary')
    if not Employee.objects.filter(pk=employee_id).exists() and not request.user.is_superuser:
        return HttpResponseForbidden('Employee does not belong to your organisation')

    emp = get_object_or_404(
        Employee.objects.select_related('user', 'work_rule'),
        id=employee_id
    )

    payrolls = Payroll.objects.filter(
        employee=emp
    ).order_by('-month')

    # -------------------------------------------------
    #  AUTO-FIX PAID DAYS (UNCHANGED LOGIC)
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
@hr_required
def leave_type_create(request):
    if request.method == 'POST':
        form = LeaveTypeForm(request.POST)
        if form.is_valid():
            leave_type = form.save(commit=False)
            leave_type.organisation = request.organisation  # <-- Set Organisation
            leave_type.save()
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
                employee_name = employee.user.get_full_name() or employee.user.get_username()
                create_user_notification(
                    approver.user,
                    employee.organisation_id,
                    f"Leave request from {employee_name} ({employee.employee_id}) needs your review.",
                    reverse('leave_application_list'),
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
def leave_approval_action(request, pk, action):
    """
    Approve or reject a leave application.

    URL Examples:
        /leaves/3/approve/
        /leaves/3/reject/
    """

    leave_app = get_object_or_404(LeaveApplication, pk=pk)
    if action not in ('approve', 'reject'):
        raise Http404()

    # Logged-in employee
    try:
        approver = request.user.employee
    except Employee.DoesNotExist:
        return HttpResponseForbidden("Employee profile not found.")

    # Check whether this employee is allowed to approve
    approval = LeaveApproval.objects.filter(
        leave=leave_app,
        approver=approver,
        status="pending"
    ).first()

    if not approval:
        messages.error(request, "You are not authorized to approve this leave.")
        return redirect("leave_application_list")

    if request.method != 'POST':
        return render(request, 'confirm_mutation.html',
                      {'title': f'{action.title()} leave request?', 'confirm_label': action.title()})

    # -------------------------
    # APPROVE
    # -------------------------
    if action == "approve":
        approval.status = "approved"
        approval.acted_at = timezone.now()
        approval.save()

        next_stage = LeaveWorkflowStage.objects.filter(
            level=approval.level + 1
        ).first()

        if next_stage:
            next_approver = (
                leave_app.employee.manager
                or Employee.objects.filter(
                    role=next_stage.role_name
                ).first()
            )

            if next_approver:
                LeaveApproval.objects.create(
                    leave=leave_app,
                    approver=next_approver,
                    level=next_stage.level,
                    status="pending",
                )
                employee = leave_app.employee
                employee_name = employee.user.get_full_name() or employee.user.get_username()
                create_user_notification(
                    next_approver.user,
                    leave_app.organisation_id,
                    f"Leave request from {employee_name} ({employee.employee_id}) needs your review.",
                    reverse('leave_application_list'),
                )
        else:
            leave_app.status = "approved"
            leave_app.approved_by = approver
            leave_app.save()
            create_user_notification(
                leave_app.employee.user,
                leave_app.organisation_id,
                "Your leave request was approved.",
                reverse('leave_application_detail', args=[leave_app.pk]),
            )

        messages.success(request, "Leave approved successfully.")
        return redirect("leave_application_list")

    # -------------------------
    # REJECT
    # -------------------------
    elif action == "reject":
        approval.status = "rejected"
        approval.acted_at = timezone.now()
        approval.save()

        leave_app.status = "rejected"
        leave_app.approved_by = approver
        leave_app.save()
        create_user_notification(
            leave_app.employee.user,
            leave_app.organisation_id,
            "Your leave request was rejected.",
            reverse('leave_application_detail', args=[leave_app.pk]),
        )

        messages.success(request, "Leave rejected successfully.")
        return redirect("leave_application_list")

    # -------------------------
    # INVALID ACTION
    # -------------------------
    messages.error(request, "Invalid approval action.")
    return redirect("leave_application_list")


@login_required
def leave_workflow_list(request):
    stages = LeaveWorkflowStage.objects.all().order_by('level')
    return render(request, 'ems/leave_workflow_list.html', {
        'stages': stages,
        'title': 'Leave Workflow Stages'
    })

@login_required
@hr_required
def leave_workflow_create(request):
    if request.method == 'POST':
        form = LeaveWorkflowStageForm(request.POST)
        if form.is_valid():
            stage = form.save(commit=False)
            stage.organisation = request.organisation  # <-- Set Organisation
            stage.save()
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
@hr_required
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

#  Download Individual Document
@login_required
def download_document(request, doc_id):
    doc = get_object_or_404(JoiningDocument, id=doc_id)
    if not (request.user.is_superuser or request.user.employee.role in ('Admin','HR')
            or doc.joining.employee.user_id == request.user.pk):
        raise PermissionDenied('Private document')
    file_path = doc.file.path

    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=os.path.basename(file_path))
    else:
        messages.error(request, "File not found.")
        return redirect('joining_detail_list')

#  Delete a Document
@login_required
@hr_required
@require_POST
def delete_document(request, doc_id):
    doc = get_object_or_404(JoiningDocument, id=doc_id)

    if request.user.is_superuser or request.user.employee.role in ('Admin', 'HR'):
        file_path = doc.file.path
        doc.delete()
        if os.path.exists(file_path):
            os.remove(file_path)
        messages.success(request, "Document deleted successfully.")
    else:
        messages.error(request, "Permission denied.")

    return redirect('joining_detail_list')

#  Download all documents as a ZIP
@login_required
def download_all_documents(request, detail_id):
    detail = get_object_or_404(JoiningDetail, id=detail_id)
    if not (request.user.is_superuser or request.user.employee.role in ('Admin','HR')
            or detail.employee.user_id == request.user.pk):
        raise PermissionDenied('Private document')
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
@hr_required
def joining_detail_create(request):
    active_org = request.organisation
    if request.method == 'POST':
        # Pass organisation parameter here
        form = JoiningDetailForm(request.POST, organisation=active_org)
        if form.is_valid():
            joining = form.save(commit=False)
            joining.organisation = active_org
            joining.save()

            # Handle document upload
            for file in request.FILES.getlist('documents'):
                JoiningDocument.objects.create(
                    organisation=active_org,
                    joining=joining,
                    file=file
                )

            messages.success(request, 'Joining details added successfully!')
            return redirect('joining_detail_list')
    else:
        # Pass organisation parameter here
        form = JoiningDetailForm(organisation=active_org)

    return render(request, 'ems/joining_detail_form.html', {
        'form': form,
        'title': 'Add Joining Details'
    })


@login_required
@hr_required
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
            resignation.organisation = request.organisation
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
@hr_required
@require_POST
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
    notification_queryset = Notification.objects.filter(recipient=request.user).order_by('-created_at', '-pk')
    unread_count = notification_queryset.filter(is_read=False).count()
    page_obj = Paginator(notification_queryset, 30).get_page(request.GET.get('page'))
    
    return render(request, 'ems/notification_list.html', {
        'notifications': page_obj.object_list,
        'page_obj': page_obj,
        'unread_count': unread_count,
        'title': 'Notifications'
    })

@login_required
@require_GET
def notification_open(request, pk):
    notification = get_object_or_404(
        Notification.objects.filter(recipient=request.user),
        pk=pk,
    )
    if not notification.is_read:
        Notification.objects.filter(
            pk=notification.pk,
            recipient=request.user,
            is_read=False,
        ).update(is_read=True)

    target = notification.link
    if target and url_has_allowed_host_and_scheme(
        target,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return redirect(target)
    return redirect('notification_list')


@login_required
@require_POST
def mark_notification_read(request, pk):
    notification = get_object_or_404(Notification, pk=pk, recipient=request.user)
    if not notification.is_read:
        Notification.objects.filter(pk=notification.pk, recipient=request.user).update(is_read=True)
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'success': True})
    
    return redirect('notification_list')

@login_required
@require_POST
def mark_all_notifications_read(request):
    Notification.objects.filter(recipient=request.user, is_read=False).update(is_read=True)
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'success': True})
    
    return redirect('notification_list')



def notify_user(user: User, message: str, link_path: str = ""):
    employee = Employee.global_objects.filter(user_id=user.pk).only('organisation_id').first()
    if employee is None:
        return None
    return create_user_notification(user, employee.organisation_id, message, link_path)


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
@hr_finance_required
def workrule_create(request):
    if request.method == "POST":
        form = WorkRuleForm(request.POST)

        if form.is_valid():
            rule = form.save(commit=False)
            rule.organisation = request.organisation
            rule.save()
            messages.success(request, "Work rule created successfully.")
            return redirect("workrule_list")

        # Show all field errors as messages
        for field, errors in form.errors.items():
            field_name = (
                form.fields[field].label
                if field in form.fields
                else field.replace("__all__", "Form")
            )

            for error in errors:
                messages.error(request, f"{field_name}: {error}")

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
@hr_finance_required
def workrule_update(request, pk):
    rule = get_object_or_404(WorkRule, pk=pk)

    if request.method == "POST":
        form = WorkRuleForm(request.POST, instance=rule)

        if form.is_valid():
            form.save()
            messages.success(request, "Work rule updated successfully.")
            return redirect("workrule_list")

        for field, errors in form.errors.items():
            field_name = (
                form.fields[field].label
                if field in form.fields
                else field.replace("__all__", "Form")
            )

            for error in errors:
                messages.error(request, f"{field_name}: {error}")

    else:
        form = WorkRuleForm(instance=rule)

    return render(request, "ems/workrule_form.html", {
        "form": form,
        "rule": rule,
        "title": "Update Work Rule"
    })


    
@login_required
@hr_finance_required
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


    
