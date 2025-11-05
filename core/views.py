import os
from urllib import request
import cv2
import ast
import json
import io
import zipfile
import base64
import logging
import tempfile
import calendar
import numpy as np
from io import BytesIO
from datetime import date, datetime, timedelta
from calendar import monthrange, month_name
from PIL import Image
from django.utils import timezone
from datetime import datetime, date
import calendar
from django.db.models import Prefetch
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
from .models import (Employee, Attendance, AttendanceSettings, DailyReport,Department, SalaryStructure, Payroll, LeaveType, 
    LeaveApplication, LeaveWorkflowStage, LeaveApproval,
    JoiningDetail, Resignation, Notification,JoiningDocument)

from .forms import ( UserRegistrationForm, EmployeeRegistrationForm, AttendanceSettingsForm,DepartmentForm, SalaryStructureForm, PayrollFilterForm,
    LeaveTypeForm, LeaveApplicationForm, LeaveApprovalForm, LeaveWorkflowStageForm,
    JoiningDetailForm, ResignationForm, NotificationForm,PayrollFilterForm
    )
from .face_system import get_face_system
from core.face_recognition_utils import get_face_embedding

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


def home(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return redirect('login')



def register(request):
    face_sys = get_face_system()
    reg_success = request.session.pop('reg_success', None)

    if request.method == "POST":
        # ✅ FIX: Only pass request.POST to the form, handle FILES separately
        form = EmployeeRegistrationForm(request.POST)

        if form.is_valid():
            # ------------------ 1. Create User ------------------
            user = User.objects.create_user(
                username=form.cleaned_data['username'],
                email=form.cleaned_data['email'],
                password=form.cleaned_data['password1'],
                first_name=form.cleaned_data['first_name'],
                last_name=form.cleaned_data['last_name']
            )

            # ------------------ 2. Create Employee (Auto-ID) ----
            employee = Employee.objects.create(
                user=user,
                department=form.cleaned_data['department'],
                position=form.cleaned_data['position'],
                manager=form.cleaned_data['manager'],
                phone_number=form.cleaned_data['phone_number'],
                role=form.cleaned_data['role'],
            )

            # ------------------ 3. Handle Face Image Upload -----
            face_image = request.FILES.get('face_image')
            if face_image:
                employee.face_image = face_image
                employee.save(update_fields=['face_image'])

            # Webcam Image (Base64)
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
                    messages.warning(request, f"Webcam image could not be saved: {e}")

            # ------------------ 4. Salary Info -------------------
            if form.cleaned_data.get('base_salary'):
                SalaryStructure.objects.create(
                    employee=employee,
                    base_salary=form.cleaned_data['base_salary'],
                    hra=form.cleaned_data.get('hra') or 0,
                    allowances=form.cleaned_data.get('allowances') or 0,
                    deductions=form.cleaned_data.get('deductions') or 0,
                )

            # ------------------ 5. Joining Details --------------
            doj = form.cleaned_data['date_of_joining']
            probation = form.cleaned_data['probation_period_months']
            confirmation_date = doj + relativedelta(months=probation)

            joining = JoiningDetail.objects.create(
                employee=employee,
                date_of_joining=doj,
                probation_period_months=probation,
                confirmation_date=confirmation_date
            )

            # ✅ FIX: Handle Multiple Documents Properly
            documents = request.FILES.getlist('documents')
            uploaded_docs = []
            if documents:
                for file in documents:
                    # Validate file type and size if needed
                    if file.size > 10 * 1024 * 1024:  # 10MB limit
                        messages.warning(request, f"File {file.name} is too large. Max 10MB allowed.")
                        continue
                    
                    # Create document record
                    doc = JoiningDocument.objects.create(joining=joining, file=file)
                    uploaded_docs.append({
                        'name': file.name,
                        'url': doc.file.url
                    })
            else:
                print("⚠ No documents uploaded")

            # ------------------ 6. Face Encoding -----------------
            if employee.face_image:
                try:
                    img_path = getattr(employee.face_image, 'path', None) or default_storage.path(employee.face_image.name)
                    emb = face_sys.get_embedding(img_path)
                    if emb is not None:
                        employee.face_encoding = json.dumps(emb.tolist())
                        employee.save(update_fields=['face_encoding'])
                    else:
                        messages.warning(request, "⚠ No face detected from the image.")
                except Exception as e:
                    messages.warning(request, f"Face encoding failed: {e}")

            # ------------------ 7. Success Response -------------
            request.session['reg_success'] = {
                "employee_id": employee.employee_id,
                "name": user.get_full_name() or user.username,
                "docs": uploaded_docs,
            }
            return redirect('register')

        else:
            messages.error(request, "❌ Please fix the errors in the form.")
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


def my_profile_view(request):
    employee = request.user.employee  # assuming employee is linked to user
    return render(request, 'ems/my_profile.html', {'employee': employee})

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


@login_required
def mark_attendance(request):
    """
    Expects a file field named 'capture' containing an image upload (multipart/form-data).
    Returns JSON describing success/failure and attendance action.
    """
    if request.method == 'POST' and request.FILES.get('capture'):
        try:
            image_file = request.FILES['capture']
            file_bytes = np.frombuffer(image_file.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            emb = get_face_embedding(frame)
            if emb is None:
                return JsonResponse({'success': False, 'message': 'No face detected in frame.'})

            emb = np.asarray(emb, dtype=np.float32).ravel()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

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
                        if isinstance(raw, (list, tuple, np.ndarray)):
                            db_emb = np.asarray(raw, dtype=np.float32)
                        elif isinstance(raw, str):
                            try:
                                db_emb = np.asarray(json.loads(raw), dtype=np.float32)
                            except Exception:
                                db_emb = None

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
                    logger.exception("Error comparing embeddings for employee id %s", getattr(emp, 'id', None))
                    continue

            if not best_match:
                return JsonResponse({'success': False, 'message': 'Face not recognized.'})

            emp = best_match
            now = timezone.now()
            today = now.date()

            lock_hours = float(settings_obj.min_hours_before_checkout) if settings_obj and getattr(settings_obj, 'min_hours_before_checkout', None) is not None else 3.0

            latest_attendance = Attendance.objects.filter(employee=emp).order_by('-timestamp').first()

            if latest_attendance and timezone.localtime(latest_attendance.timestamp).date() == today:
                if latest_attendance.attendance_type == 'check_in':
                    time_since_checkin = now - latest_attendance.timestamp
                    if time_since_checkin < timedelta(hours=lock_hours):
                        remaining = timedelta(hours=lock_hours) - time_since_checkin
                        remaining_minutes = int(remaining.total_seconds() // 60)
                        return JsonResponse({
                            'success': False,
                            'message': f'You can check out only after {lock_hours:.0f} hours. Try again in {remaining_minutes} minutes.'
                        })

                    Attendance.objects.create(
                        employee=emp,
                        attendance_type='check_out',
                        confidence_score=best_score,
                        timestamp=now
                    )
                    msg = f"Checked out successfully at {now.strftime('%H:%M:%S')}."
                    return JsonResponse({
                        'success': True,
                        'name': emp.user.get_full_name(),
                        'attendance_type': 'Check-Out',
                        'confidence': round(best_score * 100, 2),
                        'message': msg
                    })
                else:
                    return JsonResponse({
                        'success': False,
                        'message': "Already checked out today."
                    })

            # First attendance — Check-In
            Attendance.objects.create(
                employee=emp,
                attendance_type='check_in',
                confidence_score=best_score,
                timestamp=now
            )
            msg = f"Checked in successfully at {now.strftime('%H:%M:%S')}."
            return JsonResponse({
                'success': True,
                'name': emp.user.get_full_name(),
                'attendance_type': 'Check-In',
                'confidence': round(best_score * 100, 2),
                'message': msg
            })

        except Exception:
            logger.exception("Error in mark_attendance")
            return JsonResponse({'success': False, 'message': 'Internal server error.'})

    return JsonResponse({'success': False, 'message': 'Invalid request.'})


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
    from datetime import datetime, timedelta
    import csv
    from django.http import HttpResponse
    from django.db.models import Q

    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    employee_id = request.GET.get('employee')
    download = request.GET.get('download')

    if not start_date or not end_date:
        today = timezone.now().date()
        start_date = today
        end_date = today
    else:
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            start_date = end_date = timezone.now().date()

    attendance_qs = Attendance.objects.filter(timestamp__date__range=(start_date, end_date))

    if employee_id:
        attendance_qs = attendance_qs.filter(employee__id=employee_id)

    if download == "csv":
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="attendance_{start_date}_to_{end_date}.csv"'

        writer = csv.writer(response)
        writer.writerow(["Employee", "Type", "Timestamp", "Confidence", "Location"])

        for record in attendance_qs:
            writer.writerow([
                record.employee.user.get_full_name(),
                record.get_attendance_type_display(),
                record.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                f"{record.confidence_score * 100:.2f}%" if record.confidence_score else "N/A",
                record.location or "-",
            ])

        return response

    total_employees = Employee.objects.filter(is_active=True).count()
    present_employees = attendance_qs.filter(attendance_type='check_in').values('employee').distinct().count()
    absent_employees = total_employees - present_employees

    context = {
        'attendance_data': attendance_qs.order_by('-timestamp'),
        'start_date': start_date,
        'end_date': end_date,
        'total_employees': total_employees,
        'present_employees': present_employees,
        'absent_employees': absent_employees,
        'employees': Employee.objects.filter(is_active=True),
        'selected_employee': int(employee_id) if employee_id else None,
    }
    return render(request, 'reports.html', context)


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
    structures = SalaryStructure.objects.select_related('employee__user').all()
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
            messages.success(request, 'Salary structure created successfully!')
            return redirect('salary_structure_list')
    else:
        form = SalaryStructureForm()
    
    return render(request, 'ems/salary_structure_form.html', {
        'form': form,
        'title': 'Create Salary Structure'
    })


from django.db.models.signals import post_save
from django.dispatch import receiver
from datetime import date

@receiver(post_save, sender=SalaryStructure)
def create_initial_payroll(sender, instance, created, **kwargs):
    if created:  # Only when salary structure is added first time
        today = date.today()
        month_start = date(today.year, today.month, 1)

        Payroll.objects.get_or_create(
            employee=instance.employee,
            month=month_start,
            defaults={
                'basic_pay': instance.base_salary,
                'allowances': instance.allowances,
                'deductions': instance.deductions,
                'net_salary': instance.total_salary(),
                'status': 'processed',
                'processed_at': timezone.now(),
            }
        )


@login_required
@finance_required
def payroll_list(request):
    form = PayrollFilterForm(request.GET or None)
    payrolls = Payroll.objects.select_related('employee__user').order_by('-month', 'employee__user__first_name')

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
        total_basic=Sum('basic_pay'),
        total_allowances=Sum('allowances'),
        total_deductions=Sum('deductions'),
        total_net=Sum('net_salary'),
    )

    return render(request, 'ems/payroll_list.html', {
        'payrolls': payrolls,
        'form': form,
        'total_basic': totals['total_basic'] or 0,
        'total_allowances': totals['total_allowances'] or 0,
        'total_deductions': totals['total_deductions'] or 0,
        'total_net': totals['total_net'] or 0,
        'title': 'Payroll'
    })

@login_required
def generate_payroll(request):
    if request.method == 'POST':
        month = int(request.POST.get('month'))
        year = int(request.POST.get('year'))
        
        try:
            Payroll.objects.generate_monthly_salary(month, year)
            messages.success(request, f'Payroll generated successfully for {calendar.month_name[month]} {year}!')
        except Exception as e:
            messages.error(request, f'Error generating payroll: {str(e)}')
        
        return redirect('payroll_list')
    
    # Get statistics for the template
    active_employees_count = Employee.objects.filter(employment_status='active').count()
    recent_payroll_count = Payroll.objects.filter(month__year=date.today().year).count()
    
    return render(request, 'ems/generate_payroll.html', {
        'title': 'Generate Payroll',
        'active_employees_count': active_employees_count,
        'recent_payroll_count': recent_payroll_count
    })

@login_required
def my_salary(request):
    try:
        employee = request.user.employee
        payrolls = Payroll.objects.filter(employee=employee).order_by('-month')
        salary_structure = SalaryStructure.objects.filter(employee=employee).first()
    except Employee.DoesNotExist:
        payrolls = []
        salary_structure = None
        messages.error(request, 'Employee profile not found!')
    
    return render(request, 'ems/my_salary.html', {
        'payrolls': payrolls,
        'salary_structure': salary_structure,
        'title': 'My Salary'
    })


@finance_required
def pay_salary(request, pk):
    payroll = get_object_or_404(Payroll, pk=pk)

    if request.method == "POST":
        paid_date = request.POST.get('paid_date')
        amount_paid = request.POST.get('amount_paid')

        payroll.status = 'paid'
        payroll.paid_date = paid_date
        payroll.net_salary = amount_paid if amount_paid else payroll.net_salary
        payroll.save()

        # ✅ Generate Salary Slip PDF
        pdf_path = generate_salary_slip(payroll)

        # ✅ Send Email to Employee
        email = EmailMessage(
            subject=f"Salary Slip - {payroll.month.strftime('%B %Y')}",
            body=f"Dear {payroll.employee.user.first_name},\n\nYour salary for {payroll.month.strftime('%B %Y')} has been processed. Please find attached your salary slip.\n\nRegards,\nFinance Team",
            from_email="avaneeshpathak900@gmail.com",  # change to your mail
            to=[payroll.employee.user.email]
        )
        email.attach_file(pdf_path)
        email.send()

        messages.success(request, "✅ Salary paid & email sent with salary slip!")
        return redirect('payroll_list')

    return render(request, 'ems/pay_salary.html', {'payroll': payroll})

@login_required
def download_salary_slip(request, pk):
    payroll = get_object_or_404(Payroll, pk=pk)
    pdf_path = generate_salary_slip(payroll)
    return FileResponse(open(pdf_path, 'rb'), content_type='application/pdf')

@login_required
def payroll_slip_pdf(request, pk):
    payroll = get_object_or_404(Payroll.objects.select_related('employee__user'), pk=pk)

    # Security: allow Finance/Admin or the employee themselves
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

    pdf_io = build_salary_slip_pdf(payroll)
    filename = f"salary_slip_{payroll.employee.employee_id}_{payroll.month.strftime('%Y_%m')}.pdf"
    return FileResponse(pdf_io, as_attachment=True, filename=filename)

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
    """
    Returns JSON: labels (YYYY-MM), totals (sum of net_salary)
    Last 12 months by default.
    """
    from datetime import date
    today = date.today()
    months = []
    labels = []
    totals = []

    for i in range(11, -1, -1):  # last 12 months
        year = (today.year if today.month - i > 0 else today.year - 1) if (today.month - i) != 0 else today.year - 1
        month = ((today.month - i - 1) % 12) + 1
        months.append((year, month))
        labels.append(f"{year}-{str(month).zfill(2)}")

    for (y, m) in months:
        s = Payroll.objects.filter(month__year=y, month__month=m).aggregate(total=Sum('net_salary'))['total'] or 0
        totals.append(float(s))

    return JsonResponse({'labels': labels, 'totals': totals})

@login_required
@finance_required
def employee_salary_history(request, employee_id):
    emp = get_object_or_404(Employee.objects.select_related('user'), id=employee_id)
    qs = Payroll.objects.filter(employee=emp).order_by('-month')

    totals = qs.aggregate(
        total_basic=Sum('basic_pay'),
        total_allowances=Sum('allowances'),
        total_deductions=Sum('deductions'),
        total_net=Sum('net_salary'),
    )
    return render(request, 'ems/employee_salary_history.html', {
        'employee': emp,
        'payrolls': qs,
        'totals': totals,
        'title': f"Salary History - {emp.user.get_full_name()}"
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

# ============================================================
# 📊 DASHBOARD & REPORTS
# ============================================================

@login_required
def ems_dashboard(request):
    """EMS-specific dashboard"""
    try:
        employee = request.user.employee
    except Employee.DoesNotExist:
        return render(request, 'ems/ems_dashboard.html', {
            'employee_exists': False,
            'title': 'EMS Dashboard'
        })
    
    # Employee stats
    today = date.today()
    current_month = today.month
    current_year = today.year
    
    # Leave stats
    leave_stats = LeaveApplication.objects.filter(
        employee=employee,
        start_date__year=current_year
    ).values('status').annotate(count=Count('id'))
    
    # Attendance stats for current month
    attendance_count = Attendance.objects.filter(
        employee=employee,
        timestamp__month=current_month,
        timestamp__year=current_year,
        attendance_type='check_in'
    ).count()
    
    # Pending approvals (for managers)
    pending_approvals = LeaveApplication.objects.filter(
        approvals__approver=employee,
        approvals__status='pending'
    ).count()
    
    # Recent notifications
    recent_notifications = Notification.objects.filter(
        recipient=request.user
    ).order_by('-created_at')[:5]
    
    # Department stats
    department_stats = Department.objects.annotate(
        employee_count=Count('employees')  # or 'employee_set' if no related_name
    ).values('name', 'employee_count')

    
    # Payroll stats
    recent_payroll = Payroll.objects.filter(
        employee=employee
    ).order_by('-month').first()
    
    return render(request, 'ems/ems_dashboard.html', {
        'employee_exists': True,
        'employee': employee,
        'leave_stats': leave_stats,
        'attendance_count': attendance_count,
        'pending_approvals': pending_approvals,
        'recent_notifications': recent_notifications,
        'department_stats': department_stats,
        'recent_payroll': recent_payroll,
        'title': 'EMS Dashboard'
    })


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




