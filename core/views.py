import os
import cv2
import ast
import json
import base64
import logging
import tempfile
import calendar
import numpy as np
from io import BytesIO
from datetime import date, datetime, timedelta
from calendar import monthrange, month_name
from PIL import Image

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.views.decorators.http import require_GET
from django.http import JsonResponse, StreamingHttpResponse
from django.utils import timezone
from django.db.models import Count, Q, Avg
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from .models import Employee, Attendance, AttendanceSettings, DailyReport
from .forms import UserRegistrationForm, EmployeeRegistrationForm, AttendanceSettingsForm
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

# Initialize face system
face_system = get_face_system()



def home(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return redirect('login')


# ------------------ REGISTER VIEW (ArcFace 512D) ------------------
def register(request):
    face_sys = get_face_system()

    if request.method == 'POST':
        user_form = UserRegistrationForm(request.POST)
        employee_form = EmployeeRegistrationForm(request.POST, request.FILES)

        captured_image = request.POST.get('captured_image')
        if captured_image:
            try:
                fmt, imgstr = captured_image.split(';base64,')
                ext = fmt.split('/')[-1]
                data = ContentFile(base64.b64decode(imgstr), name=f"captured_face.{ext}")
                employee_form.files['face_image'] = data
                logger.info("Captured webcam image successfully decoded.")
            except Exception as e:
                logger.error(f"Error decoding captured image: {e}")

        if user_form.is_valid() and employee_form.is_valid():
            user = user_form.save()
            employee = employee_form.save(commit=False)
            employee.user = user
            employee.save()

            # Generate and save 512-D ArcFace embedding
            if employee.face_image:
                try:
                    img_path = default_storage.path(employee.face_image.name)
                    emb = face_sys.get_embedding(img_path)
                    if emb is not None:
                        employee.face_encoding = emb.tolist()
                        employee.save()
                        logger.info(f"✅ ArcFace embedding saved for {employee.user.username}")
                    else:
                        messages.warning(request, "No face detected in uploaded image. Please re-upload your photo.")
                except Exception as e:
                    logger.error(f"Error generating ArcFace embedding: {e}")
                    messages.error(request, "Error generating face embedding.")
            else:
                messages.warning(request, "No face image uploaded.")

            messages.success(request, "✅ Registration successful!")
            return redirect('dashboard')
        else:
            logger.warning("Invalid registration form data.")
            logger.debug(f"User form errors: {user_form.errors}")
            logger.debug(f"Employee form errors: {employee_form.errors}")

    else:
        user_form = UserRegistrationForm()
        employee_form = EmployeeRegistrationForm()

    return render(request, 'register.html', {
        'user_form': user_form,
        'employee_form': employee_form,
    })

@login_required
def dashboard(request):
    user = request.user
    is_admin = user.is_staff or user.is_superuser

    today = timezone.localdate()
    start_week = today - timedelta(days=today.weekday())
    start_month = today.replace(day=1)

    employees = Employee.objects.all() if is_admin else None

    # --- Select employee if given, else show all ---
    emp_id = request.GET.get('emp_id')
    if is_admin:
        if emp_id:
            employee = Employee.objects.filter(id=emp_id).first()
            attendance_qs = Attendance.objects.filter(employee=employee)
        else:
            employee = None
            attendance_qs = Attendance.objects.all()
    else:
        employee = Employee.objects.filter(user=user).first()
        attendance_qs = Attendance.objects.filter(employee=employee)

    # --- Filter attendance records ---
    today_records = attendance_qs.filter(timestamp__date=today).order_by('timestamp')
    week_records = attendance_qs.filter(timestamp__date__gte=start_week)
    month_records = attendance_qs.filter(timestamp__date__gte=start_month)

    # --- Stats ---
    avg_confidence = attendance_qs.aggregate(avg=Avg('confidence_score'))['avg'] or 0
    total_employees = Employee.objects.count()

    # --- Determine employee-specific status (only for single employee) ---
    current_status = None
    total_hours = 0
    last_checkin = None

    if employee:
        last_checkin = today_records.filter(attendance_type='check_in').last()
        last_checkout = today_records.filter(attendance_type='check_out').last()

        if last_checkin and (not last_checkout or last_checkout.timestamp < last_checkin.timestamp):
            current_status = 'Checked In'
        elif last_checkout:
            current_status = 'Checked Out'
        else:
            current_status = 'Not Checked In'

        # Calculate working hours
        checkins = today_records.filter(attendance_type='check_in')
        checkouts = today_records.filter(attendance_type='check_out')
        for ci in checkins:
            co = checkouts.filter(timestamp__gt=ci.timestamp).first()
            if co:
                total_hours += (co.timestamp - ci.timestamp).total_seconds() / 3600

    # --- Add current year and month for calendar button ---
    now = timezone.now()
    context = {
        'is_admin': is_admin,
        'employees': employees,
        'employee': employee,
        'today_attendance': today_records,
        'week_attendance': week_records,
        'month_attendance': month_records,
        'avg_confidence': round(avg_confidence, 2),
        'current_status': current_status,
        'last_checkin': last_checkin,
        'total_hours': round(total_hours, 2),
        'total_employees': total_employees,
        'selected_emp_id': int(emp_id) if emp_id else None,
        'year': now.year,
        'month': now.month,
    }

    return render(request, 'dashboard.html', context)

@login_required
def attendance_summary_api(request):
    """
    Returns live attendance data (present/absent count) for the last N days.
    Works for both admin (all employees) and specific employee dashboards.
    """
    range_days = int(request.GET.get("range", 7))
    emp_id = request.GET.get("emp_id")

    today = timezone.localdate()
    start_date = today - timedelta(days=range_days - 1)

    # --- Filter Attendance by Employee or All ---
    if emp_id:
        employee = Employee.objects.filter(id=emp_id).first()
        attendance_qs = Attendance.objects.filter(employee=employee)
    else:
        attendance_qs = Attendance.objects.all()

    # --- Prepare Daily Stats ---
    labels = []
    present = []
    absent = []

    for i in range(range_days):
        day = start_date + timedelta(days=i)
        labels.append(day.strftime('%a'))  # 'Mon', 'Tue', etc.

        if emp_id:
            # Employee-specific: present if they checked in
            day_present = attendance_qs.filter(
                timestamp__date=day,
                attendance_type="check_in"
            ).exists()
            present.append(1 if day_present else 0)
            absent.append(0 if day_present else 1)
        else:
            # Admin view: count number of employees present that day
            day_checkins = attendance_qs.filter(
                timestamp__date=day,
                attendance_type="check_in"
            ).values_list("employee", flat=True).distinct().count()

            total_employees = Employee.objects.count()
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

# ------------------ MARK ATTENDANCE (ArcFace 512D) ------------------
@login_required
def mark_attendance(request):
    if request.method == 'POST' and request.FILES.get('capture'):
        try:
            image_file = request.FILES['capture']
            file_bytes = np.frombuffer(image_file.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            emb = get_face_embedding(frame)
            if emb is None:
                return JsonResponse({'success': False, 'message': 'No face detected in frame.'})

            best_match = None
            best_score = 0.60
            employees = Employee.objects.exclude(face_encoding=None)

            for emp in employees:
                try:
                    db_emb = np.array(json.loads(emp.face_encoding), dtype=np.float32)
                    sim = float(np.dot(emb, db_emb))
                    if sim > best_score:
                        best_match = emp
                        best_score = sim
                except Exception:
                    continue

            if not best_match:
                return JsonResponse({'success': False, 'message': 'Face not recognized.'})

            emp = best_match
            now = timezone.now()
            today = now.date()

            # 🔧 Get settings (fallback if none)
            settings = AttendanceSettings.objects.first()
            lock_hours = settings.min_hours_before_checkout if settings else 3.0

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

            # ✅ First attendance — Check-In
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

        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})

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
            "confidence": round(log.confidence_score * 100, 2),
        }
        for log in logs
    ]
    return JsonResponse(data, safe=False)


def video_feed():
    """Live video stream with face detection + recognition overlay."""
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            emb = get_face_embedding(face_region)
            color = (0, 0, 255)  # Default red for unknown
            label = "Unknown"

            if emb is not None:
                best_match = None
                best_score = 0.60  # Threshold
                employees = Employee.objects.exclude(face_encoding=None)

                for emp in employees:
                    try:
                        db_emb = np.array(json.loads(emp.face_encoding), dtype=np.float32)
                        sim = float(np.dot(emb, db_emb))
                        if sim > best_score:
                            best_match = emp
                            best_score = sim
                    except Exception as e:
                        continue

                if best_match:
                    name = best_match.user.get_full_name() or best_match.user.username
                    confidence = best_score * 100
                    label = f"{name} ({confidence:.1f}%)"
                    color = (0, 255, 0)  # Green for recognized faces

            # Draw rectangle + label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 25), (x + len(label) * 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Encode and yield frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@login_required
def attendance_page(request):
    return render(request, 'attendance.html')

@login_required
def video_feed_view(request):
    return StreamingHttpResponse(video_feed(), content_type='multipart/x-mixed-replace; boundary=frame')


@staff_member_required
def attendance_reports(request):
    from datetime import datetime, timedelta
    import csv
    from django.http import HttpResponse
    from django.db.models import Q

    # Get filters from query
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    employee_id = request.GET.get('employee')
    download = request.GET.get('download')

    # Default date range (today)
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

    # Base queryset
    attendance_qs = Attendance.objects.filter(timestamp__date__range=(start_date, end_date))

    # Filter by employee if selected
    if employee_id:
        attendance_qs = attendance_qs.filter(employee__id=employee_id)

    # CSV export
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

    # Statistics
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
    """
    Returns attendance history as JSON.
    - Admin/staff: See all employees' attendance.
    - Regular users: See only their own records.
    """
    days = int(request.GET.get('days', 30))
    start_date = timezone.now().date() - timedelta(days=days)

    # If user is staff/admin → show all records
    if request.user.is_staff:
        attendance_qs = Attendance.objects.filter(timestamp__date__gte=start_date)
    else:
        # For non-staff, try to get their employee record safely
        try:
            employee = request.user.employee
            attendance_qs = Attendance.objects.filter(
                employee=employee,
                timestamp__date__gte=start_date
            )
        except Employee.DoesNotExist:
            # If user has no employee record
            attendance_qs = Attendance.objects.none()

    # Build response data
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
    """
    Renders the attendance history page in HTML (not JSON).
    Admin/staff: view all records
    Users: view only their own
    """
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
    # Validate employee access
    emp = get_object_or_404(Employee, id=employee_id)
    
    # Optional: Add permission check if needed
    # if not request.user.is_superuser and request.user != emp.user:
    #     return JsonResponse({'error': 'Permission denied'}, status=403)
    
    # Get month range
    start_date = date(year, month, 1)
    end_day = calendar.monthrange(year, month)[1]
    end_date = date(year, month, end_day)
    
    # Get present days in one query
    present_days = Attendance.objects.filter(
        employee=emp,
        timestamp__date__range=(start_date, end_date),
        attendance_type='check_in'
    ).values_list('timestamp__date', flat=True)
    
    present_day_set = {d.day for d in present_days}
    
    # Build month data
    month_data = []
    for day in range(1, end_day + 1):
        current_date = date(year, month, day)
        month_data.append({
            'day': day,
            'date': current_date.isoformat(),
            'present': day in present_day_set,
            'is_weekend': current_date.weekday() >= 5,  # 5=Saturday, 6=Sunday
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

    # Fetch attendance data
    attendance_records = Attendance.objects.filter(
        employee=employee,
        timestamp__date__range=(start_date, end_date)
    ).order_by('timestamp')

    # Get attendance settings
    settings = AttendanceSettings.objects.first()
    max_daily_hours = settings.max_daily_hours if settings else 8.0

    # Organize check-in/out per day
    daily_attendance = {}
    for record in attendance_records:
        day = record.timestamp.date()
        if day not in daily_attendance:
            daily_attendance[day] = {'check_in': None, 'check_out': None}
        if record.attendance_type == 'check_in' and not daily_attendance[day]['check_in']:
            daily_attendance[day]['check_in'] = record.timestamp
        elif record.attendance_type == 'check_out':
            daily_attendance[day]['check_out'] = record.timestamp

    # Build calendar data
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

    # Month navigation
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    previous_month = month - 1 if month > 1 else 12
    previous_year = year if month > 1 else year - 1

    first_day_offset = (calendar.weekday(year, month, 1) + 1) % 7

    # Convert total working hours to readable format
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
                'confidence': f"{log.confidence_score:.2f}%" if log.confidence_score else '—',
                'image': log.image_capture.url if log.image_capture else '',
                'notes': log.notes or ''
            }
            for log in logs
        ]
    }

    return JsonResponse(data)


