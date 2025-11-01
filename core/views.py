from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.http import JsonResponse, StreamingHttpResponse
from django.utils import timezone
from django.db.models import Count, Q, Avg
import cv2
import base64
from io import BytesIO
from PIL import Image
import json
import os
import face_recognition
import tempfile
from datetime import timedelta
import numpy as np

import logging
from django.contrib import messages
from .models import Employee, Attendance, AttendanceSettings, DailyReport
from .forms import UserRegistrationForm, EmployeeRegistrationForm, AttendanceSettingsForm
from .face_recognition import get_face_system
import os, tempfile, ast, numpy as np
from datetime import timedelta
import face_recognition
from django.utils import timezone
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import Attendance, Employee, AttendanceSettings
import logging
import base64
from django.core.files.base import ContentFile
import ast

logger = logging.getLogger(__name__)
face_system = get_face_system()

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler('registration.log')  # logs will be saved here
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)
logger.setLevel(logging.INFO)



def home(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return redirect('login')



def register(request):
    from django.core.files.storage import default_storage
    import face_recognition

    if request.method == 'POST':
        user_form = UserRegistrationForm(request.POST)
        employee_form = EmployeeRegistrationForm(request.POST, request.FILES)

        # Check if image was captured from webcam
        captured_image = request.POST.get('captured_image')
        if captured_image:
            try:
                format, imgstr = captured_image.split(';base64,')
                ext = format.split('/')[-1]
                data = ContentFile(base64.b64decode(imgstr), name=f"captured_face.{ext}")
                employee_form.files['face_image'] = data
                logger.info("Captured image attached successfully.")
            except Exception as e:
                logger.error(f"Error decoding captured image: {e}")

        if user_form.is_valid() and employee_form.is_valid():
            user = user_form.save()
            employee = employee_form.save(commit=False)
            employee.user = user
            employee.save()

            # Generate and save face encoding
            if employee.face_image:
                try:
                    img_path = default_storage.path(employee.face_image.name)
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)

                    if encodings:
                        employee.face_encoding = encodings[0].tolist()  # JSONField
                        employee.save()
                        logger.info(f"Face encoding saved for {employee.user.username}")
                    else:
                        logger.warning(f"No face detected for {employee.user.username}'s uploaded image.")
                        messages.warning(request, "No face detected in uploaded image. Please re-upload your photo.")
                except Exception as e:
                    logger.error(f"Error generating encoding: {e}")
                    messages.error(request, "Error generating face encoding.")
            else:
                messages.warning(request, "No face image uploaded.")

            messages.success(request, "✅ Registration successful!")
            return redirect('dashboard')
        else:
            logger.warning("Invalid form data.")
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
            attendance_qs = Attendance.objects.all()  # ✅ All employees' records
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


@login_required
def mark_attendance(request):
    """
    Face Recognition Attendance for:
    - Normal users (mark self)
    - Admin Kiosk (detects any employee)
    Auto determines Check-In or Check-Out from AttendanceSettings.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': 'Invalid request method.'})

    capture = request.FILES.get('capture')
    if not capture:
        return JsonResponse({'success': False, 'message': 'No image received.'})

    try:
        # Load attendance settings
        settings_obj = AttendanceSettings.objects.first()
        now = timezone.localtime()
        current_time = now.time()

        # Save temporary image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            for chunk in capture.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name

        # Extract face encodings
        unknown_image = face_recognition.load_image_file(temp_path)
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        if not unknown_encodings:
            return JsonResponse({'success': False, 'message': 'No face detected in the image.'})

        # Load all known employee encodings
        employees = Employee.objects.exclude(face_encoding__isnull=True)
        known_encodings, employee_refs = [], []

        for emp in employees:
            try:
                known_encodings.append(np.array(ast.literal_eval(emp.face_encoding)))
                employee_refs.append(emp)
            except Exception:
                continue

        # Compare with all known faces
        best_match = None
        best_confidence = 0
        tolerance = settings_obj.confidence_threshold if settings_obj else 0.45

        for i, known_encoding in enumerate(known_encodings):
            face_distances = [np.linalg.norm(known_encoding - enc) for enc in unknown_encodings]
            best_distance = min(face_distances)
            confidence = round((1 - best_distance) * 100, 2)

            if best_distance <= tolerance and confidence > best_confidence:
                best_confidence = confidence
                best_match = employee_refs[i]

        if not best_match:
            return JsonResponse({'success': False, 'message': 'Face not recognized.'})

        today = now.date()

        # Determine attendance type dynamically
        last_record = Attendance.objects.filter(employee=best_match).order_by('-timestamp').first()

        if not last_record or last_record.attendance_type == 'check_out':
            attendance_type = 'check_in'
        else:
            attendance_type = 'check_out'

        # Prevent double check-in/check-out in same day
        already_marked = Attendance.objects.filter(
            employee=best_match,
            attendance_type=attendance_type,
            timestamp__date=today
        ).exists()

        if already_marked:
            return JsonResponse({
                'success': False,
                'message': f'{best_match} already marked {attendance_type.replace("_"," ")} for today.'
            })

        # Create record
        new_record = Attendance.objects.create(
            employee=best_match,
            timestamp=now,
            attendance_type=attendance_type,
            confidence_score=best_confidence / 100.0
        )

        # Calculate working time on check-out
        working_hours_msg = ""
        if attendance_type == 'check_out' and last_record:
            time_diff = now - last_record.timestamp
            hours = int(time_diff.total_seconds() // 3600)
            minutes = int((time_diff.total_seconds() % 3600) // 60)
            working_hours_msg = f" Total working time: {hours}h {minutes}m"

        return JsonResponse({
            'success': True,
            'message': f'{best_match} - {attendance_type.replace("_"," ").title()} marked successfully!'
                       f'{working_hours_msg}',
            'confidence': f"{best_confidence}%",
        })

    except Exception as e:
        logger.exception(f"Error in kiosk attendance: {e}")
        return JsonResponse({'success': False, 'message': 'Error processing attendance.'})

    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)



from django.http import JsonResponse
from django.utils import timezone
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
    """Generator for video streaming"""
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        # Recognize faces
        employee_id, confidence, face_location = face_system.recognize_face_from_frame(small_frame)
        
        # Draw rectangle around face
        if face_location:
            top, right, bottom, left = face_location
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            color = (0, 255, 0) if employee_id else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Add label
            label = f"{employee_id or 'Unknown'} ({confidence:.2f})" if employee_id else "Unknown"
            cv2.putText(frame, label, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Encode frame as JPEG
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

from datetime import datetime
from django.utils import timezone
from django.http import HttpResponse
import csv
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



# @staff_member_required
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


