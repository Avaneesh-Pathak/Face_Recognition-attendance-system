"""Private media; ensure Nginx does NOT directly alias /private-media/.

No path traversal; all media are checked against user-owned or same-organisation objects.
Payroll documents require finance/self permissions; joining docs HR or owner.
"""
import mimetypes
from pathlib import Path
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import FileResponse, Http404
from django.views.decorators.http import require_GET
from core.models import Employee, Organisation, Attendance, JoiningDocument, Payroll

@require_GET
@login_required
def private_media(request, file_path):
    root = Path(settings.MEDIA_ROOT).resolve()
    file = (root / file_path).resolve()
    if file == root or root not in file.parents or not file.is_file():
        raise Http404()
    org = getattr(request, 'organisation', None)
    user = request.user
    if not user.is_superuser and org is None:
        raise Http404()
    is_hr = user.is_superuser or (hasattr(user, 'employee') and user.employee.role in ('HR','Admin'))
    is_finance = user.is_superuser or (hasattr(user, 'employee') and user.employee.role in ('Finance','Admin'))
    # Compare stored file name exactly, no broad 'files under my org' permission.
    allowed = False
    if file_path.startswith('faces/'):
        qs = Employee.global_objects.filter(face_image=file_path)
        if not user.is_superuser:
            qs = qs.filter(organisation=org)
            if not is_hr:
                qs = qs.filter(user=user)
        allowed = qs.exists()
    elif file_path.startswith('organisation_logos/'):
        qs = Organisation.objects.filter(logo=file_path)
        if not user.is_superuser:
            qs = qs.filter(pk=org.pk)
        allowed = qs.exists()
    elif file_path.startswith('attendance_captures/'):
        qs = Attendance.global_objects.filter(image_capture=file_path)
        if not user.is_superuser:
            qs = qs.filter(organisation=org)
            if not is_hr:
                qs = qs.filter(employee__user=user)
        allowed = qs.exists()
    elif file_path.startswith('payslips/'):
        qs = Payroll.global_objects.filter(payslip_pdf=file_path)
        if not user.is_superuser:
            qs = qs.filter(organisation=org)
            if not is_finance:
                qs = qs.filter(employee__user=user)
        allowed = qs.exists()
    # Other sensitive document categories are not served via direct URL;
    # use existing authenticated download endpoints with appropriate authorization.
    if not allowed:
        raise Http404()
    mime = mimetypes.guess_type(str(file))[0] or 'application/octet-stream'
    response = FileResponse(open(file, 'rb'), content_type=mime, as_attachment=not mime.startswith('image/'))
    response['Cache-Control'] = 'no-store, private'
    response['X-Content-Type-Options'] = 'nosniff'
    return response
