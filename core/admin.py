from django.contrib import admin
from django.utils.translation import gettext_lazy as _
from .models import Employee, Attendance, AttendanceSettings, DailyReport


@admin.register(Employee)
class EmployeeAdmin(admin.ModelAdmin):
    list_display = ('employee_id', 'user', 'department', 'position', 'is_active', 'created_at')
    list_display_links = ('employee_id', 'user')
    list_filter = ('department', 'position', 'is_active')
    search_fields = (
        'employee_id',
        'user__username',
        'user__first_name',
        'user__last_name',
        'department',
        'position',
    )
    readonly_fields = ('created_at',)
    ordering = ('employee_id',)
    fieldsets = (
        (_('Employee Information'), {
            'fields': ('employee_id', 'user', 'department', 'position', 'phone_number', 'is_active')
        }),
        (_('Face Data'), {
            'fields': ('face_image', 'face_encoding'),
        }),
        (_('Timestamps'), {
            'fields': ('created_at',),
        }),
    )


@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ('employee', 'attendance_type', 'timestamp', 'confidence_score', 'location')
    list_filter = ('attendance_type', 'timestamp')
    search_fields = (
        'employee__employee_id',
        'employee__user__username',
        'employee__user__first_name',
        'employee__user__last_name',
        'location',
    )
    readonly_fields = ('timestamp',)
    date_hierarchy = 'timestamp'
    ordering = ('-timestamp',)
    fieldsets = (
        (_('Attendance Record'), {
            'fields': (
                'employee',
                'attendance_type',
                'timestamp',
                'location',
                'confidence_score',
                'image_capture',
                'notes',
            )
        }),
    )


@admin.register(AttendanceSettings)
class AttendanceSettingsAdmin(admin.ModelAdmin):
    list_display = (
        '__str__',
        'check_in_start',
        'check_in_end',
        'check_out_start',
        'check_out_end',
        'late_threshold',
        'confidence_threshold',
        'max_daily_hours',
        'min_hours_before_checkout',
    )

    def has_add_permission(self, request):
        """Allow only one AttendanceSettings instance."""
        if AttendanceSettings.objects.exists():
            return False
        return super().has_add_permission(request)


@admin.register(DailyReport)
class DailyReportAdmin(admin.ModelAdmin):
    list_display = (
        'date',
        'total_employees',
        'present_count',
        'absent_count',
        'late_count',
        'average_hours',
    )
    readonly_fields = (
        'date',
        'total_employees',
        'present_count',
        'absent_count',
        'late_count',
        'average_hours',
    )
    date_hierarchy = 'date'
    ordering = ('-date',)
    fieldsets = (
        (_('Daily Summary'), {
            'fields': (
                'date',
                'total_employees',
                'present_count',
                'absent_count',
                'late_count',
                'average_hours',
            ),
        }),
    )
