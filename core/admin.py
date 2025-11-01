from django.contrib import admin
from .models import Employee, Attendance, AttendanceSettings, DailyReport

@admin.register(Employee)
class EmployeeAdmin(admin.ModelAdmin):
    list_display = ['employee_id', 'user', 'department', 'position', 'is_active']
    list_filter = ['department', 'position', 'is_active']
    search_fields = ['employee_id', 'user__username', 'user__first_name', 'user__last_name']
    readonly_fields = ['created_at']

@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ['employee', 'attendance_type', 'timestamp', 'confidence_score']
    list_filter = ['attendance_type', 'timestamp']
    search_fields = ['employee__employee_id', 'employee__user__username']
    readonly_fields = ['timestamp']
    date_hierarchy = 'timestamp'

@admin.register(AttendanceSettings)
class AttendanceSettingsAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        return not AttendanceSettings.objects.exists()

@admin.register(DailyReport)
class DailyReportAdmin(admin.ModelAdmin):
    list_display = ['date', 'total_employees', 'present_count', 'absent_count', 'late_count']
    readonly_fields = ['date', 'total_employees', 'present_count', 'absent_count', 'late_count', 'average_hours']
    date_hierarchy = 'date'