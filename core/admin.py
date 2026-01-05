from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from django.utils.html import format_html
from .models import (
    Employee, Attendance, AttendanceSettings, DailyReport, Department,
    SalaryStructure, Payroll, LeaveType, LeaveApplication, 
    LeaveWorkflowStage, LeaveApproval, JoiningDetail, Resignation, Notification,JoiningDocument,
     WorkRule, PayrollSettings
)


# ============================================================
# 🔧 CUSTOM ADMIN ACTIONS
# ============================================================

@admin.action(description="Mark selected employees as active")
def make_employees_active(modeladmin, request, queryset):
    queryset.update(is_active=True, employment_status='active')

@admin.action(description="Mark selected employees as inactive")
def make_employees_inactive(modeladmin, request, queryset):
    queryset.update(is_active=False, employment_status='resigned')

@admin.action(description="Process payroll for selected records")
def process_payroll(modeladmin, request, queryset):
    queryset.update(status='processed')

@admin.action(description="Approve selected leave applications")
def approve_leaves(modeladmin, request, queryset):
    queryset.update(status='approved')

@admin.action(description="Mark notifications as read")
def mark_notifications_read(modeladmin, request, queryset):
    queryset.update(is_read=True)


# ============================================================
# 👤 EMPLOYEE ADMIN
# ============================================================

class EmployeeInline(admin.StackedInline):
    model = Employee
    can_delete = False
    verbose_name_plural = 'Employee Details'
    fields = ('employee_id', 'department', 'position', 'phone_number', 'employment_status')


class CustomUserAdmin(UserAdmin):
    inlines = (EmployeeInline,)
    list_display = ('username', 'email', 'first_name', 'last_name', 'get_employee_id', 'get_department', 'is_staff')
    list_select_related = ('employee',)
    
    def get_employee_id(self, obj):
        return obj.employee.employee_id if hasattr(obj, 'employee') else '-'
    get_employee_id.short_description = 'Employee ID'
    
    def get_department(self, obj):
        return obj.employee.department if hasattr(obj, 'employee') else '-'
    get_department.short_description = 'Department'


@admin.register(Employee)
class EmployeeAdmin(admin.ModelAdmin):
    list_display = ('employee_id', 'user_full_name', 'department', 'position', 'employment_status', 'is_active', 'date_of_joining','role')
    list_filter = ('department', 'position', 'employment_status', 'is_active', 'date_of_joining')
    search_fields = ('employee_id', 'user__first_name', 'user__last_name', 'user__email', 'department')
    readonly_fields = ('created_at', 'face_encoding_preview')
    fieldsets = (
        ('Personal Information', {
            'fields': ('user', 'employee_id', 'phone_number')
        }),
        ('Employment Details', {
            'fields': ('department', 'position', 'manager', 'date_of_joining', 'date_of_resignation', 'employment_status')
        }),
        ('Face Recognition', {
            'fields': ('face_image', 'face_encoding_preview'),
            'classes': ('collapse',)
        }),
        ('System', {
            'fields': ('is_active', 'created_at'),
            'classes': ('collapse',)
        }),
    )
    actions = [make_employees_active, make_employees_inactive]

    def user_full_name(self, obj):
        return obj.user.get_full_name()
    user_full_name.short_description = 'Full Name'

    def face_encoding_preview(self, obj):
        if obj.face_encoding:
            return format_html(
                '<div style="max-height: 100px; overflow-y: auto; background: #f5f5f5; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 10px;">{}</div>',
                obj.face_encoding[:200] + '...' if len(obj.face_encoding) > 200 else obj.face_encoding
            )
        return "No encoding"
    face_encoding_preview.short_description = 'Face Encoding Preview'


# ============================================================
# 🕒 ATTENDANCE ADMIN
# ============================================================

@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ('employee', 'attendance_type', 'timestamp', 'location', 'confidence_score')
    list_filter = ('attendance_type', 'timestamp', 'location')
    search_fields = ('employee__user__first_name', 'employee__user__last_name', 'employee__employee_id')
    readonly_fields = ('timestamp',)
    date_hierarchy = 'timestamp'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('employee__user')


@admin.register(AttendanceSettings)
class AttendanceSettingsAdmin(admin.ModelAdmin):
    list_display = ('check_in_start', 'check_in_end', 'check_out_start', 'check_out_end', 'late_threshold', 'confidence_threshold')
    
    def has_add_permission(self, request):
        return not AttendanceSettings.objects.exists()


@admin.register(DailyReport)
class DailyReportAdmin(admin.ModelAdmin):
    list_display = ('date', 'total_employees', 'present_count', 'absent_count', 'late_count', 'average_hours')
    list_filter = ('date',)
    readonly_fields = ('date', 'total_employees', 'present_count', 'absent_count', 'late_count', 'average_hours')
    date_hierarchy = 'date'


# ============================================================
# 🏢 ORGANISATION ADMIN
# ============================================================

@admin.register(Department)
class DepartmentAdmin(admin.ModelAdmin):
    list_display = ('name', 'head_name', 'parent_department')
    list_filter = ('parent_department',)
    search_fields = ('name', 'head__user__first_name', 'head__user__last_name')
    
    def head_name(self, obj):
        return obj.head.user.get_full_name() if obj.head else '-'
    head_name.short_description = 'Head'


# ============================================================
# 💰 SALARY & PAYROLL ADMIN
# ============================================================

@admin.register(SalaryStructure)
class SalaryStructureAdmin(admin.ModelAdmin):
    list_display = ('employee', 'base_salary', 'hra', 'allowances', 'deductions', 'total_salary', 'effective_from')
    list_filter = ('effective_from',)
    search_fields = ('employee__user__first_name', 'employee__user__last_name', 'employee__employee_id')
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('employee__user')


class PayrollMonthFilter(admin.SimpleListFilter):
    title = 'Month'
    parameter_name = 'month'
    
    def lookups(self, request, model_admin):
        from datetime import datetime
        current_year = datetime.now().year
        months = [
            (1, 'January'), (2, 'February'), (3, 'March'), (4, 'April'),
            (5, 'May'), (6, 'June'), (7, 'July'), (8, 'August'),
            (9, 'September'), (10, 'October'), (11, 'November'), (12, 'December')
        ]
        return [(f"{current_year}-{month:02d}", f"{name} {current_year}") for month, name in months]
    
    def queryset(self, request, queryset):
        if self.value():
            year, month = map(int, self.value().split('-'))
            return queryset.filter(month__year=year, month__month=month)
        return queryset


@admin.register(Payroll)
class PayrollAdmin(admin.ModelAdmin):
    list_display = (
        'employee', 'month_display', 'net_salary', 'status',
        'present_days', 'half_days', 'paid_leave_days',
        'unpaid_leave_days', 'absent_days', 'overtime_hours',
        'processed_at', 'paid_date'
    )
    list_filter = (PayrollMonthFilter, 'status')
    search_fields = ('employee__employee_id', 'employee__user__first_name')
    readonly_fields = ('processed_at',)

    actions = ['generate_payroll']

    def month_display(self, obj):
        return obj.month.strftime('%B %Y')

    @admin.action(description="Generate Payroll for Current Month")
    def generate_payroll(self, request, queryset=None):
        import datetime
        from .models import Payroll
        today = datetime.date.today()
        Payroll.objects.generate_monthly_salary(today.month, today.year)
        self.message_user(request, "✅ Payroll successfully generated!")



@admin.register(WorkRule)
class WorkRuleAdmin(admin.ModelAdmin):
    list_display = (
        'name', 'working_days', 'saturday_is_working', 'sunday_is_working',
        'full_day_hours', 'half_day_hours', 'overtime_rate'
    )
    list_editable = ('working_days', 'saturday_is_working', 'sunday_is_working')


@admin.register(PayrollSettings)
class PayrollSettingsAdmin(admin.ModelAdmin):
    list_display = (
        'professional_tax', 'default_working_days', 'default_full_day_hours',
        'default_half_day_hours', 'default_overtime_rate',
        'default_saturday_is_working', 'default_sunday_is_working',
    )

    def has_add_permission(self, request):
        # ✅ Only 1 PayrollSettings allowed
        return not PayrollSettings.objects.exists()


# ============================================================
# 📝 LEAVE MANAGEMENT ADMIN
# ============================================================

@admin.register(LeaveType)
class LeaveTypeAdmin(admin.ModelAdmin):
    list_display = ('name', 'max_days_per_year', 'is_paid')  # ✅ updated
    list_editable = ('is_paid',)
    search_fields = ('name',)


class LeaveApprovalInline(admin.TabularInline):
    model = LeaveApproval
    extra = 0
    readonly_fields = ('acted_at',)
    can_delete = False


@admin.register(LeaveApplication)
class LeaveApplicationAdmin(admin.ModelAdmin):
    list_display = ('employee', 'leave_type', 'start_date', 'end_date', 'total_days', 'status', 'applied_on')
    list_filter = ('status', 'leave_type', 'start_date', 'applied_on')
    search_fields = ('employee__user__first_name', 'employee__user__last_name', 'reason')
    readonly_fields = ('applied_on', 'total_days_display')
    inlines = [LeaveApprovalInline]
    actions = [approve_leaves]
    
    def total_days_display(self, obj):
        return obj.total_days()
    total_days_display.short_description = 'Total Days'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('employee__user', 'leave_type')


@admin.register(LeaveWorkflowStage)
class LeaveWorkflowStageAdmin(admin.ModelAdmin):
    list_display = ('level', 'role_name', 'next_level')
    list_editable = ('role_name', 'next_level')
    ordering = ('level',)


@admin.register(LeaveApproval)
class LeaveApprovalAdmin(admin.ModelAdmin):
    list_display = ('leave', 'level', 'approver_name', 'status', 'acted_at')
    list_filter = ('level', 'status', 'acted_at')
    search_fields = ('leave__employee__user__first_name', 'leave__employee__user__last_name')
    readonly_fields = ('acted_at',)
    
    def approver_name(self, obj):
        return obj.approver.user.get_full_name() if obj.approver else '-'
    approver_name.short_description = 'Approver'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('leave__employee__user', 'approver__user')


# ============================================================
# 📄 JOINING & RESIGNATION ADMIN
# ============================================================

@admin.register(JoiningDetail)
class JoiningDetailAdmin(admin.ModelAdmin):
    list_display = ('employee', 'date_of_joining', 'probation_period_months', 'confirmation_date')
    list_filter = ('date_of_joining', 'confirmation_date')
    search_fields = ('employee__user__first_name', 'employee__user__last_name')
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('employee__user')


@admin.register(Resignation)
class ResignationAdmin(admin.ModelAdmin):
    list_display = ('employee', 'resignation_date', 'last_working_day', 'approval_status', 'approved_by_name')
    list_filter = ('resignation_date', 'approval_status')
    search_fields = ('employee__user__first_name', 'employee__user__last_name', 'reason')
    
    def approved_by_name(self, obj):
        return obj.approved_by.user.get_full_name() if obj.approved_by else '-'
    approved_by_name.short_description = 'Approved By'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('employee__user', 'approved_by__user')


# ============================================================
# 🔔 NOTIFICATIONS ADMIN
# ============================================================


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ('recipient', 'message_preview', 'is_read', 'created_at')
    list_filter = ('is_read', 'created_at')
    search_fields = ('recipient__username', 'recipient__first_name', 'recipient__last_name', 'message')
    readonly_fields = ('created_at',)
    actions = [mark_notifications_read]
    
    def message_preview(self, obj):
        return obj.message[:50] + '...' if len(obj.message) > 50 else obj.message
    message_preview.short_description = 'Message'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('recipient')


@admin.register(JoiningDocument)
class JoiningDocumentAdmin(admin.ModelAdmin):
    list_display = ('joining', 'file_name', 'uploaded_at')
    list_filter = ('uploaded_at',)
    search_fields = (
        'joining__employee__user__first_name',
        'joining__employee__user__last_name',
        'file'
    )
    readonly_fields = ('uploaded_at',)

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('joining__employee__user')

    # ✅ Display the filename instead of full path
    def file_name(self, obj):
        return obj.file.name.split('/')[-1]
    file_name.short_description = "Document Name"


# ============================================================
# 🔄 UNREGISTER DEFAULT USER & REGISTER CUSTOM
# ============================================================

admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)


# ============================================================
# 🎯 ADMIN SITE CUSTOMIZATION
# ============================================================

admin.site.site_header = "Employee Management System"
admin.site.site_title = "EMS Admin"
admin.site.index_title = "Welcome to Employee Management System"