# admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from django.utils.html import format_html
from .models import (
    Organisation, Employee, Attendance, AttendanceSettings, DailyReport, Department,
    SalaryStructure, Payroll, LeaveType, LeaveApplication, 
    LeaveWorkflowStage, LeaveApproval, JoiningDetail, Resignation, Notification, JoiningDocument,
    WorkRule, PayrollSettings, OfficeLocation
)

# ============================================================
# BASE ADMIN CLASSES
# ============================================================

# core/admin.py
import copy

class TenantBaseAdmin(admin.ModelAdmin):
    """Base class to restrict Admin Panel viewing limits."""
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        
        # Safe check: Only filter if model has 'organisation' field
        has_org_field = any(field.name == 'organisation' for field in self.model._meta.get_fields())
        
        if has_org_field and hasattr(request.user, 'employee'):
            return qs.filter(organisation=request.user.employee.organisation)
        return qs

    def save_model(self, request, obj, form, change):
        # Automatically assign the model to the logged-in admin's organisation
        if not change and hasattr(request.user, 'employee') and hasattr(obj, 'organisation_id'):
            if not obj.organisation_id:
                obj.organisation = request.user.employee.organisation
        super().save_model(request, obj, form, change)

    def get_fields(self, request, obj=None):
        """Dynamically add/remove organisation field on standard layouts."""
        fields = list(super().get_fields(request, obj))
        has_org_field = any(field.name == 'organisation' for field in self.model._meta.get_fields())
        
        if has_org_field:
            if request.user.is_superuser:
                if 'organisation' not in fields:
                    fields.insert(0, 'organisation')  # Prepend for superusers
            else:
                if 'organisation' in fields:
                    fields.remove('organisation')  # Hide from standard HR/Admins
        return fields

    def get_fieldsets(self, request, obj=None):
        """Dynamically inject organisation field into fieldset layouts (like EmployeeAdmin)."""
        fieldsets = super().get_fieldsets(request, obj)
        has_org_field = any(field.name == 'organisation' for field in self.model._meta.get_fields())
        if not has_org_field:
            return fieldsets

        # Deep copy to avoid mutating the class-level layout
        fieldsets = list(copy.deepcopy(fieldsets))

        if request.user.is_superuser:
            # Find if 'organisation' is already in any of the fieldsets
            flat_fields = []
            for name, options in fieldsets:
                flat_fields.extend(options.get('fields', []))
            
            if 'organisation' not in flat_fields:
                # Add 'organisation' as the first field of the first fieldset
                first_fieldset_fields = list(fieldsets[0][1]['fields'])
                first_fieldset_fields.insert(0, 'organisation')
                fieldsets[0][1]['fields'] = tuple(first_fieldset_fields)
        else:
            # Remove 'organisation' from all fieldsets so tenant admins cannot see or edit it
            for name, options in fieldsets:
                fields = list(options.get('fields', []))
                if 'organisation' in fields:
                    fields.remove('organisation')
                    options['fields'] = tuple(fields)
                    
        return fieldsets

# ============================================================
# CUSTOM ADMIN ACTIONS
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
# ORGANISATION ADMIN
# ============================================================

@admin.register(Organisation)
class OrganisationAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_at', 'is_active')
    search_fields = ('name',)


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
class EmployeeAdmin(TenantBaseAdmin):
    list_display = (
        'employee_id',
        'user_full_name',
        'department',
        'position',
        'employment_status',
        'is_active',
        'date_of_joining',
        'role',
        'location_type',
        'assigned_location',
    )
    list_filter = (
        'department',
        'position',
        'employment_status',
        'is_active',
        'date_of_joining',
        'role',
        'location_type',
        'assigned_location',
    )
    search_fields = (
        'employee_id',
        'user__first_name',
        'user__last_name',
        'user__email',
        'department__name',
        'assigned_location__name',
    )
    readonly_fields = ('created_at', 'face_encoding_preview')

    fieldsets = (
        ('Personal Information', {
            'fields': (
                'user',
                'employee_id',
                'phone_number',
            )
        }),
        ('Employment Details', {
            'fields': (
                'department',
                'position',
                'manager',
                'date_of_joining',
                'date_of_resignation',
                'employment_status',
                'role',
            )
        }),
        ('📍 Location Access Control', {
            'fields': (
                'location_type',
                'assigned_location',
            ),
            'description': (
                'INDOOR: Employee can mark attendance ONLY at assigned office.<br>'
                'OUTDOOR: Employee can mark attendance at ANY office location.'
            )
        }),
        ('Face Recognition', {
            'fields': (
                'face_image',
                'face_encoding_preview',
            ),
            'classes': ('collapse',)
        }),
        ('System', {
            'fields': (
                'is_active',
                'created_at',
            ),
            'classes': ('collapse',)
        }),
    )

    def user_full_name(self, obj):
        return obj.user.get_full_name()
    user_full_name.short_description = 'Full Name'

    def face_encoding_preview(self, obj):
        if obj.face_encoding:
            return format_html(
                '<div style="max-height: 100px; overflow-y: auto; '
                'background: #f5f5f5; padding: 10px; border-radius: 5px; '
                'font-family: monospace; font-size: 10px;">{}</div>',
                obj.face_encoding[:200] + '...' if len(obj.face_encoding) > 200 else obj.face_encoding
            )
        return "No encoding"
    face_encoding_preview.short_description = 'Face Encoding Preview'


# ============================================================
# 🕒 ATTENDANCE ADMIN
# ============================================================

@admin.register(Attendance)
class AttendanceAdmin(TenantBaseAdmin):
    list_display = ('employee', 'attendance_type', 'timestamp', 'location', 'confidence_score')
    list_filter = ('attendance_type', 'timestamp', 'location')
    search_fields = ('employee__user__first_name', 'employee__user__last_name', 'employee__employee_id')
    date_hierarchy = 'timestamp'


@admin.register(AttendanceSettings)
class AttendanceSettingsAdmin(TenantBaseAdmin):
    list_display = ('check_in_start', 'check_in_end', 'check_out_start', 'check_out_end', 'late_threshold', 'confidence_threshold')


@admin.register(DailyReport)
class DailyReportAdmin(TenantBaseAdmin):
    list_display = ('date', 'total_employees', 'present_count', 'absent_count', 'late_count', 'average_hours')


@admin.register(Department)
class DepartmentAdmin(TenantBaseAdmin):
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
class SalaryStructureAdmin(TenantBaseAdmin):
    list_display = (
        'employee',
        'base_salary',
        'hra',
        'allowances',
        'deductions',
        'total_salary_display'
    )
    def total_salary_display(self, obj):
        return obj.total_salary()


# Defined before PayrollAdmin to fix the order-of-declaration error
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
class PayrollAdmin(TenantBaseAdmin):
    list_display = (
        'employee',
        'month',
        'present_days',
        'absent_days',
        'overtime_hours',
        'net_salary',
        'status'
    )
    list_filter = (PayrollMonthFilter, 'status')
    search_fields = ('employee__employee_id', 'employee__user__first_name')
    readonly_fields = ('processed_at',)


@admin.register(OfficeLocation)
class OfficeLocationAdmin(TenantBaseAdmin):
    list_display = ("name", "latitude", "longitude", "radius_meters", "is_active")


@admin.register(WorkRule)
class WorkRuleAdmin(TenantBaseAdmin):
    list_display = (
        'name', 'working_days', 'saturday_is_working', 'sunday_is_working',
        'full_day_hours', 'half_day_hours', 'overtime_rate'
    )
    list_editable = ('working_days', 'saturday_is_working', 'sunday_is_working')


@admin.register(PayrollSettings)
class PayrollSettingsAdmin(TenantBaseAdmin):
    list_display = (
        'professional_tax', 'default_working_days', 'default_full_day_hours',
        'default_half_day_hours', 'default_overtime_rate',
        'default_saturday_is_working', 'default_sunday_is_working',
    )


# ============================================================
# 📝 LEAVE MANAGEMENT ADMIN
# ============================================================

@admin.register(LeaveType)
class LeaveTypeAdmin(TenantBaseAdmin):
    list_display = ('name', 'max_days_per_year', 'is_paid')
    list_editable = ('is_paid',)


@admin.register(LeaveApplication)
class LeaveApplicationAdmin(TenantBaseAdmin):
    list_display = ('employee', 'leave_type', 'start_date', 'end_date', 'total_days', 'status', 'applied_on')


@admin.register(LeaveWorkflowStage)
class LeaveWorkflowStageAdmin(TenantBaseAdmin):
    list_display = ('level', 'role_name', 'next_level')


@admin.register(LeaveApproval)
class LeaveApprovalAdmin(TenantBaseAdmin):
    list_display = ('leave', 'level', 'approver_name', 'status', 'acted_at')

    # Restored helper method to prevent E108 system error
    def approver_name(self, obj):
        return obj.approver.user.get_full_name() if obj.approver else '-'
    approver_name.short_description = 'Approver'


# ============================================================
# 📄 JOINING & RESIGNATION ADMIN
# ============================================================

@admin.register(JoiningDetail)
class JoiningDetailAdmin(TenantBaseAdmin):
    list_display = ('employee', 'date_of_joining', 'probation_period_months', 'confirmation_date')


@admin.register(JoiningDocument)
class JoiningDocumentAdmin(TenantBaseAdmin):
    list_display = ('joining', 'file_name', 'uploaded_at')

    # Restored helper method to prevent E108 system error
    def file_name(self, obj):
        return obj.file.name.split('/')[-1] if obj.file else '-'
    file_name.short_description = "Document Name"


@admin.register(Resignation)
class ResignationAdmin(TenantBaseAdmin):
    list_display = ('employee', 'resignation_date', 'last_working_day', 'approval_status', 'approved_by_name')

    def approved_by_name(self, obj):
        return obj.approved_by.user.get_full_name() if obj.approved_by else '-'
    approved_by_name.short_description = 'Approved By'


# ============================================================
# 🔔 NOTIFICATIONS ADMIN
# ============================================================

@admin.register(Notification)
class NotificationAdmin(TenantBaseAdmin):
    list_display = ('recipient', 'message_preview', 'is_read', 'created_at')

    # Restored helper method to prevent E108 system error
    def message_preview(self, obj):
        return obj.message[:50] + '...' if len(obj.message) > 50 else obj.message
    message_preview.short_description = 'Message'


# Register Django User model
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)