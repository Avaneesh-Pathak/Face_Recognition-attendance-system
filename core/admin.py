import copy
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from django.utils.html import format_html
from django.contrib.admin import RelatedOnlyFieldListFilter
from .models import (
    Organisation, Employee, Attendance, AttendanceSettings, DailyReport, Department,
    SalaryStructure, Payroll, LeaveType, LeaveApplication, 
    LeaveWorkflowStage, LeaveApproval, JoiningDetail, Resignation, Notification, JoiningDocument,
    WorkRule, PayrollSettings, OfficeLocation
)

# ============================================================
# BASE ADMIN CLASSES
# ============================================================

class TenantBaseAdmin(admin.ModelAdmin):
    """Django admin with explicit superuser/global and organisation-admin/tenant scopes."""

    @staticmethod
    def _is_org_admin(request):
        user = request.user
        try:
            return bool(user.is_active and user.is_staff and not user.is_superuser
                        and user.employee.is_active and user.employee.organisation_id
                        and user.employee.role == 'Admin')
        except (AttributeError, Employee.DoesNotExist):
            return False

    def _has_tenant_scope(self):
        return any(field.name == 'organisation' for field in self.model._meta.get_fields())

    def _owns_object(self, request, obj):
        if request.user.is_superuser:
            return True
        if not self._is_org_admin(request) or not self._has_tenant_scope():
            return False
        return getattr(obj, 'organisation_id', None) == request.user.employee.organisation_id

    def has_module_permission(self, request):
        return bool(request.user.is_superuser or (self._is_org_admin(request) and self._has_tenant_scope()))

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        if self._is_org_admin(request) and self._has_tenant_scope():
            return qs.filter(organisation_id=request.user.employee.organisation_id)
        return qs.none()

    def has_view_permission(self, request, obj=None):
        if request.user.is_superuser:
            return True
        if not self.has_module_permission(request):
            return False
        return obj is None or self._owns_object(request, obj)

    def has_add_permission(self, request):
        return bool(request.user.is_superuser or (self._is_org_admin(request) and self._has_tenant_scope()))

    def has_change_permission(self, request, obj=None):
        if request.user.is_superuser:
            return True
        if not self.has_module_permission(request):
            return False
        return obj is None or self._owns_object(request, obj)

    def has_delete_permission(self, request, obj=None):
        if request.user.is_superuser:
            return True
        if not self.has_module_permission(request):
            return False
        return obj is None or self._owns_object(request, obj)

    def save_model(self, request, obj, form, change):
        if not request.user.is_superuser:
            if not self._is_org_admin(request) or not self._has_tenant_scope():
                raise PermissionError('Organisation administrator access required')
            obj.organisation_id = request.user.employee.organisation_id
        super().save_model(request, obj, form, change)

    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        if not request.user.is_superuser and self._is_org_admin(request):
            related_model = db_field.remote_field.model
            if related_model is User:
                kwargs['queryset'] = related_model.objects.filter(
                    employee__organisation_id=request.user.employee.organisation_id)
            elif any(field.name == 'organisation' for field in related_model._meta.get_fields()):
                kwargs['queryset'] = related_model.objects.filter(
                    organisation_id=request.user.employee.organisation_id)
        return super().formfield_for_foreignkey(db_field, request, **kwargs)

    def formfield_for_manytomany(self, db_field, request, **kwargs):
        if not request.user.is_superuser and self._is_org_admin(request):
            related_model = db_field.remote_field.model
            if any(field.name == 'organisation' for field in related_model._meta.get_fields()):
                kwargs['queryset'] = related_model.objects.filter(
                    organisation_id=request.user.employee.organisation_id)
        return super().formfield_for_manytomany(db_field, request, **kwargs)

    def get_fields(self, request, obj=None):
        fields = list(super().get_fields(request, obj))
        if not request.user.is_superuser:
            fields = [name for name in fields if name != 'organisation']
        elif self._has_tenant_scope() and 'organisation' not in fields:
            fields.insert(0, 'organisation')
        return fields

    def get_fieldsets(self, request, obj=None):
        fieldsets = list(copy.deepcopy(super().get_fieldsets(request, obj)))
        if not self._has_tenant_scope():
            return fieldsets
        if request.user.is_superuser:
            flattened = [field for _, options in fieldsets for field in options.get('fields', ())]
            if 'organisation' not in flattened and fieldsets:
                first = list(fieldsets[0][1].get('fields', ()))
                first.insert(0, 'organisation')
                fieldsets[0][1]['fields'] = tuple(first)
        else:
            for _, options in fieldsets:
                options['fields'] = tuple(f for f in options.get('fields', ()) if f != 'organisation')
        return fieldsets

    class Media:
        css = {'all': ('admin/css/production_admin.css',)}

# ============================================================
# CUSTOM ADMIN ACTIONS
# ============================================================

@admin.action(description="Mark selected employees as active")
def make_employees_active(modeladmin, request, queryset):
    for employee in queryset.order_by('pk'):
        employee.is_active = True
        employee.employment_status = 'active'
        employee.save(update_fields=['is_active', 'employment_status'])

@admin.action(description="Mark selected employees as inactive")
def make_employees_inactive(modeladmin, request, queryset):
    for employee in queryset.order_by('pk'):
        employee.is_active = False
        employee.employment_status = 'resigned'
        employee.save(update_fields=['is_active', 'employment_status'])

@admin.action(description="Process payroll for selected records")
def process_payroll(modeladmin, request, queryset):
    queryset.update(status='processed')

@admin.action(description="Approve selected leave applications")
def approve_leaves(modeladmin, request, queryset):
    for leave in queryset.order_by('employee_id', 'pk'):
        leave.status = 'approved'
        leave.save(update_fields=['status'])

@admin.action(description="Mark notifications as read")
def mark_notifications_read(modeladmin, request, queryset):
    queryset.update(is_read=True)


# ============================================================
# ORGANISATION ADMIN (RESTRICTED TO SUPERUSERS ONLY)
# ============================================================

@admin.register(Organisation)
class OrganisationAdmin(admin.ModelAdmin):

    def has_module_permission(self, request):
        return request.user.is_superuser

    def has_view_permission(self, request, obj=None):
        return request.user.is_superuser

    def has_change_permission(self, request, obj=None):
        return request.user.is_superuser

    def has_add_permission(self, request):
        return request.user.is_superuser

    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser


# ============================================================
# 👤 EMPLOYEE & USER ADMIN
# ============================================================

class EmployeeInline(admin.StackedInline):
    model = Employee
    can_delete = False
    verbose_name_plural = 'Employee Details'
    fields = ('employee_id', 'department', 'position', 'phone_number', 'employment_status')

    def get_formset(self, request, obj=None, **kwargs):
        formset = super().get_formset(request, obj, **kwargs)
        # Filter options inside the Inline form when standard tenant admins create/edit a user
        if not request.user.is_superuser and hasattr(request.user, 'employee'):
            user_org = request.user.employee.organisation
            if 'department' in formset.form.base_fields:
                formset.form.base_fields['department'].queryset = Department.objects.filter(organisation=user_org)
        return formset


class CustomUserAdmin(UserAdmin):
    inlines = (EmployeeInline,)

    def has_module_permission(self, request):
        return request.user.is_superuser

    def has_view_permission(self, request, obj=None):
        return request.user.is_superuser
    list_display = (
        'username',
        'email',
        'first_name',
        'last_name',
        'get_employee_id',
        'get_department',
        'is_staff'
    )
    list_select_related = ('employee',)

    def get_employee_id(self, obj):
        return obj.employee.employee_id if hasattr(obj, 'employee') else '-'
    get_employee_id.short_description = 'Employee ID'

    def get_department(self, obj):
        return obj.employee.department if hasattr(obj, 'employee') else '-'
    get_department.short_description = 'Department'

    # ======================================================
    # TENANT FILTER
    # ======================================================

    def get_queryset(self, request):
        qs = super().get_queryset(request)

        if request.user.is_superuser:
            return qs

        if hasattr(request.user, "employee"):
            return qs.filter(
                employee__organisation=request.user.employee.organisation
            )

        return qs.none()

    # ======================================================
    # HIDE DJANGO PERMISSION SECTION
    # ======================================================

    def get_fieldsets(self, request, obj=None):

        if request.user.is_superuser:
            return super().get_fieldsets(request, obj)

        return (
            (
                None,
                {
                    "fields": (
                        "username",
                        "password",
                    )
                },
            ),
            (
                "Personal info",
                {
                    "fields": (
                        "first_name",
                        "last_name",
                        "email",
                    )
                },
            ),
            (
                "Status",
                {
                    "fields": (
                        "is_active",
                    )
                },
            ),
        )

    # ======================================================
    # READ ONLY FIELDS
    # ======================================================

    def get_readonly_fields(self, request, obj=None):

        readonly = list(super().get_readonly_fields(request, obj))

        if not request.user.is_superuser:
            readonly.extend([
                "groups",
                "user_permissions",
                "is_staff",
                "is_superuser",
            ])

        return readonly

    # ======================================================
    # SAVE
    # ======================================================

    def save_model(self, request, obj, form, change):

        if not request.user.is_superuser:
            obj.is_staff = True
            obj.is_superuser = False

        super().save_model(request, obj, form, change)

    # ======================================================
    # SECURITY
    # ======================================================

    def has_delete_permission(self, request, obj=None):

        if request.user.is_superuser:
            return True

        return False

    def has_add_permission(self, request):

        if request.user.is_superuser:
            return True

        return False

    def has_change_permission(self, request, obj=None):
        return request.user.is_superuser


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
    # RelatedOnlyFieldListFilter restricts sidebar options to only show choices linked to visible tenant entries
    list_filter = (
        ('department', RelatedOnlyFieldListFilter),
        'position',
        'employment_status',
        'is_active',
        'date_of_joining',
        'role',
        'location_type',
        ('assigned_location', RelatedOnlyFieldListFilter),
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
        return 'Enrolled' if obj.face_encoding else 'Not enrolled' 
    face_encoding_preview.short_description = 'Face Encoding Preview'


# ============================================================
# 🕒 ATTENDANCE ADMIN
# ============================================================

@admin.register(Attendance)
class AttendanceAdmin(TenantBaseAdmin):
    def has_add_permission(self, request):
        return super().has_add_permission(request) if request.user.is_superuser else False

    def has_change_permission(self, request, obj=None):
        return super().has_change_permission(request, obj) if request.user.is_superuser else False

    def has_delete_permission(self, request, obj=None):
        return super().has_delete_permission(request, obj) if request.user.is_superuser else False

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
    def has_add_permission(self, request):
        return super().has_add_permission(request) if request.user.is_superuser else False

    def has_change_permission(self, request, obj=None):
        return super().has_change_permission(request, obj) if request.user.is_superuser else False

    def has_delete_permission(self, request, obj=None):
        return super().has_delete_permission(request, obj) if request.user.is_superuser else False

    def get_actions(self, request):
        return {}

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

    def message_preview(self, obj):
        return obj.message[:50] + '...' if len(obj.message) > 50 else obj.message
    message_preview.short_description = 'Message'


# Register custom User admin and clean default Registration
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)


admin.site.site_header = 'FaceAttend Administration'
admin.site.site_title = 'FaceAttend Admin'
admin.site.index_title = 'System administration'


# Operational models intentionally omitted from the tenant-facing admin menu are
# available to the global superuser for complete system administration.
from .models import (
    Holiday, EmployeeShiftAssignment, FaceMatchStats, AttendanceMatchLog,
    EmployeeThreshold, AuditEvent,
)

@admin.register(Holiday)
class HolidayAdmin(TenantBaseAdmin):
    list_display = ('name', 'date')

@admin.register(EmployeeShiftAssignment)
class EmployeeShiftAssignmentAdmin(TenantBaseAdmin):
    list_display = ('employee', 'work_rule', 'start_date', 'end_date')

@admin.register(FaceMatchStats)
class FaceMatchStatsAdmin(TenantBaseAdmin):
    list_display = ('employee_id', 'total_attempts', 'false_accepts', 'false_rejects')

@admin.register(AttendanceMatchLog)
class AttendanceMatchLogAdmin(TenantBaseAdmin):
    list_display = ('employee_id', 'confidence', 'created_at')

@admin.register(EmployeeThreshold)
class EmployeeThresholdAdmin(TenantBaseAdmin):
    list_display = ('employee_id', 'cos_max', 'euc_max')

@admin.register(AuditEvent)
class AuditEventAdmin(admin.ModelAdmin):
    """Global-superuser-only, append-only audit viewer."""
    list_display = ('created_at', 'organisation', 'actor_name', 'action', 'entity', 'object_id')
    list_filter = ('organisation', 'action', 'entity', 'created_at')
    search_fields = ('actor_name', 'entity', 'object_id', 'request_id')
    readonly_fields = ('organisation', 'actor_id', 'actor_name', 'action', 'entity',
                       'object_id', 'before', 'after', 'request_id', 'reason', 'created_at')
    date_hierarchy = 'created_at'

    def has_module_permission(self, request):
        return request.user.is_superuser

    def has_view_permission(self, request, obj=None):
        return request.user.is_superuser

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
