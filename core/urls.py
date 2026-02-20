from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('employees/', views.employee_list, name='employee_list'),
    path('employees/<int:pk>/edit/', views.employee_update, name='employee_update'),
    path('employees/<int:pk>/delete/', views.employee_delete, name='employee_delete'),

    path('api/check-username/', views.check_username, name='check_username'),
    path("employees/<int:pk>/profile/",views.employee_profile,name="employee_profile"),
    path('my-profile/', views.my_profile_view, name='my_profile'),

    path('api/check-email/', views.check_email, name='check_email'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('attendance/', views.attendance_page, name='attendance_page'),
    # streaming endpoint — ensure views.video_feed (or adjust to actual view name)
    path('video_feed/', views.video_feed, name='video_feed'),
    path('mark_attendance/', views.mark_attendance, name='mark_attendance'),
    path('reports/', views.attendance_reports, name='reports'),
    path('settings/', views.attendance_settings, name='attendance_settings'),
    path('attendance-history/', views.attendance_history_page, name='attendance_history_all'),
    path('attendance-history/api/', views.get_attendance_history, name='attendance_history_api'),
    path('attendance/log/api/', views.attendance_log_api, name='attendance_log_api'),
    path('attendance-summary-api/', views.attendance_summary_api, name='attendance_summary_api'),
    path('attendance-calendar/<int:employee_id>/<int:year>/<int:month>/', views.attendance_calendar, name='attendance_calendar'),
    path('attendance-calendar-data/<int:employee_id>/<int:year>/<int:month>/', views.attendance_calendar_data, name='attendance_calendar_data'),
    path('attendance-day-detail/<int:emp_id>/<str:date>/', views.attendance_day_detail, name='attendance_day_detail'),
    
    # 🏢 Organization Structure URLs
    path('org-chart/', views.org_chart_page, name='org_chart'),
    path('api/org-tree/me/', views.org_tree_api_me, name='org_tree_api_me'),
    path('api/org-tree/<int:employee_id>/', views.org_tree_api, name='org_tree_api'),

    # ✅ Department URLs
    path('departments/', views.department_list, name='department_list'),
    path('departments/create/', views.department_create, name='department_create'),
    path('departments/<int:pk>/edit/', views.department_edit, name='department_edit'),
    path('departments/<int:pk>/delete/', views.department_delete, name='department_delete'),
    path('employees/<str:employee_id>/', views.employee_detail, name='employee_detail'),
    
    # 💰 Salary & Payroll URLs
    path('salary-structures/', views.salary_structure_list, name='salary_structure_list'),
    path('salary-structures/create/', views.salary_structure_create, name='salary_structure_create'),
    path('payroll/', views.payroll_list, name='payroll_list'),
    path('payroll/generate/', views.generate_payroll, name='generate_payroll'),
    path('my-salary/', views.my_salary, name='my_salary'),
    path('payroll/pay/<int:pk>/', views.pay_salary, name='pay_salary'),
    path('payroll/slip/<int:pk>/', views.payroll_slip_pdf, name='payroll_slip_pdf'),
    path('payroll/expense/', views.payroll_expense_chart, name='payroll_expense_chart'),
    path('api/payroll/expense/', views.payroll_expense_api, name='payroll_expense_api'),
    path('payroll/history/<int:employee_id>/', views.employee_salary_history, name='employee_salary_history'),
    
    # 📝 Leave Management URLs
    path('leave-types/', views.leave_type_list, name='leave_type_list'),
    path('leave-types/create/', views.leave_type_create, name='leave_type_create'),
    path('leaves/', views.leave_application_list, name='leave_application_list'),
    path('leaves/create/', views.leave_application_create, name='leave_application_create'),
    path('leaves/<int:pk>/approve/', views.leave_approval_action, name='leave_approval_action'),
    path('leave-workflow/', views.leave_workflow_list, name='leave_workflow_list'),
    path('leave-workflow/create/', views.leave_workflow_create, name='leave_workflow_create'),
    path('leaves/my-approvals/', views.my_pending_approvals, name='my_pending_approvals'),
    path('leaves/<int:pk>/', views.leave_application_detail, name='leave_application_detail'),
    path('leaves/<int:pk>/edit/', views.leave_application_update, name='leave_application_update'),

    

    
    # 📄 Joining & Resignation URLs
    path('joining-details/', views.joining_detail_list, name='joining_detail_list'),
    path('joining-details/create/', views.joining_detail_create, name='joining_detail_create'),
    path('resignations/', views.resignation_list, name='resignation_list'),
    path('resignations/create/', views.resignation_create, name='resignation_create'),
    path('resignations/<int:pk>/approve/', views.resignation_approve, name='resignation_approve'),
    path('download-document/<int:doc_id>/', views.download_document, name='download_document'),
    path('delete-document/<int:doc_id>/', views.delete_document, name='delete_document'),
    path('download-all-documents/<int:detail_id>/', views.download_all_documents, name='download_all_documents'),
    path('joining-details/<int:detail_id>/upload-documents/',views.upload_joining_documents,name='upload_joining_documents'),
    
    # 🔔 Notification URLs
    path('notifications/', views.notification_list, name='notification_list'),
    path('notifications/<int:pk>/read/', views.mark_notification_read, name='mark_notification_read'),
    path('notifications/mark-all-read/', views.mark_all_notifications_read, name='mark_all_notifications_read'),
    path('get-managers/<int:dept_id>/', views.get_managers_by_department, name='get_managers'),

    #work rule
    path("workrules/", views.workrule_list, name="workrule_list"),
    path("workrules/create/", views.workrule_create, name="workrule_create"),
    path("workrules/<int:pk>/edit/", views.workrule_update, name="workrule_update"),
    path("workrules/<int:pk>/delete/", views.workrule_delete, name="workrule_delete"),
    path("attendance/heartbeat/", views.attendance_heartbeat, name="attendance_heartbeat"),

    # CRUD Attendance Logs
    path('attendance/create/', views.create_attendance, name='attendance_create'),
    path('attendance/update/<int:att_id>/', views.update_attendance, name='attendance_update'),
    path('attendance/delete/<int:att_id>/', views.delete_attendance, name='attendance_delete'),

    # Office Location URLs
    path("office-locations/", views.office_location_list, name="office_location_list"),
    path("office-locations/create/", views.office_location_create, name="office_location_create"),
    path("office-locations/<int:pk>/edit/", views.office_location_update, name="office_location_update"),
    path("office-locations/<int:pk>/delete/", views.office_location_delete, name="office_location_delete"),

]