from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('attendance/', views.attendance_page, name='attendance_page'),
    path('video_feed/', views.video_feed_view, name='video_feed'),
    path('mark_attendance/', views.mark_attendance, name='mark_attendance'),
    path('reports/', views.attendance_reports, name='reports'),
    path('settings/', views.attendance_settings, name='attendance_settings'),
    # path('attendance-history/', views.get_attendance_history, name='attendance_history'),
    path('attendance-history/', views.attendance_history_page, name='attendance_history_all'),
    path('attendance-history/api/', views.get_attendance_history, name='attendance_history_api'),
    path('attendance/log/api/', views.attendance_log_api, name='attendance_log_api'),
    path('attendance-summary-api/', views.attendance_summary_api, name='attendance_summary_api'),




]