from .models import Notification


def notification_context(request):
    user = getattr(request, 'user', None)
    if not user or not user.is_authenticated:
        return {'recent_notifications': (), 'unread_notification_count': 0}

    notifications = Notification.objects.filter(recipient=user)
    return {
        'recent_notifications': notifications.order_by('-created_at', '-pk')[:5],
        'unread_notification_count': notifications.filter(is_read=False).count(),
    }
