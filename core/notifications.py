from .models import Employee, Notification


def create_user_notification(recipient, organisation_id, message, link_path=''):
    """Create a notification only for a user who belongs to its organisation."""
    if not recipient or not organisation_id or not message:
        return None

    recipient_profile = Employee.global_objects.filter(
        user_id=recipient.pk,
        organisation_id=organisation_id,
        is_active=True,
    ).only('pk').first()
    if recipient_profile is None:
        return None

    return Notification.global_objects.create(
        organisation_id=organisation_id,
        recipient=recipient,
        message=message[:255],
        link=link_path or None,
    )
