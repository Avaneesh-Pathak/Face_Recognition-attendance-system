from django import template
import calendar
register = template.Library()

@register.filter
def filter_attendance_type(records, attendance_type):
    """Filter attendance records by type ('check_in' or 'check_out')."""
    return [r for r in records if getattr(r, 'attendance_type', '') == attendance_type]


@register.filter
def get_confidence_scores(records):
    """Calculate average confidence score from attendance records"""
    scores = [r.confidence_score for r in records if r.confidence_score is not None]
    if not scores:
        return 0
    return sum(scores) / len(scores) * 100  # convert to %


@register.filter
def get_percentage(part, whole):
    """Safe percentage calculation."""
    try:
        return (part / whole) * 100 if whole else 0
    except:
        return 0

@register.filter
def filter_attendance_type(records, attendance_type):
    """Return queryset filtered by attendance type."""
    try:
        # Ensure records is a queryset or list
        if hasattr(records, 'filter'):
            return records.filter(attendance_type=attendance_type)
        else:
            return [r for r in records if getattr(r, 'attendance_type', None) == attendance_type]
    except Exception:
        return []


@register.filter
def get_month_name(value):
    """Return the month name from its number."""
    return calendar.month_name[value]

@register.filter
def to(value, end):
    """Generate range in template"""
    return range(value, end + 1)

