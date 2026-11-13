from django import template

register = template.Library()

@register.filter
def to_percent(value, decimals=0):
    """
    Converts 0.60 → 60 or 0.753 → 75.3 depending on decimals.
    Usage: {{ value|to_percent:1 }} or {{ value|to_percent }}
    """
    try:
        value = float(value) * 100
        format_str = f"%.{decimals}f"
        return format_str % value
    except:
        return value
