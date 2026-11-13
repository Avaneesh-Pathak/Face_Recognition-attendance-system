# core/templatetags/math_filters.py
from django import template

register = template.Library()

@register.filter
def subtract(value, arg):
    """Subtract the arg from the value"""
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def multiply(value, arg):
    """Multiply value by arg"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def divide(value, arg):
    """Divide value by arg"""
    try:
        return float(value) / float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def percentage(value, total):
    """Calculate percentage of value from total"""
    try:
        if total == 0:
            return 0
        return (float(value) / float(total)) * 100
    except (ValueError, TypeError):
        return 0