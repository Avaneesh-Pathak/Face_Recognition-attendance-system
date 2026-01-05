# core/signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from datetime import date
from django.utils import timezone
from .models import SalaryStructure, Payroll

@receiver(post_save, sender=SalaryStructure)
def create_initial_payroll(sender, instance, created, **kwargs):
    """
    ✅ Automatically generate payroll when salary structure is created for an employee.
    """
    if created:
        today = date.today()
        month_start = date(today.year, today.month, 1)

        Payroll.objects.get_or_create(
            employee=instance.employee,
            month=month_start,
            defaults={
                'basic_pay': instance.base_salary,
                'allowances': instance.allowances,
                'deductions': instance.deductions,
                'net_salary': instance.total_salary(),
                'status': 'processed',  # Can be paid later by Finance Dept
                'processed_at': timezone.now(),
            }
        )
