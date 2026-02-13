from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils.timezone import now

from .models import Payroll, SalaryStructure, Employee


@receiver(post_save, sender=Employee)
def create_initial_payroll(sender, instance, created, **kwargs):
    if not created:
        return

    employee = instance

    salary = (
        SalaryStructure.objects
        .filter(employee=employee)
        .values_list("base_salary", flat=True)
        .first()
        or 0
    )

    month = now().date().replace(day=1)

    Payroll.objects.get_or_create(
        employee=employee,
        month=month,
        defaults={
            "basic_pay": salary,
            "allowances": 0,
            "deductions": 0,
            "calculated_salary": salary,  # ✅ REQUIRED
            "net_salary": salary,         # ✅ REQUIRED
            "status": "processed",
        }
    )
