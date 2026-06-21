# signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils.timezone import now
from .models import Payroll, SalaryStructure

@receiver(post_save, sender=SalaryStructure)
def create_initial_payroll(sender, instance, created, **kwargs):
    """
    Triggered when a SalaryStructure is created.
    Listening to SalaryStructure instead of Employee resolves the database race condition 
    where the initial payroll was created with 0 salary.
    """
    if not created:
        return

    structure = instance
    employee = structure.employee
    salary = structure.base_salary or 0
    
    # First day of the current month
    month = now().date().replace(day=1)

    # Use global_objects to run safely outside request-thread limits
    Payroll.global_objects.get_or_create(
        employee=employee,
        month=month,
        defaults={
            "organisation": employee.organisation,  # Explicitly assign tenant
            "basic_pay": salary,
            "allowances": structure.hra + structure.allowances,
            "deductions": structure.deductions,
            "calculated_salary": structure.total_salary(),
            "net_salary": structure.total_salary(),
            "status": "processed",
        }
    )