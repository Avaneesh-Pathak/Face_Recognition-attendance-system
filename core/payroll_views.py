"""Locked payroll mutations. Derived attendance counts cannot be hand edited."""
from datetime import date
from decimal import Decimal, InvalidOperation
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST
from core.models import Payroll, Employee


def allowed(request):
    employee = getattr(request.user, 'employee', None)
    return request.user.is_superuser or bool(employee and employee.role in ('Admin', 'Finance'))


def locked_payroll(pk):
    record = get_object_or_404(Payroll.objects, pk=pk)
    Employee.global_objects.select_for_update().get(pk=record.employee_id)
    return get_object_or_404(Payroll.objects.select_for_update(), pk=pk)


@login_required
def pay_salary(request, pk):
    if not allowed(request):
        return HttpResponseForbidden('Finance or Admin role required')
    if request.method != 'POST':
        return render(request, 'ems/pay_salary.html', {
            'payroll': get_object_or_404(Payroll.objects, pk=pk), 'today': timezone.localdate().isoformat()})
    with transaction.atomic():
        payroll = locked_payroll(pk)
        if payroll.status == 'paid':
            messages.info(request, 'This payroll is already paid; no changes made.')
            return redirect('payroll_list')
        if payroll.status != 'processed' or payroll.month >= timezone.localdate().replace(day=1):
            messages.error(request, 'Reconcile attendance and regenerate a closed month before payment.')
            return redirect('pay_salary', pk=pk)
        try:
            paid = date.fromisoformat(request.POST.get('paid_date') or timezone.localdate().isoformat())
            if paid > timezone.localdate() or paid < payroll.month:
                raise ValueError('Payment date must be within or after the payroll period and not in the future')
            amount = Decimal(request.POST.get('amount_paid') or str(payroll.net_salary))
            if not amount.is_finite() or amount != payroll.net_salary:
                raise ValueError('Payment must exactly match the calculated net salary')
        except (ValueError, InvalidOperation) as exc:
            messages.error(request, str(exc))
            return redirect('pay_salary', pk=pk)
        payroll.status = 'paid'
        payroll.paid_date = paid
        payroll.save(update_fields=['status', 'paid_date'])
    messages.success(request, 'Salary marked paid. Payment and amount are locked in audit history.')
    return redirect('payroll_list')


@login_required
@require_POST
def payroll_update(request, pk):
    if not allowed(request):
        return HttpResponseForbidden('Finance or Admin role required')
    payroll = get_object_or_404(Payroll.objects, pk=pk)
    messages.error(request, 'Payroll is attendance-derived. Correct attendance or salary setup, then regenerate. Paid payroll is locked.')
    return redirect('employee_salary_history', payroll.employee_id)


@login_required
@require_POST
def payroll_delete(request, pk):
    if not allowed(request):
        return HttpResponseForbidden('Finance or Admin role required')
    with transaction.atomic():
        payroll = locked_payroll(pk)
        employee_id = payroll.employee_id
        if payroll.status == 'paid':
            messages.error(request, 'Paid payroll cannot be deleted.')
        else:
            payroll.delete()
            messages.success(request, 'Unpaid payroll removed; deletion recorded in audit history.')
    return redirect('employee_salary_history', employee_id)
