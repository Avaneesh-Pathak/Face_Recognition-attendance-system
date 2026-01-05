from django.core.mail import EmailMessage
from django.conf import settings

def email_payslip(payroll):
    emp = payroll.employee
    user = emp.user

    if not user.email:
        return False

    subject = f"Salary Slip – {payroll.month.strftime('%B %Y')}"
    body = f"""
Dear {user.get_full_name()},

Please find attached your salary slip for {payroll.month.strftime('%B %Y')}.

Net Salary: ₹ {payroll.net_salary}

Regards,
Accounts Department
"""

    email = EmailMessage(
        subject=subject,
        body=body,
        from_email=settings.DEFAULT_FROM_EMAIL,
        to=[user.email],
    )

    if payroll.payslip_pdf:
        email.attach_file(payroll.payslip_pdf.path)

    email.send(fail_silently=False)
    return True
