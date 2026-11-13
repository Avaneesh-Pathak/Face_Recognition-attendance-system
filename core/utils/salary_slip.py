from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
import os
from django.conf import settings

def generate_salary_slip(payroll):
    """
    ✅ Generate a PDF salary slip for the given payroll.
    Returns the file path to the generated PDF.
    """
    folder_path = os.path.join(settings.MEDIA_ROOT, "salary_slips")
    os.makedirs(folder_path, exist_ok=True)

    filename = f"SalarySlip_{payroll.employee.employee_id}_{payroll.month.strftime('%Y_%m')}.pdf"
    file_path = os.path.join(folder_path, filename)

    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4

    # Company + Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, height - 50, "Salary Slip")
    c.setFont("Helvetica", 12)
    c.drawString(30, height - 75, f"Employee: {payroll.employee.user.get_full_name()}")
    c.drawString(30, height - 95, f"Employee ID: {payroll.employee.employee_id}")
    c.drawString(30, height - 115, f"Salary Month: {payroll.month.strftime('%B %Y')}")

    # Salary Breakdown
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, height - 150, "Salary Details")
    c.setFont("Helvetica", 12)
    c.drawString(30, height - 175, f"Basic Salary: ₹{payroll.basic_pay}")
    c.drawString(30, height - 195, f"Allowances: ₹{payroll.allowances}")
    c.drawString(30, height - 215, f"Deductions: ₹{payroll.deductions}")
    
    c.setFont("Helvetica-Bold", 13)
    c.drawString(30, height - 245, f"Net Salary: ₹{payroll.net_salary}")

    c.line(30, height - 260, width - 30, height - 260)

    c.drawString(30, height - 285, f"Payment Status: {payroll.status.capitalize()}")
    c.drawString(30, height - 305, f"Paid On: {payroll.paid_date or 'Not Paid'}")

    c.showPage()
    c.save()
    return file_path
