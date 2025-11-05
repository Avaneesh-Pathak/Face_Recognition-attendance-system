# core/utils_pdf.py
from io import BytesIO
from decimal import Decimal
from datetime import date, timedelta

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit

from django.utils import timezone

# Import Models
from core.models import SalaryStructure, Attendance, LeaveApplication, PayrollSettings, Holiday

# ============================================================
# Utilities
# ============================================================
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 2.0 * cm, 2.0 * cm, 2.0 * cm, 1.8 * cm
CONTENT_W = A4[0] - (MARGIN_L + MARGIN_R)


def _month_bounds(d: date):
    first = d.replace(day=1)
    if first.month == 12:
        next_first = first.replace(year=first.year + 1, month=1)
    else:
        next_first = first.replace(month=first.month + 1)
    last = next_first - timedelta(days=1)
    return first, last


def _iter_days(start: date, end: date):
    while start <= end:
        yield start
        start += timedelta(days=1)


def _is_weekday(d: date):
    return d.weekday() < 5  # Monday–Friday


def _approved_leave_workdays(employee, first_day: date, last_day: date):
    leaves = LeaveApplication.objects.filter(
        employee=employee, status="approved",
        end_date__gte=first_day, start_date__lte=last_day
    ).values("start_date", "end_date")

    count = 0
    for lv in leaves:
        s, e = max(lv["start_date"], first_day), min(lv["end_date"], last_day)
        for d in _iter_days(s, e):
            if _is_weekday(d):
                count += 1
    return count


def _present_workdays(employee, first_day: date, last_day: date):
    qs = Attendance.objects.filter(
        employee=employee, attendance_type="check_in",
        timestamp__date__gte=first_day, timestamp__date__lte=last_day
    ).values_list("timestamp__date", flat=True).distinct()
    return sum(1 for d in qs if _is_weekday(d))


# ============================================================
# Drawing Utilities
# ============================================================
def fmt_money(val):
    return f"₹ {Decimal(val or 0):,.2f}"


def safe_str(val):
    return "-" if not val else str(val)


def draw_title_bar(p, x, y, text, width, color):
    """Draws a colored header bar for a section and resets text color to black."""
    bar_h = 0.8 * cm
    p.setFillColor(color)
    p.rect(x, y - bar_h + 1, width, bar_h, stroke=0, fill=1)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 11)
    p.drawString(x + 0.35 * cm, y - 0.55 * cm, text)
    # ✅ Reset color for upcoming text
    p.setFillColor(colors.black)
    return y - bar_h - 0.3 * cm


def draw_kv_rows(p, x_label, x_val, y, rows, line_h=0.45 * cm):
    p.setFont("Helvetica", 9)
    for label, value in rows:
        p.drawString(x_label, y, f"{label}:")
        wrapped = simpleSplit(safe_str(value), "Helvetica", 9, CONTENT_W / 2)
        for i, line in enumerate(wrapped):
            p.drawString(x_val, y - (i * line_h * 0.8), line)
        y -= max(line_h, (len(wrapped) * line_h * 0.8))
    return y


def draw_amount_two_cols(p, y, left_items, right_items):
    """Draw earnings and deductions side-by-side."""
    left_x_label, left_x_amt = MARGIN_L, MARGIN_L + 7.2 * cm
    right_x_label, right_x_amt = MARGIN_L + CONTENT_W * 0.55, MARGIN_L + CONTENT_W - 0.5 * cm
    rows = max(len(left_items), len(right_items))
    p.setFont("Helvetica", 9)
    for i in range(rows):
        if i < len(left_items):
            label, amt = left_items[i]
            p.drawString(left_x_label, y, label)
            p.drawRightString(left_x_amt, y, fmt_money(amt))
        if i < len(right_items):
            label, amt = right_items[i]
            p.drawString(right_x_label, y, label)
            p.drawRightString(right_x_amt, y, fmt_money(amt))
        y -= 0.42 * cm
    return y


# ============================================================
# Number to Words (Indian)
# ============================================================
def number_to_words(number):
    if number == 0:
        return "Zero"
    ones = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    teens = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
             "Seventeen", "Eighteen", "Nineteen"]
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]

    def lt1000(n):
        if n == 0:
            return ""
        if n < 10:
            return ones[n]
        if n < 20:
            return teens[n - 10]
        if n < 100:
            return tens[n // 10] + (" " + ones[n % 10] if n % 10 else "")
        return ones[n // 100] + " Hundred" + (" " + lt1000(n % 100) if n % 100 else "")

    if number < 1000:
        return lt1000(number)
    if number < 100000:
        return lt1000(number // 1000) + " Thousand " + lt1000(number % 1000)
    if number < 10000000:
        return lt1000(number // 100000) + " Lakh " + number_to_words(number % 100000)
    return number_to_words(number // 10000000) + " Crore " + number_to_words(number % 10000000)


# ============================================================
# PDF Generator
# ============================================================
def build_salary_slip_pdf(payroll):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # --- Colors ---
    PRIMARY = colors.HexColor("#2c3e50")
    SECONDARY = colors.HexColor("#34495e")
    ACCENT = colors.HexColor("#27ae60")
    LIGHT = colors.HexColor("#bdc3c7")

    emp = payroll.employee
    first_day, last_day = _month_bounds(payroll.month)

    # --- Salary Data ---
    basic = Decimal(payroll.basic_pay or 0)
    allowances = Decimal(payroll.allowances or 0)
    deductions = Decimal(payroll.deductions or 0)
    net = Decimal(payroll.net_salary or 0)

    try:
        ss = SalaryStructure.objects.get(employee=emp)
        hra = Decimal(ss.hra or (basic * Decimal("0.40")))
        other_allow = Decimal(ss.allowances or 0)
        ss_ded = Decimal(ss.deductions or 0)
    except SalaryStructure.DoesNotExist:
        hra = basic * Decimal("0.40")
        other_allow = ss_ded = Decimal(0)

    pf = (basic * Decimal("0.12")).quantize(Decimal("0.01"))
    settings = PayrollSettings.objects.first()
    prof_tax = settings.professional_tax if settings else Decimal("200.00")

    # --- Attendance ---
    working_days = sum(1 for d in _iter_days(first_day, last_day) if _is_weekday(d))
    present = _present_workdays(emp, first_day, last_day)
    leaves = _approved_leave_workdays(emp, first_day, last_day)
    total_present = present + leaves
    absent = max(0, working_days - total_present)
    holidays = Holiday.objects.filter(date__gte=first_day, date__lte=last_day).count()
    weekends = sum(1 for d in _iter_days(first_day, last_day) if not _is_weekday(d))

    # --- Header ---
    p.setFillColor(PRIMARY)
    p.rect(0, height - 1.8 * cm, width, 1.8 * cm, stroke=0, fill=1)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(MARGIN_L, height - 1.1 * cm, "TECHNOVATE SOLUTIONS LTD.")
    p.setFont("Helvetica", 9)
    p.drawString(MARGIN_L, height - 1.5 * cm, "123 Tech Park, Sector 62, Noida • +91-120-4567890")
    p.setFont("Helvetica-Bold", 14)
    p.drawRightString(width - MARGIN_R, height - 1.1 * cm, "Salary Slip")
    p.setFont("Helvetica", 9)
    p.drawRightString(width - MARGIN_R, height - 1.5 * cm, f"{payroll.month.strftime('%B %Y')} • {payroll.status.upper()}")

    y = height - 2.5 * cm

    # --- Employee Details ---
    y = draw_title_bar(p, MARGIN_L, y, "Employee Details", CONTENT_W, SECONDARY)
    emp_rows = [
        ("Employee Name", emp.user.get_full_name() or emp.user.username),
        ("Employee ID", safe_str(emp.employee_id)),
        ("Designation", safe_str(emp.position)),
        ("Department", emp.department.get_name_display() if getattr(emp, "department", None) else "-"),
        ("Date of Joining", emp.date_of_joining.strftime("%d-%m-%Y") if emp.date_of_joining else "-"),
        ("Pay Month", payroll.month.strftime("%B %Y")),
    ]
    y = draw_kv_rows(p, MARGIN_L + 0.3 * cm, MARGIN_L + 5.5 * cm, y, emp_rows)
    y -= 0.3 * cm

    # --- Salary Breakdown ---
    y = draw_title_bar(p, MARGIN_L, y, "Salary Breakdown", CONTENT_W, SECONDARY)
    earnings = [("Basic Salary", basic), ("HRA", hra), ("Other Allowances", other_allow)]
    if allowances:
        earnings.append(("Variable Allowances", allowances))

    deductions_list = [("PF (12%)", pf), ("Professional Tax", prof_tax)]
    if ss_ded:
        deductions_list.append(("Structured Deductions", ss_ded))
    if deductions:
        deductions_list.append(("Other Deductions / LOP", deductions))

    y = draw_amount_two_cols(p, y, earnings, deductions_list)

    total_earn = sum(a for _, a in earnings)
    total_ded = sum(a for _, a in deductions_list)
    calc_net = total_earn - total_ded

    p.setFont("Helvetica-Bold", 10)
    y -= 0.3 * cm
    p.drawString(MARGIN_L, y, "Total Earnings")
    p.drawRightString(MARGIN_L + 7.2 * cm, y, fmt_money(total_earn))
    p.drawString(MARGIN_L + CONTENT_W * 0.55, y, "Total Deductions")
    p.drawRightString(MARGIN_L + CONTENT_W - 0.5 * cm, y, fmt_money(total_ded))
    y -= 0.6 * cm

    # --- Net Salary ---
    p.setFillColor(ACCENT)
    p.setFont("Helvetica-Bold", 12)
    p.drawString(MARGIN_L, y, "NET SALARY PAYABLE")
    p.setFillColor(colors.black)
    p.drawRightString(MARGIN_L + 7.2 * cm, y, fmt_money(net))
    y -= 0.5 * cm

    # if calc_net != net:
    #     p.setFillColor(colors.red)
    #     p.setFont("Helvetica-Oblique", 8)
    #     p.drawString(MARGIN_L, y, f"⚠ Salary prorated (Calculated: {fmt_money(calc_net)}, Paid: {fmt_money(net)})")
    #     p.setFillColor(colors.black)
    #     y -= 0.3 * cm

    p.setFont("Helvetica", 9)
    p.drawString(MARGIN_L, y, f"In Words: {number_to_words(int(net))} Rupees Only")
    y -= 0.7 * cm

    # --- Attendance Summary ---
    y = draw_title_bar(p, MARGIN_L, y, "Attendance Summary", CONTENT_W, SECONDARY)
    p.setFont("Helvetica", 9)
    attendance = [
        ("Calendar Days", (last_day - first_day).days + 1),
        ("Working Days", working_days),
        ("Present Days", present),
        ("Approved Leaves", leaves),
        ("Absent Days", absent),
        ("Weekends", weekends),
        ("Holidays", holidays),
        ("Counted Present", total_present),
    ]
    y = draw_kv_rows(p, MARGIN_L + 0.3 * cm, MARGIN_L + 6.5 * cm, y, attendance)
    y -= 0.5 * cm

    # --- Footer ---
    p.setFont("Helvetica", 9)
    p.drawString(MARGIN_L, y, "Employee Signature")
    p.drawRightString(width - MARGIN_R, y, "Authorized Signatory")
    y -= 0.5 * cm
    p.setFont("Helvetica-Oblique", 8)
    p.setFillColor(LIGHT)
    p.drawCentredString(width / 2, y, "This is a computer generated document and does not require a physical signature.")
    p.setFillColor(colors.black)

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

