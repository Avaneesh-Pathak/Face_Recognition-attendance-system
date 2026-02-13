from io import BytesIO
from decimal import Decimal, ROUND_HALF_UP
import calendar

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle,
    Paragraph, Spacer, Flowable
)
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas


# =====================================================
# CONFIGURATION & THEME
# =====================================================

THEME_COLOR = colors.Color(0.12, 0.20, 0.35)
ACCENT_COLOR = colors.Color(0.96, 0.96, 0.96)
TEXT_BODY = colors.Color(0.1, 0.1, 0.1)


# =====================================================
# UTILITIES
# =====================================================

def money(val):
    try:
        return Decimal(val).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except:
        return Decimal("0.00")


def format_currency(val):
    return f"{money(val):,}"


def number_to_words(amount):
    try:
        from num2words import num2words
        amt = money(amount)
        rupees = int(amt)
        paise = int((amt - rupees) * 100)

        words = num2words(rupees, lang="en_IN").title() + " Rupees"
        if paise:
            words += f" And {num2words(paise).title()} Paise"
        return words + " Only"
    except:
        return f"{int(amount)} Rupees Only"


class HorizontalLine(Flowable):
    def __init__(self, thickness=0.5, color=colors.lightgrey):
        super().__init__()
        self.thickness = thickness
        self.color = color

    def wrap(self, w, h):
        return w, self.thickness

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)


def draw_watermark(c, doc):
    c.saveState()
    c.setFillColor(colors.lightgrey)
    c.setFont("Helvetica-Bold", 60)
    try:
        c.setFillAlpha(0.12)
    except:
        pass
    c.translate(A4[0] / 2, A4[1] / 2)
    c.rotate(45)
    c.drawCentredString(0, 0, "CONFIDENTIAL")
    c.restoreState()


# =====================================================
# MAIN PDF GENERATOR
# =====================================================

def generate_payslip_pdf(payroll):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=10 * mm,
        leftMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
        title=f"Payslip_{payroll.month.strftime('%b_%Y')}"
    )

    styles = getSampleStyleSheet()
    elements = []

    # -------------------------------------------------
    # STYLES
    # -------------------------------------------------

    styles.add(ParagraphStyle("CompTitle", fontSize=20, fontName="Helvetica-Bold", textColor=THEME_COLOR))
    styles.add(ParagraphStyle(name="HospitalHeading",fontName="Helvetica-Bold",fontSize=18,textColor=colors.black,spaceAfter=4))
    styles.add(ParagraphStyle("CompSub", fontSize=8, textColor=colors.grey))
    styles.add(ParagraphStyle("PayslipTitle", fontSize=12, textColor=colors.white, alignment=TA_CENTER))
    styles.add(ParagraphStyle("Label", fontSize=8, textColor=colors.grey))
    styles.add(ParagraphStyle("Value", fontSize=9, fontName="Helvetica-Bold", textColor=THEME_COLOR))
    styles.add(ParagraphStyle("Th", fontSize=9, fontName="Helvetica-Bold", textColor=colors.white))
    styles.add(ParagraphStyle("Td", fontSize=9, textColor=TEXT_BODY))
    styles.add(ParagraphStyle("TdMoney", fontSize=9, alignment=TA_RIGHT))
    styles.add(ParagraphStyle("TdMoneyBold", fontSize=9, fontName="Helvetica-Bold", alignment=TA_RIGHT))
    styles.add(
        ParagraphStyle(
            name="CompSubRight",
            parent=styles["CompSub"],
            alignment=TA_RIGHT
        )
    )
    emp = payroll.employee
    user = emp.user

    # -------------------------------------------------
    # HEADER
    # -------------------------------------------------

    logo_box = Table(
        [["NH"]],
        colWidths=16 * mm,
        rowHeights=16 * mm,
        style=[
            ('BACKGROUND', (0, 0), (-1, -1), THEME_COLOR),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 13),
        ]
    )

    company_text = Table(
        [
            [Paragraph("NELSON HOSPITAL", styles["HospitalHeading"])],
            [Paragraph(
                "Reg. No: RMEE2227209<br/>"
                "B1/37, Sector F, Kapoorthala,<br/>"
                "Lucknow – 226024",
                styles["CompSubRight"]   # ✅ right aligned
            )]
        ],
        style=[
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]
    )


    header = Table(
        [[logo_box, company_text]],
        colWidths=[22 * mm, 148 * mm],
        style=[
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]
    )

    elements.append(header)
    elements.append(Spacer(1, 4 * mm))


    # -------------------------------------------------
    # TITLE BAR
    # -------------------------------------------------

    elements.append(Table(
        [[Paragraph(f"PAYSLIP FOR {payroll.month.strftime('%B %Y').upper()}", styles["PayslipTitle"])]],
        colWidths=[190 * mm],
        style=[('BACKGROUND', (0, 0), (-1, -1), THEME_COLOR),
               ('PADDING', (0, 0), (-1, -1), 6)]
    ))
    elements.append(Spacer(1, 6 * mm))

    # -------------------------------------------------
    # EMPLOYEE INFO
    # -------------------------------------------------

    def info(l1, v1, l2, v2):
        return [
            Paragraph(l1, styles["Label"]), Paragraph(str(v1), styles["Value"]),
            Paragraph(l2, styles["Label"]), Paragraph(str(v2), styles["Value"])
        ]

    emp_table = Table([
        info("Name", user.get_full_name().title(), "Employee ID", emp.employee_id),
        info("Department", emp.department.name if emp.department else "-", "Designation", emp.get_role_display()),
        info("Monthly Salary", f"Rs. {format_currency(payroll.basic_pay)}", "Salary Type", "Monthly"),
        info("Date of Joining",
             emp.date_of_joining.strftime("%d-%b-%Y") if emp.date_of_joining else "-",
             "Bank Account", getattr(emp, "bank_account_number", "-"))
    ], colWidths=[30 * mm, 65 * mm, 30 * mm, 65 * mm])

    emp_table.setStyle([
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 5)
    ])

    elements.append(emp_table)
    elements.append(Spacer(1, 8 * mm))

    # -------------------------------------------------
    # ATTENDANCE
    # -------------------------------------------------

    total_days = calendar.monthrange(payroll.month.year, payroll.month.month)[1]
    paid_days = min(payroll.present_days or 0, total_days)
    lop_days = max(total_days - paid_days, 0)

    def metric(lbl, val):
        return Table([[Paragraph(lbl, styles["Label"])],
                      [Paragraph(str(val), styles["Value"])]],
                     colWidths=40 * mm,
                     style=[('BOX', (0, 0), (-1, -1), 0.5, colors.grey),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER')])

    elements.append(Table([[
        metric("TOTAL DAYS", total_days),
        metric("PAID DAYS", paid_days),
        metric("LOSS OF PAY", lop_days),
        metric("OVERTIME (HRS)", payroll.overtime_hours or 0)
    ]], colWidths=[47.5 * mm] * 4))

    elements.append(Spacer(1, 8 * mm))

    # -------------------------------------------------
    # PAY CALCULATIONS
    # -------------------------------------------------

    daily_pay = money(payroll.basic_pay) / total_days if total_days else money(0)
    paid_basic = money(daily_pay * paid_days)

    ot_rate = getattr(getattr(emp, "work_rule", None), "overtime_rate", 0)
    ot_amt = money((payroll.overtime_hours or 0) * ot_rate)

    allowances = money(payroll.allowances)
    deductions = money(payroll.deductions)

    total_earn = paid_basic + allowances + ot_amt
    total_ded = deductions

    net_val = (
        money(getattr(payroll, "actual_paid_salary", 0))
        or money(getattr(payroll, "net_salary", 0))
        or (total_earn - total_ded)
    )

    # -------------------------------------------------
    # EARNINGS & DEDUCTIONS TABLE
    # -------------------------------------------------

    data = [[
        Paragraph("EARNINGS", styles["Th"]), Paragraph("AMOUNT", styles["Th"]),
        Paragraph("DEDUCTIONS", styles["Th"]), Paragraph("AMOUNT", styles["Th"])
    ]]

    rows = [
        ("Basic Salary", paid_basic),
        ("House Rent Allowance", allowances),
        ("Overtime Pay", ot_amt),
        ("Special Allowance", 0)
    ]

    ded_rows = [
        ("Provident Fund", 0),
        ("Professional Tax", 0),
        ("Income Tax (TDS)", 0),
        ("Other Deductions", deductions)
    ]

    for e, d in zip(rows, ded_rows):
        data.append([
            Paragraph(e[0], styles["Td"]), Paragraph(format_currency(e[1]), styles["TdMoney"]),
            Paragraph(d[0], styles["Td"]), Paragraph(format_currency(d[1]), styles["TdMoney"])
        ])

    data.append([
        Paragraph("TOTAL EARNINGS", styles["TdMoneyBold"]),
        Paragraph(format_currency(total_earn), styles["TdMoneyBold"]),
        Paragraph("TOTAL DEDUCTIONS", styles["TdMoneyBold"]),
        Paragraph(format_currency(total_ded), styles["TdMoneyBold"])
    ])

    elements.append(Table(data, colWidths=[70 * mm, 25 * mm, 70 * mm, 25 * mm],
                          style=[
                              ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                              ('BACKGROUND', (0, 0), (-1, 0), THEME_COLOR),
                              ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, ACCENT_COLOR]),
                              ('BACKGROUND', (0, -1), (-1, -1), colors.whitesmoke)
                          ]))

    elements.append(Spacer(1, 6 * mm))

    # -------------------------------------------------
    # NET PAY
    # -------------------------------------------------

    elements.append(Table([
        [
            Paragraph("NET PAYABLE AMOUNT", styles["Label"]),
            Paragraph(f"Rs. {format_currency(net_val)}",
                      ParagraphStyle("Net", fontSize=16, fontName="Helvetica-Bold", alignment=TA_RIGHT))
        ],
        [Paragraph(f"In Words: <i>{number_to_words(net_val)}</i>", styles["Label"]), ""]
    ], colWidths=[100 * mm, 90 * mm],
        style=[
            ('BOX', (0, 0), (-1, -1), 1, THEME_COLOR),
            ('SPAN', (0, 1), (1, 1)),
            ('PADDING', (0, 0), (-1, -1), 8)
        ]))

    elements.append(Spacer(1, 15 * mm))

    # -------------------------------------------------
    # SIGNATURES
    # -------------------------------------------------

    elements.append(Table([
        ["_______________________", "_______________________"],
        ["Employee Signature", "Authorized Signatory"]
    ], colWidths=[95 * mm, 95 * mm],
        style=[('ALIGN', (1, 0), (1, -1), 'RIGHT')]))

    elements.append(Spacer(1, 8 * mm))
    elements.append(HorizontalLine())

    elements.append(Paragraph(
        "This is a computer-generated document and does not require a signature.",
        ParagraphStyle("Footer", fontSize=7, alignment=TA_CENTER, textColor=colors.grey)
    ))

    doc.build(elements, onFirstPage=draw_watermark, onLaterPages=draw_watermark)

    buffer.seek(0)
    filename = f"payslip_{emp.employee_id}_{payroll.month.strftime('%Y_%m')}.pdf"
    return buffer, filename
