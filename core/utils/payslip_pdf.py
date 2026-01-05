from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
import calendar

def format_currency(amount):
    """Helper to format currency beautifully"""
    if amount is None:
        return "0.00"
    return f"{float(amount):,.2f}"

def generate_payslip_pdf(payroll):
    buffer = BytesIO()
    
    # margins: top, bottom, left, right
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                            rightMargin=15*mm, leftMargin=15*mm, 
                            topMargin=15*mm, bottomMargin=15*mm)
    
    elements = []
    styles = getSampleStyleSheet()
    
    # --- Custom Styles ---
    styles.add(ParagraphStyle(name='CenterTitle', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=16, spaceAfter=10))
    styles.add(ParagraphStyle(name='RightBold', parent=styles['Normal'], alignment=TA_RIGHT, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='LeftBold', parent=styles['Normal'], alignment=TA_LEFT, fontName='Helvetica-Bold'))
    
    emp = payroll.employee
    user = emp.user

    # Colors
    HEADER_BG = colors.Color(0.17, 0.24, 0.31) # Dark Blue/Grey
    HEADER_TEXT = colors.whitesmoke
    ROW_BG = colors.whitesmoke

    # =====================================================
    # 1. HEADER SECTION
    # =====================================================
    # Placeholder for Company Name - You can replace text with an Image() if you have a logo
    company_name = "NELSON HOSPITAL" 
    
    header_data = [
        [Paragraph(f"<b>{company_name}</b>", styles['Heading2']), Paragraph("<b>PAYSLIP</b>", styles['Heading2'])]
    ]
    
    header_table = Table(header_data, colWidths=[100*mm, 80*mm])
    header_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 5*mm))
    
    # Pay Period Subtitle
    period_str = payroll.month.strftime('%B %Y')
    elements.append(Paragraph(f"Payslip for the period of <b>{period_str}</b>", styles['Normal']))
    elements.append(Spacer(1, 5*mm))

    # =====================================================
    # 2. EMPLOYEE DETAILS (Grid Layout)
    # =====================================================
    # Format: Label | Value | Label | Value
    emp_data = [
        ["Employee ID", emp.employee_id, "Role", emp.get_role_display()],
        ["Name", user.get_full_name(), "Department", emp.department.name if emp.department else "-"],
        [
            "Date of Joining",
            emp.date_of_joining.strftime("%d-%m-%Y") if emp.date_of_joining else "-",
            "Contact",
            emp.phone_number or "-"
        ],
    ]
    
    emp_table = Table(emp_data, colWidths=[35*mm, 55*mm, 35*mm, 55*mm])
    emp_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), # First col bold
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'), # Third col bold
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(emp_table)
    elements.append(Spacer(1, 8*mm))

    # =====================================================
    # 3. ATTENDANCE SUMMARY (Modern Strip)
    # =====================================================
    # Calculate days
    total_days = calendar.monthrange(payroll.month.year, payroll.month.month)[1]
    working_days = (payroll.present_days + payroll.absent_days + payroll.paid_leave_days + payroll.unpaid_leave_days)
    
    att_data = [[
        f"Total Days: {total_days}",
        f"Present: {payroll.present_days}",
        f"Paid Leave: {payroll.paid_leave_days}",
        f"Absent: {payroll.absent_days}",
        f"LOP: {payroll.unpaid_leave_days}"
    ]]
    
    att_table = Table(att_data, colWidths=[36*mm]*5)
    att_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 0), (-1, -1), ROW_BG),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(Paragraph("Attendance Details", styles['LeftBold']))
    elements.append(Spacer(1, 2*mm))
    elements.append(att_table)
    elements.append(Spacer(1, 8*mm))

    # =====================================================
    # 4. FINANCIALS (Earnings vs Deductions)
    # =====================================================
    
    # Prepare Data
    earnings_list = [
        ("Basic Salary", format_currency(payroll.basic_pay)),
        ("HRA & Allowances", format_currency(payroll.allowances)),
        (f"Overtime ({payroll.overtime_hours} hrs)", getattr(payroll, 'overtime_amount', '0.00')),
        # Add Bonus etc here if exists
    ]
    
    deductions_list = [
        ("Provident Fund", "0.00"), # Example place holders, replace with actual logic
        ("Professional Tax", "0.00"),
        ("TDS / Income Tax", "0.00"),
        ("Other Deductions", format_currency(payroll.deductions)),
    ]

    # Normalize lengths (make lists same size for the table)
    max_rows = max(len(earnings_list), len(deductions_list))
    while len(earnings_list) < max_rows: earnings_list.append(("", ""))
    while len(deductions_list) < max_rows: deductions_list.append(("", ""))

    # Build Table Data
    # Header Row
    fin_data = [[
        Paragraph("<b>EARNINGS</b>", styles['Normal']), 
        Paragraph("<b>AMOUNT</b>", styles['RightBold']),
        Paragraph("<b>DEDUCTIONS</b>", styles['Normal']), 
        Paragraph("<b>AMOUNT</b>", styles['RightBold'])
    ]]

    # Content Rows
    for i in range(max_rows):
        fin_data.append([
            earnings_list[i][0],
            earnings_list[i][1],
            deductions_list[i][0],
            deductions_list[i][1]
        ])

    # Footer Row (Totals)
    # Note: You might need to sum these up dynamically or pull from payroll object
    total_earnings = float(payroll.basic_pay) + float(payroll.allowances) # simplified
    total_deductions = float(payroll.deductions)
    
    fin_data.append([
        "Total Earnings", format_currency(total_earnings),
        "Total Deductions", format_currency(total_deductions)
    ])

    fin_table = Table(fin_data, colWidths=[65*mm, 25*mm, 65*mm, 25*mm])
    
    fin_table.setStyle(TableStyle([
        # Header Style
        ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), HEADER_TEXT),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'), # Align amounts right
        ('ALIGN', (3, 0), (3, -1), 'RIGHT'), # Align amounts right
        
        # Grid
        ('GRID', (0, 0), (-1, -2), 0.5, colors.lightgrey),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        
        # Totals Row Style
        ('fontName', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -1), (-1, -1), ROW_BG),
        ('LINEABOVE', (0, -1), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    
    elements.append(fin_table)
    elements.append(Spacer(1, 5*mm))

    # =====================================================
    # 5. NET PAY SECTION
    # =====================================================
    
    net_pay_val = payroll.actual_paid_salary if payroll.actual_paid_salary else payroll.net_salary
    
    net_data = [[
        "Net Pay (Rounded)", 
        f"Rs. {format_currency(net_pay_val)}"
    ]]
    
    net_table = Table(net_data, colWidths=[130*mm, 50*mm])
    net_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.Color(0.9, 0.95, 1)), # Very light blue
        ('BOX', (0, 0), (-1, -1), 1, colors.darkblue),
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (1, 0), (1, 0), 14), # Big font for money
        ('PADDING', (0, 0), (-1, -1), 12),
    ]))
    
    elements.append(net_table)
    elements.append(Spacer(1, 15*mm))

    # =====================================================
    # 6. FOOTER / SIGNATURES
    # =====================================================
    
    footer_data = [
        ["________________________", "________________________"],
        ["Employee Signature", "Authorized Signatory"]
    ]
    
    foot_table = Table(footer_data, colWidths=[90*mm, 90*mm])
    foot_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    
    elements.append(foot_table)
    elements.append(Spacer(1, 10*mm))
    
    elements.append(Paragraph("<i>This is a system generated payslip.</i>", styles['Normal']))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    
    filename = f"payslip_{emp.employee_id}_{payroll.month.strftime('%Y_%m')}.pdf"
    return buffer, filename