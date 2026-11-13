"""Read-only payroll integrity audit. Run for each organisation before releasing payslips."""
from decimal import Decimal
from django.core.management.base import BaseCommand, CommandError
from core.models import Organisation, Payroll

class Command(BaseCommand):
    help = 'Read-only audit of net-pay arithmetic and legacy overtime snapshots for one organisation'
    def add_arguments(self, parser):
        parser.add_argument('--organisation', type=int, required=True)
        parser.add_argument('--year', type=int, required=True)
        parser.add_argument('--month', type=int, required=True)
    def handle(self, *args, **opt):
        if not 1 <= opt['month'] <= 12 or not 2000 <= opt['year'] <= 2100:
            raise CommandError('Invalid payroll period')
        if not Organisation.objects.filter(pk=opt['organisation']).exists():
            raise CommandError('Organisation not found')
        records = Payroll.global_objects.filter(organisation_id=opt['organisation'],
                   month__year=opt['year'], month__month=opt['month'])
        issues = 0
        for row in records.iterator():
            expected = (row.basic_pay + row.allowances + row.regular_overtime_pay +
                        row.holiday_overtime_pay - row.deductions)
            reasons = []
            if row.net_salary != max(Decimal('0'),expected):
                reasons.append('net amount mismatch')
            if (row.overtime_hours > 0 and row.regular_overtime_pay == 0) or (
                row.holiday_overtime_hours > 0 and row.holiday_overtime_pay == 0):
                reasons.append('legacy/unsnapshotted overtime')
            if reasons:
                issues += 1
                self.stdout.write(f'payroll={row.pk} employee_pk={row.employee_id} '
                                  f'status={row.status} reasons={",".join(reasons)}')
        self.stdout.write(f'Audited={records.count()}, flagged={issues}. No changes made.')
