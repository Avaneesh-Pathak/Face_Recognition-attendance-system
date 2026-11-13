from django.core.management.base import BaseCommand
from django.db.models import F
from core.models import Attendance, Payroll, LeaveApplication, JoiningDetail

class Command(BaseCommand):
    help = 'Non-destructive tenant FK integrity audit before and after migrations'
    def handle(self, *args, **kwargs):
        models = [Attendance, Payroll, LeaveApplication, JoiningDetail]
        broken = 0
        for model in models:
            qs = model.global_objects.exclude(organisation_id=F('employee__organisation_id'))
            n = qs.count()
            broken += n
            self.stdout.write(f'{model.__name__}: {n} mismatches')
        if broken:
            raise RuntimeError(f'{broken} cross-organisation records need review; do not launch yet')
        self.stdout.write(self.style.SUCCESS('No tenant FK mismatches in audited tables'))
