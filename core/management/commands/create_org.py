from django.core.management.base import BaseCommand
from core.models import Organisation

class Command(BaseCommand):
    help = 'Create an organisation (superuser CLI only); does not create default admin credentials.'
    def add_arguments(self, parser):
        parser.add_argument('name')
    def handle(self, *args, **options):
        org, created = Organisation.objects.get_or_create(name=options['name'])
        self.stdout.write(f"Organisation id={org.pk} created={created}")
