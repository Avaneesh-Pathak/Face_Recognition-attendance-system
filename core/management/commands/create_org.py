from django.core.management.base import BaseCommand
from core.models import Organization


class Command(BaseCommand):
    help = 'Create a default organization for the system'

    def handle(self, *args, **options):
        # Check if organization already exists
        if Organization.objects.exists():
            org = Organization.objects.first()
            self.stdout.write(self.style.SUCCESS(f'Organization already exists: {org}'))
            return

        # Create default organization
        org = Organization.objects.create(
            name="Default Organization",
            slug="default-org",
            email="admin@defaultorg.com",
            description="Default organization for the system"
        )
        
        self.stdout.write(self.style.SUCCESS(f'Successfully created organization: {org}'))
