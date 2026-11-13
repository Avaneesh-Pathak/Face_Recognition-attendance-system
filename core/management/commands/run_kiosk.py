from django.core.management.base import BaseCommand, CommandError

class Command(BaseCommand):
    help = 'Legacy local kiosk disabled for tenant isolation; use browser attendance terminal.'
    def handle(self, *args, **kwargs):
        raise CommandError('Legacy kiosk used unscoped recognition. Use the protected browser scanner per organisation. A dedicated kiosk agent requires a separately audited device-authentication flow.')
