import threading
from django.db import models

# Thread-local storage to securely store the active organisation for the current request
_thread_locals = threading.local()

def get_current_organisation():
    """Returns the Organisation instance bound to the current request thread."""
    return getattr(_thread_locals, 'organisation', None)

def set_current_organisation(org):
    """Binds an Organisation instance to the current request thread."""
    _thread_locals.organisation = org

class TenantQuerySet(models.QuerySet):
    def for_tenant(self):
        org = get_current_organisation()
        if org:
            return self.filter(organisation=org)
        return self

class TenantManager(models.Manager):
    """
    Automatic security manager. 
    Every query (e.g., Department.objects.all()) automatically filters by the current tenant.
    """
    def get_queryset(self):
        return TenantQuerySet(self.model, using=self._db).for_tenant()