"""Context-local tenant; always set explicit organisation in security-sensitive queries."""
from contextvars import ContextVar
from django.db import models
_org = ContextVar('organisation_id', default=None)
_global = ContextVar('global_access', default=False)

def get_current_organisation():
    return _org.get()
def set_current_organisation(org):
    _org.set(org)
def is_global_access():
    return _global.get()
def set_global_access(value):
    _global.set(bool(value))
class TenantQuerySet(models.QuerySet):
    def for_tenant(self):
        if is_global_access():
            return self
        org = get_current_organisation()
        return self.filter(organisation_id=getattr(org, 'pk', org)) if org is not None else self.none()
class TenantManager(models.Manager):
    def get_queryset(self):
        return TenantQuerySet(self.model, using=self._db).for_tenant()
