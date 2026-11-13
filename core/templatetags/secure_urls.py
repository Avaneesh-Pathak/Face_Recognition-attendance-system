from django import template
from django.urls import reverse

register = template.Library()



@register.simple_tag
def secure_url(viewname, obj=None, *extra_args, **kwargs):

    args = []

    if obj is not None:

        if hasattr(obj, "secure_id"):
            args.append(obj.secure_id)
        else:
            args.append(obj)

    args.extend(extra_args)

    return reverse(viewname, args=args, kwargs=kwargs)