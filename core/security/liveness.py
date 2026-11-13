"""Server-side PAD adapter contract; client claims are never accepted as proof."""
from django.conf import settings
from django.utils.module_loading import import_string


class LivenessUnavailable(RuntimeError):
    pass


def verify_liveness(frame, request):
    if not settings.FACE_LIVENESS_VERIFIER:
        raise LivenessUnavailable('No verified PAD provider configured')
    try:
        verifier = import_string(settings.FACE_LIVENESS_VERIFIER)
        # Provider must bind its proof to this frame, user session and organisation,
        # validate freshness/replay resistance, and enforce its own bounded timeout.
        return verifier(frame=frame, request=request, organisation=request.organisation) is True
    except Exception as exc:
        raise LivenessUnavailable('PAD provider unavailable') from exc
