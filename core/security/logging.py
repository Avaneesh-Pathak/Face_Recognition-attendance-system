import json
import logging
from datetime import datetime, timezone

class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {'ts': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                   'level': record.levelname, 'logger': record.name,
                   'event': record.getMessage()[:1000]}
        for key in ('request_id', 'status_code', 'organisation_id', 'duration_ms'):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        # Deliberately don't dump exception traceback, request body, password or face embeddings.
        if record.exc_info:
            payload['exception_type'] = record.exc_info[0].__name__
        return json.dumps(payload, default=str, ensure_ascii=True)
