
import json
import logging
import numpy as np
from django.core.management.base import BaseCommand
from django.db import transaction

from core.models import Employee

logger = logging.getLogger('core')

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = np.linalg.norm(v)
    if n <= 0:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)

class Command(BaseCommand):
    help = "Sanitize existing face encodings: normalize, convert single vectors to list form, and deduplicate."

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true', help='Do not write changes')
        parser.add_argument('--max-per-employee', type=int, default=24, help='If more than this, keep the most recent N (when possible)')
        parser.add_argument('--force-mean', action='store_true', help='Replace with a single mean vector (still stored as a list with one element)')

    def handle(self, *args, **opts):
        dry = bool(opts['dry_run'])
        kmax = int(opts['max_per_employee'])
        force_mean = bool(opts['force_mean'])

        qs = Employee.objects.exclude(face_encoding__isnull=True).exclude(face_encoding__exact='')
        count = 0
        changed = 0

        for emp in qs:
            count += 1
            raw = emp.face_encoding
            arr = None
            try:
                obj = json.loads(raw)
                arr = np.asarray(obj, dtype=np.float32)
            except Exception:
                # Try comma separated
                try:
                    parts = [float(x) for x in raw.strip('[]() ').split(',') if x.strip()]
                    arr = np.asarray(parts, dtype=np.float32)
                except Exception:
                    arr = None

            if arr is None or arr.size == 0:
                continue

            # normalize & convert to list-of-vectors JSON
            if arr.ndim == 1:
                vecs = [_normalize(arr).tolist()]
            else:
                # dedupe near-identical vectors
                normed = np.vstack([_normalize(v) for v in arr])
                # simple unique by rounding
                rounded = np.round(normed, 4)
                _, unique_idx = np.unique(rounded, axis=0, return_index=True)
                normed = normed[sorted(unique_idx)]
                if normed.shape[0] > kmax:
                    normed = normed[-kmax:]  # keep last N (approx; since order unknown after unique)
                if force_mean:
                    meanv = _normalize(normed.mean(axis=0)).tolist()
                    vecs = [meanv]
                else:
                    vecs = [v.tolist() for v in normed]

            new_json = json.dumps(vecs)
            if new_json != raw:
                changed += 1
                if not dry:
                    try:
                        with transaction.atomic():
                            emp.face_encoding = new_json
                            emp.save(update_fields=['face_encoding'])
                    except Exception:
                        logger.exception("Failed saving cleaned embedding for emp %s", getattr(emp, 'pk', None))

        self.stdout.write(self.style.SUCCESS(f"Processed {count} employees; updated {changed}{' (dry-run)' if dry else ''}"))
