"""Tenant-separated and bounded InsightFace matching.

Does NOT establish presentation-attack resistance. A verified PAD/liveness provider
is required before unattended real-world biometric attendance is enabled.
"""
import json
import hashlib
import logging
import os
import threading
import time
import queue
from django.core.cache import cache
from collections import OrderedDict, deque
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from .models import Employee

logger = logging.getLogger('face_match')
_instance = None
_instance_lock = threading.Lock()


def normalize(value):
    if value is None:
        return None
    vec = np.asarray(value, dtype=np.float32).ravel()
    if not vec.size or not np.isfinite(vec).all():
        return None
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-8 else None


class BusyInference(RuntimeError):
    pass


class ModelUnavailable(RuntimeError):
    pass


class FaceRecognitionSystem:
    def __init__(self):
        self._model_pool = queue.LifoQueue(maxsize=settings.FACE_SCAN_INFERENCE_SLOTS)
        for _ in range(settings.FACE_SCAN_INFERENCE_SLOTS):
            self._model_pool.put(None)
        self._cache_lock = threading.RLock()
        self._tenant_cache = OrderedDict()
        self._votes = {}

    def _build_model(self):
        try:
            from insightface.app import FaceAnalysis
            root = settings.INSIGHTFACE_MODEL_ROOT
            weights = Path(root) / 'models' / settings.INSIGHTFACE_MODEL
            if not weights.is_dir() or not list(weights.glob('*.onnx')):
                raise ModelUnavailable('Face model weights are not provisioned')
            model = FaceAnalysis(name=settings.INSIGHTFACE_MODEL, root=root,
                                 providers=settings.FACE_EXECUTION_PROVIDERS,
                                 allowed_modules=['detection', 'recognition'])
            model.prepare(ctx_id=0 if 'CUDAExecutionProvider' in settings.FACE_EXECUTION_PROVIDERS else -1,
                          det_size=settings.INSIGHTFACE_DET_SIZE)
            return model
        except ModelUnavailable:
            raise
        except Exception as exc:
            logger.error('face_model_initialization_failed')
            raise ModelUnavailable('Face model unavailable') from exc

    def _infer(self, frame):
        try:
            model = self._model_pool.get(timeout=settings.FACE_INFERENCE_QUEUE_TIMEOUT)
        except queue.Empty as exc:
            raise BusyInference('Inference busy') from exc
        try:
            if model is None:
                model = self._build_model()
            # Each slot owns an independent model. No global inference mutex.
            return model.get(frame)
        finally:
            self._model_pool.put_nowait(model)

    def get_embedding(self, image_input):
        if isinstance(image_input, (str, os.PathLike)):
            image_input = cv2.imread(str(image_input))
        if image_input is None:
            return None
        faces = self._infer(image_input)
        if len(faces) != 1:
            return None
        face = faces[0]
        if float(getattr(face, 'det_score', 0)) < settings.FACE_DETECTION_MIN_SCORE:
            return None
        emb = normalize(face.embedding)
        return emb.tolist() if emb is not None else None

    def invalidate_tenant(self, organisation_id):
        with self._cache_lock:
            self._tenant_cache.pop(organisation_id, None)

    def load_from_db(self, organisation_id, force=False):
        if not isinstance(organisation_id, int) or organisation_id <= 0:
            raise ValueError('Explicit tenant ID required')
        now = time.monotonic()
        try:
            version = cache.get(f'face-gallery-version:{organisation_id}')
        except Exception as exc:
            raise BusyInference('Gallery state unavailable') from exc
        with self._cache_lock:
            cached = self._tenant_cache.get(organisation_id)
            if not force and cached is not None and cached[0] > now and cached[3] == version:
                self._tenant_cache.move_to_end(organisation_id)
                return cached[1], cached[2]
        vectors, ids = [], []
        # Explicit org restriction, regardless of thread context.
        qs = (Employee.global_objects.filter(organisation_id=organisation_id,
                                              is_active=True, employment_status='active',
                                              user__is_active=True, organisation__is_active=True)
              .exclude(face_encoding__isnull=True).exclude(face_encoding='')
              .only('id', 'employee_id', 'face_encoding'))
        for employee in qs.iterator(chunk_size=500):
            try:
                data = json.loads(employee.face_encoding)
                if isinstance(data, dict):
                    continue
                if data and isinstance(data[0], (int, float)):
                    data = [data]
                for item in data[:12]:
                    vec = normalize(item)
                    if vec is not None and vec.size == 512:
                        vectors.append(vec)
                        ids.append(employee.id)  # Internal PK instead of shared employee code.
            except (ValueError, TypeError, KeyError):
                logger.warning('invalid_enrollment_template', extra={'organisation_id': organisation_id})
        gallery = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32) if vectors else np.empty((0, 512), dtype=np.float32)
        owners = np.asarray(ids, dtype=np.int64)
        with self._cache_lock:
            self._tenant_cache[organisation_id] = (now + settings.FACE_MATCH_CACHE_TTL, gallery, owners, version)
            self._tenant_cache.move_to_end(organisation_id)
            while len(self._tenant_cache) > settings.FACE_MATCH_CACHE_MAX_ORGS:
                self._tenant_cache.popitem(last=False)
            return gallery, owners

    def identify_face(self, embedding, organisation_id):
        vector = normalize(embedding)
        if vector is None or vector.size != 512:
            return None, 0.0
        gallery, owners = self.load_from_db(organisation_id)
        if not gallery.size:
            return None, 0.0
        similarities = gallery @ vector
        ix = int(np.argmax(similarities))
        winner = int(owners[ix])
        best = float(similarities[ix])
        # Distinct second person (not another photo of the same person).
        different = similarities[owners != winner]
        runner_up = float(different.max()) if different.size else -1.0
        if best < settings.FACE_RECOGNITION_THRESHOLD or best - runner_up < settings.FACE_AMBIGUITY_MARGIN:
            return None, best
        return winner, best

    def _reset_votes(self, organisation_id, session_id):
        if not session_id:
            return
        digest = hashlib.sha256(f'{organisation_id}:{session_id}'.encode()).hexdigest()
        key = f'face-votes:{organisation_id}:{digest}'
        if settings.REDIS_URL:
            from django_redis import get_redis_connection
            try:
                get_redis_connection('default').delete(key)
            except Exception as exc:
                raise BusyInference('Scan state unavailable') from exc
        else:
            with self._cache_lock:
                self._votes.pop(key, None)

    def recognize_from_frame(self, frame, organisation_id=None, session_id=None):
        if organisation_id is None:
            raise ValueError('Tenant must be supplied explicitly')
        if frame is None or not hasattr(frame, 'shape'):
            self._reset_votes(organisation_id, session_id)
            return None, 0.0, {'reason': 'INVALID_FRAME'}
        h, w = frame.shape[:2]
        if max(w, h) > 1280:
            factor = 1280 / max(w, h)
            frame = cv2.resize(frame, (int(w*factor), int(h*factor)), interpolation=cv2.INTER_AREA)
        faces = self._infer(frame)
        if not faces:
            self._reset_votes(organisation_id, session_id)
            return None, 0.0, {'reason': 'NO_FACE'}
        if len(faces) != 1:
            self._reset_votes(organisation_id, session_id)
            return None, 0.0, {'reason': 'MULTIPLE_FACES'}
        face = faces[0]
        bbox = list(map(int, face.bbox[:4]))
        if bbox[2] - bbox[0] < 90 or float(getattr(face, 'det_score', 0)) < settings.FACE_DETECTION_MIN_SCORE:
            self._reset_votes(organisation_id, session_id)
            return None, 0.0, {'reason': 'TOO_FAR'}
        employee_pk, score = self.identify_face(face.embedding, organisation_id)
        if employee_pk is None:
            self._reset_votes(organisation_id, session_id)
            gallery, _ = self.load_from_db(organisation_id)
            if not gallery.size:
                return None, score, {'reason': 'NO_ENROLLED_FACES'}
            return None, score, {'reason': 'UNKNOWN'}
        # The three positive frames must be in *this scan session*; never accumulate
        # confirmations globally across kiosk devices or organisations.
        if session_id is None:
            return None, score, {'reason': 'SESSION_REQUIRED'}
        # Redis Lua ensures the 3-of-5 vote gate works across Gunicorn workers.
        # Hash the scan session so session IDs never become cache keys in plaintext.
        digest = hashlib.sha256(f'{organisation_id}:{session_id}'.encode()).hexdigest()
        key = f'face-votes:{organisation_id}:{digest}'
        if settings.REDIS_URL:
            from django_redis import get_redis_connection
            lua = """
            local previous = redis.call('LINDEX', KEYS[1], 0)
            if previous and previous ~= ARGV[1] then redis.call('DEL', KEYS[1]) end
            redis.call('LPUSH', KEYS[1], ARGV[1])
            redis.call('LTRIM', KEYS[1], 0, tonumber(ARGV[2])-1)
            redis.call('EXPIRE', KEYS[1], 15)
            local values = redis.call('LRANGE', KEYS[1], 0, -1)
            local matches = 0
            for i=1,#values do
                if values[i] == ARGV[1] then matches = matches + 1 end
            end
            if matches >= tonumber(ARGV[3]) then
                redis.call('DEL', KEYS[1])
                return 1
            end
            return 0
            """
            try:
                confirmed = bool(get_redis_connection('default').eval(
                    lua, 1, key, str(employee_pk), settings.FACE_VOTE_WINDOW,
                    settings.FACE_STABLE_VOTES))
            except Exception as exc:
                logger.error('face_vote_redis_unavailable')
                raise BusyInference('Scan state unavailable') from exc
        else:
            # Single-process, DEBUG-only fallback; never use this in production.
            now = time.monotonic()
            with self._cache_lock:
                self._votes = {k: v for k, v in self._votes.items() if now - v[0] < 15}
                previous = self._votes.get(key)
                votes = previous[1] if previous else deque(maxlen=settings.FACE_VOTE_WINDOW)
                if votes and votes[-1] != employee_pk:
                    votes.clear()
                votes.append(employee_pk)
                self._votes[key] = (now, votes)
                confirmed = sum(1 for x in votes if x == employee_pk) >= settings.FACE_STABLE_VOTES
                if confirmed:
                    self._votes.pop(key, None)
        if not confirmed:
            return None, score, {'reason': 'ACCUMULATING'}
        return employee_pk, score, {'bbox': bbox}


def get_face_system():
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = FaceRecognitionSystem()
    return _instance
