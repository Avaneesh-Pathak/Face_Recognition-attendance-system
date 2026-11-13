"""Bounded image decoding; avoid decompression bombs and polyglot uploads."""
import io
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from django.conf import settings

Image.MAX_IMAGE_PIXELS = settings.MAX_FACE_IMAGE_PIXELS

def decode_face_upload(upload):
    if upload.size > settings.MAX_FACE_UPLOAD_BYTES or upload.size < 100:
        raise ValueError('Capture must be between 100 bytes and 5 MB')
    data = upload.read(settings.MAX_FACE_UPLOAD_BYTES + 1)
    if len(data) > settings.MAX_FACE_UPLOAD_BYTES:
        raise ValueError('Capture too large')
    try:
        with Image.open(io.BytesIO(data)) as pil:
            if pil.format not in ('JPEG','PNG','WEBP'):
                raise ValueError('Capture must be JPEG, PNG, or WebP')
            if pil.width * pil.height > settings.MAX_FACE_IMAGE_PIXELS or min(pil.size) < 100:
                raise ValueError('Invalid image dimensions')
            pil.verify()
        # OpenCV produces BGR, required by InsightFace.
        buf = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('Image could not be decoded')
        h,w = frame.shape[:2]
        if max(w,h) > 1280:
            scale = 1280 / max(w,h)
            frame = cv2.resize(frame, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
        return frame
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Invalid or unsafe image') from exc


def optimize_face_upload(upload):
    """Validate, strip EXIF/metadata, resize and re-encode uploaded face as WebP."""
    from django.core.files.base import ContentFile
    from pathlib import Path
    import secrets
    if upload.size > settings.MAX_FACE_UPLOAD_BYTES:
        raise ValueError('Uploaded image is too large (max 5 MB)')
    data = upload.read(settings.MAX_FACE_UPLOAD_BYTES + 1)
    if len(data) > settings.MAX_FACE_UPLOAD_BYTES:
        raise ValueError('Uploaded image is too large (max 5 MB)')
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ('JPEG','PNG','WEBP') or im.width * im.height > settings.MAX_FACE_IMAGE_PIXELS:
                raise ValueError('Upload a valid JPEG, PNG or WebP image (max 4 MP)')
            if min(im.size) < 100:
                raise ValueError('Face image dimensions too small')
            im.load()
            im = im.convert('RGB')
            im.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            im.save(output, format='WEBP', quality=82, method=5)
            return ContentFile(output.getvalue(), name=f'face_{secrets.token_hex(12)}.webp')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Invalid or unsafe face image') from exc
