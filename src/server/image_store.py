"""Filesystem-backed image registry.

Images are stored under ``data/images/{user_id}/{uuid}.{ext}``. Image UUIDs
attached to a chat message are tracked in the ``messages.image_refs`` JSONB
column; the text ``content`` column never contains image placeholders.
"""

from __future__ import annotations

import os
import re
import uuid as _uuid
import base64
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("wabi.image_store")

_MIME_TO_EXT = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
}

_EXT_TO_MIME = {v: k.replace("jpg", "jpeg") if k == "image/jpg" else k for k, v in _MIME_TO_EXT.items()}
_EXT_TO_MIME["jpg"] = "image/jpeg"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_IMAGES_ROOT = Path(os.environ.get("WABI_IMAGES_ROOT", _PROJECT_ROOT / "data" / "images"))

_UUID_RE = re.compile(r"^[a-f0-9]{32}$", re.IGNORECASE)


def _safe_user_dir(user_id: str) -> Path:
    if not user_id or "/" in user_id or ".." in user_id:
        raise ValueError(f"invalid user_id: {user_id!r}")
    d = _IMAGES_ROOT / user_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ext_from_mime(mime_type: Optional[str]) -> str:
    if not mime_type:
        return "jpg"
    return _MIME_TO_EXT.get(mime_type.lower().strip(), "jpg")


def save_image(user_id: str, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Persist image bytes and return a short UUID handle."""
    if not image_bytes:
        raise ValueError("image_bytes is empty")
    uid = _uuid.uuid4().hex
    ext = _ext_from_mime(mime_type)
    path = _safe_user_dir(user_id) / f"{uid}.{ext}"
    path.write_bytes(image_bytes)
    logger.info("Saved image %s for user %s (%d bytes)", uid, user_id, len(image_bytes))
    return uid


def save_base64(user_id: str, data_uri_or_b64: str, mime_type: str = "image/jpeg") -> str:
    """Decode a base64 string (with or without data URI prefix) and save."""
    s = data_uri_or_b64
    detected_mime = mime_type
    if s.startswith("data:"):
        try:
            header, _, body = s.partition(",")
            if ";base64" in header:
                detected_mime = header[5:].split(";")[0] or mime_type
            s = body
        except Exception:
            pass
    raw = base64.b64decode(s)
    return save_image(user_id, raw, detected_mime)


def _find_path(user_id: str, image_uuid: str) -> Optional[Path]:
    if not _UUID_RE.match(image_uuid):
        return None
    d = _IMAGES_ROOT / user_id
    if not d.exists():
        return None
    for ext in ("jpg", "jpeg", "png", "webp", "gif"):
        p = d / f"{image_uuid}.{ext}"
        if p.exists():
            return p
    for p in d.glob(f"{image_uuid}.*"):
        return p
    return None


def load_image(user_id: str, image_uuid: str) -> Tuple[bytes, str]:
    """Load image bytes by UUID. Returns (bytes, mime_type)."""
    path = _find_path(user_id, image_uuid)
    if path is None:
        raise FileNotFoundError(f"image {image_uuid} not found for user {user_id}")
    ext = path.suffix.lstrip(".").lower()
    mime = _EXT_TO_MIME.get(ext, "application/octet-stream")
    return path.read_bytes(), mime


def image_exists(user_id: str, image_uuid: str) -> bool:
    return _find_path(user_id, image_uuid) is not None


def delete_user_images(user_id: str) -> int:
    """Remove all images for a user. Returns count deleted."""
    d = _IMAGES_ROOT / user_id
    if not d.exists():
        return 0
    count = 0
    for p in d.iterdir():
        if p.is_file():
            p.unlink()
            count += 1
    try:
        d.rmdir()
    except OSError:
        pass
    return count
