"""Migrate legacy inline-base64 image messages to the filesystem registry.

Scans the `messages` table for rows whose content still embeds
``data:image/...;base64,...`` (either as a raw string or as a serialized
multimodal JSON list). For each match:

1. Extracts the base64 body and writes the decoded bytes to
   ``data/images/{user_id}/{uuid}.{ext}``.
2. Stores ``{"uuid": ..., "description": ""}`` in the row's ``image_refs``
   JSONB column.
3. Strips the base64 payload from the ``content`` text column.

The script is idempotent — rows whose ``image_refs`` already contains an
entry, or whose content carries no base64, are skipped. Run it once per
database during the Phase 2 rollout.

Usage:
    python scripts/migrate_images.py           # dry run
    python scripts/migrate_images.py --apply   # actually write
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from typing import Optional, Tuple

# Ensure src is importable when running as a script
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from server.image_store import save_base64  # noqa: E402

logger = logging.getLogger("wabi.migrate_images")

DB_URL = os.environ.get(
    "WABI_DB_URL", "postgresql://wabi_user:wabi_password@localhost:5432/wabi_chat"
)

# Matches `data:image/xxx;base64,AAAA...` up to the closing quote/paren/bracket.
_DATA_URI_RE = re.compile(
    r"data:(image/[a-zA-Z0-9+.\-]+);base64,([A-Za-z0-9+/=]+)",
    re.DOTALL,
)


def _extract_from_multimodal_json(content: str) -> Optional[Tuple[str, str, str]]:
    """If content is a JSON list like [{"type":"text",...},{"type":"image_url",...}],
    return (text, mime_type, base64_body). Otherwise return None.
    """
    s = content.strip()
    if not s.startswith("["):
        return None
    try:
        parts = json.loads(s)
    except Exception:
        return None
    if not isinstance(parts, list):
        return None

    text = ""
    mime = "image/jpeg"
    b64 = None
    for p in parts:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            text = p.get("text", "") or text
        elif p.get("type") == "image_url":
            url = (p.get("image_url") or {}).get("url", "")
            m = _DATA_URI_RE.search(url)
            if m:
                mime = m.group(1)
                b64 = m.group(2)
    if b64 is None:
        return None
    return text, mime, b64


def _extract_inline_data_uri(content: str) -> Optional[Tuple[str, str, str]]:
    """If content contains a raw ``data:image/...;base64,...`` URI, extract it."""
    m = _DATA_URI_RE.search(content)
    if not m:
        return None
    mime = m.group(1)
    b64 = m.group(2)
    # Strip the data URI (and surrounding Markdown/HTML if present) from the text
    text = content[: m.start()] + content[m.end() :]
    text = re.sub(r"!\[[^\]]*\]\(\s*\)", "", text)  # empty markdown image
    text = re.sub(r"<img[^>]*>", "", text, flags=re.IGNORECASE)
    text = text.strip()
    return text, mime, b64


def _already_migrated(image_refs) -> bool:
    """A row whose image_refs array already has an entry was handled before."""
    if image_refs is None:
        return False
    if isinstance(image_refs, str):
        try:
            image_refs = json.loads(image_refs)
        except Exception:
            return False
    return bool(image_refs)


async def migrate(apply: bool) -> None:
    import asyncpg  # Lazy — keep module importable without asyncpg for unit tests
    pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=4)
    assert pool is not None
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, user_id, content, image_refs FROM messages "
                "WHERE content LIKE '%data:image/%;base64,%' "
                "ORDER BY id ASC"
            )
            logger.info("Found %d candidate rows", len(rows))

            migrated = 0
            skipped = 0
            failed = 0

            for row in rows:
                msg_id = row["id"]
                user_id = row["user_id"]
                content = row["content"] or ""

                if _already_migrated(row["image_refs"]):
                    skipped += 1
                    continue

                extracted = _extract_from_multimodal_json(content) or _extract_inline_data_uri(content)
                if extracted is None:
                    skipped += 1
                    continue
                text, mime, b64 = extracted

                try:
                    if apply:
                        uid = save_base64(user_id, b64, mime)
                    else:
                        uid = "<dry-run>"
                except Exception as e:
                    logger.warning("Row %s: save failed: %s", msg_id, e)
                    failed += 1
                    continue

                refs = [{"uuid": uid, "description": ""}]

                if apply:
                    await conn.execute(
                        "UPDATE messages SET content = $1, image_refs = $2::jsonb, has_image = 1 WHERE id = $3",
                        text,
                        json.dumps(refs),
                        msg_id,
                    )
                migrated += 1
                logger.info(
                    "Row %s (user=%s): %s bytes -> image_refs[0].uuid=%s",
                    msg_id,
                    user_id,
                    len(b64),
                    uid if apply else "(dry-run)",
                )

            logger.info(
                "Done. migrated=%d skipped=%d failed=%d apply=%s",
                migrated,
                skipped,
                failed,
                apply,
            )
    finally:
        await pool.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually write changes (default is dry-run)")
    args = parser.parse_args()
    asyncio.run(migrate(apply=args.apply))


if __name__ == "__main__":
    main()
