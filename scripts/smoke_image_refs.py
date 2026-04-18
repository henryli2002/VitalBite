"""Smoke test for the image_refs refactor.

Steps:
1. Create a fresh test user via REST.
2. Open a WebSocket and send a user message whose prose contains
   `[image: aaaa...]`. Verify the server stores it as plain text (no image
   reference) and the AI responds without error.
3. Hit GET /history and verify the user bubble's `image_refs` is empty even
   though the content still literally contains the `[image:` string.
4. Upload a real tiny PNG. Verify the server stores it in image_refs, the
   history returns the UUID, and GET /api/images/... serves the bytes.
5. Clean up the test user.

Run via:
    python scripts/smoke_image_refs.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys

import httpx
import websockets

BASE = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"


# 1x1 red PNG, 67 bytes
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)


async def _first_final(ws):
    """Drain WS messages until we see the first 'message' (final assistant reply)."""
    async for raw in ws:
        data = json.loads(raw)
        t = data.get("type")
        if t == "message":
            return data
        if t == "error":
            return data


async def main() -> int:
    async with httpx.AsyncClient(base_url=BASE, timeout=60) as client:
        # 1. Create user
        r = await client.post("/api/users", json={"name": "smoke_image_refs"})
        r.raise_for_status()
        user = r.json()
        uid = user["user_id"]
        print(f"[ok] created user {uid}")

        try:
            # 2. Open WS
            async with websockets.connect(f"{WS_BASE}/ws/{uid}") as ws:
                # --- CASE A: fake placeholder in user prose ---
                fake = "a" * 32
                probe_text = f"hello check [image: {fake}] please ignore"
                print(f"[.] sending text with fake placeholder...")
                await ws.send(json.dumps({
                    "type": "message",
                    "content": probe_text,
                }))
                reply = await _first_final(ws)
                if reply.get("type") == "error":
                    print(f"[FAIL] backend error: {reply.get('content')}")
                    return 1
                print(f"[ok] assistant replied ({len(reply.get('content') or '')} chars)")

                # --- CASE B: real image upload ---
                print("[.] sending real image...")
                await ws.send(json.dumps({
                    "type": "image",
                    "content": TINY_PNG_B64,
                    "text": "what is this pixel",
                    "mime_type": "image/png",
                }))
                reply2 = await _first_final(ws)
                if reply2.get("type") == "error":
                    print(f"[FAIL] image flow backend error: {reply2.get('content')}")
                    return 1
                print(f"[ok] image flow replied ({len(reply2.get('content') or '')} chars)")

            # 3. GET /history and inspect
            r = await client.get(f"/api/users/{uid}/history")
            r.raise_for_status()
            history = r.json()
            print(f"[.] history has {len(history)} messages")

            user_msgs = [m for m in history if m["role"] == "user"]
            if len(user_msgs) < 2:
                print(f"[FAIL] expected >=2 user msgs, got {len(user_msgs)}")
                return 1

            # First user message: fake placeholder should be PRESERVED as text,
            # image_refs MUST be empty (no spoofing through prose).
            probe_msg = user_msgs[0]
            if "[image:" not in probe_msg["content"]:
                print(f"[FAIL] probe text missing from content: {probe_msg['content']!r}")
                return 1
            if probe_msg.get("image_refs"):
                print(f"[FAIL] fake placeholder triggered image_refs: {probe_msg['image_refs']}")
                return 1
            print("[ok] fake '[image: ...]' in prose did NOT produce image_refs entry")

            # Second user message: real upload should produce exactly one image_ref
            real_msg = user_msgs[1]
            refs = real_msg.get("image_refs") or []
            if len(refs) != 1:
                print(f"[FAIL] real upload expected 1 image_ref, got {len(refs)}")
                return 1
            real_uuid = refs[0]["uuid"]
            print(f"[ok] real upload stored as image_refs[0].uuid={real_uuid}")

            # 4. GET the image bytes
            r = await client.get(f"/api/images/{uid}/{real_uuid}")
            if r.status_code != 200:
                print(f"[FAIL] /api/images returned {r.status_code}")
                return 1
            if len(r.content) < 50:
                print(f"[FAIL] served image too small: {len(r.content)} bytes")
                return 1
            print(f"[ok] /api/images served {len(r.content)} bytes")

            # Sanity: confirm the fake uuid does NOT resolve to an image
            r = await client.get(f"/api/images/{uid}/{fake}")
            if r.status_code != 404:
                print(f"[FAIL] fake uuid should 404, got {r.status_code}")
                return 1
            print(f"[ok] fake uuid correctly 404s")

        finally:
            # 5. Clean up
            r = await client.delete(f"/api/users/{uid}")
            print(f"[ok] cleaned up user {uid} (status={r.status_code})")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
