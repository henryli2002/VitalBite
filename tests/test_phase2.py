"""Unit tests for Phase 2: Image registry & migration.

Covers:
- image_store round-trip (save/load/exists/delete)
- placeholder regex and format helpers
- analyze_food_image tool with UUID input (mocked pipeline)
- migration script helpers (idempotency, extraction variants)
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure src/ and project root on path (the latter for `scripts.migrate_images`)
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, ".."))

os.environ.setdefault("USE_SUPERVISOR", "1")


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def images_root(tmp_path, monkeypatch):
    """Point the image store at a temp directory for the duration of the test."""
    root = tmp_path / "images"
    root.mkdir()
    monkeypatch.setenv("WABI_IMAGES_ROOT", str(root))
    # Reload the module so the env var is re-read at module level
    import importlib
    from server import image_store
    importlib.reload(image_store)
    yield root
    # Reload again with the original env so other tests aren't affected
    monkeypatch.delenv("WABI_IMAGES_ROOT", raising=False)
    importlib.reload(image_store)


# =========================================================================
# 1. image_store tests
# =========================================================================


class TestImageStore:
    def test_save_and_load_roundtrip(self, images_root):
        from server import image_store

        payload = b"\xff\xd8\xff\xe0fake-jpeg-body"
        uid = image_store.save_image("user_abc", payload, "image/jpeg")
        assert uid and len(uid) == 32

        bytes_out, mime = image_store.load_image("user_abc", uid)
        assert bytes_out == payload
        assert mime == "image/jpeg"

    def test_save_base64_strips_data_uri(self, images_root):
        from server import image_store

        raw = b"hello-png-bytes"
        b64 = base64.b64encode(raw).decode()
        data_uri = f"data:image/png;base64,{b64}"
        uid = image_store.save_base64("user_abc", data_uri)

        out, mime = image_store.load_image("user_abc", uid)
        assert out == raw
        assert mime == "image/png"

    def test_image_exists_false_for_unknown(self, images_root):
        from server import image_store

        assert image_store.image_exists("user_abc", "deadbeef" * 4) is False

    def test_load_missing_raises(self, images_root):
        from server import image_store

        with pytest.raises(FileNotFoundError):
            image_store.load_image("user_abc", "f" * 32)

    def test_invalid_uuid_returns_none_path(self, images_root):
        from server import image_store

        assert image_store.image_exists("user_abc", "not-a-uuid") is False

    def test_rejects_traversal_user_id(self, images_root):
        from server import image_store

        with pytest.raises(ValueError):
            image_store.save_image("../etc", b"x", "image/png")

    def test_delete_user_images(self, images_root):
        from server import image_store

        image_store.save_image("u1", b"a", "image/jpeg")
        image_store.save_image("u1", b"b", "image/jpeg")
        assert image_store.delete_user_images("u1") == 2
        assert image_store.delete_user_images("u1") == 0


# =========================================================================
# 2. Placeholder helpers
# =========================================================================


class TestPlaceholderHelpers:
    def test_format_without_description(self):
        from server.image_store import format_placeholder

        uid = "a" * 32
        assert format_placeholder(uid) == f"[图片: {uid}]"

    def test_format_with_description(self):
        from server.image_store import format_placeholder

        uid = "a" * 32
        assert format_placeholder(uid, "burger, 500kcal") == f"[图片: {uid} | burger, 500kcal]"

    def test_extract_uuids_plain(self):
        from server.image_store import extract_uuids

        uid = "b" * 32
        text = f"hello\n\n[图片: {uid}]"
        assert extract_uuids(text) == [uid]

    def test_extract_uuids_with_description(self):
        from server.image_store import extract_uuids

        uid = "c" * 32
        text = f"[图片: {uid} | pizza, 300kcal]"
        assert extract_uuids(text) == [uid]

    def test_extract_uuids_multiple(self):
        from server.image_store import extract_uuids

        uid1 = "a" * 32
        uid2 = "b" * 32
        text = f"[图片: {uid1}] then [图片: {uid2} | salad, 200kcal]"
        assert extract_uuids(text) == [uid1, uid2]

    def test_extract_uuids_empty(self):
        from server.image_store import extract_uuids

        assert extract_uuids("nothing here") == []
        assert extract_uuids("") == []


# =========================================================================
# 3. analyze_food_image tool with UUID input
# =========================================================================


class TestAnalyzeFoodImageTool:
    def test_tool_schema_uses_uuid(self):
        from langgraph_app.tools.food_recognition_tool import analyze_food_image

        fields = analyze_food_image.args_schema.model_fields
        assert "image_uuid" in fields
        assert "image_base64" not in fields

    @pytest.mark.asyncio
    async def test_missing_user_id_returns_error(self, images_root):
        from langgraph_app.tools.food_recognition_tool import analyze_food_image

        result = await analyze_food_image.ainvoke({"image_uuid": "a" * 32})
        data = json.loads(result)
        assert "error" in data
        assert "user_id" in data["error"]

    @pytest.mark.asyncio
    async def test_unknown_uuid_returns_error(self, images_root):
        from langgraph_app.tools.food_recognition_tool import analyze_food_image

        cfg = {"configurable": {"user_id": "u_missing"}}
        result = await analyze_food_image.ainvoke(
            {"image_uuid": "a" * 32},
            config=cfg,
        )
        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"]

    @pytest.mark.asyncio
    async def test_successful_flow_calls_pipeline_and_db(self, images_root):
        from server import image_store
        from langgraph_app.tools import food_recognition_tool

        uid = image_store.save_image("u_ok", b"fake-bytes", "image/jpeg")

        fake_result = {
            "items": [{"name": "burger"}, {"name": "fries"}],
            "total_calories": 850.0,
            "total_nutrition": {},
            "nutrition_source": "local_model",
        }

        fake_store = MagicMock()
        fake_store.update_image_description = AsyncMock(return_value=1)

        with patch.object(
            food_recognition_tool, "_run_recognition_pipeline",
            new=AsyncMock(return_value=fake_result),
        ), patch("server.db.PostgresHistoryStore", return_value=fake_store):
            cfg = {"configurable": {"user_id": "u_ok"}}
            result = await food_recognition_tool.analyze_food_image.ainvoke(
                {"image_uuid": uid},
                config=cfg,
            )

        data = json.loads(result)
        assert data["total_calories"] == 850.0
        fake_store.update_image_description.assert_awaited_once()
        args = fake_store.update_image_description.await_args
        assert args.args[0] == "u_ok"
        assert args.args[1] == uid
        desc = args.args[2]
        assert "burger" in desc
        assert "850kcal" in desc


# =========================================================================
# 4. _build_description helper
# =========================================================================


class TestBuildDescription:
    def test_joins_item_names(self):
        from langgraph_app.tools.food_recognition_tool import _build_description

        desc = _build_description({
            "items": [{"name": "汉堡"}, {"name": "薯条"}],
            "total_calories": 850,
        })
        assert "汉堡" in desc
        assert "薯条" in desc
        assert "850kcal" in desc

    def test_empty_items_uses_fallback(self):
        from langgraph_app.tools.food_recognition_tool import _build_description

        desc = _build_description({"items": [], "total_calories": 0})
        assert "meal" in desc
        assert "0kcal" in desc

    def test_caps_to_three_items(self):
        from langgraph_app.tools.food_recognition_tool import _build_description

        desc = _build_description({
            "items": [
                {"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"},
            ],
            "total_calories": 100,
        })
        assert desc.count("+") == 2  # only 3 names joined with 2 plusses


# =========================================================================
# 5. Migration script extraction helpers
# =========================================================================


class TestMigrationHelpers:
    def test_extract_from_multimodal_json(self):
        from scripts.migrate_images import _extract_from_multimodal_json

        raw = b"hello"
        b64 = base64.b64encode(raw).decode()
        payload = json.dumps([
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ])
        result = _extract_from_multimodal_json(payload)
        assert result is not None
        text, mime, body = result
        assert text == "look at this"
        assert mime == "image/png"
        assert body == b64

    def test_extract_from_plain_string_returns_none(self):
        from scripts.migrate_images import _extract_from_multimodal_json

        assert _extract_from_multimodal_json("hello world") is None

    def test_extract_inline_data_uri(self):
        from scripts.migrate_images import _extract_inline_data_uri

        raw = b"payload"
        b64 = base64.b64encode(raw).decode()
        content = f"preface data:image/jpeg;base64,{b64} trailer"
        result = _extract_inline_data_uri(content)
        assert result is not None
        text, mime, body = result
        assert "preface" in text and "trailer" in text
        assert mime == "image/jpeg"
        assert body == b64

    def test_already_migrated_skips(self):
        from scripts.migrate_images import _already_migrated

        uid = "a" * 32
        assert _already_migrated(f"hi\n[图片: {uid}]") is True
        assert _already_migrated("no placeholder here") is False
