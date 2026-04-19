"""Unit tests for Phase 1: Supervisor Agent migration.

Tests cover:
- agent_utils shared functions
- Tool definitions and interfaces
- Supervisor graph creation
- State compatibility
- chat_manager simplification
- ai.py thinking partial builder
"""

import sys
import os
import json
import base64
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("USE_SUPERVISOR", "1")


# =========================================================================
# 1. agent_utils tests
# =========================================================================


class TestBuildProfileContext:
    def test_none_profile(self):
        from langgraph_app.utils.agent_utils import build_profile_context

        assert build_profile_context(None) == ""

    def test_empty_profile(self):
        from langgraph_app.utils.agent_utils import build_profile_context

        assert build_profile_context({}) == ""

    def test_profile_with_none_values_excluded(self):
        from langgraph_app.utils.agent_utils import build_profile_context

        ctx = build_profile_context({"age": 25, "weight_kg": None, "gender": "male"})
        assert "Age: 25" in ctx
        assert "Gender" in ctx
        assert "Weight" not in ctx

    def test_full_profile(self):
        from langgraph_app.utils.agent_utils import build_profile_context

        profile = {
            "age": 30,
            "height_cm": 175,
            "weight_kg": 70,
            "gender": "male",
            "fitness_goals": "lose weight",
        }
        ctx = build_profile_context(profile)
        assert "User Profile & Health Information" in ctx
        assert "Age: 30" in ctx
        assert "Height Cm: 175" in ctx


class TestCalculateTdee:
    def test_none_profile(self):
        from langgraph_app.utils.agent_utils import calculate_tdee

        assert calculate_tdee(None) is None

    def test_missing_fields(self):
        from langgraph_app.utils.agent_utils import calculate_tdee

        assert calculate_tdee({"age": 25}) is None

    def test_male(self):
        from langgraph_app.utils.agent_utils import calculate_tdee

        tdee = calculate_tdee(
            {"age": 25, "weight_kg": 70, "height_cm": 175, "gender": "male"}
        )
        assert tdee is not None
        assert 1800 < tdee < 2400

    def test_female(self):
        from langgraph_app.utils.agent_utils import calculate_tdee

        tdee = calculate_tdee(
            {"age": 25, "weight_kg": 55, "height_cm": 160, "gender": "female"}
        )
        assert tdee is not None
        assert 1300 < tdee < 1900

    def test_active_goals_increase_tdee(self):
        from langgraph_app.utils.agent_utils import calculate_tdee

        base = calculate_tdee(
            {"age": 25, "weight_kg": 70, "height_cm": 175, "gender": "male"}
        )
        active = calculate_tdee(
            {
                "age": 25,
                "weight_kg": 70,
                "height_cm": 175,
                "gender": "male",
                "fitness_goals": "high intensity training",
            }
        )
        assert active > base


class TestDetectMealTime:
    def test_returns_valid_string(self):
        from langgraph_app.utils.agent_utils import detect_meal_time

        result = detect_meal_time("Asia/Shanghai")
        assert result in (
            "breakfast time",
            "lunch time",
            "dinner time",
            "not meal time",
        )

    def test_none_timezone_uses_default(self):
        from langgraph_app.utils.agent_utils import detect_meal_time

        result = detect_meal_time(None)
        assert isinstance(result, str)

    def test_invalid_timezone_uses_fallback(self):
        from langgraph_app.utils.agent_utils import detect_meal_time

        result = detect_meal_time("Invalid/Timezone")
        assert isinstance(result, str)


class TestBuildDailyCalRef:
    def test_with_valid_profile(self):
        from langgraph_app.utils.agent_utils import build_daily_cal_ref

        ref = build_daily_cal_ref(
            {"age": 25, "weight_kg": 70, "height_cm": 175, "gender": "male"}
        )
        assert "kcal" in ref
        assert "estimated" in ref.lower()

    def test_without_profile(self):
        from langgraph_app.utils.agent_utils import build_daily_cal_ref

        ref = build_daily_cal_ref(None)
        assert "2000" in ref


# =========================================================================
# 2. predictor decode_base64_image test
# =========================================================================


class TestDecodeBase64Image:
    def test_decodes_correctly(self):
        from langgraph_app.agents.food_recognition.predictor import decode_base64_image

        original = b"test image bytes"
        encoded = base64.b64encode(original).decode()
        result = decode_base64_image(encoded)
        assert result == original


# =========================================================================
# 3. Tool interface tests
# =========================================================================


class TestAnalyzeFoodImageTool:
    def test_tool_metadata(self):
        from langgraph_app.tools.food_recognition_tool import analyze_food_image

        assert analyze_food_image.name == "analyze_food_image"
        # Phase 2: tool takes a UUID handle instead of inline base64
        assert "image_uuid" in analyze_food_image.args_schema.model_fields

    @pytest.mark.asyncio
    async def test_returns_error_on_missing_user_id(self):
        from langgraph_app.tools.food_recognition_tool import analyze_food_image

        result = await analyze_food_image.ainvoke({"image_uuid": "a" * 32})
        data = json.loads(result)
        assert "error" in data


class TestSearchRestaurantsTool:
    def test_tool_metadata(self):
        from langgraph_app.tools.recommendation_tool import search_restaurants

        assert search_restaurants.name == "search_restaurants"
        schema_fields = search_restaurants.args_schema.model_fields
        assert "query" in schema_fields
        assert "cuisine_type" in schema_fields
        assert "lat" in schema_fields
        assert "lng" in schema_fields


class TestSupervisorToolsList:
    def test_all_tools_exported(self):
        from langgraph_app.tools import supervisor_tools

        names = [t.name for t in supervisor_tools]
        assert "analyze_food_image" in names
        assert "search_restaurants" in names

    @pytest.mark.asyncio
    async def test_recommendation_tool_guidance_preserves_full_page(self):
        from langgraph_app.tools import recommendation_tool

        fake_batch = {
            "restaurants": [
                {"name": "A", "address": "Addr A", "rating": 4.1, "types": ["restaurant"]},
                {"name": "B", "address": "Addr B", "rating": 4.2, "types": ["restaurant"]},
                {"name": "C", "address": "Addr C", "rating": 4.3, "types": ["restaurant"]},
                {"name": "D", "address": "Addr D", "rating": 4.4, "types": ["restaurant"]},
                {"name": "E", "address": "Addr E", "rating": 4.5, "types": ["restaurant"]},
            ],
            "next_page_token": None,
        }

        with patch.object(recommendation_tool, "_load_state", AsyncMock(return_value={"results": [], "next_token": None, "exhausted": False})), patch.object(
            recommendation_tool, "_save_state", AsyncMock()
        ), patch.object(
            recommendation_tool.map_tool, "search_restaurants", AsyncMock(return_value=fake_batch)
        ):
            raw = await recommendation_tool.search_restaurants.ainvoke(
                {"query": "restaurants", "page": 1},
                config={"configurable": {"user_id": "u1", "user_context": {}}},
            )

        data = json.loads(raw)
        guidance = data["display_guidance"]
        assert len(data["restaurants"]) == 5
        assert "Output every restaurant returned for the current page exactly once" in guidance
        assert "Do NOT shrink it" in guidance


# =========================================================================
# 4. Supervisor module tests
# =========================================================================


class TestSupervisorPromptBuilder:
    def test_build_system_prompt_minimal(self):
        from langgraph_app.orchestrator.supervisor import _build_system_prompt

        state = {"messages": [], "user_profile": None, "user_context": {}}
        result = _build_system_prompt(state)
        assert len(result) == 1
        assert "WABI" in result[0].content

    def test_build_system_prompt_with_profile(self):
        from langgraph_app.orchestrator.supervisor import _build_system_prompt

        state = {
            "messages": [],
            "user_profile": {
                "age": 25,
                "weight_kg": 70,
                "height_cm": 175,
                "gender": "male",
                "behavioral_notes": "tends to skip breakfast",
            },
            "user_context": {"timezone": "Asia/Shanghai"},
        }
        result = _build_system_prompt(state)
        content = result[0].content
        assert "Age: 25" in content
        assert "tends to skip breakfast" in content

    def test_prompt_contains_tool_rules(self):
        from langgraph_app.orchestrator.supervisor import _build_system_prompt

        state = {"messages": [], "user_profile": None, "user_context": {}}
        content = _build_system_prompt(state)[0].content
        assert "analyze_food_image" in content
        assert "search_restaurants" in content
        assert "must contain every restaurant returned for the current page exactly once" in content


# =========================================================================
# 5. Graph creation tests
# =========================================================================


class TestGraphCreation:
    def test_supervisor_graph_compiles(self):
        from langgraph_app.orchestrator.graph import _create_supervisor_graph

        g = _create_supervisor_graph()
        assert g is not None

    def test_legacy_graph_compiles(self):
        from langgraph_app.orchestrator.graph import _create_legacy_graph

        g = _create_legacy_graph()
        assert g is not None

    def test_feature_flag_selects_correct_graph(self):
        from langgraph_app.orchestrator.graph import USE_SUPERVISOR, graph

        assert USE_SUPERVISOR is True
        assert graph is not None


# =========================================================================
# 6. SupervisorState tests
# =========================================================================


class TestSupervisorState:
    def test_state_has_required_fields(self):
        from langgraph_app.orchestrator.supervisor_state import SupervisorState

        annotations = SupervisorState.__annotations__
        assert "messages" in annotations
        assert "user_id" in annotations
        assert "user_profile" in annotations
        assert "user_context" in annotations
        # response_channel is intentionally NOT on state — it lives on
        # RunnableConfig.configurable as a publish_thinking callable.
        assert "response_channel" not in annotations
        assert "analysis" in annotations
        assert "debug_logs" in annotations

    def test_no_legacy_fields(self):
        from langgraph_app.orchestrator.supervisor_state import SupervisorState

        annotations = SupervisorState.__annotations__
        assert "recognition_result" not in annotations
        assert "recommendation_result" not in annotations
        assert "meal_time" not in annotations


# =========================================================================
# 7. ai.py build_thinking_partial tests
# =========================================================================


class TestBuildThinkingPartial:
    def test_supervisor_with_tool_calls(self):
        from server.ai import build_thinking_partial

        mock_msg = MagicMock()
        mock_msg.tool_calls = [{"name": "analyze_food_image"}]
        result = build_thinking_partial("supervisor", {"messages": [mock_msg]})
        assert result is not None
        assert result["node"] == "tool_call"
        assert "analyze_food_image" in result["analysis"]["reasoning"]

    def test_supervisor_without_tool_calls(self):
        from server.ai import build_thinking_partial

        mock_msg = MagicMock()
        mock_msg.tool_calls = None
        result = build_thinking_partial("supervisor", {"messages": [mock_msg]})
        assert result is None

    def test_tools_node_with_content(self):
        from server.ai import build_thinking_partial

        mock_msg = MagicMock()
        mock_msg.content = '{"items": [{"name": "burger"}]}'
        result = build_thinking_partial("tools", {"messages": [mock_msg]})
        assert result is not None
        assert result["node"] == "tool_result"

    def test_legacy_router_still_works(self):
        from server.ai import build_thinking_partial

        result = build_thinking_partial(
            "router", {"analysis": {"intent": "recognition", "confidence": 0.95}}
        )
        assert result is not None
        assert result["node"] == "intent_router"

    def test_unknown_node_returns_none(self):
        from server.ai import build_thinking_partial

        result = build_thinking_partial("unknown_node", {})
        assert result is None

    def test_supervisor_streaming_builds_tool_specific_partials(self):
        from langgraph_app.orchestrator.thinking import build_thinking_partials

        mock_msg = MagicMock()
        mock_msg.tool_calls = [
            {"name": "analyze_food_image"},
            {"name": "search_restaurants"},
        ]
        result = build_thinking_partials("agent", {"messages": [mock_msg]})
        assert [item["node"] for item in result] == [
            "tool_call_analyze_food_image",
            "tool_call_search_restaurants",
        ]
        assert result[0]["analysis"]["language"] == "Chinese"
        assert result[0]["analysis"]["title"] == "查看图片"
        assert "我先看看图里有哪些食物" in result[0]["analysis"]["reasoning"]
        assert result[1]["analysis"]["title"] == "搜索餐厅"
        assert "我在根据当前位置和你的需求找更合适的餐厅" in result[1]["analysis"]["reasoning"]

    def test_tool_result_uses_structured_summary(self):
        from langgraph_app.orchestrator.thinking import build_thinking_partials

        mock_msg = MagicMock()
        mock_msg.name = "search_restaurants"
        mock_msg.content = json.dumps(
            {"restaurants": [{"name": "A"}, {"name": "B"}]}, ensure_ascii=False
        )
        result = build_thinking_partials("tools", {"messages": [mock_msg]})
        assert len(result) == 1
        assert result[0]["node"] == "tool_result_search_restaurants"
        assert result[0]["analysis"]["title"] == "候选餐厅"
        assert "目前找到 2 家比较匹配的餐厅" in result[0]["analysis"]["reasoning"]

    def test_food_result_lists_detected_items_in_chinese(self):
        from langgraph_app.orchestrator.thinking import build_thinking_partials

        mock_msg = MagicMock()
        mock_msg.name = "analyze_food_image"
        mock_msg.content = json.dumps(
            {"items": [{"name": "burger"}, {"name": "fries"}]}, ensure_ascii=False
        )
        result = build_thinking_partials(
            "tools",
            {"messages": [mock_msg]},
            context_messages=[HumanMessage(content="帮我看看这顿饭怎么样")],
        )
        assert len(result) == 1
        assert result[0]["analysis"]["title"] == "识别结果"
        assert "burger" in result[0]["analysis"]["reasoning"]
        assert "fries" in result[0]["analysis"]["reasoning"]
        assert "整理成更容易读的结论" in result[0]["analysis"]["reasoning"]

    def test_supervisor_reply_uses_english_when_user_prefers_english(self):
        from langgraph_app.orchestrator.thinking import build_thinking_partials

        mock_msg = MagicMock()
        mock_msg.tool_calls = None
        mock_msg.content = "Final answer incoming"
        result = build_thinking_partials(
            "agent",
            {"messages": [mock_msg]},
            context_messages=[HumanMessage(content="Please reply in English")],
        )
        assert len(result) == 1
        assert result[0]["node"] == "supervisor_reply"
        assert result[0]["analysis"]["language"] == "English"
        assert result[0]["analysis"]["title"] == "Writing the reply"
        assert "turning everything I gathered into the final reply" in result[0]["analysis"]["reasoning"]


# =========================================================================
# 8. db.py load_recent_messages test
# =========================================================================


class TestLoadRecentMessages:
    @pytest.mark.asyncio
    async def test_method_exists(self):
        from server.db import PostgresHistoryStore

        store = PostgresHistoryStore()
        assert hasattr(store, "load_recent_messages")


# =========================================================================
# 9. Config tests
# =========================================================================


class TestConfig:
    def test_use_supervisor_config(self):
        from langgraph_app.config import config

        assert hasattr(config, "USE_SUPERVISOR")
        assert isinstance(config.USE_SUPERVISOR, bool)

    def test_max_tool_calls_config(self):
        from langgraph_app.config import config

        assert hasattr(config, "MAX_TOOL_CALLS_PER_TURN")
        assert config.MAX_TOOL_CALLS_PER_TURN == 5


# =========================================================================
# 10. Integration: chat_manager no longer has Phase 2 logic
# =========================================================================


class TestChatManagerSimplified:
    def test_no_goalplanning_phase2_rerun_logic(self):
        """Verify the Phase 2 goalplanning re-run logic was removed.

        The old code had a block that re-pushed a full_payload to the queue
        when goalplanning was detected. This should be gone now.
        """
        import inspect
        from server.chat_manager import ChatManager

        source = inspect.getsource(ChatManager.process_message)
        # These are hallmarks of the old Phase 2 logic — none should remain
        assert "full_history" not in source
        assert "full_payload" not in source
        assert "Phase 2" not in source
        assert "detected_intent" not in source
