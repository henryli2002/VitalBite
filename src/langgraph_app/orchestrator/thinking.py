"""Helpers for translating graph/tool events into frontend thinking updates."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage

from langgraph_app.utils.utils import get_dominant_language


def _resolve_language(
    language: Optional[str],
    messages: Optional[List[BaseMessage]] = None,
) -> str:
    if language in ("Chinese", "English"):
        return language
    if messages:
        return get_dominant_language(messages, default_lang="Chinese")
    return "Chinese"


def _join_examples(items: List[str], language: str) -> str:
    if not items:
        return ""
    sep = "、" if language == "Chinese" else ", "
    return sep.join(items)


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _preview_names(rows: Any, key: str, limit: int = 3) -> List[str]:
    if not isinstance(rows, list):
        return []
    names = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = _normalize_label(row.get(key))
        if label:
            names.append(label)
        if len(names) >= limit:
            break
    return names


def _make_partial(
    node: str,
    title: str,
    reasoning: str,
    tone: str,
    language: str,
) -> Dict[str, Any]:
    return {
        "status": "partial",
        "node": node,
        "analysis": {
            "title": title,
            "reasoning": reasoning,
            "tone": tone,
            "language": language,
        },
    }

def _extract_last_message(node_output: Dict[str, Any]) -> Any:
    messages = node_output.get("messages", []) or []
    if not messages:
        return None
    return messages[-1]


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
            elif item is not None:
                text_parts.append(str(item))
        return " ".join(part for part in text_parts if part).strip()
    if content is None:
        return ""
    return str(content)


def _json_loads_maybe(raw: str) -> Optional[Any]:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _infer_tool_name(tool_name: Optional[str], content: str) -> str:
    if tool_name:
        return tool_name
    parsed = _json_loads_maybe(content)
    if isinstance(parsed, dict):
        if "items" in parsed or "total_calories" in parsed:
            return "analyze_food_image"
        if "restaurants" in parsed or "page" in parsed:
            return "search_restaurants"
    return "unknown"


def _tool_call_partial(tool_name: str, language: str) -> Dict[str, Any]:
    if language == "Chinese":
        title_map = {
            "analyze_food_image": "查看图片",
            "search_restaurants": "搜索餐厅",
        }
        reasoning_map = {
            "analyze_food_image": "我先看看图里有哪些食物，再估算每一项的营养情况。",
            "search_restaurants": "我在根据当前位置和你的需求找更合适的餐厅。",
        }
        fallback_title = "调用工具"
        fallback_reasoning = f"我正在调用工具 {tool_name}，拿到结果后继续整理。"
    else:
        title_map = {
            "analyze_food_image": "Checking the image",
            "search_restaurants": "Searching restaurants",
        }
        reasoning_map = {
            "analyze_food_image": "I’m checking what food appears in the image, then I’ll estimate the nutrition for each item.",
            "search_restaurants": "I’m looking for restaurant options that better match your request and location.",
        }
        fallback_title = "Using a tool"
        fallback_reasoning = f"I’m calling {tool_name} and will continue once I have the result."

    return _make_partial(
        node=f"tool_call_{tool_name}",
        title=title_map.get(tool_name, fallback_title),
        reasoning=reasoning_map.get(tool_name, fallback_reasoning),
        tone="active",
        language=language,
    )


def _tool_result_partial(tool_name: str, content: str, language: str) -> Dict[str, Any]:
    parsed = _json_loads_maybe(content)
    if isinstance(parsed, dict) and parsed.get("error"):
        title = "调整方案" if language == "Chinese" else "Adjusting the plan"
        reasoning = (
            f"{tool_name} 这一步没有顺利完成，我会换一种方式继续组织回复。"
            if language == "Chinese"
            else f"{tool_name} didn’t complete cleanly, so I’m adjusting the response path."
        )
        return _make_partial(
            node=f"tool_result_{tool_name}",
            title=title,
            reasoning=reasoning,
            tone="neutral",
            language=language,
        )

    if tool_name == "analyze_food_image":
        items = parsed.get("items") if isinstance(parsed, dict) else None
        count = len(items) if isinstance(items, list) else 0
        names = _preview_names(items, "name")
        names_text = _join_examples(names, language)
        if language == "Chinese":
            title = "识别结果"
            if count and names_text:
                reasoning = f"我先看到了 {names_text}，一共 {count} 样。接下来把热量和营养信息整理成更容易读的结论。"
            elif count:
                reasoning = f"我已经识别出 {count} 样食物，接下来把热量和营养信息整理出来。"
            else:
                reasoning = "我已经拿到识别结果，接下来整理营养结论。"
        else:
            title = "Recognition result"
            if count and names_text:
                reasoning = f"I found {count} food item(s): {names_text}. Next I’m turning that into a clearer nutrition summary."
            elif count:
                reasoning = f"I found {count} food item(s). Next I’m turning that into a clearer nutrition summary."
            else:
                reasoning = "I have the recognition result, and now I’m turning it into a clearer nutrition summary."
    elif tool_name == "search_restaurants":
        restaurants = parsed.get("restaurants") if isinstance(parsed, dict) else None
        count = len(restaurants) if isinstance(restaurants, list) else 0
        names = _preview_names(restaurants, "name")
        names_text = _join_examples(names, language)
        if language == "Chinese":
            title = "候选餐厅"
            if count and names_text:
                reasoning = f"目前找到 {count} 家比较匹配的餐厅，包括 {names_text}。我接着把它们整理成更方便比较的推荐。"
            elif count:
                reasoning = f"目前找到 {count} 家比较匹配的餐厅，我接着把推荐理由整理出来。"
            else:
                reasoning = "我已经拿到餐厅结果，接下来整理推荐理由。"
        else:
            title = "Restaurant candidates"
            if count and names_text:
                reasoning = f"I found {count} matching places, including {names_text}. Next I’m organizing them into an easier-to-scan recommendation list."
            elif count:
                reasoning = f"I found {count} matching places. Next I’m organizing them into an easier-to-scan recommendation list."
            else:
                reasoning = "I have the restaurant results, and now I’m organizing the recommendations."
    else:
        snippet = content[:120].strip()
        title = "继续整理" if language == "Chinese" else "Continuing"
        reasoning = (
            "我已经拿到这一步的结果，继续整理后面的回答。"
            if language == "Chinese"
            else "I have the result for this step and I’m continuing the response."
        )
        if snippet:
            reasoning = snippet

    return _make_partial(
        node=f"tool_result_{tool_name}",
        title=title,
        reasoning=reasoning,
        tone="success",
        language=language,
    )


def build_thinking_partials(
    node_name: str,
    node_output: Dict[str, Any],
    language: Optional[str] = None,
    context_messages: Optional[List[BaseMessage]] = None,
) -> List[Dict[str, Any]]:
    """Build zero or more frontend-friendly partial thinking payloads."""
    resolved_language = _resolve_language(
        language,
        context_messages or node_output.get("messages"),
    )

    # --- Supervisor architecture nodes ---
    if node_name in ("supervisor", "agent"):
        last = _extract_last_message(node_output)
        if last is None:
            return []

        tool_calls = getattr(last, "tool_calls", None) or []
        if tool_calls:
            partials = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "unknown")
                partials.append(_tool_call_partial(tool_name, resolved_language))
            return partials

        content = _stringify_content(getattr(last, "content", ""))
        if content:
            reasoning = (
                "我在把刚才拿到的信息整理成最终回答，马上发给你。"
                if resolved_language == "Chinese"
                else "I’m turning everything I gathered into the final reply now."
            )
            return [
                _make_partial(
                    node="supervisor_reply",
                    title="组织回答" if resolved_language == "Chinese" else "Writing the reply",
                    reasoning=reasoning,
                    tone="compose",
                    language=resolved_language,
                )
            ]
        return []

    if node_name in ("tools", "tool"):
        last = _extract_last_message(node_output)
        if last is None:
            return []
        content = _stringify_content(getattr(last, "content", ""))
        tool_name = _infer_tool_name(
            getattr(last, "name", None) or getattr(last, "tool_name", None),
            content,
        )
        if not content and tool_name == "unknown":
            return []
        return [_tool_result_partial(tool_name, content, resolved_language)]

    # --- Legacy architecture nodes ---
    if node_name in ("router", "intent_router"):
        analysis = node_output.get("analysis")
        if analysis:
            enriched = dict(analysis)
            enriched.setdefault("language", resolved_language)
            return [{"status": "partial", "node": "intent_router", "analysis": enriched}]
        return []

    if node_name == "chitchat":
        messages = node_output.get("messages", []) or []
        answer = ""
        if messages and isinstance(messages, list):
            first = messages[0]
            answer = getattr(first, "content", "") if first else ""
        if answer:
            return [
                {
                    "status": "partial",
                    "node": "chitchat",
                    "analysis": {
                        "reasoning": f"Answering: {str(answer)[:180]}",
                        "language": resolved_language,
                    },
                }
            ]
        return []

    if node_name == "recommendation":
        rec = node_output.get("recommendation_result") or {}
        restaurants = rec.get("restaurants") or []
        if not isinstance(restaurants, list) or not restaurants:
            return []
        names = [str(r.get("name", "")) for r in restaurants[:3] if isinstance(r, dict)]
        names = [n for n in names if n]
        if not names:
            return []
        return [
            {
                "status": "partial",
                "node": "recommendation",
                "analysis": {
                    "reasoning": f"Finding restaurants: Found restaurants: {', '.join(names)}",
                    "language": resolved_language,
                },
            }
        ]

    return []


def build_thinking_partial(
    node_name: str,
    node_output: Dict[str, Any],
    language: Optional[str] = None,
    context_messages: Optional[List[BaseMessage]] = None,
) -> Optional[Dict[str, Any]]:
    """Backward-compatible single-event helper used by older tests/callers."""

    partials = build_thinking_partials(
        node_name,
        node_output,
        language=language,
        context_messages=context_messages,
    )
    if not partials:
        return None
    first = partials[0]
    if node_name in ("supervisor", "agent") and first["node"].startswith("tool_call_"):
        return {
            "status": first["status"],
            "node": "tool_call",
            "analysis": first["analysis"],
        }
    if node_name in ("tools", "tool") and first["node"].startswith("tool_result_"):
        return {
            "status": first["status"],
            "node": "tool_result",
            "analysis": first["analysis"],
        }
    return first
