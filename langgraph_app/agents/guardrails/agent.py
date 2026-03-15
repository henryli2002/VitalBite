"""Guardrails agent for handling malicious or unsafe inputs."""

from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from langgraph_app.orchestrator.state import GraphState
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.utils.utils import (
    get_dominant_language,
)
from datetime import datetime
from time import sleep


def guardrails_node(state: GraphState) -> GraphState:
    """
    Generate a safe response to malicious or inappropriate user input.
    """
    state = state.copy()
    messages = state.setdefault("messages", [])
    
    # Check if analysis has safety reason and category
    analysis = state.get("analysis", {})
    safety_reason = analysis.get("safety_reason")
    safety_category = analysis.get("safety_category")

    # Set safety warning response
    safety_message = "抱歉，您的输入包含不当内容，我无法处理此请求。\nSorry, your input contains inappropriate content and I cannot process this request."
    if safety_reason:
        safety_message += f"\n原因 (Reason)：{safety_reason}"
        
    if safety_category == "self_harm":
        safety_message += "\n\n生命非常珍贵，请不要放弃希望。如果您或您身边的人正在经历困难，请务必寻求帮助：\n"
        safety_message += "Life is precious, please don't lose hope. If you or someone you know is going through a difficult time, please seek help:\n"
        safety_message += "📞 【中国大陆 / Mainland China】24小时心理危机干预热线 (24/7 Crisis Hotline)：400-161-9995 或 (or) 010-82951332\n"
        safety_message += "📞 【新加坡 / Singapore】SOS (Samaritans of Singapore) 热线 (Hotline)：1767\n"
        safety_message += "💬 【新加坡 / Singapore】SOS WhatsApp 求助 (WhatsApp Help)：9151 1767\n"
        safety_message += "🏥 【新加坡 / Singapore】心理健康研究所 (IMH) 危机热线 (Crisis Helpline)：6389 2222"
    elif safety_category in ["illegal", "violence"]:
        safety_message += "\n\n暴力和非法行为将带来严重后果。如果您正处于危险之中，或发现任何违法犯罪行为，请立即联系警方：\n"
        safety_message += "Violence and illegal activities have serious consequences. If you are in danger or witness any crimes, please contact the police immediately:\n"
        safety_message += "🚨 【中国大陆 / Mainland China】报警电话 (Police)：110\n"
        safety_message += "🚨 【新加坡 / Singapore】紧急报警电话 (Emergency Police)：999\n"
        safety_message += "📞 【新加坡 / Singapore】警察局非紧急热线 (Police Non-Emergency)：1800-255-0000"

    messages.append(AIMessage(content=safety_message))
    state.setdefault("message_timestamps", []).append(datetime.utcnow().isoformat())
    state["final_response"] = safety_message
    
    return state
