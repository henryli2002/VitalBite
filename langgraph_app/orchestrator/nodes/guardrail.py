"""Guardrail nodes for input and output safety checks using NeMo Guardrails."""

from typing import Dict, Any, Literal, Tuple
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.utils.logger import setup_logger
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langgraph_app.utils.utils import get_all_user_text
import re
import os
from nemoguardrails import LLMRails, RailsConfig

logger = setup_logger(__name__)

# Path to the NeMo Guardrails configuration directory
config_dir = os.path.join(os.path.dirname(__file__), "nemo_config")
config = RailsConfig.from_path(config_dir)
nemo_rails = LLMRails(config)
logger.info("NeMo Guardrails initialized successfully.")

class SafetyCheck(BaseModel):
    safe: bool
    reason: str | None = None
    category: str | None = None

IntentType = Literal["recognition", "recommendation", "chitchat", "tutorial", "goalplanning"]

# ======================================================================
# Native fallback checks (used if nemoguardrails is missing or fails)
# ======================================================================

NORMALIZE = re.compile(r'[^\w\s]')

def normalize(text: str) -> str:
    text = text.lower()
    text = NORMALIZE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

EN_INSTRUCTION = [
    r"\bignore\b",
    r"\bfollow (these|my) instructions?\b",
    r"\byou must\b",
    r"\byour task is\b",
    r"\bact as\b",
    r"\bpretend to be\b",
    r"\bfrom now on\b",
    r"\brespond with\b",
    r"\bclassify (this|it) as\b",
    r"\bdo not analyze\b",
    r"\boutput (yes|no)\b",
]

CN_INSTRUCTION = [
    r"忽略.*指令",
    r"无视.*规则",
    r"你必须",
    r"你的任务是",
    r"现在开始",
    r"扮演.*角色",
    r"假装.*是",
    r"请直接输出",
    r"请判断为",
    r"不要分析",
]

PROMPT_BREAKERS = [
    r"\bsystem\s*:",
    r"\bassistant\s*:",
    r"\buser\s*:",
    r"系统\s*[:：]",
    r"助手\s*[:：]",
    r"用户\s*[:：]",
    r"\bbegin\b",
    r"\bend\b",
    r"开始",
    r"结束",
    r"###",
    r"```",
    r"<\|.*?\|>",
]

META_CONTROL = [
    r"\bnew rule\b",
    r"\boverride\b",
    r"\bno matter what\b",
    r"新的规则",
    r"覆盖.*规则",
    r"之前的指令无效",
    r"无论如何",
]

def match(text: str, patterns)  -> Tuple[int, list]:
    """return the the content and amount of matches for the given patterns"""
    matches = []
    count = 0
    for pattern in patterns:
        if re.search(pattern, text):
            matches.append(pattern)
            count += 1
    return count, matches

def prompt_injection_risk(text: str) -> Tuple[bool, str]:
    """
    Detect if user input attempts to control downstream LLM behaviour.
    """
    t = normalize(text)

    instruction_count, instruction_matches = match(t, EN_INSTRUCTION + CN_INSTRUCTION)
    prompt_breaker_count, prompt_breaker_matches = match(t, PROMPT_BREAKERS)
    meta_control_count, meta_control_matches = match(t, META_CONTROL)  
    if instruction_count > 0 or prompt_breaker_count > 0 or meta_control_count > 0:
        reasoning = "User input contains potential prompt injection patterns: "
        if instruction_count > 0:
            reasoning += f"instruction patterns ({', '.join(instruction_matches)}); "
        if prompt_breaker_count > 0:
            reasoning += f"prompt breaker patterns ({', '.join(prompt_breaker_matches)}); "
        if meta_control_count > 0:
            reasoning += f"meta control patterns ({', '.join(meta_control_matches)}); "
        return True, reasoning
    return False, ""

def get_standard_response(category: str | None) -> str:
    """Return a standard response based on the safety failure category."""
    if category == "prompt_injection":
        return "对不起，我无法执行改变我核心指令或规则的操作。请问有其他我可以帮您的吗？"
    elif category == "food_safety_risk":
        return "请注意，这可能涉及严重的食品安全问题（如有毒或危险）。为了您的健康，请勿食用。如果您或他人已经误食了有毒物品，请立即寻求医疗救助（在新加坡请拨打 995 呼叫救护车或 999 报警）！"
    elif category == "self_harm":
        return "听起来您正在经历一段非常艰难的时期。请不要放弃，您的生命非常宝贵。如果您正在考虑伤害自己，请立即寻求专业帮助。您可以拨打心理援助热线（如新加坡的援人协会 SOS 热线：1-767，或心理卫生学院热线：6389-2222），或者在紧急情况下拨打 995 或 999 报警和呼叫急救。请记住，这个世界总有人在乎您！"
    elif category == "illegal":
        return "抱歉，您的请求涉及违法内容，我无法为您提供帮助。"
    elif category == "violence" or category == "hate":
        return "对不起，我不能处理包含暴力、仇恨或攻击性内容的请求。"
    elif category == "sexual":
        return "抱歉，我无法处理包含色情或不当内容的请求。"
    else:
        return "抱歉，您的请求涉及不当或不安全的话题，我无法为您提供帮助。"


async def run_nemo_check(text_to_check: str) -> Tuple[bool, str | None, str | None]:
    """
    Run NeMo Guardrails check asynchronously.
    Returns (is_safe, refusal_message, category).
    If it's safe, refusal_message is None.
    """
    try:
        # Generate the response using NeMo rails
        # If the input triggers a refusal rail (e.g., jailbreak), NeMo will return the predefined refusal string.
        # Otherwise, it will just pass through or return a normal response.
        response = await nemo_rails.generate_async(messages=[{"role": "user", "content": text_to_check}])
        
        # Check if the response contains our predefined refusal markers from topics.co
        response_text = ""
        if isinstance(response, str):
            response_text = response
        elif isinstance(response, dict):
            response_text = response.get("content", "")
        elif hasattr(response, "content"):
            response_text = response.content # type: ignore
        elif isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict):
             # handle list of dict case which sometimes returned by nemoguardrails
             response_text = response[0].get("content", "")
        elif isinstance(response, tuple) and len(response) > 0:
            if isinstance(response[0], dict):
                response_text = response[0].get("content", "")
        else:
             response_text = str(response)

        
        if "I cannot follow instructions that override my safety guidelines" in response_text or "无法执行改变我核心指令" in response_text:
            return False, response_text, "prompt_injection"
        elif "This sounds like a severe food safety risk" in response_text or "严重的食品安全问题" in response_text:
            return False, response_text, "food_safety_risk"
        elif "It sounds like you are going through a difficult time" in response_text or "听起来您正在经历一段非常艰难的时期" in response_text:
            return False, response_text, "self_harm"
            
        # If it doesn't match our predefined refusals, we consider it safe from the NeMo perspective
        # and let it proceed to normal routing.
        return True, None, None
        
    except Exception as e:
        logger.error(f"NeMo check failed during execution: {e}")
        return True, None, None


def _check_safety_native(text_to_check: str, intent: Any, node_name: str, messages: list | None = None) -> NodeOutput:
    """
    Fallback native safety check used if NeMo is not available.
    """
    # 1. Check for prompt injection first (fast regex check)
    is_injection, injection_reason = prompt_injection_risk(text_to_check)
    if is_injection:
        logger.warning(f"[{node_name}] Prompt injection risk detected: {injection_reason}")
        return {
            "analysis": {
                "safety_safe": False,
                "safety_reason": injection_reason,
                "safety_category": "prompt_injection",
                "intent": "chitchat"
            },
            "messages": [AIMessage(content=get_standard_response("prompt_injection"))],
            "debug_logs": [{
                "node": node_name,
                "status": "warning",
                "reason": "prompt_injection"
            }]
        }

    # 2. LLM-based safety check
    client = get_llm_client(module="guardrail")
    
    # Safety check prompt
    system_prompt = """Analyze the following text for harmful, inappropriate, or unsafe content.

Respond with a JSON object containing:
- "safe": boolean indicating if the content is safe
- "reason": string explaining why it's unsafe (null if safe)
- "category": string indicating the category of unsafe content ("self_harm", "illegal", "violence", "hate", "sexual", "food_safety_risk"), null if safe

Only mark as unsafe if the content contains:
- Hate speech, discrimination, or harassment ("hate")
- Violence or threats ("violence")
- Illegal activities ("illegal")
- Explicit sexual content ("sexual")
- Self-harm or suicide references ("self_harm")
- Dangerous, toxic, or poisoned food, or any severe food safety risk (e.g. eating poisonous mushrooms, intentionally spoiling food) ("food_safety_risk")

Be lenient with normal food-related queries, even if they mention dietary restrictions or health concerns. Only flag severe food safety risks."""

    try:
        result = client.generate_structured(
            messages=[HumanMessage(content=text_to_check)],
            schema=SafetyCheck,
            system_prompt=system_prompt
        )
        
        if not result.safe:
            logger.warning(f"[{node_name}] Safety check failed. Reason: {result.reason}, Category: {result.category}")
            return {
                "analysis": {
                    "safety_safe": False,
                    "safety_reason": result.reason,
                    "safety_category": result.category,
                    "intent": "chitchat"
                },
                "messages": [AIMessage(content=get_standard_response(result.category))],
                "debug_logs": [{
                    "node": node_name,
                    "status": "warning",
                    "reason": result.reason
                }]
            }
        
        return {
            "analysis": {
                "safety_safe": True,
                "safety_reason": None,
                "safety_category": None,
                "intent": intent
            }
        }
    except Exception as e:
        # On error, default to safe but log the issue
        logger.error(f"[{node_name}] Guardrail check encountered an error: {e}", exc_info=True)
        return {
            "analysis": {
                "safety_safe": True,
                "safety_reason": f"Safety check error: {str(e)}",
                "safety_category": None,
                "intent": intent
            },
            "debug_logs": [{
                "node": node_name,
                "status": "error",
                "error": str(e)
            }]
        }


async def _check_safety(text_to_check: str, intent: Any, node_name: str, messages: list | None = None) -> NodeOutput:
    """
    Main function to check text for harmful content and prompt injection.
    Attempts NeMo Guardrails first, falls back to native checks.
    """
    if not text_to_check:
        # Empty input is considered safe
        return {
            "analysis": {
                "safety_safe": True,
                "safety_reason": None,
                "safety_category": None,
                "intent": intent
            }
        }
        
    if nemo_rails is not None:
        logger.info(f"[{node_name}] Routing check through NeMo Guardrails...")
        is_safe, refusal_msg, category = await run_nemo_check(text_to_check)
        
        if not is_safe:
            logger.warning(f"[{node_name}] NeMo Guardrails rejected input. Category: {category}")
            return {
                "analysis": {
                    "safety_safe": False,
                    "safety_reason": "Blocked by NeMo Guardrails",
                    "safety_category": category,
                    "intent": "chitchat"
                },
                "messages": [AIMessage(content=refusal_msg)],
                "debug_logs": [{
                    "node": node_name,
                    "status": "warning",
                    "reason": category,
                    "engine": "nemo_guardrails"
                }]
            }
        # If NeMo considers it safe, we just return safe
        return {
            "analysis": {
                "safety_safe": True,
                "safety_reason": None,
                "safety_category": None,
                "intent": intent
            }
        }
    else:
        # Fallback to the native manual LLM and regex check
        return _check_safety_native(text_to_check, intent, node_name, messages)


def _extract_text(obj: Any) -> str:
    if not obj:
        return ""
    if isinstance(obj, str):
        return obj
    
    content = ""
    if isinstance(obj, dict):
        content = obj.get("content", "")
    elif hasattr(obj, "content"):
        content = obj.content
    else:
        return str(obj)
        
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
        return " ".join(texts)
    elif isinstance(content, str):
        return content
    return str(content)

async def input_guardrail_node(state: GraphState) -> NodeOutput:
    """
    Check the user's input for harmful content and prompt injection.
    """
    messages = state.get("messages", [])
    text_to_check = get_all_user_text(messages)  # type: ignore
    intent = state.get("analysis", {}).get("intent", "chitchat")
    return await _check_safety(text_to_check, intent, "input_guardrail", messages)

async def output_guardrail_node(state: GraphState) -> NodeOutput:
    """
    Check the agent's final response for harmful content.
    """
    text_to_check = _extract_text(state.get("final_response", ""))
    intent = state.get("analysis", {}).get("intent", "chitchat")
    return await _check_safety(text_to_check, intent, "output_guardrail")
