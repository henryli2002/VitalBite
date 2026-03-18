"""Guardrail nodes for input and output safety checks."""

from typing import Dict, Any, Literal, Tuple
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.utils.logger import setup_logger
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph_app.utils.utils import get_all_user_text
import re

logger = setup_logger(__name__)

class SafetyCheck(BaseModel):
    safe: bool
    reason: str | None = None
    category: str | None = None

IntentType = Literal["recognition", "recommendation", "chitchat", "tutorial", "goalplanning"]

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

def is_chinese(text: str) -> bool:
    """Check if the text contains Chinese characters."""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def get_standard_response(category: str | None, user_input: str = "") -> str:
    """Return a standard response based on the safety failure category and language."""
    chinese = is_chinese(user_input)

    if category == "prompt_injection":
        if chinese:
            return "对不起，我无法执行改变我核心指令或规则的操作。请问有其他我可以帮您的吗？"
        return "Sorry, I cannot perform actions that override my core instructions. How else can I help you today?"
        
    elif category == "food_safety_risk":
        if chinese:
            return "请注意，这可能涉及严重的食品安全问题（如误食有毒有害物质）。您的健康和安全是最重要的，请千万不要食用，并尽快寻求专业医生的建议！我无法为您提供此类建议。"
        return "Please be aware that this involves severe food safety risks. Your health is the top priority—please do not consume this and seek professional advice immediately! I cannot provide guidance on this matter."
        
    elif category == "self_harm":
        if chinese:
            return "无论您此刻正在经历多么沉重的痛苦，请相信，您在这个世界上并不孤单。总有人愿意倾听您的心声，为您提供支持和力量。如果您感到绝望，请务必联系他们寻求帮助：\n\n**新加坡援助热线：**\n- 援人协会 (SOS): 1-767 (24小时)\n- 心理健康研究所 (IMH): 6389-2222 (24小时)\n\n**中国大陆援助热线：**\n- 希望24小时热线: 400-161-9995\n- 心理危机干预热线 (北京): 010-82951332\n\n请给自己一个机会，也给愿意帮助您的人一个机会。世界依然有很多温暖等待着您。"
        return "I'm so sorry you're going through this pain right now. Please know that you are not alone, and there is help available. No matter how heavy things feel, there are people who want to support you. Please reach out to these resources:\n\n**Singapore Helplines:**\n- Samaritans of Singapore (SOS): 1-767 (24-hour)\n- IMH Mental Health Helpline: 6389-2222 (24-hour)\n\nPlease take a moment to talk to someone. Your life is precious, and there is hope."
        
    elif category == "violence" or category == "illegal":
        if chinese:
            return "我注意到您的描述涉及暴力或潜在的危险/违法行为。如果您或他人正面临紧急的人身安全威胁，请务必保持冷静，并立即联系警方寻求保护：\n\n- **新加坡报警电话**: 999\n- **中国大陆报警电话**: 110\n\n我无法为您提供与此类内容相关的协助，请以安全为重。"
        return "I noticed your message involves violence, danger, or illegal activities. If you or someone else is facing an immediate threat to physical safety, please contact the police immediately:\n\n- **Singapore Police Force**: 999\n\nI cannot assist with requests related to these topics. Please prioritize safety."
        
    elif category == "sexual":
        if chinese:
            return "抱歉，我无法处理包含色情或不当内容的请求。"
        return "I apologize, but I cannot process requests containing explicit or inappropriate sexual content."
        
    else:
        if chinese:
            return "抱歉，您的请求涉及不当或不安全的话题，我无法为您提供帮助。"
        return "I'm sorry, but your request involves inappropriate or unsafe topics, and I cannot assist you with it."

def _check_safety(text_to_check: str, intent: Any, node_name: str, messages: list | None = None) -> NodeOutput:
    """
    Generic function to check text for harmful content and prompt injection.
    
    Args:
        text_to_check: The text to be analyzed.
        intent: The current intent to be preserved in the state.
        node_name: The name of the node for logging.
        messages: The message history for context.
        
    Returns:
        NodeOutput with the safety analysis and potential short-circuit message.
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

    # 1. Check for prompt injection first (fast regex check)
    is_injection, injection_reason = prompt_injection_risk(text_to_check)
    if is_injection:
        logger.warning(f"[{node_name}] Prompt injection risk detected: {injection_reason}")
        return {
            "analysis": {
                "safety_safe": False,
                "safety_reason": injection_reason,
                "safety_category": "prompt_injection",
                "intent": "guardrails"
            },
            "messages": [AIMessage(content=get_standard_response("prompt_injection", text_to_check))],
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

    last_error = None
    structured_llm = client.with_structured_output(SafetyCheck)

    for attempt in range(3):
        try:
            messages_to_send = [SystemMessage(content=system_prompt), HumanMessage(content=text_to_check)]
            if last_error:
                error_feedback = f"Your previous response failed validation with this error: {str(last_error)}. Please correct your JSON output and ensure it strictly follows the schema."
                messages_to_send.append(SystemMessage(content=error_feedback))
                
            result = structured_llm.invoke(messages_to_send, config={"callbacks": []})
            
            if not result.safe:
                logger.warning(f"[{node_name}] Safety check failed. Reason: {result.reason}, Category: {result.category}")
                return {
                    "analysis": {
                        "safety_safe": False,
                        "safety_reason": result.reason,
                        "safety_category": result.category,
                        "intent": "guardrails"
                    },
                    "messages": [AIMessage(content=get_standard_response(result.category, text_to_check))],
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
            last_error = e
            logger.warning(f"[{node_name}] Guardrail check failed on attempt {attempt + 1}: {e}")
            if attempt < 2:
                import time
                time.sleep(1)

    # On error, default to safe but log the issue
    logger.error(f"[{node_name}] Guardrail check encountered an error after retries: {last_error}", exc_info=True)
    return {
        "analysis": {
            "safety_safe": True,
            "safety_reason": f"Safety check error: {str(last_error)}",
            "safety_category": None,
            "intent": intent
        },
        "debug_logs": [{
            "node": node_name,
            "status": "error",
            "error": str(last_error)
        }]
    }

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

def input_guardrail_node(state: GraphState) -> NodeOutput:
    """
    Check the user's input for harmful content and prompt injection.
    Only checks the latest message to avoid false positives from history.
    """
    messages = state.get("messages", [])
    
    # Get only the text of the LAST message from the user
    latest_message = messages[-1] if messages else None
    text_to_check = _extract_text(latest_message)

    intent = state.get("analysis", {}).get("intent", "chitchat")
    return _check_safety(text_to_check, intent, "input_guardrail", messages)

def output_guardrail_node(state: GraphState) -> NodeOutput:
    """
    Check the agent's final response for harmful content.
    """
    messages = state.get("messages", [])
    latest_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            latest_ai_message = msg
            break
            
    text_to_check = _extract_text(latest_ai_message) if latest_ai_message else ""
    intent = state.get("analysis", {}).get("intent", "chitchat")
    return _check_safety(text_to_check, intent, "output_guardrail")
