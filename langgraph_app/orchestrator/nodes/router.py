"""Intent routing node."""

from typing import Dict, Any, Literal
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.config import config
from langgraph_app.utils.logger import setup_logger
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph_app.utils.utils import (
    get_all_user_text,
)

import time
import re
from typing import Tuple
from time import sleep

logger = setup_logger(__name__)

class IntentAnalysis(BaseModel):
    """Structured output for intent routing."""
    intent: Literal["recognition", "recommendation", "chitchat", "tutorial", "goalplanning"]
    confidence: float
    reasoning: str


def intent_router_node(state: GraphState) -> NodeOutput:
    """
    Route user input to appropriate agent based on intent.
    This router focuses only on the high-level user goal.
    """
    client = get_llm_client(module="router")
    messages = state.get("messages", [])
    
    current_hour = time.localtime().tm_hour
    current_minute = time.localtime().tm_min
    current_time = current_hour + current_minute / 60.0
    if 7 <= current_time < 9.5:
        meal_time = "breakfast time"
    elif 11.5 <= current_time < 13.5:
        meal_time = "lunch time"
    elif 17.5 <= current_time < 19.5:
        meal_time = "dinner time"
    else:
        meal_time = "not meal time"

    system_prompt = f"""Analyze the user's intent based on the entire conversation history. Your goal is to identify the user's primary goal, not the method to achieve it.

Determine the intent based on these rules:
1.  "recognition": If the user's primary goal is to identify food, get nutritional info, or analyze a meal from one or more images. (Important: Only use this if there is actually an image of food provided. If they ask to identify a food but provide NO image, use "tutorial").
2.  "recommendation": If the user is asking about restaurants, places to eat, or food recommendations. ALSO use this if the user expresses hunger, fatigue, or a mood that strongly implies they need food recommendations right now. Also consider it's {current_hour}:{current_minute:02d} which is {meal_time}.
3.  "goalplanning": If the user wants to plan their diet, set eating goals, or discuss long-term nutrition.
4.  "tutorial": If the user asks how to use the app, for instructions, OR if they ask for image recognition but there are NO images provided in the entire conversation. ALSO use this if the user provides a food image but uses very vague or weak recognition language (e.g. "I want something like this"), to ask them if they want a recommendation or something else. ALSO use this if the user input is extremely short, minimal, noisy, or meaningless (like just emojis or random symbols), so we can guide them on how to use the assistant.
5.  "chitchat": For general conversation, greetings, follow-up questions not tied to a specific feature, off-topic questions, or if the user wants to end the conversation. ALSO use this if an image is provided but it is completely unrelated to food (e.g., a car, a landscape, a pet) or is so severely blurry/dark that no objects can be discerned. This is the default.

Respond with a JSON object containing:
- "intent": one of ["recognition", "recommendation", "goalplanning", "tutorial", "chitchat"]
- "confidence": float between 0.0 and 1.0
- "reasoning": brief explanation of why this intent was chosen."""

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            structured_llm = client.with_structured_output(IntentAnalysis)
            messages_to_send = [SystemMessage(content=system_prompt)] + messages
            
            if last_error:
                error_feedback = f"Your previous response failed validation with this error: {str(last_error)}. Please correct your JSON output and ensure it strictly follows the schema."
                messages_to_send.append(SystemMessage(content=error_feedback))
                
            result = structured_llm.invoke(messages_to_send, config={"callbacks": []})
            
            logger.info(f"[router] Intent detected: {result.intent} (confidence: {result.confidence})")
            
            return {
                "analysis": {
                    "intent": result.intent,
                    "safety_safe": True,
                    "safety_reason": None,
                },
                "debug_logs": [{
                    "node": "router",
                    "status": "success",
                    "llm_response": result.model_dump() if hasattr(result, 'model_dump') else {}
                }]
            }
        except Exception as e:
            last_error = e
            logger.warning(f"[router] Intent routing failed on attempt {attempt + 1}: {e}")
            if attempt < 2:
                sleep(1)

    logger.error(f"[router] Intent routing ultimately failed after retries: {last_error}", exc_info=True)
    return {
        "analysis": {
            "intent": "chitchat",
            "safety_safe": True,
            "safety_reason": None,
        },
        "debug_logs": [{
            "node": "router",
            "status": "error",
            "error": str(last_error)
        }]
    }
