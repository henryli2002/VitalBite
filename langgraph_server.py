"""FastAPI standalone server to encapsulate the LangGraph execution.
Serves a POST endpoint to process a chat turn and run the complete AI pipeline.
"""

import logging
import time
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Import the LangGraph workflow
from langgraph_app.orchestrator.graph import graph
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="WABI AI Engine", version="1.0.0")
logger = logging.getLogger("wabi.ai")
logging.basicConfig(level=logging.INFO)

class GraphInvokeRequest(BaseModel):
    messages: List[Dict[str, Any]]
    session_id: str
    user_id: str
    user_name: Optional[str] = None
    user_profile: Optional[Dict[str, Any]] = None
    thread_id: str
    invoke_full_history: bool = False
    full_messages: Optional[List[Dict[str, Any]]] = None

class GraphInvokeResponse(BaseModel):
    ai_text: str
    detected_intent: str

def build_langchain_messages(history: List[Dict]) -> List[BaseMessage]:
    """Convert JSON messages back into LangChain message objects."""
    messages = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        timestamp = msg.get("timestamp")
        if role == "user":
            messages.append(HumanMessage(content=content, response_metadata={"timestamp": timestamp} if timestamp else {}))
        elif role == "assistant":
            messages.append(AIMessage(content=content, response_metadata={"timestamp": timestamp} if timestamp else {}))
    return messages

@app.post("/invoke", response_model=GraphInvokeResponse)
async def invoke_graph(request: GraphInvokeRequest):
    logger.info(f"[{request.user_id}] Received invoke request via HTTP API")
    try:
        # Build initial state
        initial_messages = build_langchain_messages(request.messages)
        initial_state = {
            "messages": initial_messages,
            "session_id": request.session_id,
            "user_id": request.user_id,
            "user_name": request.user_name,
            "user_profile": request.user_profile,
        }
        
        config = {"configurable": {"thread_id": request.thread_id}}
        
        logger.info(f"[{request.user_id}] Running LangGraph (thread_id: {request.thread_id})...")
        result = await graph.ainvoke(initial_state, config=config)
        
        # Extract intent
        analysis = result.get("analysis", {})
        detected_intent = analysis.get("intent", "chitchat")
        
        # If intent is goalplanning and we have full history provided, re-invoke
        if detected_intent == "goalplanning" and request.invoke_full_history and request.full_messages:
            logger.info(f"[{request.user_id}] Goalplanning detected. Reprocessing with FULL history.")
            full_msgs = build_langchain_messages(request.full_messages)
            full_state = {
                "messages": full_msgs,
                "session_id": request.session_id,
                "user_id": request.user_id,
                "user_name": request.user_name,
                "user_profile": request.user_profile,
            }
            full_config = {"configurable": {"thread_id": f"{request.thread_id}_full"}}
            result = await graph.ainvoke(full_state, config=full_config)
            
        # Extract final AI response
        ai_text = ""
        if "messages" in result and result["messages"]:
            last_msg = result["messages"][-1]
            if isinstance(last_msg, AIMessage):
                ai_text = last_msg.content
            else:
                ai_text = str(last_msg.content)
        else:
            ai_text = "Sorry, I could not process your request."
            
        logger.info(f"[{request.user_id}] Graph complete. Intent: {detected_intent}")
        return GraphInvokeResponse(ai_text=ai_text, detected_intent=detected_intent)
        
    except Exception as e:
        logger.error(f"[{request.user_id}] Error in graph execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
