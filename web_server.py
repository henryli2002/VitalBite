"""WABI Chat Web Server — FastAPI + WebSocket interface to LangGraph.

Run with:
    cd /Users/henryli/Desktop/Project/WABI
    python -m uvicorn web_server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from models import UserCreate, UserInfo, ChatMessage, WSIncoming, WSOutgoing
from chat_manager import ChatManager

# Lazy import the graph to avoid import-time side effects
_graph = None


def get_graph():
    """Lazy-load the LangGraph graph."""
    global _graph
    if _graph is None:
        from langgraph_app.orchestrator.graph import graph
        _graph = graph
    return _graph


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="WABI Chat", version="1.0.0")
chat_manager = ChatManager()
logger = logging.getLogger("wabi.web")

# Active WebSocket connections: user_id -> WebSocket
active_connections: Dict[str, WebSocket] = {}


# ---------------------------------------------------------------------------
# Static files (frontend)
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the main chat UI."""
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# REST API — User Management
# ---------------------------------------------------------------------------

@app.post("/api/users", response_model=UserInfo)
async def create_user(body: UserCreate = None):
    """Create a new user/conversation."""
    name = body.name if body else None
    user_info = await chat_manager.create_user(name)
    return UserInfo(**user_info)


@app.get("/api/users")
async def list_users():
    """List all users."""
    users = await chat_manager.get_users()
    return [UserInfo(**u) for u in users]


@app.delete("/api/users/{user_id}")
async def delete_user(user_id: str):
    """Delete a user and their chat history."""
    success = await chat_manager.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    # Disconnect active WebSocket if any
    if user_id in active_connections:
        try:
            await active_connections[user_id].close()
        except Exception:
            pass
        active_connections.pop(user_id, None)
    return {"status": "deleted", "user_id": user_id}


@app.get("/api/users/{user_id}/history")
async def get_history(user_id: str):
    """Get chat history for a user."""
    history = await chat_manager.get_history(user_id)
    return [ChatMessage(
        role=msg["role"],
        content=msg["content"],
        timestamp=msg["timestamp"],
    ) for msg in history]


# ---------------------------------------------------------------------------
# WebSocket — Real-time Chat
# ---------------------------------------------------------------------------

@app.websocket("/ws/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time chat with a specific user."""
    await websocket.accept()
    active_connections[user_id] = websocket
    logger.info(f"WebSocket connected: {user_id}")

    try:
        while True:
            # Receive message from client
            raw = await websocket.receive_json()
            incoming = WSIncoming(**raw)

            # Send typing indicator
            await websocket.send_json(
                WSOutgoing(type="typing", content="").model_dump()
            )

            try:
                # Build content for LangGraph
                if incoming.type == "image" and incoming.content:
                    # Multimodal: image + optional text
                    content_parts = []
                    if incoming.text.strip():
                        content_parts.append({"type": "text", "text": incoming.text})
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{incoming.mime_type};base64,{incoming.content}"
                        },
                    })
                    content = content_parts
                else:
                    content = incoming.content

                # Process through LangGraph (run in thread to avoid blocking)
                graph = get_graph()
                ai_response = await asyncio.to_thread(
                    _sync_process_message, user_id, content, graph
                )

                # Send AI response
                response = WSOutgoing(
                    type="message",
                    role="assistant",
                    content=ai_response,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                await websocket.send_json(response.model_dump())

            except Exception as e:
                logger.error(f"Error processing message for {user_id}: {e}", exc_info=True)
                error_response = WSOutgoing(
                    type="error",
                    content=f"Error: {str(e)}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                await websocket.send_json(error_response.model_dump())

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {user_id}: {e}", exc_info=True)
    finally:
        active_connections.pop(user_id, None)


def _sync_process_message(user_id: str, content, graph) -> str:
    """Synchronous wrapper for chat_manager.process_message (used in to_thread)."""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            chat_manager.process_message(user_id, content, graph)
        )
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Pre-load dotenv and log startup."""
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    logger.info("WABI Chat Web Server started")
