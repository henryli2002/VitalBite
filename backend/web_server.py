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

from models import UserCreate, UserInfo, UserProfile, ChatMessage, WSIncoming, WSOutgoing
from chat_manager import ChatManager




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

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the main chat UI."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


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
    out = []
    for msg in history:
        content = msg["content"]
        if isinstance(content, list):
            text_parts = []
            for p in content:
                if p.get("type") == "text":
                    text_parts.append(p["text"])
                elif p.get("type") == "image_url":
                    url = p["image_url"]["url"]
                    text_parts.append(f"![image]({url})")
            content_str = "\n\n".join(text_parts)
        else:
            content_str = str(content)
            
        out.append(ChatMessage(
            role=msg["role"],
            content=content_str,
            timestamp=msg["timestamp"],
        ))
    return out


@app.get("/api/users/{user_id}/profile")
async def get_profile(user_id: str):
    """Get user profile."""
    profile = await chat_manager.get_profile(user_id)
    return profile


@app.put("/api/users/{user_id}/profile")
async def update_profile(user_id: str, body: UserProfile):
    """Update user profile (partial merge — only overwrites fields that are sent)."""
    # Load existing profile first, then merge
    existing = await chat_manager.get_profile(user_id)
    incoming = {k: v for k, v in body.model_dump().items() if v is not None}
    merged = {**existing, **incoming}
    await chat_manager.save_profile(user_id, merged)
    return {"status": "updated", "user_id": user_id, "profile": merged}


# ---------------------------------------------------------------------------
# WebSocket — Real-time Chat
# ---------------------------------------------------------------------------

@app.websocket("/ws/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time chat with a specific user."""
    await websocket.accept()
    active_connections[user_id] = websocket
    logger.info(f"WebSocket connected: {user_id}")

    # Try to get the real user IP
    client_ip = None
    if websocket.client:
        client_ip = websocket.client.host
    forwarded = websocket.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    try:
        while True:
            # Receive message from client
            raw = await websocket.receive_json()
            incoming = WSIncoming(**raw)

            # Send typing indicator
            await websocket.send_json(
                WSOutgoing(type="typing", role="assistant", content="").model_dump()
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

                # Pass user context along to LangGraph
                user_context = {
                    "lat": incoming.lat,
                    "lng": incoming.lng,
                    "user_ip": client_ip,
                }

                # Process through chat manager which invokes AI microservice
                ai_response = ""
                async for ai_chunk in chat_manager.process_message(user_id, content, user_context=user_context):
                    if ai_chunk.get("type") == "thinking":
                        # Forward thinking chunk to frontend
                        thinking_response = WSOutgoing(
                            type="thinking",
                            role="assistant",
                            content=ai_chunk.get("node", ""),
                            analysis=ai_chunk.get("analysis", {}),
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        )
                        await websocket.send_json(thinking_response.model_dump())
                    elif ai_chunk.get("type") == "final":
                        ai_response = ai_chunk.get("text", "")
                        break

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
                    role="assistant",
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


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Pre-load dotenv, initialize DB, and log startup."""
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    # Initialize SQLite database tables
    if hasattr(chat_manager.store, "init_db"):
        await chat_manager.store.init_db()  # type: ignore
    logger.info("WABI Chat Web Server started (SQLite DB initialized)")
