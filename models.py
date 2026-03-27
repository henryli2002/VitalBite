"""Pydantic models for the WABI Chat Web API."""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class UserCreate(BaseModel):
    """Request model for creating a new user/conversation."""
    name: Optional[str] = Field(None, description="Display name for the user")


class UserInfo(BaseModel):
    """Response model for user information."""
    user_id: str
    name: str
    created_at: str
    last_active: str
    message_count: int = 0 


class ChatMessage(BaseModel):
    """A single chat message for API transport."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str
    timestamp: str
    has_image: bool = False


class WSIncoming(BaseModel):
    """WebSocket message from client to server."""
    type: str = Field(..., description="'message' or 'image'")
    content: str = Field("", description="Text content or base64 image data")
    text: str = Field("", description="Text accompanying an image upload")
    mime_type: str = Field("image/jpeg", description="MIME type for image uploads")


class WSOutgoing(BaseModel):
    """WebSocket message from server to client."""
    type: str = Field(..., description="'message', 'error', 'typing', 'history'")
    role: str = Field("assistant", description="'user' or 'assistant'")
    content: str = ""
    timestamp: str = ""
    messages: Optional[List[ChatMessage]] = None
