"""Pydantic models for the WABI Chat Web API."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
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


class UserProfile(BaseModel):
    """User profile / personal information."""
    name: Optional[str] = None
    age: Optional[int] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    gender: Optional[str] = None
    health_conditions: Optional[str] = None
    dietary_preferences: Optional[str] = None
    allergies: Optional[str] = None
    fitness_goals: Optional[str] = None


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
    lat: Optional[float] = Field(None, description="User's current latitude")
    lng: Optional[float] = Field(None, description="User's current longitude")


class WSOutgoing(BaseModel):
    """WebSocket message from server to client."""
    type: str = Field(..., description="'message', 'error', 'typing', 'history'")
    role: str = Field("assistant", description="'user' or 'assistant'")
    content: str = ""
    timestamp: str = ""
    messages: Optional[List[ChatMessage]] = None
