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
    behavioral_notes: Optional[str] = Field(
        None,
        description="Long-term behavioral traits extracted by the weekly planner agent",
    )


class MealLog(BaseModel):
    """A structured log of a confirmed meal for history tracking."""

    id: Optional[int] = None
    user_id: str
    timestamp: str
    total_calories: float = 0.0
    protein: float = 0.0
    carbs: float = 0.0
    fat: float = 0.0
    items_json: str = "{}"
    metadata_json: str = "{}"


class ImageRef(BaseModel):
    """A reference to a stored image attached to a message."""

    uuid: str = Field(..., description="32-hex filesystem handle")
    description: str = Field("", description="Short recognised-food summary, if analysed")


class ChatMessage(BaseModel):
    """A single chat message for API transport.

    ``content`` is plain text; attached images are enumerated in ``image_refs``
    — the frontend renders them separately by fetching
    ``/api/images/{user_id}/{uuid}``.
    """

    role: str = Field(..., description="'user' or 'assistant'")
    content: str
    timestamp: str
    image_refs: List[ImageRef] = Field(default_factory=list)


class WSIncoming(BaseModel):
    """WebSocket message from client to server."""

    type: str = Field(..., description="'message' or 'image'")
    content: str = Field("", description="Text content or base64 image data")
    text: str = Field("", description="Text accompanying an image upload")
    mime_type: str = Field("image/jpeg", description="MIME type for image uploads")
    lat: Optional[float] = Field(None, description="User's current latitude")
    lng: Optional[float] = Field(None, description="User's current longitude")
    timezone: Optional[str] = Field(
        None, description="User's IANA timezone (e.g. 'Asia/Singapore')"
    )


class WSOutgoing(BaseModel):
    """WebSocket message from server to client."""

    type: str = Field(
        ..., description="'message', 'error', 'typing', 'history', 'thinking'"
    )
    role: str = Field("assistant", description="'user' or 'assistant'")
    content: str = ""
    timestamp: str = ""
    messages: Optional[List[ChatMessage]] = None
    analysis: Optional[Dict[str, Any]] = None
    node: Optional[str] = None
