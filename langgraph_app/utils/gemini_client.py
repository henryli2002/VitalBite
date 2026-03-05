"""Gemini API client wrapper for text, vision, and structured generation."""

import os
import json
from typing import Optional, Type, TypeVar, List, Any
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage

from langgraph_app.config import config as app_config

# Load environment variables
load_dotenv()

T = TypeVar('T', bound=BaseModel)


class GeminiClient:
    """Client for interacting with Google Gemini API using the new google-genai SDK."""
    
    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        module: Optional[str] = None,
    ):
        """
        Initialize Gemini client.
        
        Args:
            model_name: Name of the Gemini model to use (defaults to config value)
        """
        if model_name is None:
            model_name = app_config.GEMINI_MODEL_NAME

        sampling_params = app_config.get_sampling_params(app_config.LLM_PROVIDER, module)

        self.temperature = temperature if temperature is not None else sampling_params["temperature"]
        self.top_p = top_p if top_p is not None else sampling_params["top_p"]
        self.presence_penalty = (
            presence_penalty if presence_penalty is not None else sampling_params["presence_penalty"]
        )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _convert_messages(self, messages: List[AnyMessage]) -> List[dict]:
        """Convert LangChain messages to Gemini format."""
        gemini_contents = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "model"
            # Ignore SystemMessage here, we pass it via config
            if isinstance(msg, SystemMessage):
                continue
                
            parts = []
            if isinstance(msg.content, str):
                parts.append({"text": msg.content})
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, str):
                        parts.append({"text": part})
                    elif isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append({"text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            image_url = part.get("image_url", {}).get("url", "")
                            if "base64," in image_url:
                                mime_type = image_url.split(";")[0].split(":")[1]
                                b64_data = image_url.split("base64,")[1]
                                parts.append({
                                    "inline_data": {
                                        "data": b64_data,
                                        "mime_type": mime_type
                                    }
                                })
                        elif part.get("type") == "image" and part.get("source_type") == "base64":
                            b64_data = part.get("data")
                            if b64_data:
                                parts.append({
                                    "inline_data": {
                                        "data": b64_data,
                                        "mime_type": "image/jpeg"
                                    }
                                })
            
            gemini_contents.append({"role": role, "parts": parts})
            
        return gemini_contents

    def generate(
        self, 
        messages: List[AnyMessage], 
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text response from Gemini."""
        contents = self._convert_messages(messages)
        
        config_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
            
        config = types.GenerateContentConfig(**config_kwargs) # type: ignore
            
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents, # type: ignore
            config=config,
        )
        return response.text or ""

    def generate_structured(
        self,
        messages: List[AnyMessage],
        schema: Type[T],
        system_prompt: Optional[str] = None
    ) -> T:
        """Generate structured output from a list of messages."""
        contents = self._convert_messages(messages)
        
        config_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": self.temperature,
        }
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
            
        config = types.GenerateContentConfig(**config_kwargs) # type: ignore

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents, # type: ignore
            config=config,
        )
        
        try:
            if hasattr(response, 'parsed') and response.parsed is not None:
                if isinstance(response.parsed, schema):
                    return response.parsed
                elif isinstance(response.parsed, dict):
                    return schema(**response.parsed)
        except Exception:
            pass # Fallback to text parsing

        json_text = response.text or "{}"
        try:
            data = json.loads(json_text)
            return schema(**data)
        except json.JSONDecodeError:
            return schema()
