"""Gemini API client wrapper for text, vision, and structured generation."""

import os
import json
import base64
import io
from typing import Optional, Type, TypeVar, List, Any
from pydantic import BaseModel
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
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
    
    def generate_text(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generate text response from Gemini.
        
        Args:
            prompt: User prompt
            system_instruction: Optional system instruction
            
        Returns:
            Generated text response
        """
        if system_instruction:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        else:
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=self.top_p,
            )
            
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        return response.text or ""
    
    def generate_vision(
        self, 
        images_b64: List[str], 
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generate response from image and text prompt (multimodal).
        
        Args:
            images_b64: List of Base64 encoded image strings
            prompt: Text prompt describing what to do with the image
            system_instruction: Optional system instruction
            
        Returns:
            Generated text response
        """
        
        contents: List[Any] = [prompt]
        for image_b64 in images_b64:
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
            contents.append(image)

        if system_instruction:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        else:
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=self.top_p,
            )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        return response.text or ""
    
    def generate_structured(
        self,
        prompt: str, 
        schema: Type[T],
        images_b64: Optional[List[str]] = None,
        system_instruction: Optional[str] = None
    ) -> T:
        """
        Generate structured output conforming to a Pydantic schema.
        
        Args:
            prompt: User prompt
            schema: Pydantic BaseModel class to structure the output
            images_b64: Optional list of Base64 encoded image strings
            system_instruction: Optional system instruction
            
        Returns:
            Instance of schema class with generated data
        """
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            system_instruction=system_instruction,
            temperature=self.temperature,
        )
        
        contents: List[Any] = [prompt]
        if images_b64:
            for image_b64 in images_b64:
                image_data = base64.b64decode(image_b64)
                image = Image.open(io.BytesIO(image_data))
                contents.append(image)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        
        # New SDK might support direct parsing, but to be safe and consistent
        # with previous behavior, we handle the text response.
        # However, passing response_schema usually ensures JSON structure.
        
        # Try to use parsed response if available (some SDK versions support this)
        # Otherwise parse text.
        try:
            if hasattr(response, 'parsed') and response.parsed is not None:
                # If the SDK automatically parses it into the Pydantic model or dict
                if isinstance(response.parsed, schema):
                    return response.parsed
                elif isinstance(response.parsed, dict):
                    return schema(**response.parsed)
        except Exception:
            pass # Fallback to text parsing

        json_text = response.text or "{}"
        data = json.loads(json_text)
        return schema(**data)
