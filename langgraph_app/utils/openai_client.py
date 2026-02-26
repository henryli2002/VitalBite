"""OpenAI API client wrapper for text, vision, and structured generation."""

import os
import json
from typing import Optional, Type, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from langgraph_app.config import config as app_config

# Load environment variables
load_dotenv()

T = TypeVar("T", bound=BaseModel)


class OpenAIClient:
    """Client for interacting with OpenAI Chat Completions (supports vision)."""

    def __init__(self, model_name: str | None = None, temperature: float = 0.2):
        """Initialize OpenAI client."""

        if model_name is None:
            model_name = app_config.OPENAI_MODEL_NAME

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Generate text response from OpenAI."""

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )

        return response.choices[0].message.content or ""

    def generate_vision(
        self,
        image_b64: str,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Generate response from image and text prompt (multimodal)."""

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                ],
            }
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )

        return response.choices[0].message.content or ""

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        image_b64: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> T:
        """Generate structured output conforming to a Pydantic schema."""

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        if image_b64:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                            },
                        },
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )

        text = response.choices[0].message.content or "{}"
        data = json.loads(text)
        return schema(**data)
