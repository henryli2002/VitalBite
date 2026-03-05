"""OpenAI API client wrapper for text, vision, and structured generation."""

import os
import json
from typing import Optional, Type, TypeVar, List, Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage

from langgraph_app.config import config as app_config

# Load environment variables
load_dotenv()

T = TypeVar("T", bound=BaseModel)


class OpenAIClient:
    """Client for interacting with OpenAI Chat Completions (supports vision)."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        module: Optional[str] = None,
    ):
        """Initialize OpenAI client."""

        if model_name is None:
            model_name = app_config.OPENAI_MODEL_NAME

        sampling_params = app_config.get_sampling_params(app_config.LLM_PROVIDER, module)

        self.temperature = temperature if temperature is not None else sampling_params["temperature"]
        self.top_p = top_p if top_p is not None else sampling_params["top_p"]
        self.presence_penalty = (
            presence_penalty if presence_penalty is not None else sampling_params["presence_penalty"]
        )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def _convert_messages(self, messages: List[AnyMessage], system_prompt: Optional[str] = None) -> List[Any]:
        """Convert LangChain messages to OpenAI format."""
        openai_msgs: List[Any] = []
        if system_prompt:
            openai_msgs.append({"role": "system", "content": system_prompt})
            
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else ("system" if isinstance(msg, SystemMessage) else "assistant")
            
            # LangChain's content format is already highly compatible with OpenAI
            if isinstance(msg.content, str):
                openai_msgs.append({"role": role, "content": msg.content})
            elif isinstance(msg.content, list):
                # We map `image_url` and `text` parts directly
                formatted_content = []
                for part in msg.content:
                    if isinstance(part, str):
                        formatted_content.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        if part.get("type") == "text":
                            formatted_content.append({"type": "text", "text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            # OpenAI expects {"type": "image_url", "image_url": {"url": "..."}}
                            formatted_content.append({
                                "type": "image_url",
                                "image_url": part.get("image_url", {})
                            })
                        elif part.get("type") == "image" and part.get("source_type") == "base64":
                            # Convert legacy internal format to standard image_url
                            b64_data = part.get("data", "")
                            formatted_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
                            })
                openai_msgs.append({"role": role, "content": formatted_content})
                
        return openai_msgs

    def generate(
        self,
        messages: List[AnyMessage],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text response from OpenAI."""
        openai_msgs = self._convert_messages(messages, system_prompt)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_msgs,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
        )

        return response.choices[0].message.content or ""

    def generate_structured(
        self,
        messages: List[AnyMessage],
        schema: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """Generate structured output conforming to a Pydantic schema."""
        openai_msgs = self._convert_messages(messages, system_prompt)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_msgs,
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )

        text = response.choices[0].message.content or "{}"
        try:
            data = json.loads(text)
            return schema(**data)
        except json.JSONDecodeError:
            return schema() # Handle empty or malformed
