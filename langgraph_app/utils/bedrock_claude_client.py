"""AWS Bedrock Claude client wrapper for text, vision, and structured generation."""

import os
import json
import base64
from typing import Optional, Type, TypeVar, List, Any

import boto3
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage

from langgraph_app.config import config as app_config

# Load environment variables
load_dotenv()

T = TypeVar("T", bound=BaseModel)


class BedrockClaudeClient:
    """Client for interacting with Anthropic Claude on AWS Bedrock."""

    def __init__(
        self,
        model_name: str | None = None,
        region_name: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        max_tokens: int | None = None,
        module: Optional[str] = None,
    ):
        if model_name is None:
            model_name = app_config.BEDROCK_CLAUDE_MODEL_NAME

        region = region_name or os.getenv("AWS_REGION") or "us-east-1"

        sampling_params = app_config.get_sampling_params(app_config.LLM_PROVIDER, module)

        self.temperature = temperature if temperature is not None else sampling_params["temperature"]
        self.top_p = top_p if top_p is not None else sampling_params["top_p"]
        self.presence_penalty = (
            presence_penalty if presence_penalty is not None else sampling_params["presence_penalty"]
        )
        self._max_tokens = max_tokens if max_tokens is not None else app_config.BEDROCK_CLAUDE_MAX_TOKENS

        self.model_name = model_name
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=BotoConfig(retries={"max_attempts": 3, "mode": "standard"}),
        )

    def _convert_messages(self, messages: List[AnyMessage], system_prompt: Optional[str] = None) -> tuple[list[dict], list[dict]]:
        """Convert LangChain messages to Bedrock Converse API format.
        Returns:
            system: List of system message dicts
            messages: List of regular message dicts
        """
        bedrock_msgs = []
        system_blocks = []
        
        if system_prompt:
            system_blocks.append({"text": system_prompt})
            
        for msg in messages:
            if isinstance(msg, SystemMessage):
                if isinstance(msg.content, str):
                    system_blocks.append({"text": msg.content})
                continue
                
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            
            content_parts = []
            if isinstance(msg.content, str):
                content_parts.append({"text": msg.content})
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, str):
                        content_parts.append({"text": part})
                    elif isinstance(part, dict):
                        if part.get("type") == "text":
                            content_parts.append({"text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            if "base64," in url:
                                mime_type = url.split(";")[0].split(":")[1]
                                b64_data = url.split("base64,")[1]
                                content_parts.append({
                                    "image": {
                                        "format": mime_type.split("/")[-1],
                                        "source": {"bytes": base64.b64decode(b64_data)}
                                    }
                                })
                        elif part.get("type") == "image" and part.get("source_type") == "base64":
                            b64_data = part.get("data")
                            if b64_data:
                                content_parts.append({
                                    "image": {
                                        "format": "jpeg",
                                        "source": {"bytes": base64.b64decode(b64_data)}
                                    }
                                })
                                
            if content_parts:
                bedrock_msgs.append({"role": role, "content": content_parts})
                
        return system_blocks, bedrock_msgs

    def generate(
        self,
        messages: List[AnyMessage],
        system_prompt: Optional[str] = None,
    ) -> str:
        system_blocks, bedrock_msgs = self._convert_messages(messages, system_prompt)

        kwargs = {
            "modelId": self.model_name,
            "messages": bedrock_msgs,
            "inferenceConfig": {
                "maxTokens": self._max_tokens,
                "temperature": self.temperature,
                "topP": self.top_p,
                "stopSequences": [],
            }
        }
        if system_blocks:
            kwargs["system"] = system_blocks

        response = self.client.converse(**kwargs)

        outputs = response.get("output", {}).get("message", {}).get("content", [])
        texts = [c.get("text", "") for c in outputs if "text" in c]
        return "\n".join(texts).strip()

    def generate_structured(
        self,
        messages: List[AnyMessage],
        schema: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        system_blocks, bedrock_msgs = self._convert_messages(messages, system_prompt)

        kwargs = {
            "modelId": self.model_name,
            "messages": bedrock_msgs,
            "inferenceConfig": {
                "maxTokens": self._max_tokens,
                "temperature": self.temperature,
                "topP": self.top_p,
                "stopSequences": [],
            },
            "additionalModelRequestFields": {
                "response_format": {"type": "json_object"}
            }
        }
        if system_blocks:
            kwargs["system"] = system_blocks

        response = self.client.converse(**kwargs)

        outputs = response.get("output", {}).get("message", {}).get("content", [])
        texts = [c.get("text", "") for c in outputs if "text" in c]
        json_text = "\n".join(texts).strip() or "{}"
        try:
            data = json.loads(json_text)
            return schema(**data)
        except json.JSONDecodeError:
            return schema()
