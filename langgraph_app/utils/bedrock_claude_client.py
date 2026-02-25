"""AWS Bedrock Claude client wrapper for text, vision (base64), and structured generation."""

import os
import json
import base64
from typing import Optional, Type, TypeVar

import boto3
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv
from pydantic import BaseModel

from langgraph_app.config import config as app_config

# Load environment variables
load_dotenv()

T = TypeVar("T", bound=BaseModel)


class BedrockClaudeClient:
    """Client for interacting with Anthropic Claude on AWS Bedrock."""

    def __init__(self, model_name: str | None = None, region_name: str | None = None):
        if model_name is None:
            model_name = app_config.BEDROCK_CLAUDE_MODEL_NAME

        region = region_name or os.getenv("AWS_REGION") or "us-east-1"

        self.model_name = model_name
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=BotoConfig(retries={"max_attempts": 3, "mode": "standard"}),
        )

    def _build_messages(
        self,
        prompt: str,
        image_b64: Optional[str],
        system_instruction: Optional[str],
    ):
        messages: list[dict] = []
        if system_instruction:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_instruction}],
                }
            )

        content_parts: list[dict] = [{"type": "text", "text": prompt}]
        if image_b64:
            content_parts.append(
                {
                    "type": "image",  # Claude 3.5 Bedrock image block
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                }
            )

        messages.append({"role": "user", "content": content_parts})
        return messages

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        messages = self._build_messages(prompt, None, system_instruction)

        response = self.client.converse(
            modelId=self.model_name,
            messages=messages,
            inferenceConfig={"maxTokens": 2048},
        )

        outputs = response.get("output", {}).get("message", {}).get("content", [])
        texts = [c.get("text", "") for c in outputs if c.get("type") == "text"]
        return "\n".join(texts).strip()

    def generate_vision(
        self,
        image_b64: str,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        messages = self._build_messages(prompt, image_b64, system_instruction)

        response = self.client.converse(
            modelId=self.model_name,
            messages=messages,
            inferenceConfig={"maxTokens": 2048},
        )

        outputs = response.get("output", {}).get("message", {}).get("content", [])
        texts = [c.get("text", "") for c in outputs if c.get("type") == "text"]
        return "\n".join(texts).strip()

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        image_b64: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> T:
        messages = self._build_messages(prompt, image_b64, system_instruction)

        # Ask Claude to return JSON only.
        response = self.client.converse(
            modelId=self.model_name,
            messages=messages,
            inferenceConfig={"maxTokens": 2048},
            additionalModelRequestFields={
                "response_format": {"type": "json_object"}
            },
        )

        outputs = response.get("output", {}).get("message", {}).get("content", [])
        texts = [c.get("text", "") for c in outputs if c.get("type") == "text"]
        json_text = "\n".join(texts).strip() or "{}"
        data = json.loads(json_text)
        return schema(**data)
