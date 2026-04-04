import time
import logging
from typing import Optional, Dict, Any, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from langgraph_app.utils.logger import request_id_var, logger, log_trace


class TokenUsageCallbackHandler(BaseCallbackHandler):
    def __init__(self, node_name: str = "unknown"):
        super().__init__()
        self.node_name = node_name
        self.start_time: Optional[float] = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.calls: List[Dict[str, Any]] = []

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        self.start_time = time.perf_counter()

    def on_llm_end(self, response: LLMResult, **kwargs):
        if self.start_time:
            latency_ms = (time.perf_counter() - self.start_time) * 1000
        else:
            latency_ms = 0

        usage_metadata = None

        # Extract usage_metadata from ChatGeneration.message
        for generation in response.generations:
            for gen in generation:
                if hasattr(gen, "message") and gen.message:
                    msg = gen.message
                    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                        usage_metadata = msg.usage_metadata
                        break
            if usage_metadata:
                break

        input_tokens = usage_metadata.get("input_tokens", 0) if usage_metadata else 0
        output_tokens = usage_metadata.get("output_tokens", 0) if usage_metadata else 0
        total_tokens = usage_metadata.get("total_tokens", 0) if usage_metadata else 0

        reasoning_tokens = 0
        if usage_metadata and "output_token_details" in usage_metadata:
            reasoning_tokens = usage_metadata["output_token_details"].get(
                "reasoning", 0
            )

        cache_tokens = 0
        if usage_metadata and "input_token_details" in usage_metadata:
            cache_tokens = usage_metadata["input_token_details"].get("cache_read", 0)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += total_tokens

        call_info = {
            "node": self.node_name,
            "latency_ms": round(latency_ms, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cache_tokens": cache_tokens,
        }
        self.calls.append(call_info)

        log_trace(
            node_name=self.node_name,
            provider="gemini",
            model_name=self._get_model_name(response),
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            status="success",
            extra_meta={
                "reasoning_tokens": reasoning_tokens,
                "cache_tokens": cache_tokens,
                "request_id": request_id_var.get(),
            },
        )

        self.start_time = None

    def on_llm_error(self, error: Exception, **kwargs):
        if self.start_time:
            latency_ms = (time.perf_counter() - self.start_time) * 1000
        else:
            latency_ms = 0

        log_trace(
            node_name=self.node_name,
            provider="gemini",
            model_name="unknown",
            latency_ms=latency_ms,
            status="error",
            error_msg=str(error),
            extra_meta={"request_id": request_id_var.get()},
        )

    def _get_model_name(self, response: LLMResult) -> str:
        # Extract model name from ChatGeneration.message.response_metadata
        for generation in response.generations:
            for gen in generation:
                if hasattr(gen, "message") and gen.message:
                    msg = gen.message
                    if hasattr(msg, "response_metadata"):
                        meta = msg.response_metadata
                        if isinstance(meta, dict):
                            return meta.get("model_name", "unknown")

        return "unknown"

    def get_usage(self) -> Dict[str, Any]:
        return {
            "node": self.node_name,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }

    def reset(self):
        self.start_time = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.calls = []


def create_callback_handler(node_name: str) -> TokenUsageCallbackHandler:
    return TokenUsageCallbackHandler(node_name=node_name)
