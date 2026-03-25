import time
from typing import Any, Dict, Optional, List
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.callbacks import BaseCallbackHandler

from langgraph_app.utils.logger import request_id_var, log_trace
from langgraph_app.config import config


def _get_model_name(llm: BaseChatModel) -> Optional[str]:
    if hasattr(llm, "model"):
        return llm.model
    if hasattr(llm, "model_name"):
        return llm.model_name
    if hasattr(llm, "_model"):
        return getattr(llm._model, "name", None)
    return None


def _get_provider(llm: BaseChatModel) -> str:
    model_name = _get_model_name(llm) or ""
    if "gemini" in model_name.lower():
        return "gemini"
    if "claude" in model_name.lower() or "bedrock" in model_name.lower():
        return "claude"
    if "gpt" in model_name.lower() or "openai" in model_name.lower():
        return "openai"
    return "unknown"


class _TrackedStructuredOutput:
    """Wrapper that tracks token usage for with_structured_output calls.

    LangChain's with_structured_output(include_raw=True) returns a dict with
    'raw' (AIMessage), 'parsed' (Pydantic model), and 'parsing_error'.
    This wrapper extracts usage from 'raw' for logging and returns only 'parsed'.
    """

    def __init__(self, raw_runnable: Any, node_name: str, underlying_llm: BaseChatModel):
        self._runnable = raw_runnable
        self._node_name = node_name
        self._underlying_llm = underlying_llm

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        start_time = time.perf_counter()

        try:
            raw_result = self._runnable.invoke(input, config=config, **kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract token usage from the raw AIMessage
            if isinstance(raw_result, dict):
                raw_msg = raw_result.get("raw")
                if raw_msg and hasattr(raw_msg, "usage_metadata") and raw_msg.usage_metadata:
                    usage = raw_msg.usage_metadata
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    reasoning_tokens = usage.get("output_token_details", {}).get("reasoning", 0)
                    cache_tokens = usage.get("input_token_details", {}).get("cache_read", 0)

                    log_trace(
                        node_name=self._node_name,
                        provider=_get_provider(self._underlying_llm),
                        model_name=_get_model_name(self._underlying_llm) or "unknown",
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

                # Return only the parsed result (Pydantic model)
                parsed = raw_result.get("parsed")
                if raw_result.get("parsing_error"):
                    raise raw_result["parsing_error"]
                return parsed
            else:
                # Fallback: if result is not a dict, return as-is
                return raw_result

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            log_trace(
                node_name=self._node_name,
                provider=_get_provider(self._underlying_llm),
                model_name=_get_model_name(self._underlying_llm) or "unknown",
                latency_ms=latency_ms,
                status="error",
                error_msg=str(e),
                extra_meta={"request_id": request_id_var.get()},
            )
            raise


class TrackedChatModel(BaseChatModel):
    def __init__(self, llm: BaseChatModel, node_name: str = "unknown", **kwargs):
        super().__init__(**kwargs)
        self._llm = llm
        self._node_name = node_name

    @property
    def _underlying_llm(self) -> BaseChatModel:
        return self._llm

    @property
    def node_name(self) -> str:
        return self._node_name

    @property
    def _llm_type(self) -> str:
        return f"tracked_{self._llm._llm_type}"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResult:
        start_time = time.perf_counter()

        try:
            result = self._llm._generate(messages, stop=stop, **kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000

            self._log_usage(result, latency_ms)
            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            log_trace(
                node_name=self._node_name,
                provider=_get_provider(self._llm),
                model_name=_get_model_name(self._llm) or "unknown",
                latency_ms=latency_ms,
                status="error",
                error_msg=str(e),
                extra_meta={"request_id": request_id_var.get()},
            )
            raise

    def _log_usage(self, result: LLMResult, latency_ms: float):
        usage_metadata = None
        model_name = _get_model_name(self._llm) or "unknown"

        # Extract usage_metadata from ChatGeneration.message
        for generation in result.generations:
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

        log_trace(
            node_name=self._node_name,
            provider=_get_provider(self._llm),
            model_name=model_name,
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

    def bind(self, **kwargs: Any) -> "TrackedChatModel":
        return TrackedChatModel(self._llm.bind(**kwargs), self._node_name)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        # Use include_raw=True so we can extract usage_metadata from the raw AIMessage
        # then return only the parsed result to the caller
        kwargs.pop("include_raw", None)  # Remove if caller passed it
        raw_structured = self._llm.with_structured_output(schema, include_raw=True, **kwargs)
        return _TrackedStructuredOutput(raw_structured, self._node_name, self._llm)

    def invoke(
        self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Any:
        start_time = time.perf_counter()

        try:
            result = self._llm.invoke(input, config=config, **kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000

            self._log_invoke_usage(result, latency_ms)
            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            log_trace(
                node_name=self._node_name,
                provider=_get_provider(self._llm),
                model_name=_get_model_name(self._llm) or "unknown",
                latency_ms=latency_ms,
                status="error",
                error_msg=str(e),
                extra_meta={"request_id": request_id_var.get()},
            )
            raise

    def _log_invoke_usage(self, result: Any, latency_ms: float):
        if hasattr(result, "usage_metadata") and result.usage_metadata:
            usage = result.usage_metadata
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            reasoning_tokens = usage.get("output_token_details", {}).get("reasoning", 0)
            cache_tokens = usage.get("input_token_details", {}).get("cache_read", 0)

            log_trace(
                node_name=self._node_name,
                provider=_get_provider(self._llm),
                model_name=_get_model_name(self._llm) or "unknown",
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
        elif isinstance(result, dict) and "raw" in result:
            raw = result["raw"]
            if hasattr(raw, "usage_metadata") and raw.usage_metadata:
                usage = raw.usage_metadata
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                reasoning_tokens = usage.get("output_token_details", {}).get(
                    "reasoning", 0
                )
                cache_tokens = usage.get("input_token_details", {}).get("cache_read", 0)

                log_trace(
                    node_name=self._node_name,
                    provider=_get_provider(self._llm),
                    model_name=_get_model_name(self._llm) or "unknown",
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

    @property
    def model_name(self) -> str:
        return _get_model_name(self._llm) or "unknown"

    def get_model_names(self) -> List[str]:
        return (
            self._llm.get_model_names()
            if hasattr(self._llm, "get_model_names")
            else [self.model_name]
        )

    def _llm_config(self) -> Dict[str, Any]:
        return getattr(self._llm, "_llm_config", {})


def get_tracked_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    module: Optional[str] = None,
    node_name: Optional[str] = None,
) -> TrackedChatModel:
    from langgraph_app.utils.llm_factory import get_llm_client

    llm = get_llm_client(provider=provider, model_name=model_name, module=module)
    return TrackedChatModel(llm, node_name=node_name or module or "unknown")
