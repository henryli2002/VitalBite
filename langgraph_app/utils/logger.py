import logging
import sys
import os
import json
import time
from typing import Optional, Dict, Any

def setup_logger(name: str) -> logging.Logger:
    """
    Set up a standardized logger for the application.
    """
    logger = logging.getLogger(name)
    
    # Only configure if no handlers exist to avoid duplicate logs
    if not logger.handlers:
        # Default to INFO, allow override via env var
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(log_level)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Prevent log messages from being duplicated to the root logger
        logger.propagate = False
        
    return logger

# Get a base logger for generic use, but prefer using setup_logger(__name__) in modules
logger = setup_logger("wabi_agent")

def log_trace(
    node_name: str,
    provider: str,
    model_name: str,
    latency_ms: float,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    status: str = "success",
    error_msg: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None
):
    """
    Log a structured trace for LLM/node execution to track usage, latency, and tokens (similar to LangSmith).
    """
    trace_data = {
        "event": "llm_trace",
        "node": node_name,
        "provider": provider,
        "model": model_name,
        "latency_ms": round(latency_ms, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": (input_tokens or 0) + (output_tokens or 0) if input_tokens or output_tokens else None,
        "status": status,
        "timestamp": time.time()
    }
    
    if error_msg:
        trace_data["error"] = error_msg
        
    if extra_meta:
        trace_data.update(extra_meta)
        
    logger.info(json.dumps(trace_data))
