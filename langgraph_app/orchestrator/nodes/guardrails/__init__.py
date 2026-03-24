"""Guardrails system for prompt injection and content safety detection.

This package implements a defense-in-depth approach to safety detection following
the first principle: treat all untrusted input as potentially malicious.

Architecture:
- config.py: Shared types and constants
- normalizer.py: Text normalization for defeating obfuscation
- detectors.py: Multiple detector implementations
- scorer.py: Weighted scoring aggregation
- responses.py: Localized response templates
- nodes.py: Public LangGraph node functions

Quick Start:
    from langgraph_app.orchestrator.nodes.guardrails import (
        input_guardrail_node,
        output_guardrail_node,
    )

Testing:
    from langgraph_app.orchestrator.nodes.guardrails import test_normalization, test_detection
"""

from langgraph_app.orchestrator.nodes.guardrails.nodes import (
    input_guardrail_node,
    output_guardrail_node,
)
from langgraph_app.orchestrator.nodes.guardrails.normalizer import TextNormalizer
from langgraph_app.orchestrator.nodes.guardrails.scorer import (
    get_scorer,
    SecurityScorer,
)
from langgraph_app.orchestrator.nodes.guardrails.config import (
    SafetyCategory,
    ThreatLevel,
    DetectionResult,
    SecurityScore,
)

# Utility functions for testing
from langgraph_app.orchestrator.nodes.guardrails.normalizer import (
    TextNormalizer as test_normalization,
)
from langgraph_app.orchestrator.nodes.guardrails.scorer import (
    get_scorer as test_detection,
)


__all__ = [
    # Nodes
    "input_guardrail_node",
    "output_guardrail_node",
    # Core classes
    "TextNormalizer",
    "SecurityScorer",
    # Types
    "SafetyCategory",
    "ThreatLevel",
    "DetectionResult",
    "SecurityScore",
    # Testing utilities
    "test_normalization",
    "test_detection",
]
