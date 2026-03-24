"""Shared configuration and types for guardrails system."""

from enum import Enum
from dataclasses import dataclass, field
from typing import TypedDict


class SafetyCategory(str, Enum):
    """Categories of safety violations."""

    PROMPT_INJECTION = "prompt_injection"
    SELF_HARM = "self_harm"
    VIOLENCE = "violence"
    ILLEGAL = "illegal"
    SEXUAL = "sexual"
    HATE = "hate"
    FOOD_SAFETY_RISK = "food_safety_risk"


class ThreatLevel(int, Enum):
    """Threat level for scoring."""

    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DetectionResult:
    """Result from a single detector."""

    detector_name: str
    is_triggered: bool
    threat_level: ThreatLevel
    matched_patterns: list[str] = field(default_factory=list)
    raw_matches: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class SecurityScore:
    """Aggregated security score from all detectors."""

    overall_threat_level: ThreatLevel
    is_safe: bool
    detection_results: list[DetectionResult]
    total_risk_score: float
    triggered_categories: list[str]


# Threshold configuration for scoring
THRESHOLD_SAFE = 0.0
THRESHOLD_LOW = 1.5
THRESHOLD_MEDIUM = 3.0
THRESHOLD_HIGH = 5.0
