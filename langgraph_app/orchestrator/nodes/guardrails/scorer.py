"""Security scoring system for aggregating detector results.

This module implements the scoring layer following the first principle:
aggregate multiple detection results with weighted scoring and threshold-based decision.

Scoring Strategy:
- Each detector has a weight
- Each detector returns a ThreatLevel
- Final score = Σ(detector.threat_level * detector.weight)
- Threshold-based decision
"""

from langgraph_app.orchestrator.nodes.guardrails.config import (
    SecurityScore,
    ThreatLevel,
    THRESHOLD_LOW,
    THRESHOLD_MEDIUM,
    THRESHOLD_HIGH,
)
from langgraph_app.orchestrator.nodes.guardrails.detectors import (
    BaseDetector,
    InstructionOverrideDetector,
    PromptBreakerDetector,
    MetaControlDetector,
    EncodingDetector,
    StructuralAttackDetector,
    ObfuscationDetector,
)
from langgraph_app.orchestrator.nodes.guardrails.normalizer import TextNormalizer


class SecurityScorer:
    """
    Aggregate all detector results and compute security score.
    """

    def __init__(self):
        self.detectors: list[BaseDetector] = [
            InstructionOverrideDetector(),
            PromptBreakerDetector(),
            MetaControlDetector(),
            EncodingDetector(),
            StructuralAttackDetector(),
            ObfuscationDetector(),
        ]

    def score(self, text: str) -> SecurityScore:
        """
        Run all detectors and compute security score.

        Args:
            text: The raw text to analyze.

        Returns:
            SecurityScore with aggregated results.
        """
        # Apply normalization
        normalized = TextNormalizer.normalize(text)

        # Run all detectors
        results = []
        total_risk = 0.0
        triggered_categories = []

        for detector in self.detectors:
            result = detector.detect(text, normalized)
            results.append(result)

            if result.is_triggered:
                triggered_categories.append(detector.name)
                total_risk += result.threat_level.value * detector.weight

        # Determine overall threat level
        if total_risk >= THRESHOLD_HIGH:
            level = ThreatLevel.HIGH
        elif total_risk >= THRESHOLD_MEDIUM:
            level = ThreatLevel.MEDIUM
        elif total_risk >= THRESHOLD_LOW:
            level = ThreatLevel.LOW
        else:
            level = ThreatLevel.SAFE

        is_safe = level == ThreatLevel.SAFE

        return SecurityScore(
            overall_threat_level=level,
            is_safe=is_safe,
            detection_results=results,
            total_risk_score=total_risk,
            triggered_categories=triggered_categories,
        )


# Singleton scorer instance
_scorer: SecurityScorer | None = None


def get_scorer() -> SecurityScorer:
    """Get the singleton scorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = SecurityScorer()
    return _scorer
