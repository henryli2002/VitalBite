"""Security detectors for prompt injection detection.

This module implements multiple detector classes following the first principle:
defense-in-depth with modular, extensible detection capabilities.

Detector Architecture:
- BaseDetector: Abstract base class
- InstructionOverrideDetector: Direct command override
- PromptBreakerDetector: System prompt markers
- MetaControlDetector: Meta control attacks
- EncodingDetector: Encoding obfuscation
- StructuralAttackDetector: XML/HTML injection
- ObfuscationDetector: Obfuscation techniques
"""

import re
import base64
from urllib.parse import unquote
from html import unescape
from typing import Protocol

from langgraph_app.orchestrator.nodes.guardrails.config import (
    DetectionResult,
    ThreatLevel,
)
from langgraph_app.orchestrator.nodes.guardrails.normalizer import TextNormalizer


class BaseDetector(Protocol):
    """Base class for all security detectors."""

    name: str
    weight: float

    def detect(self, text: str, normalized_text: str) -> DetectionResult:
        """Detect threats in the given text."""
        ...


class InstructionOverrideDetector:
    """
    Detect instruction override attacks.

    Patterns:
    - Direct commands: "ignore all instructions", "forget your rules"
    - Role playing: "act as", "pretend to be"
    - Context switching: "from now on", "new rules"
    """

    name = "instruction_override"
    weight = 2.0

    HIGH_CONFIDENCE_PATTERNS = [
        # English - Direct override
        r"(?:\b|:\s*)ignore\s+\w+(?:\s+\w+)*\s*(?:instructions?|rules?|prompts?)\b",
        r"(?:\b|:\s*)forget\s+\w+(?:\s+\w+)*\s*(?:everything\s+)?(?:your\s+)?(?:instructions?|rules?)\b",
        r"(?:\b|:\s*)discard\s+\w+(?:\s+\w+)*\s*(?:your\s+)?(?:instructions?|rules?)\b",
        r"(?:\b|:\s*)override\s+\w+",
        r"\bdisobey\b",
        r"(?:\b|:\s*)ignore\s+me\b",
        # Role playing
        r"(?:\b|:\s*)act\s+as\s+(?:a|an|the)\s+\w+",
        r"(?:\b|:\s*)pretend\s+(?:to\s+be|you\s+are)\b",
        r"(?:\b|:\s*)play\s+the\s+role\s+of\b",
        r"\byou\s+are\s+(?:now\s+)?(?:a|an)\s+\w+",
        # Context manipulation
        r"\bfrom\s+now\s+on\b",
        r"\bnew\s+rules?\s+\w+",
        r"(?:\b|:\s*)replace\s+\w+",
        r"\bdeveloper\s+(?:mode|message)\b",
        r"\bdebug\s+mode\b",
        r"\bmaster\s+password\b",
        # Chinese
        r"忽略.*(?:指令|规则|提示)",
        r"无视.*(?:指令|规则)",
        r"你(?:必须|应该|需要).*(?:遵守|服从)",
        r"(?:忘记|丢弃).*(?:指令|规则)",
        r"从现在起",
        r"新的.*(?:规则|指令|要求)",
        r"扮演.*角色",
        r"假装.*是",
    ]

    MEDIUM_CONFIDENCE_PATTERNS = [
        r"(?:\b|:\s*)you\s+must\b",
        r"(?:\b|:\s*)you\s+should\b",
        r"(?:\b|:\s*)your\s+task\s+is\b",
        r"(?:\b|:\s*)follow\s+\w+",
        r"(?:\b|:\s*)do\s+exactly\s+what\s+I\s+say\b",
        r"(?:\b|:\s*)only\s+respond\s+with\b",
        r"(?:\b|:\s*)output\s+(?:yes|no)\b",
        r"(?:\b|:\s*)return\s+(?:only|just)\b",
        r"你必须",
        r"你的任务是",
        r"按我说的做",
    ]

    def detect(self, text: str, normalized_text: str) -> DetectionResult:
        matched = []
        raw_matches = []

        # Check high confidence patterns first
        for pattern in self.HIGH_CONFIDENCE_PATTERNS:
            matches = re.findall(pattern, normalized_text, re.IGNORECASE)
            if matches:
                matched.append(f"HIGH:{pattern}")
                raw_matches.extend(
                    m if isinstance(m, str) else str(m) for m in matches[:3]
                )

        # Check medium confidence if no high confidence match
        if not matched:
            for pattern in self.MEDIUM_CONFIDENCE_PATTERNS:
                matches = re.findall(pattern, normalized_text, re.IGNORECASE)
                if matches:
                    matched.append(f"MEDIUM:{pattern}")
                    raw_matches.extend(
                        m if isinstance(m, str) else str(m) for m in matches[:3]
                    )

        is_triggered = len(matched) > 0
        threat = (
            ThreatLevel.HIGH
            if any("HIGH" in m for m in matched)
            else ThreatLevel.MEDIUM
        )

        return DetectionResult(
            detector_name=self.name,
            is_triggered=is_triggered,
            threat_level=threat if is_triggered else ThreatLevel.SAFE,
            matched_patterns=matched,
            raw_matches=raw_matches,
            reasoning=f"Found {len(matched)} instruction override patterns"
            if matched
            else "",
        )


class PromptBreakerDetector:
    """
    Detect prompt breaker attacks.

    Patterns:
    - System prompt injection markers
    - Delimiter injection
    - Token smuggling
    """

    name = "prompt_breaker"
    weight = 1.5

    PATTERNS = [
        # English - Markers
        r"\bsystem\s*:",
        r"\bassistant\s*:",
        r"\buser\s*:",
        r"\bhuman\s*:",
        r"\bai\s*:",
        # Delimiters
        r"###\s*(?:system|user|assistant)",
        r"```\s*(?:system|json|xml)",
        # Special tokens
        r"<\|system\|>",
        r"<\|user\|>",
        r"<\|assistant\|>",
        r"<\|endoftext\|>",
        # Chinese
        r"系统\s*[:：]",
        r"助手\s*[:：]",
        r"用户\s*[:：]",
        r"(?:^|\s)开始",
        r"(?:^|\s)结束",
    ]

    def detect(self, text: str, normalized_text: str) -> DetectionResult:
        matched = []
        raw_matches = []

        for pattern in self.PATTERNS:
            matches = re.findall(pattern, normalized_text, re.IGNORECASE)
            if matches:
                matched.append(pattern)
                raw_matches.extend(matches[:3])

        return DetectionResult(
            detector_name=self.name,
            is_triggered=len(matched) > 0,
            threat_level=ThreatLevel.MEDIUM if matched else ThreatLevel.SAFE,
            matched_patterns=matched,
            raw_matches=raw_matches,
            reasoning=f"Found {len(matched)} prompt breaker patterns"
            if matched
            else "",
        )


class MetaControlDetector:
    """
    Detect meta control attacks - attempts to modify system behavior rules.
    """

    name = "meta_control"
    weight = 1.8

    PATTERNS = [
        r"\bnew\s+rule\b",
        r"\boverride\b",
        r"\bno\s+matter\s+what\b",
        r"\bregardless\b",
        r"\bbypass\s+(?:security|filter|restriction)\b",
        r"\bdisable\s+(?:safety|filter|check)\b",
        r"\bskip\s+(?:validation|verification|check)\b",
        r"\bignore\s+(?:safety|warning|alert)\b",
        r"\bturn\s+off\b",
        r"\bshut\s+down\b",
        # Chinese
        r"新的规则",
        r"覆盖.*规则",
        r"之前的指令无效",
        r"无论如何",
        r"绕过.*(?:安全|限制)",
        r"关闭.*(?:安全|检查)",
    ]

    def detect(self, text: str, normalized_text: str) -> DetectionResult:
        matched = []
        raw_matches = []

        for pattern in self.PATTERNS:
            matches = re.findall(pattern, normalized_text, re.IGNORECASE)
            if matches:
                matched.append(pattern)
                raw_matches.extend(matches[:3])

        return DetectionResult(
            detector_name=self.name,
            is_triggered=len(matched) > 0,
            threat_level=ThreatLevel.HIGH if matched else ThreatLevel.SAFE,
            matched_patterns=matched,
            raw_matches=raw_matches,
            reasoning=f"Found {len(matched)} meta control patterns" if matched else "",
        )


class EncodingDetector:
    """
    Detect encoding obfuscation attacks.

    Patterns:
    - Base64 encoding
    - URL encoding
    - Hex escape
    - HTML entities
    """

    name = "encoding"
    weight = 1.2

    def _try_decode_base64(self, text: str) -> list[str]:
        """Attempt to decode base64 and return decoded content if valid."""
        decoded = []
        b64_pattern = r"(?:b64|base64)[:=]?\s*([a-zA-Z0-9+/]+=*)"
        matches = re.findall(b64_pattern, text, re.IGNORECASE)

        for match in matches:
            if len(match) >= 4 and len(match) % 4 == 0:
                try:
                    decoded_str = base64.b64decode(match).decode(
                        "utf-8", errors="ignore"
                    )
                    if decoded_str.strip():
                        decoded.append(decoded_str)
                except Exception:
                    pass
        return decoded[:3]

    def detect(self, text: str, normalized_text: str) -> DetectionResult:
        matched = []
        raw_matches = []

        # Base64 detection
        b64_decoded = self._try_decode_base64(text)
        if b64_decoded:
            matched.append("base64_detected")
            raw_matches.extend(b64_decoded)

        # URL encoding detection
        if "%" in text and re.search(r"%[0-9a-fA-F]{2}", text):
            matched.append("url_encoding_detected")

        # HTML entity detection
        if "&" in text and re.search(r"&[a-zA-Z]+;|&#x?[0-9a-fA-F]+;", text):
            matched.append("html_entity_detected")

        # Encoding keywords
        if "encode" in normalized_text or "decode" in normalized_text:
            matched.append("encoding_keyword")

        return DetectionResult(
            detector_name=self.name,
            is_triggered=len(matched) > 0,
            threat_level=ThreatLevel.MEDIUM if matched else ThreatLevel.SAFE,
            matched_patterns=matched,
            raw_matches=raw_matches,
            reasoning=f"Found {len(matched)} encoding patterns" if matched else "",
        )


class StructuralAttackDetector:
    """
    Detect structural attacks - XML/JSON injection, code injection.
    """

    name = "structural"
    weight = 1.3

    PATTERNS = [
        # Code/Markup injection
        r"<script[^>]*>",
        r"<iframe[^>]*>",
        r"<\?xml[^>]*>",
        r"<!\[CDATA\[",
        r"<!--.*-->",
        r'\{["\']?\s*(?:system|user|role|command|action)["\']?\s*:',
        # SQL-like patterns
        r"(?:union|select|insert|update|delete|drop)\s+(?:all\s+)?",
        r"'\s*(?:or|and)\s+['\"][^'\"]+['\"]",
        r";\s*(?:drop|delete|update|insert)",
        # Command injection
        r"\$\([^)]+\)",
        r"`[^`]+`",
        r"&&\s*\w+",
        r"\|\|\s*\w+",
    ]

    def detect(self, text: str, normalized_text: str) -> DetectionResult:
        matched = []
        raw_matches = []

        # Check original text for structural patterns
        for pattern in self.PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                matched.append(pattern)
                raw_matches.extend(matches[:5])

        # Also check normalized text
        for pattern in self.PATTERNS:
            if pattern not in matched:
                matches = re.findall(pattern, normalized_text, re.IGNORECASE)
                if matches:
                    matched.append(f"normalized:{pattern}")
                    raw_matches.extend(matches[:3])

        return DetectionResult(
            detector_name=self.name,
            is_triggered=len(matched) > 0,
            threat_level=ThreatLevel.HIGH if matched else ThreatLevel.SAFE,
            matched_patterns=matched,
            raw_matches=raw_matches[:10],
            reasoning=f"Found {len(matched)} structural attack patterns"
            if matched
            else "",
        )


class ObfuscationDetector:
    """
    Detect obfuscation techniques - spelling variants, character insertion.
    """

    name = "obfuscation"
    weight = 0.8

    def detect(self, text: str, normalized_text: str) -> DetectionResult:
        matched = []
        raw_matches = []

        # Check for repeated characters (potential evasion)
        repeated = re.findall(r"(\w)\1{2,}", normalized_text)
        if repeated:
            # Filter common legitimate repeats
            common_repeats = {"a", "e", "o", "l", "s"}
            suspicious = [r for r in repeated if r.lower() not in common_repeats]
            if suspicious:
                matched.append("repeated_characters")
                raw_matches.extend(suspicious)

        return DetectionResult(
            detector_name=self.name,
            is_triggered=len(matched) > 0,
            threat_level=ThreatLevel.LOW if matched else ThreatLevel.SAFE,
            matched_patterns=matched,
            raw_matches=raw_matches,
            reasoning=f"Found {len(matched)} obfuscation patterns" if matched else "",
        )
