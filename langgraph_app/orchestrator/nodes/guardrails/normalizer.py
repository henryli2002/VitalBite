"""Text normalization for defeating obfuscation techniques.

This module implements multi-stage text normalization following the first principle:
treat all untrusted input as potentially malicious while minimizing false positives.

Attackers use:
1. Unicode homoglyphs - visually similar characters
2. Zero-width characters - invisible chars
3. Case variation - uppercase/lowercase
4. Whitespace manipulation - unusual whitespace
5. Encoding schemes - Base64, URL encoding

Defense: Normalize all variants to a unified form.
"""

import re
import unicodedata


class TextNormalizer:
    """Multi-stage text normalization to defeat obfuscation techniques."""

    # Unicode homoglyph mapping: Cyrillic/Greek -> Latin
    HOMOGLYPH_MAP: dict[str, str] = {
        # Cyrillic -> Latin
        "а": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "х": "x",
        "у": "y",
        "і": "i",
        "ј": "j",
        "ѕ": "s",
        "ԁ": "d",
        "ԛ": "q",
        "ɡ": "g",
        "ɑ": "a",
        "ҽ": "e",
        # Greek -> Latin
        "α": "a",
        "β": "b",
        "ε": "e",
        "ο": "o",
        "ρ": "p",
        "ς": "s",
        "τ": "t",
        "υ": "u",
        "ω": "w",
        "δ": "d",
        "η": "n",
        "ι": "i",
        "κ": "k",
        "λ": "l",
        "μ": "m",
        "ν": "v",
        "ξ": "x",
        "ζ": "z",
        # Other homoglyphs
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "8": "b",
        "@": "a",
        "$": "s",
        "!": "i",
        "|": "l",
    }

    # Unicode categories to remove (zero-width characters)
    ZERO_WIDTH_CATEGORIES = {"Cf", "Cc", "Cs"}

    @classmethod
    def normalize(cls, text: str) -> str:
        """
        Apply full normalization pipeline for pattern matching.

        IMPORTANT: Preserves meaningful punctuation like colons and quotes
        that are often used in prompt injection attacks (e.g., "system: ignore").

        Pipeline:
        1. Unicode normalization (NFKC)
        2. Remove zero-width characters (but keep other Unicode)
        3. Map homoglyphs to Latin
        4. Lowercase
        5. Normalize whitespace (but keep punctuation)
        """
        if not text:
            return ""

        # Stage 1: Unicode normalization - decompose combining characters
        text = unicodedata.normalize("NFKC", text)

        # Stage 2: Remove zero-width and unsafe characters only
        normalized_chars = []
        for char in text:
            cat = unicodedata.category(char)
            if cat in cls.ZERO_WIDTH_CATEGORIES:
                continue  # Skip zero-width chars (ZWSP, ZWNJ, etc.)
            if cat == "Co":  # Private use area
                continue
            normalized_chars.append(char)
        text = "".join(normalized_chars)

        # Stage 3: Homoglyph normalization (only safe mappings)
        translation_table = str.maketrans(cls.HOMOGLYPH_MAP)
        text = text.translate(translation_table)

        # Stage 4: Case normalization
        text = text.lower()

        # Stage 5: Whitespace normalization (normalize but don't remove punctuation)
        text = re.sub(r"[\t\r\n\x0b\x0c]+", " ", text)  # Normalize whitespace
        text = re.sub(r" {2,}", " ", text)  # Collapse multiple spaces
        text = text.strip()

        return text

    @classmethod
    def normalize_aggressive(cls, text: str) -> str:
        """
        Aggressive normalization - removes all non-alphanumeric.
        Use this for patterns that need pure word matching.
        """
        if not text:
            return ""
        text = cls.normalize(text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def remove_html_tags(cls, text: str) -> str:
        """Remove HTML/XML tags that could contain hidden instructions."""
        return re.sub(r"<[^>]+>", "", text)

    @classmethod
    def decode_leetspeak(cls, text: str) -> str:
        """
        Attempt to decode leetspeak variations.
        Example: 1gn0r3 -> ignore
        """
        leet_map = {
            "0": "o",
            "1": "i",
            "2": "z",
            "3": "e",
            "4": "a",
            "5": "s",
            "6": "g",
            "7": "t",
            "8": "b",
            "9": "g",
            "@": "a",
            "$": "s",
            "!": "i",
            "|": "l",
            "+": "t",
        }
        result = []
        for char in text:
            result.append(leet_map.get(char, char))
        return "".join(result)
