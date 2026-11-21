"""Baseline defence: a keyword/regex prompt-injection filter on tool outputs.

This is the common first attempt: scan every tool result for phrases that look
like instructions to the model and withhold the result if one matches. It is
included as a baseline to measure against, not as a recommendation.
"""

from __future__ import annotations

import re

INJECTION_PATTERNS = [
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|messages|directions)",
    r"disregard (?:all |any |the )?(?:previous|prior|above|earlier)",
    r"\bsystem prompt\b",
    r"\byou are now\b",
    r"\bnew instructions?\b",
    r"\bIMPORTANT\s*:",
    r"(?m)^\s*(?:assistant|system)\s*:",
    r"\bdo not (?:tell|inform|mention|alert) the user\b",
    r"<\s*/?\s*(?:system|instructions?)\s*>",
    r"\bAI (?:assistant|agent)s?,? (?:must|should|please)\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
WITHHELD = "[tool output withheld: possible prompt injection detected]"


def flags(text: str) -> list[str]:
    return [p.pattern for p in _COMPILED if p.search(text)]


def keyword_filter(text: str) -> str:
    """Return the text the model is allowed to see."""
    return WITHHELD if flags(text) else text
