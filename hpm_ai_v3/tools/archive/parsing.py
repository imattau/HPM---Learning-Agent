"""
parsing.py - Minimal NLP substrate for HPM agents.
Only provides tokenization, POS tagging, and dependency parsing.
No hardcoded concept mappings or domain-specific extraction.
"""

import warnings
from typing import Dict, Any, List, Optional

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False
    nlp = None
    warnings.warn("spaCy not installed. Using regex fallback for basic tokenization.")

from .registry import ToolRegistry


def tokenize(text: str) -> Dict[str, Any]:
    """Return tokens with basic metadata (text, POS, dependency)."""
    if SPACY_AVAILABLE and nlp:
        doc = nlp(text)
        tokens = [{
            "text": t.text,
            "pos": t.pos_,
            "dep": t.dep_,
            "lemma": t.lemma_,
            "is_stop": t.is_stop
        } for t in doc]
        return {"tokens": tokens, "status": "success"}
    else:
        # Simple regex fallback
        import re
        words = re.findall(r'\b\w+\b', text)
        tokens = [{"text": w, "pos": "UNKNOWN", "dep": None, "lemma": w, "is_stop": False} for w in words]
        return {"tokens": tokens, "status": "success", "fallback": True}


def extract_numbers(text: str) -> Dict[str, Any]:
    """Extract numeric values using regex only. No unit normalization or concept mapping."""
    import re
    pattern = re.compile(r'([\d\.]+)\s*([a-zA-Z/\^2\*]+)?')
    results = []
    for match in pattern.finditer(text):
        try:
            value = float(match.group(1))
            unit = match.group(2) if match.group(2) else ""
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end].strip()
            results.append({"value": value, "unit": unit, "context": context})
        except ValueError:
            continue
    return {"numbers": results, "status": "success"}


def register_parsing_tools():
    ToolRegistry.register(
        name="tokenize",
        tool_fn=tokenize,
        input_keys=["text"],
        output_key="tokens",
        cost=0.005,
        description="Tokenize text and return POS/dependency tags (spaCy if available)."
    )
    ToolRegistry.register(
        name="extract_numbers",
        tool_fn=extract_numbers,
        input_keys=["text"],
        output_key="numbers",
        cost=0.005,
        description="Extract numeric values and adjacent unit strings from text."
    )
    print("[ParsingTools] Registered minimal NLP substrate tools.")


register_parsing_tools()
