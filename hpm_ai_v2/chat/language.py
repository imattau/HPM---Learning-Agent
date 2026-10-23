"""Question wording shared by the planner and the evidence selector."""
from __future__ import annotations

from hpm_ai_v2.agents.text_structure import content_stems, stem

# Words that ask rather than say: a question's subject matter is what is left once they are removed.
QUESTION_WORDS = frozenset(stem(w) for w in (
    "how", "what", "why", "when", "where", "which", "who", "whom", "whose",
    "explain", "describe", "tell", "show",
))


def content_terms(text: str) -> set[str]:
    """Content-word stems of a question, without the question words themselves."""
    return {t for t in content_stems(text) if t not in QUESTION_WORDS}
