# hpm_ai_v3/tools/tool_selector.py
"""
ToolSelector - Uses LM embeddings to bias pattern selection toward
semantically relevant tools. Acts as a pattern evaluator in HPM terms.
"""
import numpy as np
import urllib.parse
from typing import Any, Dict, List, Optional

TOOL_DESCRIPTIONS = {
    "textblob.TextBlob": "sentiment analysis polarity positive negative opinion text",
    "re.findall": "extract pattern match numbers regex search text",
    "re.search": "find pattern match regex text search",
    "builtins.str.split": "split words tokenize count whitespace text",
    "builtins.str.lower": "lowercase convert string text",
    "builtins.str.upper": "uppercase convert string text",
    "spacy.nlp": "entity noun parse sentence structure named entity",
    "math.sqrt": "square root numeric calculation math",
    "math.factorial": "factorial numeric calculation math",
    "math.floor": "floor round down numeric math",
    "math.gcd": "greatest common divisor numeric math",
    "sympy.sympify": "evaluate expression arithmetic symbolic math",
    "operator.add": "add sum two numbers arithmetic",
    "operator.mul": "multiply product two numbers arithmetic",
    "language_model": "language tokenize embed extract text nlp",
}


class ToolSelector:
    """
    Biases population pattern selection using LM semantic similarity.
    Selector only boosts relevant patterns — never suppresses.
    """
    def __init__(self, lm, alpha: float = 0.5):
        """
        lm: LanguageModelPattern instance (provides _embed())
        alpha: bias strength — 0.0 = no effect, 1.0 = strong bias
        """
        self.lm = lm
        self.alpha = alpha
        self._cache: Dict[str, List[float]] = {}

    def _embed(self, text: str) -> List[float]:
        if text not in self._cache:
            # LanguageModelPattern has _embed(text)
            self._cache[text] = self.lm._embed(text)
        return self._cache[text]

    def _cosine(self, a: List[float], b: List[float]) -> float:
        va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom < 1e-8:
            return 0.0
        # dot(va, vb) / denom can be slightly > 1 or < -1 due to precision
        sim = float(np.dot(va, vb) / denom)
        return float(np.clip(sim, 0.0, 1.0))

    def score(self, task_text: str, patterns: List[Any]) -> np.ndarray:
        """Return similarity score [0,1] per pattern."""
        task_emb = self._embed(task_text)
        scores = []
        for p in patterns:
            desc = self._pattern_description(p)
            pat_emb = self._embed(desc)
            scores.append(self._cosine(task_emb, pat_emb))
        return np.array(scores)

    def apply(self, task_text: str, weights: np.ndarray,
              patterns: List[Any]) -> np.ndarray:
        """Return adjusted weights: weights * (1 + alpha * similarity)."""
        if not task_text or len(patterns) == 0:
            return weights
        similarity = self.score(task_text, patterns)
        return weights * (1.0 + self.alpha * similarity)

    def _pattern_description(self, pattern: Any) -> str:
        tool_name = getattr(pattern, 'tool_name', None)
        if tool_name and tool_name in TOOL_DESCRIPTIONS:
            return TOOL_DESCRIPTIONS[tool_name]
        module = getattr(pattern, 'module', None)
        function = getattr(pattern, 'function', None)
        if module and function:
            key = f"{module}.{function}"
            if key in TOOL_DESCRIPTIONS:
                return TOOL_DESCRIPTIONS[key]
            return f"{module} {function} tool call function"
        action = getattr(pattern, 'action_type', str(pattern))
        return action
