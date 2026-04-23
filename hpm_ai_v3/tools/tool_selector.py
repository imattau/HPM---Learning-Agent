# hpm_ai_v3/tools/tool_selector.py
"""
ToolSelector - Uses LM embeddings to bias pattern selection toward
semantically relevant tools. Acts as a pattern evaluator in HPM terms.
"""
import numpy as np
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
    # Innate tools (Fix 4: differentiate math vs text)
    "arithmetic": "evaluate numeric expression calculate math calculation numbers addition subtraction",
    "float": "convert to decimal float numeric number",
    "int": "convert to integer whole number",
    "str": "convert to string text character",
    "split": "split text into words tokens whitespace partition",
    "index": "get item from list or string at position index sequence",
    "re_findall": "regex pattern match extraction find all text search",
    "extract_numbers": "extract numbers numeric values from text",
    "get_type": "check data type object class category",
}


class ToolSelector:
    """
    Biases population pattern selection using LM semantic similarity.
    Selector only boosts relevant patterns — never suppresses.
    """
    def __init__(self, lm, alpha: float = 5.0, decay_rate: float = 0.998):
        """
        lm: LanguageModelPattern instance (provides _embed())
        alpha: initial bias strength (Fix 4: increased from 2.0 to 5.0)
        decay_rate: per-episode decay for alpha
        """
        self.lm = lm
        # Bind for cache clearing
        if hasattr(lm, '_tool_selector'):
            lm._tool_selector = self
        
        self.base_alpha = alpha
        self.current_alpha = alpha
        self.decay_rate = decay_rate
        self.episode_count = 0
        self._cache: Dict[str, List[float]] = {}

    def clear_cache(self):
        """Clear the embedding cache. Call this when LM parameters change."""
        self._cache = {}

    def _embed(self, input_val: Any) -> List[float]:
        """Convert input to string and get embedding from LM."""
        # Always convert to string for hashing to avoid TypeError: unhashable type: 'list'
        if isinstance(input_val, (list, tuple)):
            safe_key = " ".join(map(str, input_val))
        else:
            safe_key = str(input_val)

        try:
            if safe_key not in self._cache:
                # LanguageModelPattern has _embed(text)
                self._cache[safe_key] = self.lm._embed(safe_key)
            return self._cache[safe_key]
        except TypeError as e:
            # Fallback for unexpected unhashable types
            print(f"  [ToolSelector] Warning: Unhashable type {type(safe_key)} encountered. Bypassing cache.")
            return self.lm._embed(str(safe_key))

    def _cosine(self, a: List[float], b: List[float]) -> float:
        va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom < 1e-8:
            return 0.0
        sim = float(np.dot(va, vb) / denom)
        return float(np.clip(sim, 0.0, 1.0))

    def score(self, task_text: Any, patterns: List[Any]) -> np.ndarray:
        """Return similarity score [0,1] per pattern."""
        task_emb = self._embed(task_text)
        scores = []
        for p in patterns:
            desc = self._pattern_description(p)
            pat_emb = self._embed(desc)
            scores.append(self._cosine(task_emb, pat_emb))
        return np.array(scores)

    def apply(self, task_text: Any, weights: np.ndarray,
              patterns: List[Any]) -> np.ndarray:
        """Return adjusted weights: weights * (1 + alpha * similarity)."""
        if not task_text or len(patterns) == 0:
            return weights
        
        # Decay alpha based on episode count
        self.current_alpha = self.base_alpha * (self.decay_rate ** self.episode_count)
        self.episode_count += 1
        
        similarity = self.score(task_text, patterns)
        return weights * (1.0 + self.current_alpha * similarity)

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
