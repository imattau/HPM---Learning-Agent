from typing import Dict, List, Optional, Tuple
import numpy as np

from hpm_ai_v6.hpm_model.evaluators.epistemic import EpistemicEvaluator


class UtilityAgent:
    """
    Thin utility layer over trained V6 agents.
    Provides simple practical tasks without training any patterns itself.
    """

    def __init__(self, word_agent, phrase_agent, semantic_agent, tag_fn, contextual_agent=None, syntactic_agent=None):
        self.word_agent = word_agent
        self.phrase_agent = phrase_agent
        self.semantic_agent = semantic_agent
        self.contextual_agent = contextual_agent
        self.syntactic_agent = syntactic_agent
        self.tag_fn = tag_fn
        self.epistemic = EpistemicEvaluator()

    @staticmethod
    def _clean_words(text: str) -> List[str]:
        import string
        return [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation)]

    def predict_next_word(self, prefix: str) -> Optional[str]:
        words = self._clean_words(prefix)
        if not words:
            return None

        if self.contextual_agent is not None:
            contextual_guess = self.contextual_agent.predict_next(words)
            if contextual_guess is not None:
                return contextual_guess

        last_word = words[-1]
        if last_word not in self.word_agent.word_cells:
            return None

        best_name = None
        best_weight = -1.0
        weights = self.word_agent.get_weights()
        for idx, pattern in enumerate(self.word_agent.patterns):
            if pattern.dim != 1 or pattern.source is None or pattern.target is None:
                continue
            if pattern.source.name != f"word_{last_word}":
                continue
            weight = float(weights[idx]) if idx < len(weights) else 0.0
            if weight > best_weight:
                best_weight = weight
                best_name = pattern.target.name.removeprefix("word_")
        return best_name

    def grammar_score(self, sentence: str) -> Dict[str, float]:
        if self.syntactic_agent is not None and hasattr(self.syntactic_agent, "grammar_score"):
            try:
                return self.syntactic_agent.grammar_score(sentence)
            except Exception:
                pass

        tags = self.tag_fn(self._clean_words(sentence))
        seq = [self.phrase_agent.pos_cells[t] for t in tags if t in self.phrase_agent.pos_cells]
        population = list(self.phrase_agent.pos_cells.values())
        if len(seq) < 2 or not self.phrase_agent.patterns:
            return {"mean_nll": 0.0, "matched_patterns": 0.0}

        weights = self.phrase_agent.get_weights()
        total_nll = 0.0
        total_weight = 0.0
        matched = 0
        for idx, pattern in enumerate(self.phrase_agent.patterns):
            result = self.epistemic.evaluate(
                pattern,
                {
                    "observation_seq": seq,
                    "population": population,
                },
            )
            count = int(result.metadata.get("count", 0))
            if count <= 0:
                continue
            weight = float(weights[idx]) if idx < len(weights) else 0.0
            weight = max(weight, 1e-6)
            total_nll += float(result.metadata.get("nll", 0.0)) * weight
            total_weight += weight
            matched += 1

        mean_nll = total_nll / total_weight if total_weight > 0.0 else 0.0
        return {"mean_nll": mean_nll, "matched_patterns": float(matched)}

    def topic_boundaries(self, sentences: List[str], threshold: float = 0.75) -> List[int]:
        if len(sentences) < 2:
            return []

        embeddings = []
        for sentence in sentences:
            cell = self.semantic_agent._get_or_create_sent_cell(sentence)
            if cell is None:
                continue
            embeddings.append(cell.as_numpy())

        boundaries: List[int] = []
        for idx in range(len(embeddings) - 1):
            a = embeddings[idx]
            b = embeddings[idx + 1]
            sim = float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))
            if sim < threshold:
                boundaries.append(idx + 1)
        return boundaries
