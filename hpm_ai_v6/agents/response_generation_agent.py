from typing import Dict, List, Optional, Sequence, Tuple
import math
import random
import string


class ResponseGenerationAgent:
    """
    Autoregressive text generator over trained V6 agents.
    Uses contextual decoding with phrase and semantic reranking, plus bigram fallback.
    """

    def __init__(
        self,
        contextual_agent,
        word_agent,
        phrase_agent,
        semantic_agent,
        tag_fn,
        max_length: int = 50,
        stop_tokens: Optional[set[str]] = None,
        repetition_penalty: float = 0.35,
        recent_window: int = 6,
    ):
        self.contextual_agent = contextual_agent
        self.word_agent = word_agent
        self.phrase_agent = phrase_agent
        self.semantic_agent = semantic_agent
        self.tag_fn = tag_fn
        self.max_length = max_length
        self.stop_tokens = stop_tokens or {".", "!", "?"}
        self.repetition_penalty = repetition_penalty
        self.recent_window = max(1, recent_window)
        self.rng = random.Random(0)

    @staticmethod
    def _clean_words(text: str) -> List[str]:
        return [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation)]

    def _word_distribution(self, prefix_tokens: Sequence[str]) -> List[Tuple[str, float]]:
        if not prefix_tokens:
            return []
        last_word = prefix_tokens[-1]
        if last_word not in self.word_agent.word_cells:
            return []

        weights = self.word_agent.get_weights()
        candidates: List[Tuple[str, float]] = []
        for idx, pattern in enumerate(self.word_agent.patterns):
            if pattern.dim != 1 or pattern.source is None or pattern.target is None:
                continue
            if pattern.source.name != f"word_{last_word}":
                continue
            weight = float(weights[idx]) if idx < len(weights) else 0.0
            candidates.append((pattern.target.name.removeprefix("word_"), weight))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates

    def next_token_distribution(self, prefix: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        tokens = self._clean_words(prefix)
        candidates = self.contextual_agent.predict_next_distribution(tokens)
        if not candidates:
            candidates = self._word_distribution(tokens)
        candidates = self._apply_phrase_bias(tokens, candidates)
        candidates = self._apply_semantic_bias(prefix, candidates)
        candidates.sort(key=lambda item: item[1], reverse=True)
        if top_k is not None:
            candidates = candidates[:top_k]
        return candidates

    def _apply_repetition_penalty(
        self,
        prefix_tokens: Sequence[str],
        candidates: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        if not candidates or not prefix_tokens:
            return candidates

        last_token = prefix_tokens[-1]
        recent_tokens = list(prefix_tokens[-self.recent_window :])
        recent_counts = {token: recent_tokens.count(token) for token in set(recent_tokens)}

        reranked: List[Tuple[str, float]] = []
        for token, score in candidates:
            adjusted = score
            if token == last_token:
                adjusted -= self.repetition_penalty * 2.0
            if token in recent_counts:
                adjusted -= self.repetition_penalty * recent_counts[token]

            if len(prefix_tokens) >= 2 and token == prefix_tokens[-2]:
                adjusted -= self.repetition_penalty

            reranked.append((token, adjusted))
        return reranked

    def _apply_phrase_bias(self, prefix_tokens: Sequence[str], candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if not candidates or not self.phrase_agent.patterns:
            return candidates

        prefix_tags = self.tag_fn(list(prefix_tokens))
        prev_tag = prefix_tags[-1] if prefix_tags else None
        weights = self.phrase_agent.get_weights()
        reranked: List[Tuple[str, float]] = []
        for token, score in candidates:
            token_tag = self.tag_fn([token])[0]
            bonus = 0.0
            if prev_tag is not None:
                for idx, pattern in enumerate(self.phrase_agent.patterns):
                    if pattern.source is None or pattern.target is None:
                        continue
                    if pattern.source.name == f"pos_{prev_tag}" and pattern.target.name == f"pos_{token_tag}":
                        bonus += float(weights[idx]) if idx < len(weights) else 0.0
            reranked.append((token, score + 0.2 * bonus))
        return reranked

    def _apply_semantic_bias(self, prefix_text: str, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if not candidates:
            return candidates

        prefix_cell = self.semantic_agent._get_or_create_sent_cell(prefix_text)
        if prefix_cell is None:
            return candidates

        reranked: List[Tuple[str, float]] = []
        for token, score in candidates:
            continuation_cell = self.semantic_agent._get_or_create_sent_cell(f"{prefix_text} {token}")
            if continuation_cell is None:
                reranked.append((token, score))
                continue
            sim = prefix_cell.similarity(continuation_cell)
            reranked.append((token, score + 0.1 * float(sim)))
        return reranked

    def _sample(self, candidates: List[Tuple[str, float]], temperature: float) -> Optional[str]:
        if not candidates:
            return None
        if temperature <= 0.0:
            return candidates[0][0]

        weights = [max(score, 1e-9) for _, score in candidates]
        logits = [math.log(weight) / max(temperature, 1e-6) for weight in weights]
        max_logit = max(logits)
        probs = [math.exp(logit - max_logit) for logit in logits]
        total = sum(probs)
        target = self.rng.random() * total
        running = 0.0
        for (token, _), prob in zip(candidates, probs):
            running += prob
            if running >= target:
                return token
        return candidates[-1][0]

    def generate(self, seed_text: str, max_length: Optional[int] = None, temperature: float = 0.0) -> str:
        tokens = self._clean_words(seed_text)
        if not tokens:
            return ""

        limit = max_length or self.max_length
        for _ in range(limit):
            prefix = " ".join(tokens)
            candidates = self.next_token_distribution(prefix)
            candidates = self._apply_repetition_penalty(tokens, candidates)
            candidates.sort(key=lambda item: item[1], reverse=True)
            next_token = self._sample(candidates, temperature=temperature)
            if next_token is None:
                break
            tokens.append(next_token)
            if next_token in self.stop_tokens:
                break
        return " ".join(tokens)
