from typing import Any, Dict, List, Optional, Sequence, Tuple
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
        reasoning_agent: Optional[Any] = None,
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
        self.reasoning_agent = reasoning_agent
        self.max_length = max_length
        self.stop_tokens = stop_tokens or {".", "!", "?"}
        self.repetition_penalty = repetition_penalty
        self.recent_window = max(1, recent_window)
        self.no_repeat_ngram_size = 3
        self.rng = random.Random(0)
        self._reasoning_guidance_cache: Dict[str, List[str]] = {}

    def clear_reasoning_guidance_cache(self) -> None:
        self._reasoning_guidance_cache.clear()

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

    def next_token_distribution(
        self,
        prefix: str,
        top_k: Optional[int] = None,
        reasoning_focus_terms: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float]]:
        tokens = self._clean_words(prefix)
        candidates = self.contextual_agent.predict_next_distribution(tokens)
        if not candidates:
            candidates = self._word_distribution(tokens)
        candidates = self._apply_phrase_bias(tokens, candidates)
        candidates = self._apply_semantic_bias(prefix, candidates)
        candidates = self._apply_reasoning_bias(prefix, candidates, reasoning_focus_terms=reasoning_focus_terms)
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

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        tokens: List[str] = []
        for raw in text.lower().replace("->", " ").replace("|", " ").split():
            token = raw.strip(string.punctuation)
            if token:
                tokens.append(token)
        return tokens

    def _reasoning_focus_terms(self, prefix_text: str) -> List[str]:
        if self.reasoning_agent is None:
            return []

        if getattr(self.reasoning_agent, "_dirty", False):
            return []

        cached = self._reasoning_guidance_cache.get(prefix_text)
        if cached is not None:
            return cached

        focus_terms: List[str] = []
        try:
            trace = self.reasoning_agent.reason_with_trace(prefix_text)
        except Exception:
            self._reasoning_guidance_cache[prefix_text] = focus_terms
            return focus_terms

        seen = set()

        def add_text(value: Optional[str]) -> None:
            if not value:
                return
            for token in self._tokenize_text(value):
                if token not in seen:
                    seen.add(token)
                    focus_terms.append(token)

        for term in trace.get("terms", []) or []:
            add_text(str(term))

        for anchor in (trace.get("anchors") or {}).values():
            if isinstance(anchor, dict):
                add_text(str(anchor.get("label", "")))

        chosen_path = trace.get("chosen_path")
        candidate_paths = trace.get("candidate_paths") or []
        paths = [chosen_path] if isinstance(chosen_path, dict) else []
        paths.extend(path for path in candidate_paths if isinstance(path, dict))
        for path in paths:
            for node in path.get("nodes", []) or []:
                if isinstance(node, dict):
                    add_text(str(node.get("label", "")))
            for step in path.get("steps", []) or []:
                if isinstance(step, dict):
                    source = step.get("source") or {}
                    target = step.get("target") or {}
                    if isinstance(source, dict):
                        add_text(str(source.get("label", "")))
                    if isinstance(target, dict):
                        add_text(str(target.get("label", "")))

        subgraph = trace.get("explanatory_subgraph")
        if isinstance(subgraph, dict):
            effect = subgraph.get("effect")
            if isinstance(effect, dict):
                add_text(str(effect.get("label", "")))
            for root in subgraph.get("root_causes", []) or []:
                if isinstance(root, dict):
                    add_text(str(root.get("label", "")))
            for edge in subgraph.get("edges", []) or []:
                if isinstance(edge, dict):
                    source = edge.get("source") or {}
                    target = edge.get("target") or {}
                    if isinstance(source, dict):
                        add_text(str(source.get("label", "")))
                    if isinstance(target, dict):
                        add_text(str(target.get("label", "")))

        self._reasoning_guidance_cache[prefix_text] = focus_terms
        return focus_terms

    def _best_sentence_anchor_text(self, seed_text: str) -> Optional[str]:
        sentences = getattr(self.semantic_agent, "sent_cells", None)
        if not sentences:
            return None

        seed_cell = self.semantic_agent._get_or_create_sent_cell(seed_text)
        if seed_cell is None:
            return None

        seed_tokens = self._clean_words(seed_text)
        best_text: Optional[str] = None
        best_score = 0.0
        for sentence_text, sent_cell in sentences.items():
            if sent_cell is None:
                continue
            try:
                similarity = float(seed_cell.similarity(sent_cell))
            except Exception:
                continue
            if similarity < 0.35:
                continue

            sentence_tokens = self._clean_words(sentence_text)
            overlap = len(set(seed_tokens) & set(sentence_tokens))
            score = similarity + 0.04 * overlap
            if score > best_score:
                best_score = score
                best_text = sentence_text

        return best_text

    def _anchor_suffix_tokens(self, seed_text: str, max_tokens: int = 6) -> List[str]:
        anchor_text = self._best_sentence_anchor_text(seed_text)
        if not anchor_text:
            return []

        seed_tokens = self._clean_words(seed_text)
        anchor_tokens = self._clean_words(anchor_text)
        if not seed_tokens or not anchor_tokens:
            return []

        prefix_len = 0
        for seed_token, anchor_token in zip(seed_tokens, anchor_tokens):
            if seed_token != anchor_token:
                break
            prefix_len += 1

        if prefix_len == 0:
            return []

        suffix = anchor_tokens[prefix_len : prefix_len + max_tokens]
        return suffix if len(suffix) >= 2 else []

    def _banned_next_tokens(self, prefix_tokens: Sequence[str]) -> set[str]:
        n = max(int(self.no_repeat_ngram_size), 2)
        if len(prefix_tokens) < n - 1:
            return {prefix_tokens[-1]} if prefix_tokens else set()

        prefix = tuple(prefix_tokens[-(n - 1) :])
        banned: set[str] = {prefix_tokens[-1]}
        for idx in range(len(prefix_tokens) - n + 1):
            if tuple(prefix_tokens[idx : idx + n - 1]) == prefix:
                banned.add(prefix_tokens[idx + n - 1])
        return banned

    def _apply_reasoning_bias(
        self,
        prefix_text: str,
        candidates: List[Tuple[str, float]],
        reasoning_focus_terms: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float]]:
        focus_terms = list(reasoning_focus_terms) if reasoning_focus_terms is not None else self._reasoning_focus_terms(prefix_text)
        if not candidates or not focus_terms:
            return candidates

        focus_set = set(focus_terms)
        reranked: List[Tuple[str, float]] = []
        for token, score in candidates:
            adjusted = score
            token_l = token.lower()
            if token_l in focus_set:
                adjusted += 0.25
            else:
                for focus in focus_terms:
                    if focus in token_l or token_l in focus:
                        adjusted += 0.12
                        break
            reranked.append((token, adjusted))
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
        anchor_tokens = self._anchor_suffix_tokens(seed_text)
        if anchor_tokens:
            tokens.extend(anchor_tokens)
            limit = max(0, limit - len(anchor_tokens))

        reasoning_focus_terms = self._reasoning_focus_terms(seed_text)
        for _ in range(limit):
            prefix = " ".join(tokens)
            candidates = self.next_token_distribution(prefix, reasoning_focus_terms=reasoning_focus_terms)
            candidates = self._apply_repetition_penalty(tokens, candidates)
            banned_tokens = self._banned_next_tokens(tokens)
            if banned_tokens:
                filtered = [item for item in candidates if item[0] not in banned_tokens]
                if filtered:
                    candidates = filtered
            candidates.sort(key=lambda item: item[1], reverse=True)
            next_token = self._sample(candidates, temperature=temperature)
            if next_token is None:
                break
            tokens.append(next_token)
            if next_token in self.stop_tokens:
                break
        return " ".join(tokens)
