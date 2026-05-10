from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.agents.social_agent import SocialAgent
from hpm_ai_v6.hpm_model.fields.pattern_field import DynamicPatternField


class ContextualAgent(SocialAgent):
    """
    Context-window word predictor.
    Learns transitions from an n-gram context cell to the next word cell.
    """

    def __init__(
        self,
        context_length: int = 3,
        shared_field: Optional[DynamicPatternField] = None,
        **kwargs,
    ):
        self.context_length = context_length
        self.word_cells: Dict[str, Cell] = {}
        self.context_cells: Dict[Tuple[str, ...], Cell] = {}
        super().__init__(patterns=[], shared_field=shared_field, **kwargs)

    def _get_or_create_word_cell(self, word: str) -> Cell:
        if word not in self.word_cells:
            self.word_cells[word] = Cell(
                name=f"word_{word}",
                dim=0,
                embedding=np.random.randn(16) * 0.1,
            )
        return self.word_cells[word]

    def _get_or_create_context_cell(self, tokens: Sequence[str]) -> Cell:
        key = tuple(tokens)
        if key not in self.context_cells:
            token_cells = [self._get_or_create_word_cell(token) for token in key]
            embedding = np.mean([cell.as_numpy() for cell in token_cells], axis=0)
            self.context_cells[key] = Cell(
                name=f"ctx_{'|'.join(key)}",
                dim=0,
                embedding=embedding,
            )
        return self.context_cells[key]

    def _ensure_pattern(self, context_tokens: Sequence[str], next_word: str) -> Cell:
        ctx_cell = self._get_or_create_context_cell(context_tokens)
        next_cell = self._get_or_create_word_cell(next_word)
        name = f"ctx_{'|'.join(context_tokens)}->{next_word}"
        for pattern in self.patterns:
            if pattern.name == name:
                return pattern

        new_pattern = Cell(
            name=name,
            dim=1,
            embedding=next_cell.as_numpy() - ctx_cell.as_numpy(),
            source=ctx_cell,
            target=next_cell,
        )
        self.patterns.append(new_pattern)
        self._refresh_learner()
        return new_pattern

    def _refresh_learner(self):
        from hpm_ai_v6.hpm_model.dynamics.meta_rule import MetaPatternRule
        from hpm_ai_v6.hpm_model.dynamics.learning import HPMLearner

        old_weights = self.meta_rule.get_weights_tensor() if hasattr(self, "meta_rule") else torch.zeros(0, dtype=torch.float32)
        self.meta_rule = MetaPatternRule(patterns=self.patterns, learning_rate=0.2)

        if len(old_weights) > 0:
            new_weights = torch.ones(len(self.patterns), dtype=torch.float32) / (len(self.patterns) + 1e-9)
            new_weights[: len(old_weights)] = old_weights
            self.meta_rule.set_weights_tensor(new_weights / (new_weights.sum() + 1e-9))

        self.learner = HPMLearner(meta_rule=self.meta_rule)

    def process_words(self, words: List[str]):
        if len(words) <= self.context_length:
            return

        contexts: List[Cell] = []
        for end_idx in range(self.context_length, len(words)):
            context_tokens = words[end_idx - self.context_length : end_idx]
            next_word = words[end_idx]
            context_cell = self._get_or_create_context_cell(context_tokens)
            self._ensure_pattern(context_tokens, next_word)
            contexts.append(context_cell)

        if len(contexts) > 1:
            self.perceive(contexts, list(self.context_cells.values()), context={})

    def predict_next(self, context_tokens: Sequence[str]) -> Optional[str]:
        if len(context_tokens) < self.context_length:
            return None

        key = tuple(context_tokens[-self.context_length :])
        ctx_cell = self.context_cells.get(key)
        if ctx_cell is None:
            return self.predict_next_nearest(context_tokens)

        best_word = None
        best_weight = -1.0
        weights = self.get_weights()
        for idx, pattern in enumerate(self.patterns):
            if pattern.source is None or pattern.target is None:
                continue
            if pattern.source.name != ctx_cell.name:
                continue
            weight = float(weights[idx]) if idx < len(weights) else 0.0
            if weight > best_weight:
                best_weight = weight
                best_word = pattern.target.name.removeprefix("word_")
        return best_word

    def predict_next_nearest(self, context_tokens: Sequence[str]) -> Optional[str]:
        if len(context_tokens) < self.context_length or not self.context_cells:
            return None

        token_cells = [self._get_or_create_word_cell(token) for token in context_tokens[-self.context_length :]]
        query = np.mean([cell.as_numpy() for cell in token_cells], axis=0)

        best_context = None
        best_sim = -1.0
        for ctx_cell in self.context_cells.values():
            sim = ctx_cell.similarity(query)
            if sim > best_sim:
                best_sim = sim
                best_context = ctx_cell

        if best_context is None:
            return None

        best_word = None
        best_weight = -1.0
        weights = self.get_weights()
        for idx, pattern in enumerate(self.patterns):
            if pattern.source is None or pattern.target is None:
                continue
            if pattern.source.name != best_context.name:
                continue
            weight = float(weights[idx]) if idx < len(weights) else 0.0
            if weight > best_weight:
                best_weight = weight
                best_word = pattern.target.name.removeprefix("word_")
        return best_word
