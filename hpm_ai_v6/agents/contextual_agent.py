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

        query_embedding = next_cell.as_tensor() - ctx_cell.as_tensor()
        restored = self.restore_pattern_from_archive(name, query_embedding=query_embedding)
        if restored is not None:
            self._refresh_learner()
            return restored

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

    def _paging_lookup(self):
        lookup = {cell.name: cell for cell in self.word_cells.values()}
        lookup.update({cell.name: cell for cell in self.context_cells.values()})
        return lookup

    def _refresh_learner(self):
        from hpm_ai_v6.hpm_model.dynamics.meta_rule import MetaPatternRule
        from hpm_ai_v6.hpm_model.dynamics.learning import HPMLearner

        old_weights = self.get_weights_dict() if hasattr(self, "meta_rule") else {}
        self.meta_rule = MetaPatternRule(patterns=self.patterns, learning_rate=0.2)

        if old_weights:
            new_weights = torch.ones(len(self.patterns), dtype=torch.float32) / (len(self.patterns) + 1e-9)
            for i, pattern in enumerate(self.patterns):
                if pattern.name in old_weights:
                    new_weights[i] = float(old_weights[pattern.name])
            self.meta_rule.set_weights_tensor(new_weights / (new_weights.sum() + 1e-9))

        self.learner = HPMLearner(meta_rule=self.meta_rule)

    def process_words(self, words: List[str]):
        if len(words) <= self.context_length:
            return

        if self.word_cells:
            sample_dim = next(iter(self.word_cells.values())).as_numpy().shape[0]
            self.drop_incompatible_patterns(sample_dim)

        contexts: List[Cell] = []
        for end_idx in range(self.context_length, len(words)):
            context_tokens = words[end_idx - self.context_length : end_idx]
            next_word = words[end_idx]
            context_cell = self._get_or_create_context_cell(context_tokens)
            self._ensure_pattern(context_tokens, next_word)
            contexts.append(context_cell)

        if len(contexts) > 1:
            self.perceive(contexts, list(self.context_cells.values()), context={})

    def _distribution_from_context_cell(self, ctx_cell: Cell) -> List[Tuple[str, float]]:
        weights = self.get_weights()
        candidates: List[Tuple[str, float]] = []
        for idx, pattern in enumerate(self.patterns):
            if pattern.source is None or pattern.target is None:
                continue
            if pattern.source.name != ctx_cell.name:
                continue
            weight = float(weights[idx]) if idx < len(weights) else 0.0
            candidates.append((pattern.target.name.removeprefix("word_"), weight))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates

    def predict_next_distribution(self, context_tokens: Sequence[str]) -> List[Tuple[str, float]]:
        if len(context_tokens) < self.context_length:
            return []

        key = tuple(context_tokens[-self.context_length :])
        ctx_cell = self.context_cells.get(key)
        if ctx_cell is not None:
            return self._distribution_from_context_cell(ctx_cell)
        return self.predict_next_nearest_distribution(context_tokens)

    def predict_next(self, context_tokens: Sequence[str]) -> Optional[str]:
        candidates = self.predict_next_distribution(context_tokens)
        return candidates[0][0] if candidates else None

    def predict_next_nearest_distribution(self, context_tokens: Sequence[str]) -> List[Tuple[str, float]]:
        if len(context_tokens) < self.context_length or not self.context_cells:
            return []

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
            return []

        return self._distribution_from_context_cell(best_context)

    def predict_next_nearest(self, context_tokens: Sequence[str]) -> Optional[str]:
        candidates = self.predict_next_nearest_distribution(context_tokens)
        return candidates[0][0] if candidates else None
