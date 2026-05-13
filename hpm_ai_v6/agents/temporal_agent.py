from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.core.temporal_cell import TemporalCell


class TemporalAgent:
    def __init__(self, threshold: float = 0.01, max_depth: int = 4):
        self.threshold = threshold
        self.max_depth = max_depth
        self._temporal_index: Dict[str, List[TemporalCell]] = {}
        self._pair_index: Dict[Tuple[str, str], TemporalCell] = {}

    @property
    def temporal_index(self) -> Dict[str, List[TemporalCell]]:
        return self._temporal_index

    @staticmethod
    def _relation_name(edge: Any) -> str:
        relation = str(getattr(edge, "relation", "") or "").lower()
        if relation in {"causes", "cause", "causal", "causal_relation"}:
            return relation
        name = str(getattr(edge, "name", "") or "").lower()
        if name.startswith("causal_"):
            return "causal_relation"
        return relation

    def _coerce_edge(self, edge: Any) -> Optional[Tuple[Cell, Cell, float]]:
        source = getattr(edge, "source", None)
        target = getattr(edge, "target", None)
        if source is None or target is None:
            return None
        relation = self._relation_name(edge)
        if relation and relation not in {"causes", "cause", "causal", "causal_relation"}:
            return None
        score = getattr(edge, "score", None)
        if score is None:
            score = getattr(edge, "effect_magnitude", None)
        if score is None:
            score = getattr(edge, "weight", None)
        if score is None:
            score = 0.0
        return source, target, float(score)

    @staticmethod
    def _cell_key(cell: Cell) -> str:
        return getattr(cell, "name", str(cell))

    def _make_temporal_cell(
        self,
        cause: Cell,
        effect: Cell,
        onset_weight: float,
        duration_weight: float,
    ) -> TemporalCell:
        import numpy as np

        cause_vec = cause.as_numpy()
        effect_vec = effect.as_numpy()
        size = max(len(cause_vec), len(effect_vec)) or 1
        embedding = np.zeros(size, dtype=float)
        return TemporalCell(
            name=f"temporal:{cause.name}->{effect.name}",
            dim=3,
            weight=float(onset_weight),
            embedding=embedding,
            cause=cause,
            effect=effect,
            onset_weight=float(onset_weight),
            duration_weight=float(duration_weight),
        )

    def _reindex(self, cells: Sequence[TemporalCell]) -> None:
        index: Dict[str, List[TemporalCell]] = defaultdict(list)
        pair_index: Dict[Tuple[str, str], TemporalCell] = {}
        for cell in cells:
            index[cell.cause.name].append(cell)
            pair_index[(cell.cause.name, cell.effect.name)] = cell
        for entries in index.values():
            entries.sort(key=lambda item: (item.duration_weight, -item.onset_weight, item.name))
        self._temporal_index = dict(index)
        self._pair_index = pair_index

    def _concurrency_key(self, left: TemporalCell, right: TemporalCell) -> bool:
        return (
            left.cause.name == right.cause.name
            or left.effect.name == right.effect.name
            or left.cause.name == right.effect.name
            or right.cause.name == left.effect.name
        )

    def _refresh_concurrency(self) -> None:
        all_cells = [cell for cells in self._temporal_index.values() for cell in cells]
        for cell in all_cells:
            cell.concurrent = []
        for i, left in enumerate(all_cells):
            for right in all_cells[i + 1 :]:
                if self._concurrency_key(left, right):
                    if right not in left.concurrent:
                        left.concurrent.append(right)
                    if left not in right.concurrent:
                        right.concurrent.append(left)

    def _candidate_paths(self, causal_edges: Sequence[Any]) -> List[Tuple[Cell, Cell, float, float]]:
        parsed: List[Tuple[Cell, Cell, float]] = []
        for edge in causal_edges:
            coerced = self._coerce_edge(edge)
            if coerced is not None:
                parsed.append(coerced)

        adjacency: Dict[str, List[Tuple[Cell, Cell, float]]] = defaultdict(list)
        for source, target, score in parsed:
            adjacency[source.name].append((source, target, score))

        candidates: Dict[Tuple[str, str], Tuple[Cell, Cell, float, float]] = {}

        def register(cause: Cell, effect: Cell, onset_weight: float, duration_weight: float) -> None:
            key = (cause.name, effect.name)
            candidate = (cause, effect, float(onset_weight), float(duration_weight))
            existing = candidates.get(key)
            if existing is None:
                candidates[key] = candidate
                return
            existing_cause, existing_effect, existing_onset, existing_duration = existing
            if duration_weight < existing_duration or (
                duration_weight == existing_duration and onset_weight > existing_onset
            ):
                candidates[key] = candidate
                return
            if duration_weight == existing_duration and onset_weight > existing_onset:
                candidates[key] = candidate

        for source, target, score in parsed:
            register(source, target, score, 1.0)

        def dfs(start: Cell, current: Cell, onset: float, depth: int, visited: set[str]) -> None:
            if depth >= self.max_depth:
                return
            for _, target, score in adjacency.get(current.name, []):
                if target.name in visited:
                    continue
                next_onset = min(onset, score)
                duration = float(depth + 1)
                register(start, target, next_onset, duration)
                dfs(start, target, next_onset, depth + 1, visited | {target.name})

        for source, target, score in parsed:
            dfs(source, target, score, 1, {source.name, target.name})

        return list(candidates.values())

    def build_temporal_cells(self, causal_edges: Sequence[Any]) -> List[TemporalCell]:
        cells: List[TemporalCell] = []
        for cause, effect, onset_weight, duration_weight in self._candidate_paths(causal_edges):
            cell = self._make_temporal_cell(cause, effect, onset_weight, duration_weight)
            cell.lapsed = cell.onset_weight < self.threshold
            cells.append(cell)
        self._reindex(cells)
        self._refresh_concurrency()
        return [cell for cells in self._temporal_index.values() for cell in cells]

    def update_temporal_cells(self, causal_edges: Sequence[Any]) -> None:
        candidates = self._candidate_paths(causal_edges)
        existing = dict(self._pair_index)

        for cause, effect, onset_weight, duration_weight in candidates:
            key = (cause.name, effect.name)
            matched = existing.get(key)
            if matched is None:
                matched = self._make_temporal_cell(cause, effect, onset_weight, duration_weight)
                self._temporal_index.setdefault(cause.name, []).append(matched)
                self._pair_index[key] = matched
            else:
                matched.onset_weight = 0.7 * float(matched.onset_weight) + 0.3 * float(onset_weight)
                matched.duration_weight = min(float(matched.duration_weight), float(duration_weight))
                matched.weight = matched.onset_weight
            matched.lapsed = matched.onset_weight < self.threshold

        self._reindex([cell for cells in self._temporal_index.values() for cell in cells])
        self._refresh_concurrency()
