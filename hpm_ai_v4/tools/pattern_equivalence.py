"""Pattern-agnostic equivalence helpers for reuse-first learning."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _digest(parts: Sequence[str]) -> str:
    joined = "\u241f".join(str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()


def _canonical_sequence(seq: Sequence[Any]) -> List[str]:
    return [str(item).strip().lower() for item in seq if str(item).strip()]


@dataclass(frozen=True)
class MatchResult:
    status: str
    candidate_signature: str
    similarity: float
    alignment_score: float
    matched_index: int = -1
    composition_parts: Tuple[Tuple[Any, ...], ...] = ()
    composition_coverage: float = 0.0
    residual_sequence: Tuple[Any, ...] = ()
    residual_fraction: float = 1.0
    exact_match: bool = False
    reason: str = ""


@dataclass(frozen=True)
class CompositionResult:
    status: str
    component_count: int
    components: Tuple[Tuple[Any, ...], ...] = ()
    residual_sequence: Tuple[Any, ...] = ()
    coverage: float = 0.0
    exact_match: bool = False
    reason: str = ""


@dataclass
class PatternEquivalenceIndex:
    """Lightweight pattern equivalence memory.

    The index is intentionally conservative: exact sequence reuse is cheap,
    while equivalence is estimated from residual size and pattern likelihood.
    """

    seen_sequence_signatures: set[str] = field(default_factory=set)
    seen_pattern_signatures: set[str] = field(default_factory=set)
    sequence_profiles: Dict[str, List[str]] = field(default_factory=dict)

    @staticmethod
    def sequence_signature(seq: Sequence[Any]) -> str:
        return _digest(_canonical_sequence(seq))

    @staticmethod
    def pattern_signature(pattern: Any, *, decimals: int = 4) -> str:
        pieces: List[np.ndarray] = []
        for name in ("A", "B", "pi"):
            arr = getattr(pattern, name, None)
            if arr is None:
                continue
            pieces.append(np.asarray(arr, dtype=np.float32).ravel())
        if not pieces:
            return _digest([str(getattr(pattern, "id", "")), str(getattr(pattern, "latent_dim", "")), str(getattr(pattern, "obs_dim", ""))])
        fingerprint = np.concatenate(pieces)
        rounded = np.round(fingerprint, decimals=decimals)
        return hashlib.sha1(rounded.tobytes()).hexdigest()

    @staticmethod
    def pattern_similarity(left: Any, right: Any) -> float:
        """Cosine similarity over flattened pattern parameters."""
        def get_params(pat: Any) -> np.ndarray:
            if getattr(pat, "complexity", 0) >= 2 or getattr(pat, "latent_dim", 0) > 1:
                return np.concatenate([
                    np.asarray(getattr(pat, "A", np.zeros((0, 0))), dtype=np.float32).ravel(),
                    np.asarray(getattr(pat, "B", np.zeros((0, 0))), dtype=np.float32).ravel(),
                    np.asarray(getattr(pat, "pi", np.zeros(0)), dtype=np.float32).ravel(),
                ])
            return np.asarray(getattr(pat, "B", np.zeros((0, 0))), dtype=np.float32).ravel()

        p_i = get_params(left)
        p_j = get_params(right)
        if len(p_i) != len(p_j):
            max_len = max(len(p_i), len(p_j))
            tmp_i = np.zeros(max_len, dtype=np.float32)
            tmp_j = np.zeros(max_len, dtype=np.float32)
            tmp_i[: len(p_i)] = p_i
            tmp_j[: len(p_j)] = p_j
            p_i, p_j = tmp_i, tmp_j
        denom = float(np.linalg.norm(p_i) * np.linalg.norm(p_j) + 1e-12)
        if denom <= 0.0:
            return 0.0
        return float(np.dot(p_i, p_j) / denom)

    def register_sequence(self, seq: Sequence[Any]) -> bool:
        key = self.sequence_signature(seq)
        if key in self.seen_sequence_signatures:
            return False
        self.seen_sequence_signatures.add(key)
        self.sequence_profiles[key] = _canonical_sequence(seq)
        return True

    def register_pattern(self, pattern: Any, *, decimals: int = 4) -> bool:
        key = self.pattern_signature(pattern, decimals=decimals)
        if key in self.seen_pattern_signatures:
            return False
        self.seen_pattern_signatures.add(key)
        return True

    def _known_sequence_profiles(self) -> List[Tuple[str, ...]]:
        return [tuple(tokens) for tokens in self.sequence_profiles.values() if tokens]

    def compose_sequence(
        self,
        seq: Sequence[Any],
        *,
        min_parts: int = 2,
    ) -> CompositionResult:
        raw = list(seq)
        if not raw:
            return CompositionResult(status="novel", component_count=0, reason="empty_sequence")

        canon = _canonical_sequence(raw)
        known = set(self._known_sequence_profiles())
        if not known:
            return CompositionResult(status="novel", component_count=0, reason="no_known_components")

        components: List[Tuple[Any, ...]] = []
        residual: List[Any] = []
        i = 0
        while i < len(raw):
            best_end = -1
            for j in range(len(raw), i, -1):
                if tuple(canon[i:j]) in known:
                    best_end = j
                    break
            if best_end > i:
                components.append(tuple(raw[i:best_end]))
                i = best_end
            else:
                residual.append(raw[i])
                i += 1

        coverage = float(sum(len(component) for component in components) / max(1, len(raw)))
        if coverage >= 1.0 and len(components) >= min_parts:
            return CompositionResult(
                status="composed",
                component_count=len(components),
                components=tuple(components),
                residual_sequence=(),
                coverage=coverage,
                exact_match=False,
                reason="all_parts_known",
            )
        if components and residual:
            return CompositionResult(
                status="partial",
                component_count=len(components),
                components=tuple(components),
                residual_sequence=tuple(residual),
                coverage=coverage,
                exact_match=False,
                reason="mixed_known_and_novel_parts",
            )
        if coverage >= 1.0 and len(components) == 1:
            return CompositionResult(
                status="exact",
                component_count=1,
                components=tuple(components),
                residual_sequence=(),
                coverage=coverage,
                exact_match=True,
                reason="single_known_component",
            )
        return CompositionResult(
            status="novel",
            component_count=len(components),
            components=tuple(components),
            residual_sequence=tuple(residual),
            coverage=coverage,
            exact_match=False,
            reason="no_composable_structure",
        )

    def _reuse_window(self, pattern: Any, obs_seq: Sequence[Any], *, min_residual: int, keep_fraction: float) -> Tuple[str, Tuple[Any, ...], float]:
        if not obs_seq:
            return "novel", (), 0.0
        if len(obs_seq) <= min_residual:
            return "near", tuple(obs_seq), 0.0

        if hasattr(pattern, "residual_observations"):
            try:
                residual = list(pattern.residual_observations(obs_seq, min_residual=min_residual, keep_fraction=keep_fraction))
            except Exception:
                residual = list(obs_seq)
        else:
            residual = list(obs_seq[-max(min_residual, int(np.ceil(len(obs_seq) * keep_fraction))):])

        if not residual:
            return "exact", (), 1.0
        residual_fraction = float(len(residual) / max(1, len(obs_seq)))
        if residual_fraction <= 0.20:
            status = "exact"
        elif residual_fraction <= 0.50:
            status = "equivalent"
        elif residual_fraction <= 0.80:
            status = "near"
        else:
            status = "novel"
        return status, tuple(residual), residual_fraction

    def classify(
        self,
        pattern: Any,
        obs_seq: Sequence[Any],
        *,
        min_residual: int = 4,
        keep_fraction: float = 0.35,
    ) -> MatchResult:
        seq_sig = self.sequence_signature(obs_seq)
        pat_sig = self.pattern_signature(pattern)
        if seq_sig in self.seen_sequence_signatures:
            return MatchResult(
                status="exact",
                candidate_signature=pat_sig,
                similarity=1.0,
                alignment_score=1.0,
                residual_sequence=(),
                residual_fraction=0.0,
                exact_match=True,
                reason="sequence_signature_seen",
            )

        composition = self.compose_sequence(obs_seq)
        if composition.status in {"exact", "composed", "partial"}:
            residual_fraction = float(len(composition.residual_sequence) / max(1, len(obs_seq)))
            similarity = float(max(0.0, 1.0 - residual_fraction))
            return MatchResult(
                status=composition.status,
                candidate_signature=pat_sig,
                similarity=similarity,
                alignment_score=similarity,
                matched_index=-1,
                composition_parts=composition.components,
                composition_coverage=composition.coverage,
                residual_sequence=composition.residual_sequence,
                residual_fraction=residual_fraction,
                exact_match=composition.exact_match,
                reason=composition.reason,
            )

        status, residual, residual_fraction = self._reuse_window(
            pattern,
            obs_seq,
            min_residual=min_residual,
            keep_fraction=keep_fraction,
        )
        similarity = float(max(0.0, 1.0 - residual_fraction))
        alignment_score = similarity
        if hasattr(pattern, "log_likelihood") and residual:
            try:
                ll = float(pattern.log_likelihood(list(residual)))
                alignment_score = float(np.clip(1.0 / (1.0 + np.exp(-ll / max(1, len(residual)) + 1.5)), 0.0, 1.0))
            except Exception:
                pass
        exact_match = status == "exact"
        return MatchResult(
            status=status,
            candidate_signature=pat_sig,
            similarity=similarity,
            alignment_score=alignment_score,
            composition_parts=(),
            composition_coverage=0.0,
            residual_sequence=residual,
            residual_fraction=float(residual_fraction),
            exact_match=exact_match,
            reason="residual_fraction" if status != "exact" else "registered_residual",
        )

    def match_pattern(
        self,
        pattern: Any,
        candidates: Sequence[Any],
        *,
        threshold: float = 0.8,
    ) -> MatchResult:
        if not candidates:
            return MatchResult(
                status="novel",
                candidate_signature=self.pattern_signature(pattern),
                similarity=0.0,
                alignment_score=0.0,
                matched_index=-1,
                composition_parts=(),
                composition_coverage=0.0,
                residual_sequence=(),
                residual_fraction=1.0,
                exact_match=False,
                reason="no_candidates",
            )

        target_sig = self.pattern_signature(pattern)
        for idx, candidate in enumerate(candidates):
            if self.pattern_signature(candidate) == target_sig:
                return MatchResult(
                    status="exact",
                    candidate_signature=target_sig,
                    similarity=1.0,
                    alignment_score=1.0,
                    matched_index=idx,
                    composition_parts=(),
                    composition_coverage=0.0,
                    residual_sequence=(),
                    residual_fraction=0.0,
                    exact_match=True,
                    reason="pattern_signature_seen",
                )

        best_idx = -1
        best_sim = 0.0
        for idx, candidate in enumerate(candidates):
            sim = self.pattern_similarity(pattern, candidate)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx >= 0 and best_sim >= threshold:
            status = "equivalent" if best_sim >= 0.92 else "near"
            residual_fraction = float(max(0.0, 1.0 - best_sim))
            return MatchResult(
                status=status,
                candidate_signature=self.pattern_signature(candidates[best_idx]),
                similarity=float(best_sim),
                alignment_score=float(best_sim),
                matched_index=best_idx,
                composition_parts=(),
                composition_coverage=0.0,
                residual_sequence=(),
                residual_fraction=residual_fraction,
                exact_match=False,
                reason="structural_similarity",
            )

        return MatchResult(
            status="novel",
            candidate_signature=target_sig,
            similarity=float(best_sim),
            alignment_score=float(best_sim),
            matched_index=-1,
            composition_parts=(),
            composition_coverage=0.0,
            residual_sequence=(),
            residual_fraction=1.0,
            exact_match=False,
            reason="below_threshold",
        )
