"""NLP polygraphs for structural language processing in HPM v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import PolygraphGenerator, PolygraphView
from ..core.state import State
from ..shared_vocab import UnifiedVocabulary


@dataclass(slots=True)
class NLPPolygraphGenerator(PolygraphGenerator):
    """Generates multiple views from a natural language query."""

    name: str = "nlp_polygraph"

    def generate(self, raw: Any, *, context: dict[str, Any] | None = None) -> list[PolygraphView]:
        if not isinstance(raw, str):
            token_id = float(UnifiedVocabulary.get_id(str(raw)))
            return [PolygraphView(name="token_view", state=State(value=(token_id,), context={**context, "view": "tokens"}))]
            
        context = context or {}
        tokens = context.get("tokens", [])
        canonical_tokens = context.get("canonical_tokens", [])
        skeleton = context.get("skeleton", [])
        delta = context.get("delta", ())
        
        views = []
        # Origin state for all structural views
        origin = State(value=(), context={**context, "view": "origin"})
        
        # 1. Token View (Raw surface tokens)
        if tokens:
            token_values = tuple(float(UnifiedVocabulary.get_id(t)) for t in tokens)
            v_state = State(value=token_values, context={**context, "view": "tokens"})
            views.append(PolygraphView(
                name="token_view",
                state=v_state,
                states=[origin, v_state]
            ))
        
        # 2. Canonical View (Synonym-normalized tokens)
        if canonical_tokens:
            canonical_values = tuple(float(UnifiedVocabulary.get_id(t)) for t in canonical_tokens)
            v_state = State(value=canonical_values, context={**context, "view": "canonical"})
            views.append(PolygraphView(
                name="canonical_view",
                state=v_state,
                states=[origin, v_state]
            ))
        
        # 3. Skeleton View (POS-based structure)
        if skeleton:
            skeleton_values = tuple(float(UnifiedVocabulary.get_id(s)) for s in skeleton)
            v_state = State(value=skeleton_values, context={**context, "view": "skeleton"})
            views.append(PolygraphView(
                name="skeleton_view",
                state=v_state,
                states=[origin, v_state]
            ))
            
        # 4. Skeleton Bigram View (sequential ordering constraints)
        ngrams = context.get("skeleton_ngrams", [])
        if ngrams:
            ngram_values = tuple(float(UnifiedVocabulary.get_id(ng)) for ng in ngrams)
            v_state = State(value=ngram_values, context={**context, "view": "skeleton_bigram"})
            views.append(PolygraphView(
                name="skeleton_bigram_view",
                state=v_state,
                states=[origin, v_state]
            ))

        # 5. Delta View (Structural changes)
        if delta:
            v_state = State(value=delta, context={**context, "view": "delta"})
            views.append(PolygraphView(
                name="delta_view",
                state=v_state,
                states=[origin, v_state]
            ))
        
        # 5. Semantic View (Synonym/Semantic Candidate View)
        candidates = context.get("semantic_candidates", [])
        if candidates:
            for cand in candidates[:3]:
                cand_id = float(UnifiedVocabulary.get_id(cand))
                v_state = State(value=(cand_id,), context={**context, "view": "semantic", "candidate": cand})
                views.append(PolygraphView(
                    name=f"semantic_view_{cand}",
                    state=v_state,
                    states=[origin, v_state]
                ))
            
        return views


@dataclass(slots=True)
class StructuralNLPPolygraphGenerator(PolygraphGenerator):
    """Lightweight NLP polygraph — skeleton and bigram views only.

    Use instead of NLPPolygraphGenerator when semantic candidate views would
    cause view engine explosion (e.g. large corpora with WordNet KB lookup).
    """

    name: str = "structural_nlp_polygraph"

    def generate(self, raw: Any, *, context: dict[str, Any] | None = None) -> list[PolygraphView]:
        if not isinstance(raw, str):
            token_id = float(UnifiedVocabulary.get_id(str(raw)))
            return [PolygraphView(name="token_view", state=State(value=(token_id,), context={**(context or {}), "view": "tokens"}))]

        context = context or {}
        tokens = context.get("tokens", [])
        canonical_tokens = context.get("canonical_tokens", [])
        content_vector = context.get("content_vector", ())
        skeleton = context.get("skeleton", [])
        ngrams = context.get("skeleton_ngrams", [])
        views = []
        origin = State(value=(), context={**context, "view": "origin"})

        # 1. Token view — discriminates specific surface forms
        if tokens:
            token_values = tuple(float(UnifiedVocabulary.get_id(t)) for t in tokens)
            v_state = State(value=token_values, context={**context, "view": "tokens"})
            views.append(PolygraphView(name="token_view", state=v_state, states=[origin, v_state]))

        # 2. Canonical view — synonym-normalised tokens
        if canonical_tokens:
            canonical_values = tuple(float(UnifiedVocabulary.get_id(t)) for t in canonical_tokens)
            v_state = State(value=canonical_values, context={**context, "view": "canonical"})
            views.append(PolygraphView(name="canonical_view", state=v_state, states=[origin, v_state]))

        # 3. Content word view — mean-pooled unit vector of domain noun embeddings
        #    Similar intents cluster in embedding space; MAE on unit vectors ≈ cosine distance
        if content_vector:
            v_state = State(value=content_vector, context={**context, "view": "content"})
            views.append(PolygraphView(name="content_view", state=v_state, states=[origin, v_state]))

        # 5. Skeleton view — POS-based structure
        if skeleton:
            skeleton_values = tuple(float(UnifiedVocabulary.get_id(s)) for s in skeleton)
            v_state = State(value=skeleton_values, context={**context, "view": "skeleton"})
            views.append(PolygraphView(name="skeleton_view", state=v_state, states=[origin, v_state]))

        # 4. Skeleton bigram view — sequential ordering constraints
        if ngrams:
            ngram_values = tuple(float(UnifiedVocabulary.get_id(ng)) for ng in ngrams)
            v_state = State(value=ngram_values, context={**context, "view": "skeleton_bigram"})
            views.append(PolygraphView(name="skeleton_bigram_view", state=v_state, states=[origin, v_state]))

        return views


@dataclass(slots=True)
class InterconnectedNLPPolygraphGenerator(PolygraphGenerator):
    """Structural NLP polygraph with bridge-anchor metadata on each view."""

    name: str = "interconnected_nlp_polygraph"

    def generate(self, raw: Any, *, context: dict[str, Any] | None = None) -> list[PolygraphView]:
        base_views = StructuralNLPPolygraphGenerator().generate(raw, context=context)
        context = context or {}
        view_anchor_map = context.get("atis_view_anchor_map", {})
        enriched: list[PolygraphView] = []
        for view in base_views:
            metadata = view_anchor_map.get(view.name, {})
            enriched.append(PolygraphView(
                name=view.name,
                state=view.state,
                context=view.context,
                states=view.states,
                leaf_keys=tuple(metadata.get("leaf_keys", ())),
                anchor_ids=tuple(metadata.get("anchor_ids", ())),
                concept_ids=tuple(metadata.get("concept_ids", ())),
            ))
        return enriched
