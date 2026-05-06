"""NLP polygraphs for Symbolic Pattern Matching in HPM v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import PolygraphGenerator, PolygraphView
from ..core.state import State
from ..adapter.clt import UnifiedVocabulary


@dataclass(slots=True)
class NLPPolygraphGenerator(PolygraphGenerator):
    """Generates multiple views from a natural language query."""

    name: str = "nlp_polygraph"

    def generate(self, raw: Any, *, context: dict[str, Any] | None = None) -> list[PolygraphView]:
        if not isinstance(raw, str):
            # If raw is already a token, we handle it as a single-view state
            token_id = float(UnifiedVocabulary.get_id(str(raw)))
            return [PolygraphView(name="token_view", state=State(value=(token_id,), context={**context, "view": "tokens"}))]
            
        context = context or {}
        tokens = context.get("tokens", [])
        canonical_tokens = context.get("canonical_tokens", [])
        
        views = []
        
        # 1. Token View (Raw surface tokens)
        token_values = tuple(float(UnifiedVocabulary.get_id(t)) for t in tokens)
        views.append(PolygraphView(
            name="token_view",
            state=State(value=token_values, context={**context, "view": "tokens"})
        ))
        
        # 2. Canonical View (Synonym-normalized tokens)
        canonical_values = tuple(float(UnifiedVocabulary.get_id(t)) for t in canonical_tokens)
        views.append(PolygraphView(
            name="canonical_view",
            state=State(value=canonical_values, context={**context, "view": "canonical"})
        ))
        
        # 3. Skeleton View (Placeholders and high-signal words only)
        skeleton = [t for t in canonical_tokens if t.startswith("PARAM_") or len(t) > 3]
        skeleton_values = tuple(float(UnifiedVocabulary.get_id(t)) for t in skeleton)
        views.append(PolygraphView(
            name="skeleton_view",
            state=State(value=skeleton_values, context={**context, "view": "skeleton"})
        ))
        
        # 4. Synonym/Semantic Candidate View (New Task)
        # In a real system, this would call an external dictionary (WordNet/spaCy)
        # Here we use the context-provided 'semantic_candidates' or a simple proxy
        candidates = context.get("semantic_candidates", [])
        if candidates:
            candidate_values = tuple(float(UnifiedVocabulary.get_id(c)) for c in candidates)
            views.append(PolygraphView(
                name="semantic_view",
                state=State(value=candidate_values, context={**context, "view": "semantic"})
            ))
            
        return views
