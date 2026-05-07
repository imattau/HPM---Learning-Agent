"""NLP polygraphs for structural language processing in HPM v5."""

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
            token_id = float(UnifiedVocabulary.get_id(str(raw)))
            return [PolygraphView(name="token_view", state=State(value=(token_id,), context={**context, "view": "tokens"}))]
            
        context = context or {}
        tokens = context.get("tokens", [])
        canonical_tokens = context.get("canonical_tokens", [])
        skeleton = context.get("skeleton", [])
        delta = context.get("delta", ())
        
        views = []
        
        # 1. Token View (Raw surface tokens)
        if tokens:
            token_values = tuple(float(UnifiedVocabulary.get_id(t)) for t in tokens)
            views.append(PolygraphView(
                name="token_view",
                state=State(value=token_values, context={**context, "view": "tokens"})
            ))
        
        # 2. Canonical View (Synonym-normalized tokens)
        if canonical_tokens:
            canonical_values = tuple(float(UnifiedVocabulary.get_id(t)) for t in canonical_tokens)
            views.append(PolygraphView(
                name="canonical_view",
                state=State(value=canonical_values, context={**context, "view": "canonical"})
            ))
        
        # 3. Skeleton View (POS-based structure)
        if skeleton:
            skeleton_values = tuple(float(UnifiedVocabulary.get_id(s)) for s in skeleton)
            views.append(PolygraphView(
                name="skeleton_view",
                state=State(value=skeleton_values, context={**context, "view": "skeleton"})
            ))
            
        # 4. Delta View (Structural changes)
        if delta:
            views.append(PolygraphView(
                name="delta_view",
                state=State(value=delta, context={**context, "view": "delta"})
            ))
        
        # 5. Semantic View (Synonym/Semantic Candidate View)
        # We try to map ALL tokens to their semantic candidates' canonical IDs if possible
        # For simplicity in this benchmark, we just use the candidates directly.
        candidates = context.get("semantic_candidates", [])
        if candidates:
            # We take the first candidate as the primary hypothesis for the state value
            # This allows matching against patterns learned from that candidate.
            for cand in candidates[:3]: # Try first few as separate views? 
                # Actually, PolygraphGenerator expects a list of views.
                # We can have multiple semantic views!
                cand_id = float(UnifiedVocabulary.get_id(cand))
                views.append(PolygraphView(
                    name=f"semantic_view_{cand}",
                    state=State(value=(cand_id,), context={**context, "view": "semantic", "candidate": cand})
                ))
            
        return views
