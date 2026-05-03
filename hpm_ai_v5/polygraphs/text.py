from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import spacy
from ..core import State
from .base import PolygraphView, PolygraphGenerator

@dataclass(slots=True)
class TextPolygraphGenerator(PolygraphGenerator):
    """Generate multiple linguistic views from raw text."""
    model_name: str = "en_core_web_sm"
    _nlp: Any = None # lazy load

    def __post_init__(self):
        self._nlp = spacy.load(self.model_name)

    def generate(self, raw: Any, *, context: dict[str, Any] | None = None) -> list[PolygraphView]:
        if not isinstance(raw, str):
            raise TypeError("TextPolygraphGenerator expects a string")
        doc = self._nlp(raw)
        context = dict(context or {})
        context.setdefault("domain", "text")
        views = []
        
        # 1. Token view (integer IDs from a fixed vocabulary)
        token_ids = [token.orth for token in doc]
        views.append(PolygraphView(
            name="tokens",
            state=State(value=tuple(token_ids), context={**context, "view": "tokens", "vocab": "spacy"}),
            context={**context, "view": "tokens"},
        ))
        
        # 2. POS tag view (universal tagset IDs)
        tag_map = {"NOUN": 0, "VERB": 1, "ADJ": 2, "ADV": 3, "DET": 4, "ADP": 5, "CONJ": 6, "PRON": 7, "NUM": 8, "PUNCT": 9}
        tag_ids = [tag_map.get(token.pos_, 0) for token in doc]
        views.append(PolygraphView(
            name="pos_tags",
            state=State(value=tuple(tag_ids), context={**context, "view": "pos"}),
            context={**context, "view": "pos"},
        ))
        
        # 3. Dependency view: flattened (head_idx, dep_id) per token
        dep_ids = [token.dep for token in doc]
        head_indices = [token.head.i for token in doc]
        flat_deps = []
        for i, (head, dep) in enumerate(zip(head_indices, dep_ids)):
            flat_deps.append(head)
            flat_deps.append(dep)
        views.append(PolygraphView(
            name="dependencies",
            state=State(value=tuple(flat_deps), context={**context, "view": "dep"}),
            context={**context, "view": "dep"},
        ))
        
        # 4. Lemma view (lemmas mapped to integers using spacy hash)
        lemma_ids = [token.lemma for token in doc]
        views.append(PolygraphView(
            name="lemmas",
            state=State(value=tuple(lemma_ids), context={**context, "view": "lemma"}),
            context={**context, "view": "lemma"},
        ))
        return views
