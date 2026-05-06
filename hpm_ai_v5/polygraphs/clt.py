"""Polygraph generation for Cross-Language Transfer (CLT)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..adapter.clt import UnifiedVocabulary
from ..core import State
from .base import PolygraphGenerator, PolygraphView


@dataclass(slots=True)
class CLTPolygraphGenerator(PolygraphGenerator):
    """Generates structural views from unified cross-language ASTs."""

    name: str = "clt_polygraph"

    def generate(self, raw: Any, *, context: dict[str, Any] | None = None) -> list[PolygraphView]:
        if not isinstance(raw, str):
            return []
            
        context = context or {}
        views = []
        
        # 1. Unified AST View
        unified_ast = context.get("unified_ast", ())
        if unified_ast:
            # Map strings to IDs
            id_seq = tuple(UnifiedVocabulary.get_id(t) for t in unified_ast)
            views.append(PolygraphView(
                name="unified_nodes",
                state=State(value=id_seq, context={**context, "view": "unified_nodes"})
            ))
            
            # 2. Control Flow Skeleton (only branch/loop constructs)
            skeleton_nodes = ["U_IF", "U_WHILE", "U_FOR", "U_TRY", "U_CATCH"]
            skeleton = tuple(UnifiedVocabulary.get_id(n) for n in unified_ast if n in skeleton_nodes)
            if skeleton:
                views.append(PolygraphView(
                    name="control_skeleton",
                    state=State(value=skeleton, context={**context, "view": "control_skeleton"})
                ))

            # 3. Functional Skeleton (Control Flow + Actions)
            func_nodes = ["U_IF", "U_WHILE", "U_FOR", "U_TRY", "U_CATCH", "U_CALL", "U_RETURN", "U_ASSIGN", "U_THROW"]
            func_skeleton = tuple(UnifiedVocabulary.get_id(n) for n in unified_ast if n in func_nodes)
            if func_skeleton:
                views.append(PolygraphView(
                    name="functional_skeleton",
                    state=State(value=func_skeleton, context={**context, "view": "functional_skeleton"})
                ))

        return views
