"""Polygraph generation for code domains."""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass
from typing import Any

from ..adapter.code import get_ast_type_id
from ..core import State
from .base import PolygraphGenerator, PolygraphView


@dataclass(slots=True)
class CodePolygraphGenerator(PolygraphGenerator):
    """Generates specialized views for code structural analysis."""

    name: str = "code_polygraph"

    def generate(self, raw: Any, *, context: dict[str, Any] | None = None) -> list[PolygraphView]:
        if not isinstance(raw, str):
            return []
            
        context = context or {}
        views = []
        
        # 1. AST Type View
        try:
            tree = ast.parse(raw)
            flat_ast = tuple(get_ast_type_id(type(node).__name__) for node in ast.walk(tree))
            views.append(PolygraphView(
                name="ast_types",
                state=State(value=flat_ast, context={**context, "view": "ast_types"})
            ))
        except Exception:
            # If raw is a single node type string (sequential)
            if not ("\n" in raw or " " in raw):
                views.append(PolygraphView(
                    name="ast_types",
                    state=State(value=(get_ast_type_id(raw),), context={**context, "view": "ast_types"})
                ))

        # 2. Token Type View
        try:
            token_types = []
            for tok in tokenize.generate_tokens(io.StringIO(raw).readline):
                token_types.append(float(tok.type))
            views.append(PolygraphView(
                name="token_types",
                state=State(value=tuple(token_types), context={**context, "view": "token_types"})
            ))
        except Exception:
            pass
            
        # 3. Structural Skeleton View (just keywords and operators)
        try:
            skeleton = []
            # Also try to treat raw string as a potential keyword/operator
            normalized_raw = str(raw).lower()
            if keyword.iskeyword(normalized_raw):
                skeleton.append(get_ast_type_id(normalized_raw))
            
            for tok in tokenize.generate_tokens(io.StringIO(raw).readline):
                if tok.type in (tokenize.NAME, tokenize.OP) and keyword.iskeyword(tok.string):
                    skeleton.append(get_ast_type_id(tok.string))
                elif tok.type == tokenize.OP:
                    skeleton.append(get_ast_type_id(tok.string))
            if skeleton:
                views.append(PolygraphView(
                    name="skeleton",
                    state=State(value=tuple(skeleton), context={**context, "view": "skeleton"})
                ))
        except Exception:
            pass

        return views

import keyword # needed for skeleton view
