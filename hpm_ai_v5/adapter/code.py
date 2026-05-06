"""Code-domain adapters for structural pattern recognition."""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass, field
from typing import Any, Mapping

from .packet import AdapterPacket
from ..core import State


@dataclass(slots=True)
class CodeTokenizer:
    """Convert raw code string into a sequence of token type IDs."""
    
    name: str = "code_tokenizer"
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=lambda: ["tokens"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw_code = packet.raw
        if not isinstance(raw_code, str):
            return packet
            
        tokens = []
        try:
            for tok in tokenize.generate_tokens(io.StringIO(raw_code).readline):
                # We store (type, string) but for the core we might just want type
                # or a canonicalized string.
                tokens.append((tok.type, tok.string))
        except Exception:
            pass
            
        packet.context["tokens"] = tokens
        return packet


@dataclass(slots=True)
class ASTFlattener:
    """Linearise an AST into a sequence of node types and structural hints."""
    
    name: str = "ast_flattener"
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=lambda: ["flat_ast"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw_code = packet.raw
        if not isinstance(raw_code, str):
            return packet
            
        try:
            tree = ast.parse(raw_code)
            flat_ast = []
            for node in ast.walk(tree):
                flat_ast.append(type(node).__name__)
            packet.context["flat_ast"] = tuple(flat_ast)
        except Exception:
            packet.context["flat_ast"] = ()
            
        return packet


@dataclass(slots=True)
class CanonicalRenamer:
    """Normalise identifiers in code to remove surface syntax noise."""
    
    name: str = "canonical_renamer"
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=lambda: ["canonical_code"])

    def _canonicalize_python(self, code: str) -> str:
        try:
            tree = ast.parse(code)
            # Collect all names first to ensure consistent mapping
            names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    names.add(node.id)
                elif isinstance(node, ast.FunctionDef):
                    names.add(node.name)
                elif isinstance(node, ast.arg):
                    names.add(node.arg)
            
            sorted_names = sorted(list(names))
            name_map = {name: f"SYM_{i}" for i, name in enumerate(sorted_names)}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    node.id = name_map[node.id]
                elif isinstance(node, ast.FunctionDef):
                    node.name = name_map[node.name]
                elif isinstance(node, ast.arg):
                    node.arg = name_map[node.arg]
            
            return ast.unparse(tree)
        except Exception:
            return code

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw_code = packet.raw
        if not isinstance(raw_code, str):
            return packet
            
        packet.context["canonical_code"] = self._canonicalize_python(raw_code)
        return packet


AST_TYPE_MAP: dict[str, float] = {}

def get_ast_type_id(type_name: str) -> float:
    if type_name not in AST_TYPE_MAP:
        AST_TYPE_MAP[type_name] = float(len(AST_TYPE_MAP) + 1)
    return AST_TYPE_MAP[type_name]

@dataclass(slots=True)
class CodeStateAdapter:
    """Convert code context into a HPM State for the core."""
    
    name: str = "code_state"
    requires: list[str] = field(default_factory=lambda: ["ast_flattener", "code_tokenizer"])
    provides: list[str] = field(default_factory=lambda: ["states"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        # If the raw input is already a single node type string (from sequential feeding)
        if isinstance(packet.raw, str) and not ("\n" in packet.raw or " " in packet.raw):
            state_val = (get_ast_type_id(packet.raw),)
        else:
            flat_ast = packet.context.get("flat_ast", ())
            state_val = tuple(get_ast_type_id(t) for t in flat_ast)
        
        packet.states.append(State(
            value=state_val,
            context={
                **packet.context,
                "ast_depth": len(state_val)
            }
        ))
        return packet


@dataclass(slots=True)
class CodeRefinementAdapter:
    """Validate and refine predicted structural nodes."""
    
    name: str = "code_refinement"
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=lambda: ["validated_output"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        if packet.core_action is None:
            return packet
            
        action_val = packet.core_action.value
        context = packet.context
        
        # Example refinement: if we just saw a 'For' node, we expect a 'Name' or 'Store'
        last_node = context.get("raw_observation")
        
        # Simple structural snap logic
        refined_val = action_val
        # if last_node == "For" and action_val != get_ast_type_id("Name"):
        #    ...
            
        packet.validated_output = refined_val
        packet.log(self.name, {"refined": refined_val}, role="adapter")
        return packet
