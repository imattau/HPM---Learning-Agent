"""Cross-Language Transfer (CLT) adapters using Tree-sitter."""

from __future__ import annotations

import io
import keyword
import tokenize
from dataclasses import dataclass, field
from typing import Any, Mapping

import tree_sitter_python as tspython
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

from .packet import AdapterPacket
from ..core import State


# Universal Vocabulary for Structural Invariants
U_IF = "U_IF"
U_WHILE = "U_WHILE"
U_FOR = "U_FOR"
U_TRY = "U_TRY"
U_CATCH = "U_CATCH"
U_THROW = "U_THROW"
U_NULL = "U_NULL"
U_ASSIGN = "U_ASSIGN"
U_RETURN = "U_RETURN"
U_RESOURCE = "U_RESOURCE"
U_CALL = "U_CALL"
U_SYM = "U_SYM"

# Tree-sitter node type mappings
TS_MAP = {
    # Python
    "if_statement": U_IF,
    "while_statement": U_WHILE,
    "for_statement": U_FOR,
    "with_statement": U_RESOURCE,
    "try_statement": U_TRY,
    "except_clause": U_CATCH,
    "raise_statement": U_THROW,
    "none": U_NULL,
    "assignment": U_ASSIGN,
    "return_statement": U_RETURN,
    "call": U_CALL,
    "identifier": U_SYM,
    
    # Java
    "if_statement": U_IF,
    "while_statement": U_WHILE,
    "for_statement": U_FOR,
    "try_with_resources_statement": U_RESOURCE,
    "try_statement": U_TRY,
    "catch_clause": U_CATCH,
    "throw_statement": U_THROW,
    "null_literal": U_NULL,
    "assignment_expression": U_ASSIGN,
    "return_statement": U_RETURN,
    "method_invocation": U_CALL,
}


class UnifiedVocabulary:
    """Shared mapping for string tokens to numeric IDs."""
    _type_map: dict[str, float] = {}

    @classmethod
    def get_id(cls, name: str) -> float:
        if name not in cls._type_map:
            cls._type_map[name] = float(len(cls._type_map) + 1)
        return cls._type_map[name]


@dataclass(slots=True)
class LanguageDetector:
    """Determine the source language of a code string."""
    
    name: str = "language_detector"
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=lambda: ["language"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw_code = packet.raw
        if not isinstance(raw_code, str):
            packet.context["language"] = "unknown"
            return packet
            
        # Heuristics
        if "def " in raw_code or "import " in raw_code:
            packet.context["language"] = "python"
        elif "public class" in raw_code or "System.out.println" in raw_code or "String[] args" in raw_code:
            packet.context["language"] = "java"
        else:
            # Default to python for small snippets if ambiguous
            packet.context["language"] = "python"
            
        return packet


@dataclass(slots=True)
class UnifiedASTFlattener:
    """Linearise AST using tree-sitter with unified node names."""
    
    name: str = "unified_ast_flattener"
    requires: list[str] = field(default_factory=lambda: ["language_detector"])
    provides: list[str] = field(default_factory=lambda: ["unified_ast"])
    
    _parsers: dict[str, Parser] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        py_lang = Language(tspython.language())
        java_lang = Language(tsjava.language())
        
        self._parsers["python"] = Parser(py_lang)
        self._parsers["java"] = Parser(java_lang)

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw_code = packet.raw
        lang = packet.context.get("language", "python")
        parser = self._parsers.get(lang)
        
        if not parser or not isinstance(raw_code, str):
            packet.context["unified_ast"] = ()
            return packet
            
        tree = parser.parse(bytes(raw_code, "utf8"))
        flat_ast = []
        
        # Traverse tree
        cursor = tree.walk()
        reached_root = False
        while not reached_root:
            node_type = cursor.node.type
            unified = TS_MAP.get(node_type)
            if unified:
                flat_ast.append(unified)
            
            if cursor.goto_first_child():
                continue
            if cursor.goto_next_sibling():
                continue
            
            retracing = True
            while retracing:
                if not cursor.goto_parent():
                    retracing = False
                    reached_root = True
                if cursor.goto_next_sibling():
                    retracing = False
        
        packet.context["unified_ast"] = tuple(flat_ast)
        return packet


@dataclass(slots=True)
class UnifiedStateAdapter:
    """Map unified AST nodes to numeric IDs for HPM core."""
    
    name: str = "clt_state"
    requires: list[str] = field(default_factory=lambda: ["unified_ast_flattener"])
    provides: list[str] = field(default_factory=lambda: ["states"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        unified_ast = packet.context.get("unified_ast", ())
        state_val = tuple(UnifiedVocabulary.get_id(t) for t in unified_ast)
        
        packet.states.append(State(
            value=state_val,
            context={**packet.context, "clt_depth": len(state_val)}
        ))
        return packet


@dataclass(slots=True)
class CLTRefinementAdapter:
    """Validate and refine cross-language structural predictions with core feedback."""
    
    name: str = "clt_refinement"
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=lambda: ["validated_output", "feedback"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        if packet.core_action is None or packet.core_action.selected_pattern is None:
            return packet
            
        pattern = packet.core_action.selected_pattern
        context = packet.context
        lang = context.get("language", "python")
        node_name = context.get("raw_observation")
        
        # Simple structural feedback logic
        # If the pattern was learned in Python but fits the Java structure,
        # we give it a 'Coherence Reward'.
        is_valid = True
        
        # Example rule: U_TRY should eventually lead to U_CATCH in many patterns
        if node_name == "U_TRY" and lang == "java":
             # If the pattern predicted by core matches the expected universal sequence
             pass
             
        if is_valid:
            # Feedback into the core: reward the pattern for being coherent in the new language
            pattern.reward(0.05) # "Coherence Reward"
            packet.context["clt_feedback"] = "coherent"
        else:
            pattern.reward(-0.05) # "Structural Penalty"
            packet.context["clt_feedback"] = "incoherent"
            
        packet.validated_output = packet.core_action.value
        return packet
