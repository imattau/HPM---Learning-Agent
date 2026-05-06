"""NLP adapters for Symbolic Pattern Matching in HPM v5."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import spacy
from .base import Adapter
from .packet import AdapterPacket
from .clt import UnifiedVocabulary
from ..core.state import State


@dataclass(slots=True)
class NLPTokenizer(Adapter):
    """Tokenizer for natural language queries using spaCy."""

    name: str = "nlp_tokenizer"
    nlp: Any = field(default_factory=lambda: spacy.blank("en"), init=False)
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=lambda: ["tokens", "pos_tags"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw = packet.raw
        if not isinstance(raw, str):
            packet.context["tokens"] = [str(raw)]
            packet.context["pos_tags"] = ["UNKNOWN"]
            return packet

        doc = self.nlp(raw)
        tokens = [t.text.lower() for t in doc]
        pos_tags = [t.pos_ for t in doc]
        
        packet.context["tokens"] = tokens
        packet.context["pos_tags"] = pos_tags
        return packet


@dataclass(slots=True)
class CanonicalPhraser(Adapter):
    """Replaces synonyms and variable phrases with placeholders."""

    name: str = "canonical_phraser"
    synonyms: dict[str, str] = field(default_factory=dict)
    placeholders: dict[str, str] = field(default_factory=dict)
    requires: list[str] = field(default_factory=lambda: ["nlp_tokenizer"])
    provides: list[str] = field(default_factory=lambda: ["canonical_tokens", "state"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        tokens = packet.context.get("tokens", [])
        canonical_tokens = []
        bindings = {}
        
        for token in tokens:
            canon = self.synonyms.get(token, token)
            if token in self.placeholders:
                placeholder = self.placeholders[token]
                canonical_tokens.append(placeholder)
                bindings[placeholder] = token
            else:
                canonical_tokens.append(canon)

        packet.context["canonical_tokens"] = canonical_tokens
        packet.context["parameter_bindings"] = bindings
        
        if canonical_tokens:
            t = canonical_tokens[0]
            state_value = (float(UnifiedVocabulary.get_id(t)),)
        else:
            state_value = ()
        
        packet.states.append(State(value=state_value, context=packet.context))
        return packet


@dataclass(slots=True)
class ToolSchemaEncoder(Adapter):
    """Encodes tool signatures into HPM patterns."""

    name: str = "tool_schema_encoder"
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=lambda: ["tool_pattern"])

    def encode(self, tool_name: str, params: list[str]) -> tuple[float, ...]:
        tool_id = float(UnifiedVocabulary.get_id(tool_name))
        param_ids = [float(UnifiedVocabulary.get_id(p)) for p in params]
        return (tool_id,) + tuple(param_ids)

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        return packet


@dataclass(slots=True)
class KnowledgeBaseLookup(Adapter):
    """Simulates external dictionary lookup for synonyms and concepts."""

    name: str = "kb_lookup"
    # Simulated external knowledge base
    KNOWLEDGE_BASE: dict[str, list[str]] = field(default_factory=lambda: {
        "check": ["if", "validate", "verify"],
        "keep": ["while", "loop", "repeat"],
        "perform": ["call", "run", "execute"],
        "store": ["set", "assign"],
        "output": ["return", "yield"],
    })
    requires: list[str] = field(default_factory=lambda: ["nlp_tokenizer"])
    provides: list[str] = field(default_factory=lambda: ["semantic_candidates"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        tokens = packet.context.get("tokens", [])
        all_candidates = []
        for token in tokens:
            candidates = self.KNOWLEDGE_BASE.get(token.lower(), [])
            all_candidates.extend(candidates)
        
        packet.context["semantic_candidates"] = list(set(all_candidates))
        return packet


@dataclass(slots=True)
class NL2CodeBridgeAdapter(Adapter):
    """Maps functional NL tokens to Universal Code structural IDs (U_*)."""

    name: str = "nl2code_bridge"
    requires: list[str] = field(default_factory=lambda: ["canonical_phraser"])
    provides: list[str] = field(default_factory=lambda: ["bridge_tokens", "states"])

    # Heuristic mapping from functional concepts to universal structural IDs
    CONCEPT_MAP: dict[str, str] = field(default_factory=lambda: {
        "if": "U_IF",
        "whether": "U_IF",
        "check": "U_IF",
        "conditional": "U_IF",
        "loop": "U_WHILE",
        "while": "U_WHILE",
        "keep": "U_WHILE",
        "repeat": "U_FOR",
        "for": "U_FOR",
        "set": "U_ASSIGN",
        "assign": "U_ASSIGN",
        "store": "U_ASSIGN",
        "call": "U_CALL",
        "run": "U_CALL",
        "invoke": "U_CALL",
        "execute": "U_CALL",
        "send": "U_CALL",
        "mail": "U_CALL",
        "return": "U_RETURN",
        "output": "U_RETURN",
        "error": "U_THROW",
        "fail": "U_THROW",
        "try": "U_TRY",
        "handle": "U_CATCH",
    })

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        tokens = packet.context.get("canonical_tokens", [])
        bridge_tokens = []
        for token in tokens:
            concept = self.CONCEPT_MAP.get(token.lower())
            if concept:
                bridge_tokens.append(concept)
        
        packet.context["bridge_tokens"] = bridge_tokens
        
        if bridge_tokens:
            state_values = tuple(float(UnifiedVocabulary.get_id(t)) for t in bridge_tokens)
            packet.states.append(State(
                value=state_values,
                context={**packet.context, "bridge_depth": len(state_values)}
            ))
        
        return packet
