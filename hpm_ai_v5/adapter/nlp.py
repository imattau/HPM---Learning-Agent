"""NLP adapters for structural language processing in HPM v5."""

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
    # Use blank model as we don't have internet to download full models
    nlp: Any = field(default_factory=lambda: spacy.blank("en"), init=False)
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=lambda: ["tokens", "pos_tags", "lemmas"])

    # Simple rule-based POS tagger for structural benchmark
    POS_MAP: dict[str, str] = field(default_factory=lambda: {
        "what": "PRON", "is": "VERB", "the": "DET", "a": "DET", "an": "DET",
        "in": "ADP", "from": "ADP", "to": "ADP", "on": "ADP", "for": "ADP",
        "show": "VERB", "tell": "VERB", "book": "VERB", "reserve": "VERB",
        "buy": "VERB", "purchase": "VERB", "acquire": "VERB", "want": "VERB",
        "flights": "NOUN", "flight": "NOUN", "weather": "NOUN", "forecast": "NOUN",
        "conditions": "NOUN", "city": "NOUN", "town": "NOUN", "status": "NOUN",
        "plane": "NOUN", "journey": "NOUN", "laptop": "NOUN", "ticket": "NOUN",
        "me": "PRON", "it": "PRON", "i": "PRON", "you": "PRON", "user": "NOUN", "task": "NOUN",
        "and": "CCONJ", "or": "CCONJ", "if": "SCONJ", "while": "SCONJ",
        "authenticated": "ADJ", "active": "ADJ", "valid": "ADJ", "immediately": "ADV",
        "like": "VERB", "would": "AUX",
    })

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        raw = packet.raw
        if not isinstance(raw, str):
            packet.context["tokens"] = [str(raw)]
            packet.context["pos_tags"] = ["UNKNOWN"]
            packet.context["lemmas"] = [str(raw).lower()]
            return packet

        doc = self.nlp(raw)
        tokens = []
        pos_tags = []
        lemmas = []
        
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            t = token.text.lower()
            tokens.append(t)
            # Use POS_MAP or fallback to NOUN for unknown words in this simple implementation
            pos = self.POS_MAP.get(t, "NOUN")
            pos_tags.append(pos)
            lemmas.append(t) # Simple lemma for now
        
        packet.context["tokens"] = tokens
        packet.context["pos_tags"] = pos_tags
        packet.context["lemmas"] = lemmas
        return packet


@dataclass(slots=True)
class CanonicalPhraser(Adapter):
    """Map natural language phrases to a fixed concept vocabulary."""

    name: str = "canonical_phraser"
    concept_map: dict[str, list[str]] = field(default_factory=lambda: {
        "BUY": ["buy", "purchase", "acquire", "get", "obtain"],
        "SELL": ["sell", "sell off", "dispose", "vend"],
        "WEATHER": ["weather", "forecast", "conditions", "temperature"],
        "CITY": ["city", "town", "municipality", "metropolis"],
        "FLIGHT": ["flight", "plane", "air travel", "journey"],
        "BOOK": ["book", "reserve", "schedule", "arrange"],
    })
    requires: list[str] = field(default_factory=lambda: ["nlp_tokenizer"])
    provides: list[str] = field(default_factory=lambda: ["canonical_tokens", "state"])
    
    _lemma_to_concept: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self):
        for concept, lemmas in self.concept_map.items():
            for lemma in lemmas:
                self._lemma_to_concept[lemma] = concept

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        lemmas = packet.context.get("lemmas", [])
        canonical_tokens = []
        
        for lemma in lemmas:
            concept = self._lemma_to_concept.get(lemma, lemma)
            canonical_tokens.append(concept)

        packet.context["canonical_tokens"] = canonical_tokens
        
        if canonical_tokens:
            t = canonical_tokens[0]
            # Map first token to state for basic HPM use
            state_value = (float(UnifiedVocabulary.get_id(t)),)
        else:
            state_value = ()
        
        packet.states.append(State(value=state_value, context=packet.context))
        return packet


@dataclass(slots=True)
class SkeletonExtractor(Adapter):
    """Extract grammatical skeleton as sequence of POS tag groups."""

    name: str = "skeleton_extractor"
    # Map coarse POS tags to skeleton types
    POS_GROUP: dict[str, str] = field(default_factory=lambda: {
        "NOUN": "N", "PROPN": "N",
        "VERB": "V",
        "ADJ": "A", "ADV": "A",
        "DET": "D", "PRON": "P",
        "ADP": "R", "CCONJ": "C", "SCONJ": "C",
        "NUM": "NUM",
        "PUNCT": "PUNCT",
    })
    collapse_repeats: bool = True
    requires: list[str] = field(default_factory=lambda: ["nlp_tokenizer"])
    provides: list[str] = field(default_factory=lambda: ["skeleton", "state"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        pos_tags = packet.context.get("pos_tags", [])
        skeleton = []
        
        for pos in pos_tags:
            group = self.POS_GROUP.get(pos, pos)
            if self.collapse_repeats and skeleton and skeleton[-1] == group:
                continue
            skeleton.append(group)
            
        packet.context["skeleton"] = skeleton
        packet.context["skeleton_str"] = " ".join(skeleton)
        
        state_value = tuple(float(UnifiedVocabulary.get_id(s)) for s in skeleton)
        packet.states.append(State(value=state_value, context=packet.context))
        return packet


@dataclass(slots=True)
class SkeletonNgramAdapter(Adapter):
    """Encode skeleton as bigram/trigram transition IDs, capturing word-order constraints."""

    name: str = "skeleton_ngram"
    n: int = 2
    requires: list[str] = field(default_factory=lambda: ["skeleton_extractor"])
    provides: list[str] = field(default_factory=lambda: ["skeleton_ngrams", "state"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        skeleton = packet.context.get("skeleton", [])
        if len(skeleton) < self.n:
            return packet
        ngrams = [
            "_".join(skeleton[i : i + self.n])
            for i in range(len(skeleton) - self.n + 1)
        ]
        packet.context["skeleton_ngrams"] = ngrams
        # Store in context only — polygraph picks this up via skeleton_bigram_view.
        # Not appended to packet.states to avoid displacing the skeleton state in the main engine.
        return packet


@dataclass(slots=True)
class DeltaEncoder(Adapter):
    """Compute structural deltas between consecutive skeleton states."""

    name: str = "delta_encoder"
    requires: list[str] = field(default_factory=lambda: ["skeleton_extractor"])
    provides: list[str] = field(default_factory=lambda: ["delta", "state"])
    
    _previous_skeleton: tuple[float, ...] | None = field(default=None, init=False)

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        # We need to find the skeleton state value
        current_skeleton = None
        for state in reversed(packet.states):
            if state.context.get("domain") == "nlp_skeleton" or "skeleton" in state.context:
                current_skeleton = state.value
                break
        
        if current_skeleton is None:
            return packet

        delta = ()
        if self._previous_skeleton is not None:
            # Simple delta: first differing position (old, new)
            min_len = min(len(self._previous_skeleton), len(current_skeleton))
            diff_idx = -1
            for i in range(min_len):
                if self._previous_skeleton[i] != current_skeleton[i]:
                    diff_idx = i
                    break
            if diff_idx == -1 and len(self._previous_skeleton) != len(current_skeleton):
                diff_idx = min_len
            
            if diff_idx != -1:
                old = self._previous_skeleton[diff_idx] if diff_idx < len(self._previous_skeleton) else -1.0
                new = current_skeleton[diff_idx] if diff_idx < len(current_skeleton) else -1.0
                delta = (old, new)
        
        self._previous_skeleton = current_skeleton
        packet.context["delta"] = delta
        packet.states.append(State(value=delta, context=packet.context))
        return packet

    def reset(self):
        self._previous_skeleton = None


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
        "validate": ["check", "if", "verify"],
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
