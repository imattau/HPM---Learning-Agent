"""NLP adapters for structural language processing in HPM v5."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import spacy
from .base import Adapter
from .packet import AdapterPacket
from ..core.state import State
from ..shared_vocab import UnifiedVocabulary


@dataclass(slots=True)
class StartOfEpisodeAdapter(Adapter):
    """Prepends a null START state to every step, enabling single-sentence recognition."""

    name: str = "start_of_episode"
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=lambda: ["start_state"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        # Prepend START state to packet.states
        # Use an empty tuple value to represent the origin
        packet.states.insert(0, State(value=(), context={**packet.context, "domain": "start"}))
        return packet


@dataclass(slots=True)
class NLPTokenizer(Adapter):
    """Tokenizer for natural language queries using spaCy."""

    name: str = "nlp_tokenizer"
    nlp: Any = field(default_factory=lambda: spacy.load("en_core_web_sm"), init=False)
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=lambda: ["tokens", "pos_tags", "lemmas"])

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
        token_vectors = []

        for token in doc:
            if token.is_punct or token.is_space:
                continue
            tokens.append(token.text.lower())
            pos_tags.append(token.pos_)
            lemmas.append(token.lemma_.lower())
            token_vectors.append(token.vector)

        ent_types = []
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            ent_types.append(token.ent_type_ if token.ent_type_ else "")

        packet.context["tokens"] = tokens
        packet.context["pos_tags"] = pos_tags
        packet.context["lemmas"] = lemmas
        packet.context["ent_types"] = ent_types
        packet.context["token_vectors"] = token_vectors
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
        "FLIGHT": ["flight", "plane", "air travel", "journey", "fly", "trip"],
        "BOOK": ["book", "reserve", "schedule", "arrange"],
        "MEAL": ["meal", "food", "dinner", "lunch", "breakfast", "snack"],
        "AIRFARE": ["airfare", "fare", "price", "cost", "ticket", "charge"],
        "AIRPORT": ["airport", "hub", "terminal", "airfield"],
        "DISTANCE": ["distance", "mile", "kilometer", "length"],
        "CAPACITY": ["capacity", "seat", "seating", "size", "hold", "passenger"],
        "TIME": ["time", "arrive", "depart", "leave", "schedule", "when", "arrival", "departure"],
    })
    requires: list[str] = field(default_factory=lambda: ["nlp_tokenizer"])
    provides: list[str] = field(default_factory=lambda: ["canonical_tokens", "state"])
    
    _lemma_to_concept: dict[str, str] = field(default_factory=dict, init=False)
    _wordnet_cache: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self):
        for concept, lemmas in self.concept_map.items():
            for lemma in lemmas:
                self._lemma_to_concept[lemma] = concept

    def _get_concept(self, lemma: str) -> str:
        # 1. Hardcoded map override
        if lemma in self._lemma_to_concept:
            return self._lemma_to_concept[lemma]

        # 2. WordNet Lemma Normalization (cached)
        if lemma in self._wordnet_cache:
            return self._wordnet_cache[lemma]

        result = lemma.upper()
        try:
            import nltk
            if "/home/mattthomson/nltk_data" not in nltk.data.path:
                nltk.data.path.append("/home/mattthomson/nltk_data")
            from nltk.corpus import wordnet
            synsets = wordnet.synsets(lemma)
            if synsets:
                result = synsets[0].lemmas()[0].name().replace("_", " ").upper()
        except Exception:
            pass

        self._wordnet_cache[lemma] = result
        return result

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        lemmas = packet.context.get("lemmas", [])
        canonical_tokens = []
        
        for lemma in lemmas:
            concept = self._get_concept(lemma)
            canonical_tokens.append(concept)

        packet.context["canonical_tokens"] = canonical_tokens
        
        if canonical_tokens:
            # Map all tokens to state for full structural matching
            state_value = tuple(float(UnifiedVocabulary.get_id(t)) for t in canonical_tokens)
        else:
            state_value = ()
        
        packet.states.append(State(value=state_value, context=packet.context))
        return packet


@dataclass(slots=True)
class NamedEntityCanonicaliser(Adapter):
    """Replace entity tokens with their NER type tag for entity-invariant matching.

    'boston' -> 'GPE', 'united' -> 'ORG', 'thursday' -> 'DATE'.
    Tokens without entity types are left unchanged.
    Updates tokens, lemmas, and canonical_tokens in context.
    """

    name: str = "ner_canonicaliser"
    requires: list[str] = field(default_factory=lambda: ["nlp_tokenizer"])
    provides: list[str] = field(default_factory=lambda: ["tokens", "lemmas"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        tokens = packet.context.get("tokens", [])
        ent_types = packet.context.get("ent_types", [])
        canonical = packet.context.get("canonical_tokens", list(tokens))

        new_tokens = []
        new_lemmas = []
        new_canonical = []
        for i, token in enumerate(tokens):
            ent = ent_types[i] if i < len(ent_types) else ""
            if ent:
                new_tokens.append(ent)
                new_lemmas.append(ent)
                new_canonical.append(ent)
            else:
                new_tokens.append(token)
                new_lemmas.append(token)
                new_canonical.append(canonical[i] if i < len(canonical) else token)

        packet.context["tokens"] = new_tokens
        packet.context["lemmas"] = new_lemmas
        packet.context["canonical_tokens"] = new_canonical
        return packet


@dataclass(slots=True)
class ContentWordExtractor(Adapter):
    """Extract discriminative content words and their mean-pooled spaCy vector.

    Filters tokens to NOUN/PROPN POS tags, excluding NE placeholder tags (GPE, ORG, etc.)
    which carry no intent signal. Computes a mean-pooled embedding vector from the content
    word vectors — semantically similar utterances produce geometrically close vectors,
    making MAE distance meaningful for intent classification.
    """

    name: str = "content_word_extractor"
    requires: list[str] = field(default_factory=lambda: ["ner_canonicaliser"])
    provides: list[str] = field(default_factory=lambda: ["content_words", "content_vector"])

    _NE_TAGS: frozenset[str] = field(
        default_factory=lambda: frozenset({"GPE", "ORG", "DATE", "TIME", "PERSON", "FAC", "LOC", "QUANTITY"}),
        init=False,
    )

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        import numpy as np
        tokens = packet.context.get("tokens", [])
        pos_tags = packet.context.get("pos_tags", [])
        token_vectors = packet.context.get("token_vectors", [])

        content_words = []
        content_vecs = []
        for i, (t, p) in enumerate(zip(tokens, pos_tags)):
            if p in ("NOUN", "PROPN") and t not in self._NE_TAGS:
                content_words.append(t)
                if i < len(token_vectors):
                    v = token_vectors[i]
                    if v is not None and v.any():
                        content_vecs.append(v)

        packet.context["content_words"] = content_words

        if content_vecs:
            mean_vec = np.mean(content_vecs, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm  # unit vector → cosine distance ≈ L2 distance
            packet.context["content_vector"] = tuple(float(x) for x in mean_vec)
        else:
            packet.context["content_vector"] = ()

        return packet


@dataclass(slots=True)
class SkeletonExtractor(Adapter):
    """Extract grammatical skeleton as sequence of POS tag groups."""

    name: str = "skeleton_extractor"
    # Map coarse POS tags to skeleton types
    POS_GROUP: dict[str, str] = field(default_factory=lambda: {
        "NOUN": "N", "PROPN": "N",
        "VERB": "V", "AUX": "V",
        "ADJ": "A", "ADV": "A",
        "DET": "D", "PRON": "P",
        "ADP": "R", "PART": "R", "CCONJ": "C", "SCONJ": "C",
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
    """WordNet-backed synonym lookup for semantic candidate views."""

    name: str = "kb_lookup"
    max_candidates: int = 5
    requires: list[str] = field(default_factory=lambda: ["nlp_tokenizer"])
    provides: list[str] = field(default_factory=lambda: ["semantic_candidates"])
    _cache: dict[str, list[str]] = field(default_factory=dict, init=False)

    def _synonyms(self, token: str) -> list[str]:
        if token in self._cache:
            return self._cache[token]
        result: list[str] = []
        try:
            import nltk
            if "/home/mattthomson/nltk_data" not in nltk.data.path:
                nltk.data.path.append("/home/mattthomson/nltk_data")
            from nltk.corpus import wordnet
            syns: set[str] = set()
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    name = lemma.name().replace("_", " ").lower()
                    if name != token:
                        syns.add(name)
            result = list(syns)[: self.max_candidates]
        except Exception:
            pass
        self._cache[token] = result
        return result

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        tokens = packet.context.get("tokens", [])
        all_candidates: list[str] = []
        for token in tokens:
            all_candidates.extend(self._synonyms(token.lower()))
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
