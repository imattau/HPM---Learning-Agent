"""Structured text storage for ReaderAgent: what an HFN document hierarchy holds.

A source becomes one passage node. Its body is a polygraph of paragraph nodes
joined by ``next`` edges, each paragraph a polygraph of sentence nodes joined
by ``next`` edges, each sentence composed of word macros. Sentence nodes carry
their exact source offsets, so a node is the citation, not a copy of it.

Above the sentences sits a shared concept layer: one ``term`` node per word
stem that occurs in two or more sentences, whose children are those sentences,
across sources. HFN nodes have no parent references, so reverse lookups live in
``StructureIndex``, which is derived from the nodes and rebuilt on load; it is
never separate state.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Any, Iterable, Optional

import numpy as np

from hfn.hfn import HFN
from hpm_ai_v2.domains.text_domain import tokenise, tokenise_raw

MIN_STEM_LENGTH = 3
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@lru_cache(maxsize=None)
def stem(word: str) -> str:
    """Porter stem, so release/released/releasing share one concept."""
    from nltk.stem import PorterStemmer

    return PorterStemmer().stem(word.lower())


def content_stems(text: str) -> list[str]:
    """Distinct content-word stems of ``text``, in order of first appearance."""
    seen: dict[str, None] = {}
    for token in tokenise(text):
        s = stem(token)
        if len(s) >= MIN_STEM_LENGTH:
            seen.setdefault(s, None)
    return list(seen)


def structure_id(kind: str, source_id: str, *parts: Any) -> str:
    digest = hashlib.sha1(":".join([source_id, kind, *map(str, parts)]).encode()).hexdigest()[:12]
    return f"{kind}_{digest}"


def paragraph_index(text: str, start: int) -> int:
    """Index of the blank-line-separated paragraph containing offset ``start``."""
    return sum(1 for match in _PARAGRAPH_BREAK.finditer(text) if match.end() <= start)


def build_structured_passage(
    agent,
    text: str,
    spans: list[tuple[str, int, int]],
    sentence_ids: list[str],
    source_id: str,
    sentence_weight: float,
) -> tuple[list[HFN], list[HFN]]:
    """Build paragraph and sentence nodes for ``spans`` of ``text``.

    Returns ``(paragraph_nodes, sentence_nodes)``. Paragraphs follow blank-line
    breaks in ``text``; consecutive sentences and consecutive paragraphs are
    linked by ``next`` edges.
    """
    groups: dict[int, list[tuple[tuple[str, int, int], str, int]]] = {}
    for ordinal, (span, node_id) in enumerate(zip(spans, sentence_ids)):
        groups.setdefault(paragraph_index(text, span[1]), []).append((span, node_id, ordinal))
    paragraphs: list[HFN] = []
    sentences: list[HFN] = []
    for para_ordinal, (_key, members) in enumerate(sorted(groups.items())):
        nodes = []
        for (sentence, start, end), node_id, ordinal in members:
            if node_id in agent.patterns:
                raise ValueError(f"sentence node already exists: {node_id}")
            tokens = tokenise_raw(sentence)
            if not tokens:
                continue
            nodes.append(agent.build_sentence_node(
                tokens,
                node_id=node_id,
                mu_text=sentence,
                initial_weight=sentence_weight,
                metadata={
                    "source_id": source_id, "start": start, "end": end,
                    "ordinal": ordinal, "stems": content_stems(sentence),
                },
            ))
        if not nodes:
            continue
        first, last = nodes[0].metadata, nodes[-1].metadata
        paragraphs.append(agent.build_paragraph_node(
            nodes,
            node_id=structure_id("paragraph", source_id, para_ordinal, first["start"]),
            metadata={
                "source_id": source_id, "ordinal": para_ordinal,
                "start": first["start"], "end": last["end"],
            },
            chain_relation="next",
        ))
        sentences.extend(nodes)
    return paragraphs, sentences


def _create_term(agent, term: str, members: list[HFN]) -> HFN:
    node = HFN(
        mu=np.mean([child.mu for child in members], axis=0),
        sigma=np.ones(agent.m_dim) * 0.2,
        id=f"term_{term}",
        use_diag=True,
    )
    node.relation_type = "term"
    node.metadata = {"type": "term", "stem": term}
    for child in members:
        node.add_child(child)
    agent.observer.register(node, protected=False, initial_weight=0.5)
    agent.patterns[node.id] = node
    return node


def ensure_term(agent, term: str) -> HFN:
    """The concept node for a stem, created on demand so a relation has endpoints."""
    node = agent.patterns.get(f"term_{term}")
    if node is None:
        node = _create_term(agent, term, [agent.patterns[i] for i in agent.term_sentences[term]])
    return node


def link_terms(agent, sentence_nodes: Iterable[HFN]) -> list[str]:
    """Attach sentences to shared stem concepts; returns ids of terms created.

    A stem seen in only one sentence has no linking value yet, so it is held in
    ``agent.term_sentences`` and becomes a node when a second sentence uses it.
    A term's Gaussian is the centroid of its first two sentences and is not
    moved afterwards; later sentences are added as children only.
    """
    index: dict[str, list[str]] = agent.term_sentences
    created = []
    for sentence in sentence_nodes:
        for term in sentence.metadata.get("stems", ()):
            members = index.setdefault(term, [])
            if sentence.id in members:
                continue
            members.append(sentence.id)
            node = agent.patterns.get(f"term_{term}")
            if node is not None:
                node.add_child(sentence)
            elif len(members) >= 2:
                created.append(_create_term(agent, term, [agent.patterns[member] for member in members]).id)
    return created


def default_role_extractor():
    from hpm_ai_v2.agents.propositions import HeuristicRoleExtractor

    return HeuristicRoleExtractor()


def link_propositions(agent, text: str, sentence_nodes: Iterable[HFN]) -> list[str]:
    """Store each sentence's extracted relations as ``proposition`` nodes.

    A proposition is an HFN relation node: ``inputs`` are the concept (term)
    nodes filling its roles, ``relation_params`` name the predicate and the
    role order, and it is ``supported_by`` the sentence it came from. Only
    relations whose predicate and arguments all occur in the sentence are kept.
    """
    from hpm_ai_v2.agents.propositions import COPULA, ROLE_ORDER

    extractor = getattr(agent, "role_extractor", None)
    if extractor is None:
        extractor = agent.role_extractor = default_role_extractor()
    created = []
    for sentence in sentence_nodes:
        meta = sentence.metadata
        stems = set(meta["stems"])
        for k, prop in enumerate(extractor.extract(text[meta["start"]:meta["end"]])):
            roles = {r: v for r, v in prop.roles.items() if r in ROLE_ORDER and v[0] in stems}
            if prop.predicate != COPULA and prop.predicate not in stems:
                continue
            if prop.voice != "passive" and len(roles) < 2 or not roles:
                continue
            ordered = [r for r in ROLE_ORDER if r in roles]
            node = HFN(
                mu=sentence.mu.copy(),
                sigma=np.ones(agent.m_dim) * 0.2,
                id=structure_id("proposition", meta["source_id"], sentence.id, k),
                use_diag=True,
            )
            node.relation_type = "proposition"
            node.metadata = {
                "type": "proposition", "source_id": meta["source_id"], "sentence_id": sentence.id,
                "predicate": prop.predicate, "lemma": prop.lemma, "voice": prop.voice,
                "roles": {r: roles[r][0] for r in ordered},
                "role_text": {r: roles[r][1] for r in ordered},
            }
            node.add_relation(inputs=[ensure_term(agent, roles[r][0]) for r in ordered])
            node.relation_params = {"predicate": prop.predicate, "roles": ordered}
            node.add_child(sentence)
            node.add_edge(node, sentence, "supported_by")
            agent.observer.register(node, protected=False, initial_weight=0.5)
            agent.patterns[node.id] = node
            created.append(node.id)
    return created


def rebuild_term_sentences(agent) -> dict[str, list[str]]:
    """Recompute the stem to sentence-id map from stored sentence nodes."""
    sentences = [
        node for node in agent.patterns.values()
        if node.metadata.get("type") == "sentence" and "stems" in node.metadata
    ]
    sentences.sort(key=lambda node: (node.metadata["source_id"], node.metadata["ordinal"]))
    index: dict[str, list[str]] = {}
    for node in sentences:
        for term in node.metadata["stems"]:
            index.setdefault(term, []).append(node.id)
    agent.term_sentences = index
    return index


class StructureIndex:
    """Read-only reverse lookups over the stored hierarchy (HFN has no parent links)."""

    def __init__(self, agent):
        self.agent = agent
        self._paragraphs: dict[str, list[HFN]] = {}
        self._order: dict[str, list[str]] = {}
        self._position: dict[str, tuple[str, int]] = {}
        for node in agent.patterns.values():
            if node.metadata.get("type") == "paragraph" and "source_id" in node.metadata:
                self._paragraphs.setdefault(node.metadata["source_id"], []).append(node)
        for source_id, paragraphs in self._paragraphs.items():
            paragraphs.sort(key=lambda node: node.metadata["ordinal"])
            ordered = [child.id for node in paragraphs for child in node.children()]
            self._order[source_id] = ordered
            for position, sentence_id in enumerate(ordered):
                self._position[sentence_id] = (source_id, position)

    def position(self, sentence_id: str) -> Optional[tuple[str, int]]:
        """``(source_id, reading-order position)`` of a stored sentence."""
        return self._position.get(sentence_id)

    def sentence_ids(self, source_id: str) -> list[str]:
        """Sentence node ids of a source in reading order, across paragraphs."""
        return list(self._order.get(source_id, ()))

    def paragraphs(self, source_id: str) -> list[HFN]:
        return list(self._paragraphs.get(source_id, ()))

    def source_of(self, sentence_id: str) -> Optional[str]:
        found = self._position.get(sentence_id)
        return found[0] if found else None

    def neighbours(self, sentence_id: str, before: int = 1, after: int = 1) -> list[str]:
        """Adjacent sentences in reading order (the sentence itself excluded)."""
        found = self._position.get(sentence_id)
        if found is None:
            return []
        source_id, position = found
        order = self._order[source_id]
        low, high = max(0, position - before), min(len(order), position + after + 1)
        return [sid for sid in order[low:high] if sid != sentence_id]

    def is_shared(self, term: str) -> bool:
        """True when a stem occurs in two or more sentences (a term node alone does not say so:
        a relation argument mentioned once also has one)."""
        return len(self.agent.term_sentences.get(term, ())) >= 2

    def terms_of(self, sentence_id: str) -> list[str]:
        """Stems of a sentence that are shared across two or more sentences, with a term node."""
        node = self.agent.patterns.get(sentence_id)
        if node is None:
            return []
        return [
            t for t in node.metadata.get("stems", ())
            if f"term_{t}" in self.agent.patterns and self.is_shared(t)
        ]

    def concepts_of(self, sentence_id: str) -> list[str]:
        """Every stem of a sentence that has a concept node: shared ones and relation arguments."""
        node = self.agent.patterns.get(sentence_id)
        if node is None:
            return []
        return [t for t in node.metadata.get("stems", ()) if f"term_{t}" in self.agent.patterns]

    def sentences_for_term(self, term: str) -> list[str]:
        node = self.agent.patterns.get(f"term_{term}")
        return [child.id for child in node.children()] if node is not None else []

    def sentences_with_stem(self, term: str) -> list[str]:
        """Every sentence mentioning a stem, including stems seen only once."""
        return list(self.agent.term_sentences.get(term, ()))

    def sentences_for_query(self, text: str) -> dict[str, int]:
        """Sentences sharing content stems with ``text``, with shared-stem counts.

        Word forms meet here: "release" finds a sentence that says "released".
        """
        counts: dict[str, int] = {}
        for term in content_stems(text):
            for sentence_id in self.sentences_with_stem(term):
                counts[sentence_id] = counts.get(sentence_id, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def propositions(self, source_id: Optional[str] = None) -> list[HFN]:
        """Stored relation nodes, optionally of one source, in a stable order."""
        found = [
            n for n in self.agent.patterns.values()
            if n.relation_type == "proposition" and (source_id is None or n.metadata["source_id"] == source_id)
        ]
        return sorted(found, key=lambda n: n.id)

    def find(self, agent: Optional[str] = None, predicate: Optional[str] = None,
             patient: Optional[str] = None) -> list[HFN]:
        """Propositions matching the given words (stemmed); omitted roles match anything.

        ``find(predicate="release", patient="oxygen")`` answers "what releases oxygen".
        """
        want = {k: stem(v) for k, v in (("AGENT", agent), ("PATIENT", patient)) if v}
        pred = stem(predicate) if predicate else None
        return [
            n for n in self.propositions()
            if (pred is None or n.metadata["predicate"] == pred)
            and all(n.metadata["roles"].get(role) == key for role, key in want.items())
        ]

    def relations_of(self, term: str) -> list[tuple[str, str, Optional[str], str]]:
        """``(role, predicate, other argument stem or None, proposition id)`` for a concept."""
        out = []
        for n in self.propositions():
            for role, key in n.metadata["roles"].items():
                if key == term:
                    other = next((k for r, k in n.metadata["roles"].items() if r != role), None)
                    out.append((role, n.metadata["predicate"], other, n.id))
        return out

    def related(self, sentence_id: str) -> dict[str, int]:
        """Other sentences sharing concepts with this one, with shared-term counts."""
        counts: dict[str, int] = {}
        for term in self.terms_of(sentence_id):
            for other in self.sentences_for_term(term):
                if other != sentence_id:
                    counts[other] = counts.get(other, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
