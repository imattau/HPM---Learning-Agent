# hpm_ai_v6/agents/syntactic_rule_agent.py
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hpm_ai_v6.hpm_model.core.cell import Cell

# spaCy universal POS tag set — deterministic embedding index
_POS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]
_POS_INDEX: Dict[str, int] = {tag: i for i, tag in enumerate(_POS_TAGS)}
_EMB_DIM = 16


def _pos_embedding(pos: str) -> np.ndarray:
    """Deterministic unit-basis 16-dim embedding for a POS tag."""
    idx = _POS_INDEX.get(pos, len(_POS_TAGS))
    emb = np.zeros(_EMB_DIM, dtype=float)
    emb[idx % _EMB_DIM] = 1.0
    return emb


class SyntacticRuleAgent:
    """
    HPM agent that learns POS transition rules from a corpus and emits dim-3
    subgraph-derivation Cells consumed by ReasoningAgent._build_forward_rule_patterns.

    Interface matches StubAgent in test_reasoning_agent.py:
      - .patterns: List[Cell]
      - .get_weights() -> List[float]
      - ._paging_lookup() -> Dict[str, Cell]
    """

    def __init__(self, min_prob: float = 0.15, nlp: Any = None):
        self.min_prob = min_prob
        self._nlp = nlp  # injectable for testing; loaded lazily if None
        self.patterns: List[Cell] = []
        self._weights: List[float] = []
        self._word_pos: Dict[str, str] = {}
        self._pos_node_cells: Dict[str, Cell] = {}
        self._pos_edge_cells: Dict[Tuple[str, str], Cell] = {}

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def get_weights(self) -> List[float]:
        return list(self._weights)

    def _paging_lookup(self) -> Dict[str, Cell]:
        lookup: Dict[str, Cell] = {}
        for cell in self._pos_node_cells.values():
            lookup[cell.name] = cell
        for cell in self._pos_edge_cells.values():
            lookup[cell.name] = cell
        return lookup

    # ------------------------------------------------------------------
    # Corpus learning
    # ------------------------------------------------------------------

    def learn_from_corpus(self, sentences: List[str]) -> None:
        """Tag sentences, count bigram transitions, emit dim-3 rule cells."""
        self.patterns = []
        self._weights = []

        if not sentences:
            return

        nlp = self._get_nlp()
        pos_sequences: List[List[str]] = []

        for doc in nlp.pipe(sentences, batch_size=64):
            tokens = [
                (tok.text.lower(), tok.pos_)
                for tok in doc
                if tok.pos_ != "SPACE"
            ]
            for word, pos in tokens:
                self._word_pos[word] = pos
                self._get_or_create_pos_node(pos)
            pos_sequences.append([pos for _, pos in tokens])

        bigram_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for seq in pos_sequences:
            for i in range(len(seq) - 1):
                bigram_counts[(seq[i], seq[i + 1])] += 1

        probs = self._compute_probs(bigram_counts)
        self._emit_rule_cells(probs)

    def _compute_probs(
        self, counts: Dict[Tuple[str, str], int]
    ) -> Dict[Tuple[str, str], float]:
        row_totals: Dict[str, int] = defaultdict(int)
        for (a, _b), count in counts.items():
            row_totals[a] += count
        return {
            (a, b): count / row_totals[a]
            for (a, b), count in counts.items()
        }

    def _get_or_create_pos_node(self, pos: str) -> Cell:
        if pos not in self._pos_node_cells:
            self._pos_node_cells[pos] = Cell(
                name=f"pos_{pos}",
                dim=0,
                embedding=_pos_embedding(pos),
            )
        return self._pos_node_cells[pos]

    def _get_or_create_pos_edge_cell(
        self, pos_a: str, pos_b: str, prob: float
    ) -> Cell:
        key = (pos_a, pos_b)
        if key not in self._pos_edge_cells:
            src = self._get_or_create_pos_node(pos_a)
            tgt = self._get_or_create_pos_node(pos_b)
            self._pos_edge_cells[key] = Cell(
                name=f"pos_edge_{pos_a}_{pos_b}",
                dim=1,
                embedding=tgt.as_numpy() - src.as_numpy(),
                source=src,
                target=tgt,
                weight=prob,
            )
        return self._pos_edge_cells[key]

    def _emit_rule_cells(self, probs: Dict[Tuple[str, str], float]) -> None:
        """For each qualifying trigram (A,B,C), emit one dim-3 rule Cell."""
        all_pos = sorted({pos for pair in probs for pos in pair})
        for pos_a in all_pos:
            for pos_b in all_pos:
                p_ab = probs.get((pos_a, pos_b), 0.0)
                if p_ab < self.min_prob:
                    continue
                for pos_c in all_pos:
                    p_bc = probs.get((pos_b, pos_c), 0.0)
                    if p_bc < self.min_prob:
                        continue
                    joint = p_ab * p_bc
                    src_edge = self._get_or_create_pos_edge_cell(pos_a, pos_b, p_ab)
                    tgt_edge = self._get_or_create_pos_edge_cell(pos_b, pos_c, p_bc)
                    rule = Cell(
                        name=f"syn_rule_{pos_a}_{pos_b}_{pos_c}",
                        dim=3,
                        embedding=tgt_edge.as_numpy() - src_edge.as_numpy(),
                        source=src_edge,
                        target=tgt_edge,
                        weight=joint,
                        metadata={
                            "rule_type": "subgraph_derivation",
                            "pair_mode": "chain",
                            "antecedent_edges": [
                                {
                                    "source_var": "X",
                                    "target_var": "Y",
                                    "relation": f"pos_{pos_a}",
                                },
                                {
                                    "source_var": "Y",
                                    "target_var": "Z",
                                    "relation": f"pos_{pos_b}",
                                },
                            ],
                            "consequent": {"source_var": "X", "target_var": "Z"},
                            "pos_chain": [pos_a, pos_b, pos_c],
                            "transition_prob": joint,
                        },
                    )
                    self.patterns.append(rule)
                    self._weights.append(joint)

    # ------------------------------------------------------------------
    # Edge tagging (advisory — EdgeRecord is a frozen dataclass)
    # ------------------------------------------------------------------

    def tag_edges(self, edge_index: Dict[str, List]) -> None:
        """
        Walk existing EdgeRecord lists and look up POS for each source word.
        EdgeRecord is frozen — callers use agent.get_pos(word) directly
        to enrich their own mutable structures.
        """
        for _key, records in edge_index.items():
            for record in records:
                src_name = getattr(record.source, "name", "")
                if src_name.startswith("word_"):
                    _ = self._word_pos.get(src_name[len("word_"):])

    def get_pos(self, word: str) -> Optional[str]:
        """Return the POS tag learned for a word, or None if unknown."""
        return self._word_pos.get(word.lower())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_nlp(self) -> Any:
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
        return self._nlp
