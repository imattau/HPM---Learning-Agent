# SyntacticRuleAgent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `SyntacticRuleAgent` that learns POS bigram transition probabilities from a corpus and emits dim-3 `Cell` subgraph-derivation rules, giving the `ReasoningAgent` structural role constraints that reward valid sequences (DET→NOUN→VERB) and suppress invalid ones.

**Architecture:** The agent produces dim-3 Cells with `metadata["rule_type"]="subgraph_derivation"` and `metadata["antecedent_edges"]` encoding POS role chains. These cells flow automatically into `ReasoningAgent._build_forward_rule_patterns()` via the agent registry — no reasoning logic changes required. A single one-line addition registers `"syntactic"` in `_iter_reasoning_agents`. Each dim-3 cell's weight equals the joint bigram probability P(B|A)×P(C|B), so high-probability role sequences get high-weight rules.

**Tech Stack:** Python 3.10+, numpy, spaCy `en_core_web_sm` (already in requirements.txt). Mock-nlp pattern for test isolation — CI requires no model download.

---

## File Structure

```
hpm_ai_v6/agents/syntactic_rule_agent.py      CREATE  — main agent
hpm_ai_v6/tests/test_syntactic_rule_agent.py  CREATE  — TDD tests
hpm_ai_v6/agents/reasoning_agent.py           MODIFY  — line 181 only (add "syntactic")
requirements.txt                               VERIFY  — spacy>=3.7 already present
```

---

## Data Flow

```
sentences: List[str]
    │ spaCy nlp.pipe()
    ▼
word_pos: Dict[str,str]  +  pos_sequences: List[List[str]]
    │ _count_transitions()
    ▼
bigram_counts: Dict[(str,str), int]
    │ _compute_probs()
    ▼
probs: Dict[(str,str), float]    e.g. ("DET","NOUN") -> 0.72
    │ _emit_rule_cells()  [triple loop over qualifying trigrams A,B,C]
    ▼
self.patterns: List[Cell dim=3]   self._weights: List[float]
    │ registered as reader.agents["syntactic"]
    ▼
ReasoningAgent.refresh()
    → _iter_reasoning_agents() yields ("syntactic", agent)
    → all_patterns accumulates dim-3 cells
    → _build_forward_rule_patterns() picks up cells with antecedent_edges
    → _forward_rule_patterns sorted by weight descending
    │ at query time
    ▼
_match_subgraph_templates() matches edges by relation="pos_<TAG>"
    → high-weight rules derive valid POS-ordered paths
    → low-weight / absent rules suppress invalid orderings
```

---

## Dim-3 Cell Structure (per qualifying trigram A→B→C)

```python
Cell(
    name=f"syn_rule_{pos_A}_{pos_B}_{pos_C}",
    dim=3,
    embedding=tgt_edge.as_numpy() - src_edge.as_numpy(),  # dim-1 edge difference
    source=pos_edge_cell(A,B),   # dim-1 Cell for A→B
    target=pos_edge_cell(B,C),   # dim-1 Cell for B→C
    weight=P(B|A) * P(C|B),
    metadata={
        "rule_type": "subgraph_derivation",
        "pair_mode": "chain",
        "antecedent_edges": [
            {"source_var": "X", "target_var": "Y", "relation": f"pos_{pos_A}"},
            {"source_var": "Y", "target_var": "Z", "relation": f"pos_{pos_B}"},
        ],
        "consequent": {"source_var": "X", "target_var": "Z"},
        "pos_chain": [pos_A, pos_B, pos_C],
        "transition_prob": P(B|A) * P(C|B),
    },
)
```

---

## Task 1: Write Failing Tests (RED)

**Files:**
- Create: `hpm_ai_v6/tests/test_syntactic_rule_agent.py`

- [ ] **Step 1.1: Create the test file**

```python
# hpm_ai_v6/tests/test_syntactic_rule_agent.py
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from hpm_ai_v6.agents.syntactic_rule_agent import SyntacticRuleAgent
from hpm_ai_v6.hpm_model.core.cell import Cell


def make_mock_nlp(sentences_to_pos):
    """Returns a mock spaCy nlp whose .pipe() yields pre-specified token/POS pairs.
    sentences_to_pos: List[List[Tuple[str, str]]] — (text, pos_) per sentence.
    """
    def nlp_pipe(texts, **kwargs):
        for token_list in sentences_to_pos:
            doc = MagicMock()
            doc.__iter__ = MagicMock(return_value=iter([
                MagicMock(text=t, pos_=p) for t, p in token_list
            ]))
            yield doc
    mock = MagicMock()
    mock.pipe = nlp_pipe
    return mock


SIMPLE_NLP = make_mock_nlp([
    [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
    [("the", "DET"), ("dog", "NOUN"), ("ran", "VERB")],
])


class TestSyntacticRuleAgentInterface:
    def test_has_patterns_list(self):
        agent = SyntacticRuleAgent(nlp=SIMPLE_NLP)
        assert isinstance(agent.patterns, list)

    def test_get_weights_returns_list(self):
        agent = SyntacticRuleAgent(nlp=SIMPLE_NLP)
        assert isinstance(agent.get_weights(), list)

    def test_paging_lookup_returns_dict(self):
        agent = SyntacticRuleAgent(nlp=SIMPLE_NLP)
        assert isinstance(agent._paging_lookup(), dict)

    def test_weights_parallel_to_patterns(self):
        agent = SyntacticRuleAgent(nlp=SIMPLE_NLP)
        agent.learn_from_corpus(["the cat sat", "the dog ran"])
        assert len(agent.get_weights()) == len(agent.patterns)


class TestCorpusLearning:
    def setup_method(self):
        self.nlp = make_mock_nlp([
            [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
            [("the", "DET"), ("dog", "NOUN"), ("ran", "VERB")],
        ])
        self.agent = SyntacticRuleAgent(min_prob=0.1, nlp=self.nlp)
        self.agent.learn_from_corpus(["the cat sat", "the dog ran"])

    def test_produces_dim3_cells(self):
        assert all(p.dim == 3 for p in self.agent.patterns)

    def test_cells_have_subgraph_derivation_rule_type(self):
        for p in self.agent.patterns:
            assert p.metadata.get("rule_type") == "subgraph_derivation"

    def test_cells_have_two_antecedent_edges(self):
        for p in self.agent.patterns:
            assert "antecedent_edges" in p.metadata
            assert len(p.metadata["antecedent_edges"]) == 2

    def test_det_noun_verb_chain_exists(self):
        names = {p.name for p in self.agent.patterns}
        assert "syn_rule_DET_NOUN_VERB" in names

    def test_weights_are_probabilities(self):
        for w in self.agent.get_weights():
            assert 0.0 <= w <= 1.0

    def test_high_prob_transition_has_high_weight(self):
        det_noun_verb = next(
            p for p in self.agent.patterns if p.name == "syn_rule_DET_NOUN_VERB"
        )
        # DET->NOUN is 100%, NOUN->VERB is 100% -> joint ~1.0
        assert det_noun_verb.weight > 0.5

    def test_cell_has_source_and_target(self):
        for p in self.agent.patterns:
            assert p.source is not None
            assert p.target is not None

    def test_embedding_is_difference_of_pos_edge_embeddings(self):
        for p in self.agent.patterns:
            expected = p.target.as_numpy() - p.source.as_numpy()
            np.testing.assert_allclose(p.as_numpy(), expected, atol=1e-6)

    def test_antecedent_relations_use_pos_prefix(self):
        for p in self.agent.patterns:
            for edge_spec in p.metadata["antecedent_edges"]:
                assert edge_spec["relation"].startswith("pos_")


class TestTagEdges:
    def setup_method(self):
        self.nlp = make_mock_nlp([
            [("cat", "NOUN"), ("sat", "VERB")],
        ])
        self.agent = SyntacticRuleAgent(nlp=self.nlp)
        self.agent.learn_from_corpus(["cat sat"])

    def test_tag_edges_runs_without_error(self):
        from hpm_ai_v6.agents.reasoning_agent import EdgeRecord
        src = Cell(name="word_cat", dim=0, embedding=np.zeros(16))
        tgt = Cell(name="word_sat", dim=0, embedding=np.ones(16))
        edge = Cell(name="w_cat->sat", dim=1, embedding=np.ones(16),
                    source=src, target=tgt)
        record = EdgeRecord(
            pattern=edge, source=src, target=tgt,
            score=0.5, raw_weight=0.5, agent_name="word",
            source_key="word:cat", target_key="word:sat",
            relation="lexical_transition",
        )
        # EdgeRecord is frozen — tag_edges is advisory only; must not raise
        self.agent.tag_edges({"word:cat": [record]})

    def test_paging_lookup_contains_pos_cells(self):
        lookup = self.agent._paging_lookup()
        assert any("pos_" in k for k in lookup)

    def test_get_pos_returns_tag_for_known_word(self):
        assert self.agent.get_pos("cat") == "NOUN"

    def test_get_pos_returns_none_for_unknown_word(self):
        assert self.agent.get_pos("xyzzy") is None


class TestMinProbThreshold:
    def test_high_min_prob_excludes_low_prob_transitions(self):
        # With min_prob=0.99, joint P(B|A)*P(C|B) must be >= 0.99*0.99 ~= 0.98
        # DET->NOUN=1.0, NOUN->VERB=0.5 -> joint=0.5 < 0.99 -> excluded
        nlp = make_mock_nlp([
            [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
            [("the", "DET"), ("cat", "NOUN"), ("ran", "VERB")],
        ])
        agent = SyntacticRuleAgent(min_prob=0.99, nlp=nlp)
        agent.learn_from_corpus(["the cat sat", "the cat ran"])
        assert len(agent.patterns) == 0

    def test_empty_corpus_produces_empty_patterns(self):
        nlp = make_mock_nlp([])
        agent = SyntacticRuleAgent(nlp=nlp)
        agent.learn_from_corpus([])
        assert agent.patterns == []
        assert agent.get_weights() == []

    def test_learn_from_corpus_is_idempotent_on_second_call(self):
        nlp = make_mock_nlp([
            [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
            [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
        ])
        agent = SyntacticRuleAgent(min_prob=0.1, nlp=nlp)
        agent.learn_from_corpus(["the cat sat"])
        count_first = len(agent.patterns)
        agent.learn_from_corpus(["the cat sat"])
        assert len(agent.patterns) == count_first  # reset, not accumulate


class TestReasoningAgentIntegration:
    def test_syntactic_rules_appear_in_forward_rule_patterns(self):
        from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent

        nlp = make_mock_nlp([
            [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
            [("the", "DET"), ("dog", "NOUN"), ("ran", "VERB")],
        ])
        syn_agent = SyntacticRuleAgent(min_prob=0.1, nlp=nlp)
        syn_agent.learn_from_corpus(["the cat sat", "the dog ran"])

        reader = SimpleNamespace(agents={"syntactic": syn_agent})
        ra = ReasoningAgent(reader=reader)
        ra.refresh()

        assert len(ra._forward_rule_patterns) > 0
        rule_names = {cell.name for _, cell in ra._forward_rule_patterns}
        assert "syn_rule_DET_NOUN_VERB" in rule_names

    def test_forward_rules_sorted_by_weight_descending(self):
        from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent

        nlp = make_mock_nlp([
            [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
        ])
        syn_agent = SyntacticRuleAgent(min_prob=0.01, nlp=nlp)
        syn_agent.learn_from_corpus(["the cat sat"])

        reader = SimpleNamespace(agents={"syntactic": syn_agent})
        ra = ReasoningAgent(reader=reader)
        ra.refresh()

        weights = [w for w, _ in ra._forward_rule_patterns]
        assert weights == sorted(weights, reverse=True)
```

- [ ] **Step 1.2: Run tests to confirm all fail**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python3 -m pytest hpm_ai_v6/tests/test_syntactic_rule_agent.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'hpm_ai_v6.agents.syntactic_rule_agent'`

---

## Task 2: Implement the Agent (GREEN)

**Files:**
- Create: `hpm_ai_v6/agents/syntactic_rule_agent.py`

- [ ] **Step 2.1: Create the agent file**

```python
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
```

- [ ] **Step 2.2: Run tests to confirm GREEN**

```bash
python3 -m pytest hpm_ai_v6/tests/test_syntactic_rule_agent.py -v --ignore-glob="*Integration*"
```

Expected: all non-integration tests pass.

---

## Task 3: Register Agent in ReasoningAgent

**Files:**
- Modify: `hpm_ai_v6/agents/reasoning_agent.py` line 181

- [ ] **Step 3.1: Add "syntactic" to the agent registry tuple**

Find this line (around line 181):
```python
for name in ("word", "contextual", "semantic", "phrase", "char", "causal"):
```

Change to:
```python
for name in ("word", "contextual", "semantic", "phrase", "char", "causal", "syntactic"):
```

- [ ] **Step 3.2: Verify existing reasoning tests still pass**

```bash
python3 -m pytest hpm_ai_v6/tests/test_reasoning_agent.py -v
```

Expected: all 33 tests pass.

---

## Task 4: Integration Tests

**Files:**
- Already in: `hpm_ai_v6/tests/test_syntactic_rule_agent.py` (class `TestReasoningAgentIntegration`)

- [ ] **Step 4.1: Run integration tests**

```bash
python3 -m pytest hpm_ai_v6/tests/test_syntactic_rule_agent.py -v -k "Integration"
```

Expected: both integration tests pass.

- [ ] **Step 4.2: Run full test suite**

```bash
python3 -m pytest hpm_ai_v6/tests/ -v
```

Expected: all tests pass, no regressions.

---

## Task 5: Commit

- [ ] **Step 5.1: Commit all changes**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
git add hpm_ai_v6/agents/syntactic_rule_agent.py
git add hpm_ai_v6/tests/test_syntactic_rule_agent.py
git add hpm_ai_v6/agents/reasoning_agent.py
git commit -m "Add SyntacticRuleAgent: POS bigram rules as dim-3 subgraph-derivation cells"
```

---

## Critical Details

**`_iter_reasoning_agents` change:** The one-line change to `reasoning_agent.py` registers the new agent exactly as a plugin. The `_agent_relation` method has no case for `"syntactic"` — it falls through to the default `"transition"` string, which is correct since the agent produces dim-3 rule cells, not dim-1 edges requiring a specific relation label.

**EdgeRecord immutability:** `EdgeRecord` is a frozen dataclass. `tag_edges` is advisory. Callers use `agent.get_pos(word)` to enrich their own mutable structures. The method must not raise.

**`learn_from_corpus` is destructive:** Calling it twice resets patterns rather than accumulating. Concatenate corpora before calling if incremental learning is needed.

**`_paging_lookup` scope:** Returns only POS node cells (dim=0) and POS edge cells (dim=1), not the dim-3 rule cells. `ReasoningAgent` uses `_paging_lookup` to populate `node_index`; dim-3 cells enter via `getattr(agent, "patterns", [])`.

**`min_prob` semantics:** Default 0.15 admits diverse transitions. Invalid sequences are suppressed by *absence* of a high-weight supporting rule, not by explicit low-weight penalty cells.

**Test isolation:** All tests use `make_mock_nlp` — no network access and no spaCy model download needed in CI.

**Computational complexity:** `_emit_rule_cells` is O(17³) = 4913 max iterations (17 universal POS tags). Negligible.
