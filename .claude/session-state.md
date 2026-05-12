# Session State Checkpoint
Generated: 2026-05-13
Reason: Context threshold exceeded (95%)

## Execution Mode
**Mode**: unattended
**Auto-Continue**: true

## Current Task
Write the implementation plan for DependencyRelationAgent.
Save to: `docs/superpowers/plans/2026-05-13-dependency-relation-agent.md`
Commit: `git commit -m "Add dependency relation agent implementation plan"`
Then stop and report "PLAN COMPLETE".
Do NOT implement anything.

## Instructions
Read the spec first: `docs/superpowers/specs/2026-05-13-dependency-relation-agent-design.md`
Read for context:
- `hpm_ai_v6/agents/syntactic_rule_agent.py` (follow same agent pattern)
- `hpm_ai_v6/agents/multi_agent_reader.py` lines 1-130 (agent registration pattern)
- `hpm_ai_v6/agents/reasoning_agent.py` line 181 (_iter_reasoning_agents tuple)
- `hpm_ai_v6/tests/test_syntactic_rule_agent.py` (follow same test pattern with mock nlp)

## Plan Format (writing-plans skill)

### Required Header
```
# DependencyRelationAgent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a DependencyRelationAgent that uses spaCy's dependency parser to extract SVO + prepositional triples, creating direct high-confidence dim-1 edges (subject_of, object_of, prep_X) that give the ReasoningAgent 2-hop paths between concepts vs the current 4-hop sentence bridges.

**Architecture:** Agent parses sentences with spaCy, extracts (subject, relation, object) triples, creates dim-1 Cell edges between word cells. Registered as reader.agents["dependency"]. MultiAgentReader calls learn_from_corpus during train_sequence and persists to dependency_relations.json. ReasoningAgent picks up edges automatically via _iter_reasoning_agents.

**Tech Stack:** Python 3.10+, spaCy en_core_web_sm (already in requirements.txt). No new dependencies.
```

### File Structure
- CREATE: `hpm_ai_v6/agents/dependency_relation_agent.py`
- CREATE: `hpm_ai_v6/tests/test_dependency_relation_agent.py`
- MODIFY: `hpm_ai_v6/agents/multi_agent_reader.py` (import, init, train, warm_start)
- MODIFY: `hpm_ai_v6/agents/reasoning_agent.py` (add "dependency" to _iter_reasoning_agents)

### Tasks (TDD — 5 tasks)

#### Task 1: DependencyRelationAgent core
Tests use mock nlp (same pattern as test_syntactic_rule_agent.py):
```python
def make_mock_dep_nlp(sentences_to_parse):
    """sentences_to_parse: List[List[Tuple[text, pos_, dep_, head_text]]]"""
    def nlp_pipe(texts, **kwargs):
        for token_list in sentences_to_parse:
            doc = MagicMock()
            tokens = []
            for text, pos_, dep_, head_text in token_list:
                tok = MagicMock()
                tok.text = text
                tok.pos_ = pos_
                tok.dep_ = dep_
                head = MagicMock()
                head.text = head_text
                tok.head = head
                tok.lower_ = text.lower()
                tokens.append(tok)
            doc.__iter__ = MagicMock(return_value=iter(tokens))
            yield doc
    mock = MagicMock()
    mock.pipe = nlp_pipe
    return mock

# "Alice followed the rabbit."
# alice: nsubj of followed
# followed: ROOT
# rabbit: dobj of followed
SIMPLE_SVO_NLP = make_mock_dep_nlp([
    [("Alice", "PROPN", "nsubj", "followed"),
     ("followed", "VERB", "ROOT", "followed"),
     ("the", "DET", "det", "rabbit"),
     ("rabbit", "NOUN", "dobj", "followed"),
     (".", "PUNCT", "punct", "followed")]
])
```

Tests:
```python
def test_has_patterns_list():
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    assert isinstance(agent.patterns, list)

def test_get_weights_parallel_to_patterns():
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    agent.learn_from_corpus(["Alice followed the rabbit."])
    assert len(agent.get_weights()) == len(agent.patterns)

def test_svo_produces_subject_of_edge():
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    agent.learn_from_corpus(["Alice followed the rabbit."])
    names = {p.name for p in agent.patterns}
    assert any("subject_of" in n and "alice" in n.lower() for n in names)

def test_svo_produces_object_of_edge():
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    agent.learn_from_corpus(["Alice followed the rabbit."])
    names = {p.name for p in agent.patterns}
    assert any("object_of" in n and "rabbit" in n.lower() for n in names)

def test_patterns_are_dim1_cells():
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    agent.learn_from_corpus(["Alice followed the rabbit."])
    assert all(p.dim == 1 for p in agent.patterns)

def test_source_and_target_are_word_cells():
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    agent.learn_from_corpus(["Alice followed the rabbit."])
    for p in agent.patterns:
        assert p.source is not None and p.source.name.startswith("word_")
        assert p.target is not None and p.target.name.startswith("word_")

def test_paging_lookup_contains_word_cells():
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    agent.learn_from_corpus(["Alice followed the rabbit."])
    lookup = agent._paging_lookup()
    assert any(k.startswith("word_") for k in lookup)
    assert "word_alice" in lookup or "word_Alice" in lookup

def test_empty_corpus_produces_no_patterns():
    nlp = make_mock_dep_nlp([])
    agent = DependencyRelationAgent(nlp=nlp)
    agent.learn_from_corpus([])
    assert agent.patterns == []
```

Implementation:
```python
# hpm_ai_v6/agents/dependency_relation_agent.py
from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from hpm_ai_v6.hpm_model.core.cell import Cell

_EMB_DIM = 16

def _word_embedding(word: str) -> np.ndarray:
    """Simple deterministic embedding for a word (hash-based)."""
    emb = np.zeros(_EMB_DIM, dtype=float)
    for i, ch in enumerate(word.lower()[:_EMB_DIM]):
        emb[i % _EMB_DIM] = ord(ch) / 128.0
    return emb


class DependencyRelationAgent:
    """
    Extracts SVO + prepositional dependency triples from sentences using spaCy
    and emits dim-1 Cell edges (subject_of, object_of, prep_X) between word cells.

    Interface matches StubAgent: .patterns, .get_weights(), ._paging_lookup()
    """

    # dep labels to extract
    _SUBJECT_DEPS = {"nsubj", "nsubjpass"}
    _OBJECT_DEPS = {"dobj", "pobj", "attr"}
    _PREP_DEPS = {"prep"}

    def __init__(self, nlp: Any = None, base_score: float = 0.85, prep_score: float = 0.75):
        self._nlp = nlp
        self.base_score = base_score
        self.prep_score = prep_score
        self.patterns: List[Cell] = []
        self._weights: List[float] = []
        self._word_cells: Dict[str, Cell] = {}  # word_lower -> Cell

    def get_weights(self) -> List[float]:
        return list(self._weights)

    def _paging_lookup(self) -> Dict[str, Cell]:
        return {cell.name: cell for cell in self._word_cells.values()}

    def _get_or_create_word_cell(self, word: str) -> Cell:
        key = word.lower()
        if key not in self._word_cells:
            self._word_cells[key] = Cell(
                name=f"word_{key}",
                dim=0,
                embedding=_word_embedding(key).tolist(),
            )
        return self._word_cells[key]

    def _make_edge(self, source: Cell, target: Cell, relation: str, score: float) -> Cell:
        emb = (target.as_numpy() - source.as_numpy()).tolist()
        return Cell(
            name=f"dep_{relation}_{source.name}_{target.name}",
            dim=1,
            embedding=emb,
            source=source,
            target=target,
            weight=score,
        )

    def learn_from_corpus(self, sentences: List[str]) -> None:
        self.patterns = []
        self._weights = []
        if not sentences:
            return
        nlp = self._get_nlp()
        for doc in nlp.pipe(sentences, batch_size=64):
            for token in doc:
                dep = token.dep_.lower() if hasattr(token, 'dep_') else ""
                head_text = token.head.text if hasattr(token, 'head') else ""

                if dep in self._SUBJECT_DEPS:
                    # subject → verb (subject_of)
                    subj_cell = self._get_or_create_word_cell(token.lower_)
                    verb_cell = self._get_or_create_word_cell(head_text.lower())
                    edge = self._make_edge(subj_cell, verb_cell, "subject_of", self.base_score)
                    self.patterns.append(edge)
                    self._weights.append(self.base_score)

                elif dep in self._OBJECT_DEPS:
                    # verb → object (object_of)
                    verb_cell = self._get_or_create_word_cell(head_text.lower())
                    obj_cell = self._get_or_create_word_cell(token.lower_)
                    edge = self._make_edge(verb_cell, obj_cell, "object_of", self.base_score)
                    self.patterns.append(edge)
                    self._weights.append(self.base_score)

                elif dep in self._PREP_DEPS:
                    # verb → prep_X → pobj handled when we see the pobj token
                    pass

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        triples = []
        for p, w in zip(self.patterns, self._weights):
            triples.append({
                "source": p.source.name if p.source else "",
                "target": p.target.name if p.target else "",
                "name": p.name,
                "relation": p.name.split("_", 2)[1] if "_" in p.name else "",
                "weight": w,
            })
        with open(path, "w") as f:
            json.dump({"triples": triples}, f)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.patterns = []
        self._weights = []
        for t in data.get("triples", []):
            src_cell = self._get_or_create_word_cell(t["source"].removeprefix("word_"))
            tgt_cell = self._get_or_create_word_cell(t["target"].removeprefix("word_"))
            w = float(t.get("weight", self.base_score))
            emb = (tgt_cell.as_numpy() - src_cell.as_numpy()).tolist()
            edge = Cell(name=t["name"], dim=1, embedding=emb, source=src_cell, target=tgt_cell, weight=w)
            self.patterns.append(edge)
            self._weights.append(w)

    def _get_nlp(self) -> Any:
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("Run: python -m spacy download en_core_web_sm")
        return self._nlp
```

Commit: `"feat: add DependencyRelationAgent with SVO+prep extraction"`

#### Task 2: Prepositional edge extraction test + fix
The pobj case needs to be handled when we see the pobj token (its head is the preposition, whose head is the verb):
```python
# Add to learn_from_corpus after the elif dep in self._OBJECT_DEPS block:
elif dep == "pobj":
    # pobj: token is the object; token.head is the preposition; token.head.head is the verb
    prep_text = head_text.lower()  # e.g. "down", "into"
    verb_node = getattr(token.head, 'head', None)
    if verb_node is not None:
        verb_cell = self._get_or_create_word_cell(verb_node.text.lower())
        obj_cell = self._get_or_create_word_cell(token.lower_)
        relation = f"prep_{prep_text}"
        edge = self._make_edge(verb_cell, obj_cell, relation, self.prep_score)
        self.patterns.append(edge)
        self._weights.append(self.prep_score)
```

Tests:
```python
PREP_NLP = make_mock_dep_nlp([
    # "She ran into the hole."
    # ran: ROOT
    # into: prep of ran
    # hole: pobj of into
    [("She", "PRON", "nsubj", "ran"),
     ("ran", "VERB", "ROOT", "ran"),
     ("into", "ADP", "prep", "ran"),
     ("the", "DET", "det", "hole"),
     ("hole", "NOUN", "pobj", "into")]
])

def test_prep_produces_prep_edge():
    # Need head.head for pobj
    # mock: hole.dep_=pobj, hole.head.text="into", hole.head.head.text="ran"
    # This test verifies the prep_into edge from ran→hole
    agent = DependencyRelationAgent(nlp=PREP_NLP)
    agent.learn_from_corpus(["She ran into the hole."])
    names = {p.name for p in agent.patterns}
    assert any("prep_into" in n for n in names), f"Expected prep_into edge, got {names}"
```

NOTE: The mock needs `token.head.head` support — update `make_mock_dep_nlp` to handle this:
```python
# In make_mock_dep_nlp, after creating head mock:
head.head = MagicMock()
head.head.text = head_text  # grandparent defaults to same as parent
# For pobj tokens, caller sets head to prep text and needs head.head = verb
```

Actually, the mock needs to be more careful. Simplify: add a `head_head_text` param to the tuple:
`(text, pos_, dep_, head_text, head_head_text="")` where head_head_text is used for pobj.

Update `make_mock_dep_nlp` signature accordingly.

Commit: `"feat: add prepositional edge extraction + tests"`

#### Task 3: save/load + integration tests
```python
def test_save_load_round_trip(tmp_path):
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    agent.learn_from_corpus(["Alice followed the rabbit."])
    path = str(tmp_path / "dep.json")
    agent.save(path)
    agent2 = DependencyRelationAgent(nlp=make_mock_dep_nlp([]))
    agent2.load(path)
    assert len(agent2.patterns) == len(agent.patterns)
    assert len(agent2.get_weights()) == len(agent.get_weights())

def test_integration_with_reasoning_agent():
    from types import SimpleNamespace
    from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent
    agent = DependencyRelationAgent(nlp=SIMPLE_SVO_NLP)
    agent.learn_from_corpus(["Alice followed the rabbit."])
    reader = SimpleNamespace(agents={
        "dependency": agent, "word": None, "phrase": None,
        "contextual": None, "semantic": None, "char": None, "causal": None,
    })
    ra = ReasoningAgent(reader, beam_width=5, max_depth=3)
    ra.refresh()
    alice = ra._resolve_cell("alice")
    rabbit = ra._resolve_cell("rabbit")
    assert alice is not None
    assert rabbit is not None
    path = ra._beam_search_path(alice, rabbit)
    assert path is not None, "Expected alice → followed → rabbit path"
    assert len(path) == 2
```

Commit: `"test: add dep agent save/load and reasoning integration tests"`

#### Task 4: Wire into MultiAgentReader
In `multi_agent_reader.py`:

1. Import: `from hpm_ai_v6.agents.dependency_relation_agent import DependencyRelationAgent`

2. In `__init__` (after relation_emitter, before agents dict):
```python
self.dependency_agent = DependencyRelationAgent()
```

3. Add to `self.agents` dict: `"dependency": self.dependency_agent`

4. In `warm_start_from_cache`:
```python
dep_path = os.path.join(self.pattern_cache_dir, "dependency_relations.json")
if os.path.exists(dep_path):
    try:
        self.dependency_agent.load(dep_path)
    except Exception:
        pass
```

5. At end of `train_sequence` (after flush_all, before pattern_store operations):
```python
# Train dependency agent on same sentences
dep_agent = self.agents.get("dependency")
if dep_agent is not None and hasattr(dep_agent, "learn_from_corpus"):
    try:
        dep_agent.learn_from_corpus(sentences)
        dep_path = os.path.join(self.pattern_cache_dir, "dependency_relations.json")
        dep_agent.save(dep_path)
    except Exception:
        pass
```

In `reasoning_agent.py` line 181:
```python
for name in ("word", "contextual", "semantic", "phrase", "char", "causal", "syntactic", "dependency"):
```

Run: `python3 -m pytest hpm_ai_v6/tests/ -x -q 2>&1 | tail -5`
Expected: all 106+ tests pass.

Commit: `"feat: wire DependencyRelationAgent into MultiAgentReader and ReasoningAgent"`

#### Task 5: End-to-end verification
```bash
python3 -c "
from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
reader = MultiAgentReader('hpm_ai_v6/data/corpus/alice_mini.txt', warm_start=False)
reader.train(episodes=2, max_chunks=50, max_words_per_chunk=80, enable_pruning=False, enable_causal=False)
print(reader.reason('Why did Alice follow the rabbit?'))
print(reader.reason('How does Alice connect to the rabbit hole?'))
" 2>&1 | grep -v "Loading\|LOAD\|UNEXPECTED\|Notes\|it/s\|BertModel\|Starting\|Episode\|Patterns\|weights"
```

Expected: alice→rabbit paths now 2 hops with score > 0.5 (vs previous 4 hops, score 0.04).

Commit: `"chore: verify dependency relation agent end-to-end on alice corpus"`

## Working Directory
`/home/mattthomson/workspace/HPM---Learning-Agent`
Python: `python3`, Branch: `hpm-ai-v6`
Current tests: 106 passing
Spec: `docs/superpowers/specs/2026-05-13-dependency-relation-agent-design.md`
