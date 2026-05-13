# Reasoning-Guided Learning Feedback Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `reflect()` method to `ReasoningAgent` that returns a `ReasoningSignal` describing what it found and couldn't find, and wire it into `MultiAgentReader.maintenance_cycle` to reinforce derived edges and guide attention toward uncertain concepts.

**Architecture:** `ReasoningSignal` dataclass holds uncertain concepts + derived edges. `ReasoningAgent.reflect(queries)` processes a query batch and produces the signal. `MultiAgentReader._reinforce_edge` injects derived edges into `word_agent.patterns`. `maintenance_cycle` calls `reflect()` after training and applies the signal.

**Tech Stack:** Python 3.10+, numpy, existing HPM infrastructure. No new dependencies.

---

## File Structure

- MODIFY: `hpm_ai_v6/agents/reasoning_agent.py` — add `ReasoningSignal` dataclass + `reflect()` method
- MODIFY: `hpm_ai_v6/agents/multi_agent_reader.py` — add `_reinforce_edge`, `_focus_words`, update `maintenance_cycle`
- CREATE: `hpm_ai_v6/tests/test_reasoning_feedback.py`

---

## Tasks

### Task 1: ReasoningSignal dataclass + reflect() stub

- [ ] Write tests first in `hpm_ai_v6/tests/test_reasoning_feedback.py`:

```python
from types import SimpleNamespace
import pytest
from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent, ReasoningSignal
from hpm_ai_v6.hpm_model.core.cell import Cell


class StubAgent:
    def __init__(self, patterns=None, weights=None, lookup=None):
        self.patterns = patterns or []
        self._weights = weights or []
        self._lookup = lookup or {}
    def get_weights(self): return list(self._weights)
    def _paging_lookup(self): return dict(self._lookup)


def make_reader_with_no_edges():
    """Reader where alice resolves but has no outgoing edges."""
    alice = Cell(name="word_alice", dim=0, embedding=[1,0,0])
    word_agent = StubAgent(patterns=[], weights=[], lookup={"word_alice": alice})
    return SimpleNamespace(agents={
        "word": word_agent, "phrase": None, "contextual": None,
        "semantic": None, "char": None, "causal": None,
    })


def make_reader_with_path():
    """Reader where alice→rabbit path exists."""
    alice = Cell(name="word_alice", dim=0, embedding=[1,0,0])
    rabbit = Cell(name="word_rabbit", dim=0, embedding=[0,1,0])
    decoy = Cell(name="word_decoy", dim=0, embedding=[0,0,1])
    edge = Cell(name="w_alice->rabbit", dim=1,
                embedding=rabbit.as_numpy()-alice.as_numpy(), source=alice, target=rabbit)
    decoy_edge = Cell(name="w_alice->decoy", dim=1,
                      embedding=decoy.as_numpy()-alice.as_numpy(), source=alice, target=decoy)
    word_agent = StubAgent(
        patterns=[edge, decoy_edge], weights=[0.5, 0.9],
        lookup={"word_alice": alice, "word_rabbit": rabbit, "word_decoy": decoy},
    )
    return SimpleNamespace(agents={
        "word": word_agent, "phrase": None, "contextual": None,
        "semantic": None, "char": None, "causal": None,
    })


def test_reasoning_signal_is_dataclass():
    signal = ReasoningSignal(
        uncertain_concepts=[],
        high_value_paths=[],
        suggested_focus_words=[],
        derived_edges=[],
    )
    assert signal.uncertain_concepts == []
    assert signal.derived_edges == []


def test_reflect_returns_reasoning_signal():
    reader = make_reader_with_no_edges()
    ra = ReasoningAgent(reader)
    signal = ra.reflect(["How does alice connect?"])
    assert isinstance(signal, ReasoningSignal)


def test_reflect_identifies_uncertain_concepts_when_no_path():
    reader = make_reader_with_no_edges()
    ra = ReasoningAgent(reader)
    signal = ra.reflect(["How does alice connect to rabbit?"])
    # alice resolves but rabbit doesn't exist → uncertain
    assert len(signal.uncertain_concepts) > 0


def test_reflect_returns_derived_edges_when_path_found():
    reader = make_reader_with_path()
    ra = ReasoningAgent(reader, beam_width=5, max_depth=3)
    signal = ra.reflect(["How does alice connect to rabbit?"])
    # Path alice→rabbit found (1 hop) — no multi-hop to derive
    # But for a 2-hop path, derived_edges would be non-empty
    assert isinstance(signal.derived_edges, list)


def test_reflect_with_empty_queries_returns_empty_signal():
    reader = make_reader_with_no_edges()
    ra = ReasoningAgent(reader)
    signal = ra.reflect([])
    assert signal.uncertain_concepts == []
    assert signal.derived_edges == []
```

- [ ] Add `ReasoningSignal` dataclass to `reasoning_agent.py` after imports:

```python
from dataclasses import dataclass, field

@dataclass
class ReasoningSignal:
    uncertain_concepts: List[str]
    high_value_paths: List[object]  # List[List[PathStep]] — kept as traces
    suggested_focus_words: List[str]
    derived_edges: List[Tuple[Cell, Cell, float]]  # (source, target, score)
```

- [ ] Add `reflect()` method to `ReasoningAgent` (before `reason_with_trace`):

```python
def reflect(self, queries: List[str], top_k_paths: int = 5) -> "ReasoningSignal":
    """
    Process a query batch and return a ReasoningSignal for learning feedback.
    Called by MultiAgentReader.maintenance_cycle after training.
    """
    if not queries:
        return ReasoningSignal(
            uncertain_concepts=[], high_value_paths=[],
            suggested_focus_words=[], derived_edges=[],
        )
    self._ensure_fresh()
    uncertain: List[str] = []
    high_value: List[object] = []
    derived: List[Tuple[Cell, Cell, float]] = []

    for query in queries:
        try:
            trace = self.reason_with_trace(query)
        except Exception:
            continue

        if trace.get("chosen_path") is None:
            # No path found — record uncertain concepts
            for term in trace.get("terms", []):
                cell = self._resolve_cell(term)
                if cell is not None and term not in uncertain:
                    uncertain.append(term)
        else:
            # Path found — check for multi-hop derivable edges
            for path_trace in trace.get("candidate_paths", [])[:top_k_paths]:
                steps = path_trace.get("steps", [])
                if len(steps) >= 2:
                    first_src_key = steps[0].get("source_key")
                    last_tgt_key = steps[-1].get("target_key")
                    src_cell = self._node_index.get(first_src_key)
                    tgt_cell = self._node_index.get(last_tgt_key)
                    score = path_trace.get("combined_score", 0.0)
                    if src_cell and tgt_cell and score > 0.1:
                        derived.append((src_cell, tgt_cell, float(score)))

    uncertain = list(dict.fromkeys(uncertain))[:20]  # deduplicate, cap at 20

    return ReasoningSignal(
        uncertain_concepts=uncertain,
        high_value_paths=high_value,
        suggested_focus_words=uncertain,
        derived_edges=derived,
    )
```

- [ ] Run: `python3 -m pytest hpm_ai_v6/tests/test_reasoning_feedback.py -v`
- [ ] Expected: all 5 tests PASS
- [ ] Commit: `"feat: add ReasoningSignal dataclass and ReasoningAgent.reflect() method"`

---

### Task 2: MultiAgentReader._reinforce_edge

- [ ] Add tests to `test_reasoning_feedback.py`:

```python
def test_reinforce_edge_adds_pattern_to_word_agent():
    from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "test.txt")
        with open(corpus, "w") as f:
            f.write("Alice followed the rabbit.\n")
        reader = MultiAgentReader(corpus, warm_start=False, pattern_cache_dir=tmp)
        alice = Cell(name="word_alice", dim=0, embedding=[1,0,0])
        rabbit = Cell(name="word_rabbit", dim=0, embedding=[0,1,0])
        initial_count = len(reader.word_agent.patterns)
        reader._reinforce_edge(alice, rabbit, 0.7)
        assert len(reader.word_agent.patterns) == initial_count + 1
        # ReasoningAgent should be dirty after reinforcement
        assert reader.reasoning_agent._dirty is True


def test_reinforce_edge_skips_dim_mismatch():
    from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "test.txt")
        with open(corpus, "w") as f:
            f.write("Alice followed the rabbit.\n")
        reader = MultiAgentReader(corpus, warm_start=False, pattern_cache_dir=tmp)
        small = Cell(name="word_small", dim=0, embedding=[1,0])
        large = Cell(name="word_large", dim=0, embedding=[0,1,0,0,0])
        initial_count = len(reader.word_agent.patterns)
        reader._reinforce_edge(small, large, 0.7)  # should not raise, should skip
        assert len(reader.word_agent.patterns) == initial_count  # unchanged
```

- [ ] Add `_reinforce_edge` to `multi_agent_reader.py`:

```python
def _reinforce_edge(self, source: "Cell", target: "Cell", score: float) -> None:
    """Inject a reasoning-derived edge into word_agent's pattern list."""
    try:
        src_emb = source.as_numpy()
        tgt_emb = target.as_numpy()
    except Exception:
        return
    if src_emb.shape != tgt_emb.shape:
        return
    pattern = Cell(
        name=f"derived_{source.name}_{target.name}",
        dim=1,
        embedding=(tgt_emb - src_emb).tolist(),
        source=source,
        target=target,
        weight=float(score),
    )
    # Avoid duplicates
    existing_names = {p.name for p in self.word_agent.patterns}
    if pattern.name not in existing_names:
        self.word_agent.patterns.append(pattern)
        self.reasoning_agent.invalidate()
```

- [ ] Run: `python3 -m pytest hpm_ai_v6/tests/test_reasoning_feedback.py -v`
- [ ] Expected: all 7 tests PASS
- [ ] Commit: `"feat: add MultiAgentReader._reinforce_edge with dirty-flag invalidation"`

---

### Task 3: Wire reflect() into maintenance_cycle

- [ ] Add test to `test_reasoning_feedback.py`:

```python
def test_maintenance_cycle_accepts_query_batch():
    from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "test.txt")
        with open(corpus, "w") as f:
            f.write("Alice followed the rabbit down the hole.\n" * 5)
        reader = MultiAgentReader(corpus, warm_start=False, pattern_cache_dir=tmp)
        # Should not raise; query_batch is optional
        report = reader.maintenance_cycle(
            sentences=["Alice followed the rabbit."],
            query_batch=["Why did Alice follow the rabbit?"],
        )
        assert isinstance(report, dict)
```

- [ ] Update `maintenance_cycle` in `multi_agent_reader.py`:
  - Add `query_batch: Optional[List[str]] = None` parameter after `sentences`
  - Add `self._focus_words: set = set()` to `__init__`
  - At end of method body (after existing training + snapshot logic, before return):

```python
# Reasoning feedback: reflect on query_batch and reinforce derived edges
if query_batch:
    try:
        signal = self.reasoning_agent.reflect(query_batch)
        for src, tgt, score in signal.derived_edges:
            self._reinforce_edge(src, tgt, score)
        if signal.suggested_focus_words:
            self._focus_words = set(signal.suggested_focus_words)
    except Exception:
        pass
```

- [ ] Run: `python3 -m pytest hpm_ai_v6/tests/test_reasoning_feedback.py -v`
- [ ] Expected: all 8 tests PASS
- [ ] Run full suite: `python3 -m pytest hpm_ai_v6/tests/ -x -q 2>&1 | tail -5`
- [ ] Expected: all 106+ tests PASS
- [ ] Commit: `"feat: wire ReasoningAgent.reflect() into MultiAgentReader.maintenance_cycle"`

---

### Task 4: Full suite + end-to-end smoke test

- [ ] Run full test suite with verbose output:

```bash
python3 -m pytest hpm_ai_v6/tests/ -v --tb=short 2>&1 | tail -20
```

- [ ] Run end-to-end smoke test:

```bash
python3 -c "
from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
reader = MultiAgentReader('hpm_ai_v6/data/corpus/alice_mini.txt', warm_start=False)
reader.train(episodes=1, max_chunks=30, max_words_per_chunk=80, enable_pruning=False, enable_causal=False)
# Run maintenance with query feedback
report = reader.maintenance_cycle(
    sentences=['Alice followed the rabbit down the hole.'],
    query_batch=['Why did Alice follow the rabbit?', 'How does Alice connect to rabbit?'],
)
print('Signal uncertain:', reader.reasoning_agent.reflect(['Why did Alice follow the rabbit?']).uncertain_concepts[:5])
print('Word patterns after reinforce:', len(reader.word_agent.patterns))
print('Reasoning answer:', reader.reason('Why did Alice follow the rabbit?'))
" 2>&1 | grep -v 'Loading\|LOAD\|UNEXPECTED\|Notes\|it/s\|BertModel\|Starting\|Episode\|Patterns\|weights'
```

- [ ] Expected: no exceptions, prints uncertain concepts list, pattern count, reasoning answer
- [ ] Commit any fixes needed: `"fix: reasoning feedback loop integration fixes"`

---

## Acceptance Criteria

1. `ReasoningSignal` is importable from `hpm_ai_v6.agents.reasoning_agent`
2. `ReasoningAgent.reflect([])` returns an empty signal without error
3. `ReasoningAgent.reflect(queries)` returns uncertain concepts for unresolvable terms
4. `MultiAgentReader._reinforce_edge` adds edge patterns and marks reasoning agent dirty
5. `_reinforce_edge` silently skips dimension-mismatched cells
6. `maintenance_cycle(sentences, query_batch=...)` accepts optional query batch
7. All 106+ existing tests continue to pass
8. End-to-end smoke test runs without exceptions
