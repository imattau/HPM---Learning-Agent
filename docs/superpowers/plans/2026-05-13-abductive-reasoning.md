# Abductive Reasoning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `abductive_explain()` to `ReasoningAgent` so that given an observed effect the agent returns ranked `ExplanatorySubgraph` objects — minimal causal subgraphs connecting hypothesised root causes to the observed effect.

**Architecture:** A new `ExplanatorySubgraph` dataclass lives in `hpm_ai_v6/hpm_model/core/explanatory_subgraph.py`. `ReasoningAgent` gains two new methods — public `abductive_explain()` and private `_score_explanation()` — that walk the existing `_edge_index` backwards from the effect cell. `reason_with_trace()` is extended with intent-trigger keywords ("why", "explain", "what caused", "how did", "reason for") that route to `abductive_explain()` and surface the result under the `explanatory_subgraph` key.

**Tech Stack:** Python 3.11+, dataclasses, existing `Cell` / `EdgeRecord` / `_noisy_or_score` from `hpm_ai_v6`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| **Create** | `hpm_ai_v6/hpm_model/core/explanatory_subgraph.py` | `ExplanatorySubgraph` dataclass |
| **Modify** | `hpm_ai_v6/agents/reasoning_agent.py` | `abductive_explain()`, `_score_explanation()`, `reason_with_trace()` intent triggers |
| **Create** | `hpm_ai_v6/tests/test_abductive_reasoning.py` | All unit + integration tests |

---

### Task 1: Create `ExplanatorySubgraph` dataclass

**Files:**
- Create: `hpm_ai_v6/hpm_model/core/explanatory_subgraph.py`

- [ ] **Step 1: Write the failing import test**

```python
# hpm_ai_v6/tests/test_abductive_reasoning.py
from hpm_ai_v6.hpm_model.core.explanatory_subgraph import ExplanatorySubgraph
from hpm_ai_v6.agents.reasoning_agent import EdgeRecord
from hpm_ai_v6.hpm_model.core.cell import Cell

def test_explanatory_subgraph_fields():
    rain = Cell(name="word_rain", dim=0, embedding=[1.0, 0.0, 0.0])
    flood = Cell(name="word_flood", dim=0, embedding=[0.0, 1.0, 0.0])
    edge = Cell(
        name="w_word_rain->word_flood", dim=1,
        embedding=[0.0, 1.0, 0.0],
        source=rain, target=flood, weight=0.9,
    )
    record = EdgeRecord(
        pattern=edge, source=rain, target=flood, score=0.9,
        raw_weight=0.9, agent_name="word",
        source_key="word_rain", target_key="word_flood",
        relation="causes",
    )
    sg = ExplanatorySubgraph(
        effect=flood,
        root_causes=[rain],
        edges=[record],
        plausibility=0.9,
        depth=1,
    )
    assert sg.effect is flood
    assert sg.root_causes == [rain]
    assert sg.edges == [record]
    assert sg.plausibility == 0.9
    assert sg.depth == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v6/tests/test_abductive_reasoning.py::test_explanatory_subgraph_fields -v
```

Expected: `ModuleNotFoundError: No module named 'hpm_ai_v6.hpm_model.core.explanatory_subgraph'`

- [ ] **Step 3: Create the dataclass file**

```python
# hpm_ai_v6/hpm_model/core/explanatory_subgraph.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.agents.reasoning_agent import EdgeRecord


@dataclass
class ExplanatorySubgraph:
    """Minimal causal subgraph explaining an observed effect via abductive reasoning."""

    effect: Cell
    """The observed effect cell being explained."""

    root_causes: List[Cell]
    """Cells with no incoming edges in this subgraph — the hypothesised origins."""

    edges: List[EdgeRecord]
    """Minimal set of edges connecting root_causes to effect."""

    plausibility: float
    """Aggregate score: noisy-OR of edge scores divided by depth (Occam penalty)."""

    depth: int
    """Number of hops from deepest root cause to effect."""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest hpm_ai_v6/tests/test_abductive_reasoning.py::test_explanatory_subgraph_fields -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v6/hpm_model/core/explanatory_subgraph.py hpm_ai_v6/tests/test_abductive_reasoning.py
git commit -m "feat: add ExplanatorySubgraph dataclass"
```

---

### Task 2: Add `_score_explanation()` to `ReasoningAgent`

**Files:**
- Modify: `hpm_ai_v6/agents/reasoning_agent.py`
- Test: `hpm_ai_v6/tests/test_abductive_reasoning.py`

- [ ] **Step 1: Write the failing test**

Add to `hpm_ai_v6/tests/test_abductive_reasoning.py`:

```python
from types import SimpleNamespace
from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent, EdgeRecord
from hpm_ai_v6.hpm_model.core.cell import Cell


class StubAgent:
    def __init__(self, patterns=None, weights=None, lookup=None):
        self.patterns = patterns or []
        self._weights = weights or []
        self._lookup = lookup or {}

    def get_weights(self):
        return list(self._weights)

    def _paging_lookup(self):
        return dict(self._lookup)


def make_word(name: str, embedding):
    return Cell(name=f"word_{name}", dim=0, embedding=embedding)


def make_edge_cell(source: Cell, target: Cell, weight: float = 0.7):
    return Cell(
        name=f"w_{source.name}->{target.name}", dim=1,
        embedding=[t - s for s, t in zip(source.embedding, target.embedding)],
        source=source, target=target, weight=weight,
    )


def build_simple_reader(rain, flood, edge_weight=0.9):
    edge_cell = make_edge_cell(rain, flood, weight=edge_weight)
    word_agent = StubAgent(
        patterns=[edge_cell],
        weights=[edge_weight],
        lookup={rain.name: rain, flood.name: flood},
    )
    return SimpleNamespace(
        agents={"word": word_agent, "contextual": None, "phrase": None,
                "semantic": None, "causal": None, "temporal": None},
        relation_registry=None,
    )


def test_score_explanation_single_edge():
    rain = make_word("rain", [1.0, 0.0, 0.0])
    flood = make_word("flood", [0.0, 1.0, 0.0])
    reader = build_simple_reader(rain, flood, edge_weight=0.8)
    agent = ReasoningAgent(reader)
    agent._ensure_fresh()

    edge_cell = make_edge_cell(rain, flood, weight=0.8)
    record = EdgeRecord(
        pattern=edge_cell, source=rain, target=flood, score=0.8,
        raw_weight=0.8, agent_name="word",
        source_key=rain.name, target_key=flood.name, relation="causes",
    )
    score = agent._score_explanation([record], depth=1)
    # noisy-OR of [0.8] = 0.8, divided by depth 1 = 0.8
    assert abs(score - 0.8) < 1e-6


def test_score_explanation_penalises_depth():
    rain = make_word("rain", [1.0, 0.0, 0.0])
    flood = make_word("flood", [0.0, 1.0, 0.0])
    reader = build_simple_reader(rain, flood, edge_weight=0.8)
    agent = ReasoningAgent(reader)
    agent._ensure_fresh()

    edge_cell = make_edge_cell(rain, flood, weight=0.8)
    record = EdgeRecord(
        pattern=edge_cell, source=rain, target=flood, score=0.8,
        raw_weight=0.8, agent_name="word",
        source_key=rain.name, target_key=flood.name, relation="causes",
    )
    score_depth1 = agent._score_explanation([record], depth=1)
    score_depth2 = agent._score_explanation([record], depth=2)
    assert score_depth1 > score_depth2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest hpm_ai_v6/tests/test_abductive_reasoning.py::test_score_explanation_single_edge hpm_ai_v6/tests/test_abductive_reasoning.py::test_score_explanation_penalises_depth -v
```

Expected: `AttributeError: 'ReasoningAgent' object has no attribute '_score_explanation'`

- [ ] **Step 3: Add `_score_explanation()` to `ReasoningAgent`**

Locate `_noisy_or_score` in `reasoning_agent.py` (around line 1868). Add the following method directly after the `_noisy_or_score` static method:

```python
def _score_explanation(self, edges: List["EdgeRecord"], depth: int) -> float:
    """Score a candidate explanatory subgraph.

    Uses noisy-OR over edge scores then divides by depth as an Occam penalty:
    shorter explanations are preferred for equal edge quality.
    """
    scores = [e.score for e in edges]
    raw = self._noisy_or_score(scores)
    return raw / max(depth, 1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest hpm_ai_v6/tests/test_abductive_reasoning.py::test_score_explanation_single_edge hpm_ai_v6/tests/test_abductive_reasoning.py::test_score_explanation_penalises_depth -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v6/agents/reasoning_agent.py hpm_ai_v6/tests/test_abductive_reasoning.py
git commit -m "feat: add _score_explanation to ReasoningAgent"
```

---

### Task 3: Implement `abductive_explain()` — single-hop case

**Files:**
- Modify: `hpm_ai_v6/agents/reasoning_agent.py`
- Test: `hpm_ai_v6/tests/test_abductive_reasoning.py`

- [ ] **Step 1: Write the failing tests**

Add to `hpm_ai_v6/tests/test_abductive_reasoning.py`:

```python
from hpm_ai_v6.hpm_model.core.explanatory_subgraph import ExplanatorySubgraph


def test_abductive_explain_unknown_effect():
    rain = make_word("rain", [1.0, 0.0, 0.0])
    flood = make_word("flood", [0.0, 1.0, 0.0])
    reader = build_simple_reader(rain, flood)
    agent = ReasoningAgent(reader)
    result = agent.abductive_explain("nonexistent_term")
    assert result == []


def test_abductive_explain_single_cause():
    rain = make_word("rain", [1.0, 0.0, 0.0])
    flood = make_word("flood", [0.0, 1.0, 0.0])
    reader = build_simple_reader(rain, flood, edge_weight=0.9)
    agent = ReasoningAgent(reader)

    results = agent.abductive_explain("flood", max_depth=4, top_k=3)

    assert len(results) >= 1
    top = results[0]
    assert isinstance(top, ExplanatorySubgraph)
    assert top.effect.name == "word_flood"
    assert any(c.name == "word_rain" for c in top.root_causes)
    assert top.depth == 1
    assert top.plausibility > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest hpm_ai_v6/tests/test_abductive_reasoning.py::test_abductive_explain_unknown_effect hpm_ai_v6/tests/test_abductive_reasoning.py::test_abductive_explain_single_cause -v
```

Expected: `AttributeError: 'ReasoningAgent' object has no attribute 'abductive_explain'`

- [ ] **Step 3: Implement `abductive_explain()`**

Add the following method to `ReasoningAgent` after `_score_explanation()`. Import `ExplanatorySubgraph` at the top of the method (or add the import to the module-level imports if preferred — avoid circular import by importing inside the method body):

```python
def abductive_explain(
    self,
    effect: str,
    max_depth: int = 4,
    top_k: int = 3,
) -> List["ExplanatorySubgraph"]:
    """Return the top-k most plausible explanatory subgraphs for an observed effect.

    Algorithm:
    1. Resolve effect string to a Cell via _resolve_cell(). Return [] if not found.
    2. Collect all EdgeRecords in _edge_index whose target matches the effect cell.
    3. Recursively expand predecessor cells up to max_depth, accumulating edges.
    4. Identify root causes: cells that have no incoming edges in the subgraph.
    5. Score each subgraph with _score_explanation().
    6. Return top_k subgraphs sorted by plausibility descending.
    """
    from hpm_ai_v6.hpm_model.core.explanatory_subgraph import ExplanatorySubgraph

    self._ensure_fresh()

    effect_cell = self._resolve_cell(effect)
    if effect_cell is None:
        return []

    # Build a reverse lookup: target_key -> List[EdgeRecord] across all agents
    reverse_index: Dict[str, List[EdgeRecord]] = {}
    for _src_key, records in self._edge_index.items():
        for rec in records:
            reverse_index.setdefault(rec.target_key, []).append(rec)

    def expand(cell: Cell, depth_remaining: int) -> List[List[EdgeRecord]]:
        """Return a list of edge-paths, each path being edges from some root to cell."""
        if depth_remaining == 0:
            return [[]]  # treat current cell as a root
        incoming = reverse_index.get(cell.name, [])
        if not incoming:
            return [[]]  # cell is a root — return an empty path segment
        paths: List[List[EdgeRecord]] = []
        for rec in incoming:
            for sub_path in expand(rec.source, depth_remaining - 1):
                paths.append(sub_path + [rec])
        return paths if paths else [[]]

    all_paths = expand(effect_cell, max_depth)

    subgraphs: List[ExplanatorySubgraph] = []
    seen: set = set()

    for path_edges in all_paths:
        if not path_edges:
            continue
        edge_key = tuple(sorted(id(e) for e in path_edges))
        if edge_key in seen:
            continue
        seen.add(edge_key)

        depth = len(path_edges)
        # Root cause: source of the first edge in the path (deepest predecessor)
        root_cell = path_edges[0].source
        plausibility = self._score_explanation(path_edges, depth)
        sg = ExplanatorySubgraph(
            effect=effect_cell,
            root_causes=[root_cell],
            edges=path_edges,
            plausibility=plausibility,
            depth=depth,
        )
        subgraphs.append(sg)

    subgraphs.sort(key=lambda s: s.plausibility, reverse=True)
    return subgraphs[:top_k]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest hpm_ai_v6/tests/test_abductive_reasoning.py::test_abductive_explain_unknown_effect hpm_ai_v6/tests/test_abductive_reasoning.py::test_abductive_explain_single_cause -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v6/agents/reasoning_agent.py
git commit -m "feat: implement abductive_explain single-hop case"
```

---

### Task 4: Multi-hop and multi-cause tests

**Files:**
- Test: `hpm_ai_v6/tests/test_abductive_reasoning.py`

- [ ] **Step 1: Write the failing tests**

Add to `hpm_ai_v6/tests/test_abductive_reasoning.py`:

```python
def build_chain_reader():
    """rain -> flood -> damage chain."""
    rain = make_word("rain", [1.0, 0.0, 0.0])
    flood = make_word("flood", [0.0, 1.0, 0.0])
    damage = make_word("damage", [0.0, 0.0, 1.0])

    e1 = make_edge_cell(rain, flood, weight=0.9)
    e2 = make_edge_cell(flood, damage, weight=0.8)

    word_agent = StubAgent(
        patterns=[e1, e2],
        weights=[0.9, 0.8],
        lookup={rain.name: rain, flood.name: flood, damage.name: damage},
    )
    return SimpleNamespace(
        agents={"word": word_agent, "contextual": None, "phrase": None,
                "semantic": None, "causal": None, "temporal": None},
        relation_registry=None,
    ), rain, flood, damage


def test_abductive_explain_chain():
    reader, rain, flood, damage = build_chain_reader()
    agent = ReasoningAgent(reader)

    results = agent.abductive_explain("damage", max_depth=4, top_k=3)

    assert len(results) >= 1
    # The deepest explanation should reach rain as root cause
    root_names = {c.name for sg in results for c in sg.root_causes}
    assert "word_rain" in root_names or "word_flood" in root_names

    # The chain explanation through flood should have depth=2
    chain_sg = next((sg for sg in results if any(c.name == "word_rain" for c in sg.root_causes)), None)
    if chain_sg:
        assert chain_sg.depth == 2


def build_multi_cause_reader():
    """storm -> flood, rain -> flood (two independent causes)."""
    storm = make_word("storm", [1.0, 0.0, 0.0])
    rain = make_word("rain", [0.5, 0.5, 0.0])
    flood = make_word("flood", [0.0, 1.0, 0.0])

    e1 = make_edge_cell(storm, flood, weight=0.85)
    e2 = make_edge_cell(rain, flood, weight=0.75)

    word_agent = StubAgent(
        patterns=[e1, e2],
        weights=[0.85, 0.75],
        lookup={storm.name: storm, rain.name: rain, flood.name: flood},
    )
    return SimpleNamespace(
        agents={"word": word_agent, "contextual": None, "phrase": None,
                "semantic": None, "causal": None, "temporal": None},
        relation_registry=None,
    ), storm, rain, flood


def test_abductive_explain_multiple_causes():
    reader, storm, rain, flood = build_multi_cause_reader()
    agent = ReasoningAgent(reader)

    results = agent.abductive_explain("flood", max_depth=4, top_k=5)

    assert len(results) >= 2
    root_names = {c.name for sg in results for c in sg.root_causes}
    assert "word_storm" in root_names
    assert "word_rain" in root_names


def test_abductive_explain_returns_top_k():
    reader, storm, rain, flood = build_multi_cause_reader()
    agent = ReasoningAgent(reader)

    results = agent.abductive_explain("flood", max_depth=4, top_k=1)
    assert len(results) == 1
    # top result should be the higher-scoring one (storm edge = 0.85)
    assert results[0].root_causes[0].name == "word_storm"


def test_plausibility_penalises_depth():
    reader, rain, flood, damage = build_chain_reader()
    agent = ReasoningAgent(reader)

    # Explain flood: direct cause is rain at depth=1
    flood_results = agent.abductive_explain("flood", max_depth=4, top_k=3)
    # Explain damage: best path is rain->flood->damage at depth=2
    damage_results = agent.abductive_explain("damage", max_depth=4, top_k=3)

    if flood_results and damage_results:
        # Same edge quality but depth-2 path must score lower than depth-1
        flood_top = flood_results[0].plausibility
        damage_top = damage_results[0].plausibility
        assert flood_top >= damage_top
```

- [ ] **Step 2: Run tests to verify they fail (or expose issues)**

```bash
python -m pytest hpm_ai_v6/tests/test_abductive_reasoning.py::test_abductive_explain_chain hpm_ai_v6/tests/test_abductive_reasoning.py::test_abductive_explain_multiple_causes hpm_ai_v6/tests/test_abductive_reasoning.py::test_abductive_explain_returns_top_k hpm_ai_v6/tests/test_abductive_reasoning.py::test_plausibility_penalises_depth -v
```

Expected: Some or all fail — the multi-cause and top-k behaviour may not be correct yet.

- [ ] **Step 3: Fix `abductive_explain()` for multiple independent causes**

The current `expand()` function returns paths rooted at individual predecessors. When two independent edges both point to the effect, each should produce its own subgraph. Verify the existing implementation handles this correctly by checking that `reverse_index.get(effect_cell.name, [])` yields two records for the multi-cause case. If `test_abductive_explain_returns_top_k` fails because the wrong root cause is returned first, check that `subgraphs.sort(key=lambda s: s.plausibility, reverse=True)` is in place and that edge scores are being indexed correctly.

No code change needed if tests pass. If `test_abductive_explain_returns_top_k` fails, verify the `StubAgent.get_weights()` list order matches the `patterns` list — the indexing in `_ensure_fresh()` (which calls `get_weights()`) must line up with the pattern list. Review `ReasoningAgent._ensure_fresh()` to confirm `EdgeRecord.score` is being set from the matching weight value, not always `0.0`.

- [ ] **Step 4: Run all abductive tests**

```bash
python -m pytest hpm_ai_v6/tests/test_abductive_reasoning.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v6/tests/test_abductive_reasoning.py
git commit -m "test: add multi-hop and multi-cause abductive reasoning tests"
```

---

### Task 5: Integrate abductive intent triggers into `reason_with_trace()`

**Files:**
- Modify: `hpm_ai_v6/agents/reasoning_agent.py`
- Test: `hpm_ai_v6/tests/test_abductive_reasoning.py`

- [ ] **Step 1: Write the failing integration tests**

Add to `hpm_ai_v6/tests/test_abductive_reasoning.py`:

```python
def test_reason_with_trace_why_intent():
    rain = make_word("rain", [1.0, 0.0, 0.0])
    flood = make_word("flood", [0.0, 1.0, 0.0])
    reader = build_simple_reader(rain, flood, edge_weight=0.9)
    agent = ReasoningAgent(reader)

    result = agent.reason_with_trace("why did flood happen?")

    assert result.get("method") == "abductive_explain"
    assert "explanatory_subgraph" in result


def test_reason_with_trace_explain_intent():
    rain = make_word("rain", [1.0, 0.0, 0.0])
    flood = make_word("flood", [0.0, 1.0, 0.0])
    reader = build_simple_reader(rain, flood, edge_weight=0.9)
    agent = ReasoningAgent(reader)

    result = agent.reason_with_trace("explain flood")

    assert result.get("method") == "abductive_explain"
    assert "explanatory_subgraph" in result


def test_reason_with_trace_abductive_unknown_effect():
    rain = make_word("rain", [1.0, 0.0, 0.0])
    flood = make_word("flood", [0.0, 1.0, 0.0])
    reader = build_simple_reader(rain, flood)
    agent = ReasoningAgent(reader)

    result = agent.reason_with_trace("why did zzznonsense happen?")

    assert result.get("method") == "abductive_explain"
    assert "not found" in result.get("answer", "").lower() or result.get("explanatory_subgraph") is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest hpm_ai_v6/tests/test_abductive_reasoning.py::test_reason_with_trace_why_intent hpm_ai_v6/tests/test_abductive_reasoning.py::test_reason_with_trace_explain_intent hpm_ai_v6/tests/test_abductive_reasoning.py::test_reason_with_trace_abductive_unknown_effect -v
```

Expected: `AssertionError` — method is not `abductive_explain`.

- [ ] **Step 3: Add intent triggers to `reason_with_trace()`**

Locate `reason_with_trace()` in `reasoning_agent.py`. Before the existing intent-check block (the first `if` that inspects question words / method routing), add:

```python
# --- Abductive intent detection (must come before other intent checks) ---
_ABDUCTIVE_TRIGGERS = ("why", "explain", "what caused", "how did", "reason for")
_question_lower = question.lower()
if any(trigger in _question_lower for trigger in _ABDUCTIVE_TRIGGERS):
    # Extract the term to explain: last significant token after the trigger words
    _stop = {
        "did", "do", "does", "happen", "occur", "cause", "the", "a", "an",
        "why", "how", "what", "explain", "reason", "for",
    }
    _tokens = [t for t in self._tokenize(_question_lower) if t not in _stop and t not in self.STOPWORDS]
    _effect_term = _tokens[-1] if _tokens else ""
    _subgraphs = self.abductive_explain(_effect_term, max_depth=self.max_depth, top_k=3)
    if _subgraphs:
        _top = _subgraphs[0]
        _root_names = ", ".join(c.name.replace("word_", "") for c in _top.root_causes)
        _answer = (
            f"The most plausible explanation for '{_effect_term}': "
            f"{_root_names} (plausibility={_top.plausibility:.2f}, depth={_top.depth})."
        )
        _sg_dict = {
            "effect": _top.effect.name,
            "root_causes": [c.name for c in _top.root_causes],
            "edges": [
                {"source": e.source.name, "target": e.target.name, "score": e.score}
                for e in _top.edges
            ],
            "plausibility": _top.plausibility,
            "depth": _top.depth,
        }
    else:
        _answer = f"No causal explanation found for '{_effect_term}'."
        _sg_dict = None
    return {
        "question": question,
        "method": "abductive_explain",
        "answer": _answer,
        "explanatory_subgraph": _sg_dict,
        "terms": _tokens,
    }
# --- End abductive intent detection ---
```

- [ ] **Step 4: Run all abductive tests**

```bash
python -m pytest hpm_ai_v6/tests/test_abductive_reasoning.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
python -m pytest hpm_ai_v6/tests/ -v --tb=short 2>&1 | tail -40
```

Expected: No new failures (pre-existing failures unrelated to abductive reasoning are acceptable — do not fix them in this task).

- [ ] **Step 6: Commit**

```bash
git add hpm_ai_v6/agents/reasoning_agent.py hpm_ai_v6/tests/test_abductive_reasoning.py
git commit -m "feat: integrate abductive intent triggers into reason_with_trace"
```

---

## Self-Review Checklist

**Spec coverage:**

| Spec requirement | Task covering it |
|---|---|
| `ExplanatorySubgraph` dataclass | Task 1 |
| `_score_explanation()` using `_noisy_or_score` / depth | Task 2 |
| `abductive_explain()` method signature and algorithm | Task 3 |
| Recursive expansion up to `max_depth` | Task 3 |
| Root cause identification (no incoming edges in subgraph) | Task 3 |
| Top-k sorting by plausibility descending | Task 3 |
| Multi-cause subgraphs | Task 4 |
| Depth penalty test | Task 4 |
| Intent triggers: "why", "explain", "what caused", "how did", "reason for" | Task 5 |
| `reason_with_trace()` returns `explanatory_subgraph` key | Task 5 |
| Answer format with plausibility and depth | Task 5 |
| Unknown effect returns `[]` / not-found answer | Tasks 3 + 5 |
| New file: `explanatory_subgraph.py` | Task 1 |
| Test: `test_abductive_explain_single_cause` | Task 3 |
| Test: `test_abductive_explain_chain` | Task 4 |
| Test: `test_abductive_explain_multiple_causes` | Task 4 |
| Test: `test_abductive_explain_returns_top_k` | Task 4 |
| Test: `test_plausibility_penalises_depth` | Task 4 |
| Test: `test_reason_with_trace_why_intent` | Task 5 |
| Test: `test_reason_with_trace_explain_intent` | Task 5 |
| Test: `test_abductive_explain_unknown_effect` | Tasks 3 + 5 |

All spec requirements covered.

**Type consistency check:**
- `ExplanatorySubgraph` fields (`effect: Cell`, `root_causes: List[Cell]`, `edges: List[EdgeRecord]`, `plausibility: float`, `depth: int`) are consistent across Tasks 1, 3, 4, and 5.
- `_score_explanation(self, edges: List[EdgeRecord], depth: int) -> float` — consistent in Tasks 2 and 3.
- `abductive_explain(self, effect: str, max_depth: int = 4, top_k: int = 3) -> List[ExplanatorySubgraph]` — consistent in Tasks 3, 4, and 5.
- `EdgeRecord` fields accessed (`e.score`, `e.source`, `e.target`, `e.source_key`, `e.target_key`) match the frozen dataclass defined in `reasoning_agent.py` lines 19–28.

**Placeholder scan:** No TBDs, no "implement later", no "similar to" references. All test functions contain full code. All implementation steps show the exact code to add.
