# Implementation Plan: KnowledgeFrontier

**Goal:** Add a `KnowledgeFrontier` class to the quiz CLI that uses WordNet's semantic graph to progressively expand Wikipedia fetch targets, always prioritising concepts with the sparsest pattern coverage in the HPM agents.

**Spec:** `docs/superpowers/specs/2026-05-17-knowledge-frontier-design.md`

**Architecture:**
- `KnowledgeFrontier` manages three sets: `known_seeds`, `frontier`, `exhausted`
- WordNet expansion at configurable `hop_depth` (1–5) yields candidate topics
- Edge density scoring (pager index hits) ranks candidates — fewest hits = most to learn
- State persisted to `hpm_ai_v6/data/quiz_banks/knowledge_frontier.json`
- Integration replaces direct model-nominated Wikipedia fetches in `quiz_cli.main()`

**Tech Stack:** `nltk.corpus.wordnet` (already a project dependency via NLTK); `json` stdlib; existing `MultiAgentReader` / `pattern_pager` API.

**Verified API signatures (from codebase):**
- `reader.agents` — `dict[str, agent]`
- `agent.pattern_pager` — may be `None`; has `iter_index_payloads()` → `Iterable[dict]`
- `_nominate_uncertain_topics(dataset_agent, reasoning_agent, n, ...)` — returns `List[str]`
- `train_on_weak_topics(reader, topics, ...)` — `topics` is `List[Tuple[str, str]]`

---

## File Map

| File | Action |
|---|---|
| `hpm_ai_v6/cli/quiz_cli.py` | Add `KnowledgeFrontier` class; update `main()` |
| `hpm_ai_v6/data/quiz_banks/knowledge_frontier.json` | Auto-created at first run |
| `hpm_ai_v6/tests/test_knowledge_frontier.py` | Create — unit tests |

---

## Task 1 — `KnowledgeFrontier` class

### Steps

- [ ] Write failing tests in `test_knowledge_frontier.py` → run → confirm FAIL
- [ ] Implement `KnowledgeFrontier` in `quiz_cli.py` (or `knowledge_frontier.py`)
- [ ] Run tests → confirm PASS
- [ ] Commit

### Failing tests (write first)

```python
# hpm_ai_v6/tests/test_knowledge_frontier.py
import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch


def _make_reader(edge_counts: dict):
    """Build a mock reader whose agents return controlled edge counts."""
    reader = MagicMock()
    agent = MagicMock()
    # iter_index_payloads yields dicts with "name" key
    payloads = [{"name": term} for term, count in edge_counts.items() for _ in range(count)]
    agent.pattern_pager.iter_index_payloads.return_value = iter(payloads)
    reader.agents = {"agent0": agent}
    return reader


class TestKnowledgeFrontierInit:
    def test_default_state(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        assert kf.known_seeds == set()
        assert kf.frontier == []
        assert kf.exhausted == set()
        assert kf.hop_depth == 1

    def test_load_missing_path_returns_default(self, tmp_path):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier.load(str(tmp_path / "nonexistent.json"))
        assert kf.hop_depth == 1
        assert kf.known_seeds == set()

    def test_save_and_load_roundtrip(self, tmp_path):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        path = str(tmp_path / "kf.json")
        kf = KnowledgeFrontier()
        kf.known_seeds = {"dog", "cat"}
        kf.frontier = ["animal", "mammal"]
        kf.exhausted = {"pet"}
        kf.hop_depth = 3
        kf.save(path)
        kf2 = KnowledgeFrontier.load(path)
        assert kf2.known_seeds == {"dog", "cat"}
        assert sorted(kf2.frontier) == ["animal", "mammal"]
        assert kf2.exhausted == {"pet"}
        assert kf2.hop_depth == 3


class TestIncrementHop:
    def test_increments(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        kf.increment_hop()
        assert kf.hop_depth == 2

    def test_capped_at_5(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        kf.hop_depth = 5
        kf.increment_hop()
        assert kf.hop_depth == 5


class TestEdgeDensity:
    def test_counts_matching_payloads(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        reader = _make_reader({"dog": 3, "cat": 1, "animal": 0})
        kf = KnowledgeFrontier()
        # _edge_density matches first word of term against payload names
        count = kf._edge_density("dog", reader)
        assert count == 3

    def test_zero_for_unknown_term(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        reader = _make_reader({"dog": 2})
        kf = KnowledgeFrontier()
        assert kf._edge_density("zebra", reader) == 0

    def test_skips_agents_without_pager(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        reader = MagicMock()
        agent = MagicMock()
        agent.pattern_pager = None
        reader.agents = {"a": agent}
        kf = KnowledgeFrontier()
        assert kf._edge_density("anything", reader) == 0


class TestWordnetCandidates:
    def test_returns_set_of_strings(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        candidates = kf._wordnet_candidates("dog", hop_depth=1)
        assert isinstance(candidates, set)
        # WordNet should yield at least one hypernym for "dog"
        assert len(candidates) > 0
        for c in candidates:
            assert isinstance(c, str)

    def test_hop2_returns_more_than_hop1(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        hop1 = kf._wordnet_candidates("dog", hop_depth=1)
        hop2 = kf._wordnet_candidates("dog", hop_depth=2)
        assert len(hop2) >= len(hop1)

    def test_filters_pos_prefixes(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        candidates = kf._wordnet_candidates("dog", hop_depth=2)
        for c in candidates:
            assert not c.startswith("pos_")
            assert not c.startswith("word_")
            assert len(c) > 1


class TestAddLearnedSeeds:
    def test_new_seeds_added_to_known(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        reader = _make_reader({})
        with patch.object(kf, "_wordnet_candidates", return_value={"mammal", "canine", "pet"}):
            kf.add_learned_seeds(["dog"], reader)
        assert "dog" in kf.known_seeds

    def test_already_known_seeds_not_re_expanded(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        kf.known_seeds = {"dog"}
        reader = _make_reader({})
        with patch.object(kf, "_wordnet_candidates", return_value=set()) as mock_wn:
            kf.add_learned_seeds(["dog"], reader)
        mock_wn.assert_not_called()

    def test_candidates_added_to_frontier(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        reader = _make_reader({})
        candidates = {"mammal", "canine", "pet", "hound", "carnivore", "animal", "canis", "wolf"}
        with patch.object(kf, "_wordnet_candidates", return_value=candidates):
            kf.add_learned_seeds(["dog"], reader)
        # Top 8 added; here all 8 are candidates
        assert len(kf.frontier) <= 8
        assert all(c in candidates for c in kf.frontier)

    def test_exhausted_terms_excluded_from_frontier(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        kf.exhausted = {"mammal"}
        reader = _make_reader({})
        with patch.object(kf, "_wordnet_candidates", return_value={"mammal", "canine"}):
            kf.add_learned_seeds(["dog"], reader)
        assert "mammal" not in kf.frontier


class TestNextTopics:
    def test_returns_n_topics(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        kf.frontier = ["alpha", "beta", "gamma", "delta", "epsilon"]
        reader = _make_reader({})  # all zero density → equal ranking
        topics = kf.next_topics(reader, n=3)
        assert len(topics) == 3

    def test_returned_topics_removed_from_frontier(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        kf.frontier = ["alpha", "beta", "gamma"]
        reader = _make_reader({})
        topics = kf.next_topics(reader, n=2)
        for t in topics:
            assert t not in kf.frontier

    def test_returns_fewer_if_frontier_small(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        kf.frontier = ["only"]
        reader = _make_reader({})
        topics = kf.next_topics(reader, n=4)
        assert len(topics) == 1

    def test_prefers_sparser_topics(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        kf.frontier = ["dense", "sparse"]
        reader = MagicMock()
        agent = MagicMock()
        # dense has 10 edges, sparse has 0
        def side_effect():
            return iter([{"name": "dense"}] * 10)
        agent.pattern_pager.iter_index_payloads.side_effect = side_effect
        reader.agents = {"a": agent}
        topics = kf.next_topics(reader, n=1)
        assert topics == ["sparse"]
```

### Run to confirm fail

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v6/tests/test_knowledge_frontier.py -x 2>&1 | tail -10
```

### Implementation: `KnowledgeFrontier` class (add to `quiz_cli.py`)

```python
# ── KnowledgeFrontier ────────────────────────────────────────────────────────

import json as _json
from pathlib import Path as _Path

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "but", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "it", "its", "this", "that", "with",
}

_DEFAULT_FRONTIER_PATH = (
    _Path(__file__).parent.parent / "data" / "quiz_banks" / "knowledge_frontier.json"
)


class KnowledgeFrontier:
    """Progressively expands Wikipedia fetch targets via WordNet semantic graph.

    Seeds come from _nominate_uncertain_topics(); the frontier is scored by
    pager edge density so the sparsest (least-known) concepts are fetched first.
    """

    def __init__(self):
        self.known_seeds: set[str] = set()
        self.frontier: list[str] = []
        self.exhausted: set[str] = set()
        self.hop_depth: int = 1

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | None = None) -> None:
        target = _Path(path) if path else _DEFAULT_FRONTIER_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "known_seeds": sorted(self.known_seeds),
            "frontier": self.frontier,
            "exhausted": sorted(self.exhausted),
            "hop_depth": self.hop_depth,
        }
        target.write_text(_json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | None = None) -> "KnowledgeFrontier":
        target = _Path(path) if path else _DEFAULT_FRONTIER_PATH
        kf = cls()
        if not target.exists():
            return kf
        try:
            data = _json.loads(target.read_text())
            kf.known_seeds = set(data.get("known_seeds", []))
            kf.frontier = data.get("frontier", [])
            kf.exhausted = set(data.get("exhausted", []))
            kf.hop_depth = int(data.get("hop_depth", 1))
        except Exception:
            pass  # corrupt file → fresh state
        return kf

    # ── hop depth ────────────────────────────────────────────────────────────

    def increment_hop(self) -> None:
        self.hop_depth = min(self.hop_depth + 1, 5)

    # ── internal helpers ─────────────────────────────────────────────────────

    def _wordnet_candidates(self, term: str, hop_depth: int) -> set[str]:
        try:
            from nltk.corpus import wordnet
        except ImportError:
            return set()

        synsets = wordnet.synsets(term.lower().replace(" ", "_"))[:3]
        candidates: set[str] = set()

        def _name(syn) -> str:
            return syn.lemmas()[0].name().replace("_", " ")

        for syn in synsets:
            for hop1 in syn.hypernyms() + syn.hyponyms():
                candidates.add(_name(hop1))
                if hop_depth >= 2:
                    for hop2 in hop1.hypernyms() + hop1.hyponyms():
                        candidates.add(_name(hop2))

        # Filter noise
        filtered = set()
        for c in candidates:
            if len(c) <= 1:
                continue
            if c.startswith("pos_") or c.startswith("word_"):
                continue
            if c.lower() in _STOPWORDS:
                continue
            filtered.add(c)
        return filtered

    def _edge_density(self, term: str, reader) -> int:
        word = term.lower().split()[0]
        count = 0
        for agent in reader.agents.values():
            pager = getattr(agent, "pattern_pager", None)
            if pager is None:
                continue
            for payload in pager.iter_index_payloads():
                if word in str(payload.get("name", "")).lower():
                    count += 1
        return count

    # ── public API ───────────────────────────────────────────────────────────

    def add_learned_seeds(self, terms: list[str], reader) -> None:
        """Expand frontier from new seed terms via WordNet."""
        for term in terms:
            key = term.lower().strip()
            if key in self.known_seeds:
                continue
            self.known_seeds.add(key)
            candidates = self._wordnet_candidates(key, self.hop_depth)
            # Remove already known / exhausted
            candidates -= self.known_seeds
            candidates -= self.exhausted
            # Remove terms already in frontier
            existing = set(self.frontier)
            candidates -= existing
            # Score by edge density (ascending = most to learn)
            scored = sorted(candidates, key=lambda c: self._edge_density(c, reader))
            self.frontier.extend(scored[:8])

    def next_topics(self, reader, n: int = 4) -> list[str]:
        """Return up to n frontier topics with the lowest edge density."""
        if not self.frontier:
            return []
        # Re-score frontier live
        scored = sorted(self.frontier, key=lambda c: self._edge_density(c, reader))
        chosen = scored[:n]
        # Remove chosen from frontier
        chosen_set = set(chosen)
        self.frontier = [t for t in self.frontier if t not in chosen_set]
        return chosen
```

### Run tests

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v6/tests/test_knowledge_frontier.py -v 2>&1 | tail -30
```

### Commit

```bash
git add hpm_ai_v6/cli/quiz_cli.py hpm_ai_v6/tests/test_knowledge_frontier.py
git commit -m "feat: add KnowledgeFrontier class with WordNet expansion and edge density scoring"
```

---

## Task 2 — Unit tests (complete file)

The tests are written above in Task 1 (TDD order). The complete file is:

**`hpm_ai_v6/tests/test_knowledge_frontier.py`** — content shown in Task 1 failing-tests block above.

No additional tests are required beyond what is specified there. Confirm all pass before proceeding to Task 3.

### Run full suite

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v6/tests/test_knowledge_frontier.py -v 2>&1 | tail -40
```

---

## Task 3 — Integration into `quiz_cli.main()`

### Steps

- [ ] Identify the section of `main()` that calls `_nominate_uncertain_topics` and fetches Wikipedia
- [ ] Write a failing integration test that confirms `frontier.next_topics` is called during the loop
- [ ] Replace model-nominated fetch logic with frontier-driven fetch
- [ ] Run tests → confirm PASS
- [ ] Commit

### Failing integration test (add to `test_knowledge_frontier.py`)

```python
class TestQuizCliIntegration:
    """Smoke-test that main() wires KnowledgeFrontier correctly."""

    def test_frontier_consulted_in_loop(self, monkeypatch):
        """next_topics must be called at least once when auto loop runs one iteration."""
        from unittest.mock import MagicMock, patch, call
        import hpm_ai_v6.cli.quiz_cli as cli_mod

        mock_frontier = MagicMock()
        mock_frontier.next_topics.return_value = ["test_topic"]
        mock_frontier.known_seeds = set()
        mock_frontier.frontier = []
        mock_frontier.exhausted = set()
        mock_frontier.hop_depth = 1

        with patch.object(cli_mod, "KnowledgeFrontier") as MockKF:
            MockKF.load.return_value = mock_frontier
            # Patch reader, quiz_agent, etc. so main() can run one iteration
            with patch.object(cli_mod, "MultiAgentReader", return_value=MagicMock()):
                with patch.object(cli_mod, "_run_quiz_round", return_value=([], [])):
                    with patch.object(cli_mod, "_nominate_uncertain_topics", return_value=["dog"]):
                        with patch.object(cli_mod, "train_on_weak_topics"):
                            # Run one loop iteration then break
                            try:
                                cli_mod.main(["--auto", "--rounds", "1"])
                            except SystemExit:
                                pass

        mock_frontier.next_topics.assert_called()
```

### Run to confirm fail

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v6/tests/test_knowledge_frontier.py::TestQuizCliIntegration -x 2>&1 | tail -10
```

### Integration changes in `quiz_cli.py` — `main()` diff

Replace the block that nominates uncertain topics and fetches Wikipedia directly:

**Before (approximate):**
```python
# Nominate uncertain topics and fetch Wikipedia
uncertain = _nominate_uncertain_topics(dataset_agent, reasoning_agent, n=4, ...)
if uncertain:
    train_on_weak_topics(reader, [(t, t) for t in uncertain], ...)
```

**After:**
```python
# 1. Initialise frontier (top of main(), before loop)
frontier = KnowledgeFrontier.load()

# ... inside the loop, after train_on_weak_topics for failed questions:

# 2. Get uncertain topics as seeds for frontier expansion
seeds = _nominate_uncertain_topics(dataset_agent, reasoning_agent, n=4, ...)

# 3. Expand frontier from seeds via WordNet
frontier.add_learned_seeds(seeds, reader)

# 4. Get next fetch targets from frontier (fewest edges = most to learn)
next_topics = frontier.next_topics(reader, n=4)

# 5. Fetch Wikipedia for frontier targets
if next_topics:
    train_on_weak_topics(reader, [(t, t) for t in next_topics], ...)

# 6. Expand hop depth and save frontier state
frontier.increment_hop()
frontier.save()
```

### Run full test suite

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v6/tests/test_knowledge_frontier.py -v 2>&1 | tail -40
```

### Commit

```bash
git add hpm_ai_v6/cli/quiz_cli.py hpm_ai_v6/tests/test_knowledge_frontier.py
git commit -m "feat: integrate KnowledgeFrontier into quiz_cli main() loop"
```

---

## Self-Review Checklist

- [ ] No placeholder comments (`TODO`, `FIXME`, `...`, `pass # stub`)
- [ ] All methods in spec are covered by at least one test
- [ ] `save()` / `load()` roundtrip tested
- [ ] `increment_hop()` cap at 5 tested
- [ ] Edge density scoring direction confirmed (ascending = sparse = priority)
- [ ] `exhausted` terms excluded from frontier candidates
- [ ] WordNet unavailable (ImportError) handled gracefully
- [ ] Integration test verifies `next_topics` is called in `main()`
- [ ] All three commits reference the correct files
