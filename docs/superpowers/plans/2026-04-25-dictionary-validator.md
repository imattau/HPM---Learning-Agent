# DictionaryValidator — Implementation Plan

**Date:** 2026-04-25
**Spec:** `docs/superpowers/specs/2026-04-25-dictionary-validator-design.md`
**Approach:** TDD — tests first, then implementation, then commit.

---

## Task 1: DictionaryValidator ABC + SimpleWordList

### Files
- `hpm_ai_v4/dictionary.py` — CREATE
- `hpm_ai_v4/tests/test_dictionary.py` — CREATE

### Step 1a — Write tests first

Create `hpm_ai_v4/tests/test_dictionary.py`:

```python
import pytest
from hpm_ai_v4.dictionary import SimpleWordList


def test_contains_exact():
    d = SimpleWordList(word_set={"hello", "world", "cat"})
    assert d.contains("hello")
    assert not d.contains("xyz")


def test_contains_case_insensitive():
    d = SimpleWordList(word_set={"Hello"})
    assert d.contains("hello")
    assert d.contains("HELLO")


def test_is_prefix_true():
    d = SimpleWordList(word_set={"hello", "help", "world"})
    assert d.is_prefix("hel")
    assert d.is_prefix("helo") == False  # not a valid prefix


def test_is_prefix_full_word():
    d = SimpleWordList(word_set={"cat"})
    assert d.is_prefix("cat")


def test_completions_basic():
    d = SimpleWordList(word_set={"hello", "help", "helm", "world"})
    comps = d.completions("hel")
    assert len(comps) <= 5
    assert all(c.startswith("hel") for c in comps)


def test_completions_empty_prefix():
    d = SimpleWordList(word_set={"cat", "dog"})
    comps = d.completions("")
    assert len(comps) <= 5


def test_completions_no_match():
    d = SimpleWordList(word_set={"cat"})
    assert d.completions("xyz") == []


def test_score_word_valid():
    d = SimpleWordList(word_set={"cat"})
    assert d.score_word("cat") == 1.0


def test_score_word_invalid():
    d = SimpleWordList(word_set={"cat"})
    assert d.score_word("xyz") == 0.0


def test_from_empty():
    d = SimpleWordList()
    assert not d.contains("anything")
    assert not d.is_prefix("a")
```

### Step 1b — Run (expect failure)

```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_dictionary.py -v
```

Expected: `ModuleNotFoundError: No module named 'hpm_ai_v4.dictionary'`

### Step 1c — Implement `hpm_ai_v4/dictionary.py`

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Set


class DictionaryValidator(ABC):
    """Abstract base class for pluggable lexical validation."""

    @abstractmethod
    def contains(self, word: str) -> bool:
        """Return True if word is in the dictionary (case-insensitive)."""
        ...

    @abstractmethod
    def is_prefix(self, prefix: str) -> bool:
        """Return True if prefix is a valid prefix of any word in the dictionary."""
        ...

    @abstractmethod
    def completions(self, prefix: str, max_suggestions: int = 5) -> List[str]:
        """Return up to max_suggestions words that start with prefix."""
        ...

    @abstractmethod
    def score_word(self, word: str) -> float:
        """Return 1.0 if word is in the dictionary, 0.0 otherwise."""
        ...


class SimpleWordList(DictionaryValidator):
    """
    Trie-based dictionary for O(L) prefix and membership checks.

    Construction:
        SimpleWordList()                        # empty
        SimpleWordList(word_set={"cat", ...})   # from a Python set
        SimpleWordList(filepath="words.txt")    # one word per line
    """

    _END = "__end__"

    def __init__(
        self,
        word_set: Optional[Set[str]] = None,
        filepath: Optional[str] = None,
    ):
        self._trie: dict = {}
        if filepath is not None:
            with open(filepath, "r", encoding="utf-8") as fh:
                for line in fh:
                    word = line.strip().lower()
                    if word:
                        self._insert(word)
        if word_set is not None:
            for word in word_set:
                self._insert(word.lower())

    def _insert(self, word: str) -> None:
        node = self._trie
        for ch in word:
            node = node.setdefault(ch, {})
        node[self._END] = {}

    def _find_node(self, prefix: str) -> Optional[dict]:
        """Return the trie node at the end of prefix, or None if not found."""
        node = self._trie
        for ch in prefix:
            if ch not in node:
                return None
            node = node[ch]
        return node

    def contains(self, word: str) -> bool:
        node = self._find_node(word.lower())
        return node is not None and self._END in node

    def is_prefix(self, prefix: str) -> bool:
        return self._find_node(prefix.lower()) is not None

    def completions(self, prefix: str, max_suggestions: int = 5) -> List[str]:
        prefix = prefix.lower()
        node = self._find_node(prefix)
        if node is None:
            return []
        results: List[str] = []
        self._dfs(node, prefix, results, max_suggestions)
        return results

    def _dfs(self, node: dict, current: str, results: List[str], limit: int) -> None:
        if len(results) >= limit:
            return
        if self._END in node:
            results.append(current)
        for ch, child in node.items():
            if ch == self._END:
                continue
            if len(results) >= limit:
                return
            self._dfs(child, current + ch, results, limit)

    def score_word(self, word: str) -> float:
        return 1.0 if self.contains(word) else 0.0
```

### Step 1d — Run (expect all pass)

```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_dictionary.py -v
```

### Step 1e — Commit

```bash
git add hpm_ai_v4/dictionary.py hpm_ai_v4/tests/test_dictionary.py
git commit -m "feat: add DictionaryValidator ABC and SimpleWordList with trie"
```

---

## Task 2: Reasoner Integration

### Files
- `hpm_ai_v4/agents/reasoning.py` — MODIFY
- `hpm_ai_v4/tests/test_dictionary.py` — APPEND tests

### Step 2a — Add tests to `test_dictionary.py`

Append to the existing test file:

```python
from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.agents.reasoning import Reasoner


def test_reasoner_accepts_dictionary():
    agent = HPMAgent()
    d = SimpleWordList(word_set={"cat", "the"})
    r = Reasoner(agent, dictionary=d)
    assert r.dictionary is d


def test_reasoner_plan_with_dictionary_returns_list():
    agent = HPMAgent()
    for obs in [0, 2, 0, 2, 0]:
        agent.perceive_and_learn(obs)
    d = SimpleWordList(word_set={"cat"})
    r = Reasoner(agent, dictionary=d)
    result = r.plan(goal_state=2, horizon=5, num_rollouts=5)
    assert isinstance(result, list)


def test_reasoner_plan_without_dictionary_unchanged():
    agent = HPMAgent()
    for obs in [0, 2, 0]:
        agent.perceive_and_learn(obs)
    r = Reasoner(agent)
    assert r.dictionary is None
    result = r.plan(goal_state=2, horizon=3, num_rollouts=3)
    assert isinstance(result, list)
```

### Step 2b — Run (expect failure)

```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_dictionary.py::test_reasoner_accepts_dictionary -v
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'dictionary'`

### Step 2c — Modify `hpm_ai_v4/agents/reasoning.py`

Change `__init__`:

```python
# Before
def __init__(self, agent):
    self.agent = agent

# After
def __init__(self, agent, dictionary=None):
    self.agent = agent
    self.dictionary = dictionary
```

Change `plan()` signature and body — add word-boundary bonus after computing `score`:

```python
def plan(self, goal_state: int, horizon: int = 5, num_rollouts: int = 10,
         use_dictionary: bool = True) -> List[int]:
    """Search for a sequence of observations that reaches the goal state using stochastic rollouts.

    NOTE on dictionary use: observations are character CLASS IDs (0=letter, 1=digit,
    2=space, 3=punct, 4=newline), not actual character IDs. It is therefore impossible
    to reconstruct actual words from the rollout sequence. When use_dictionary=True and
    a dictionary is attached, a small bonus (0.1) is awarded for rollouts that contain
    more than two observations and do not end with a space — a proxy for word-like
    production. Full lexical validation requires a raw-character pipeline (TextAdapter,
    obs_dim=256).
    """
    best_seq = []
    best_score = -np.inf

    curr_obs_base = self.agent.obs_buffer[-20:] if self.agent.obs_buffer else []

    for _ in range(num_rollouts):
        seq = []
        curr_obs = list(curr_obs_base)
        for _ in range(horizon):
            relevant = self.get_relevant_patterns(curr_obs, top_k=1)
            if not relevant:
                break

            dist = relevant[0].predict_next_distribution(curr_obs)
            dist = dist / (dist.sum() + 1e-12)
            action = np.random.choice(len(dist), p=dist)

            seq.append(int(action))
            curr_obs.append(int(action))

        if seq:
            score = -abs(seq[-1] - goal_state)

            # Word-boundary bonus (char-class level only — see docstring)
            if self.dictionary is not None and use_dictionary:
                word_bonus = 0.1 if len(seq) > 2 and seq[-1] != 2 else 0.0
                score += word_bonus

            if score > best_score:
                best_score = score
                best_seq = seq

    return best_seq
```

Also update `explain()` to mention dictionary if present:

```python
def explain(self, pattern: HierarchicalPattern) -> str:
    """Translate pattern structure into human-readable description."""
    if pattern.complexity >= 1:
        likely_obs = np.argmax(pattern.B[np.argmax(pattern.pi)])
        base = (f"Hierarchical pattern (ID:{pattern.id}) predicts observation {likely_obs} "
                f"via {pattern.latent_dim} latent states.")
        if self.dictionary is not None:
            base += " Dictionary active: rewards word-like sequences at boundaries."
        return base
    else:
        return f"Pattern (ID:{pattern.id}) is unknown."
```

### Step 2d — Run (expect all pass)

```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_dictionary.py -v
```

### Step 2e — Commit

```bash
git add hpm_ai_v4/agents/reasoning.py hpm_ai_v4/tests/test_dictionary.py
git commit -m "feat: integrate optional DictionaryValidator into Reasoner"
```

---

## Task 3: InstitutionalField Integration

### Precondition check

`InstitutionalField` in `hpm_ai_v4/field.py` **does** have an `evaluate(self, pattern, obs_seq)` method — confirmed by source inspection. Proceed.

### Files
- `hpm_ai_v4/field.py` — MODIFY
- `hpm_ai_v4/tests/test_dictionary.py` — APPEND tests

### Step 3a — Add tests

Append to test file:

```python
from hpm_ai_v4.field import InstitutionalField
from hpm_ai_v4.pattern import HierarchicalPattern


def test_institutional_field_accepts_dictionary():
    field = InstitutionalField()
    d = SimpleWordList(word_set={"cat"})
    p = HierarchicalPattern(pattern_id=99, obs_dim=5)
    p.complexity = 2
    # Should not raise; return value is a float
    result = field.evaluate(p, [0, 0, 2, 0, 2], dictionary=d)
    assert isinstance(result, float)


def test_institutional_field_no_dictionary_unchanged():
    field = InstitutionalField()
    p = HierarchicalPattern(pattern_id=98, obs_dim=5)
    p.complexity = 2
    result = field.evaluate(p, [0, 0, 2], dictionary=None)
    assert isinstance(result, float)


def test_institutional_field_lexical_bonus_applied():
    field = InstitutionalField()
    d = SimpleWordList(word_set={"cat"})
    p = HierarchicalPattern(pattern_id=97, obs_dim=5)
    p.complexity = 2
    # obs_seq with word boundaries (class 0 followed by class 2)
    obs_with_boundaries = [0, 0, 2, 0, 0, 2]
    result_with = field.evaluate(p, obs_with_boundaries, dictionary=d)
    result_without = field.evaluate(p, obs_with_boundaries, dictionary=None)
    # With dictionary, bonus >= 0
    assert result_with >= result_without
```

### Step 3b — Run (expect failure)

```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_dictionary.py::test_institutional_field_accepts_dictionary -v
```

Expected: `TypeError: evaluate() got an unexpected keyword argument 'dictionary'`

### Step 3c — Modify `hpm_ai_v4/field.py`

Change `InstitutionalField.evaluate()`:

```python
def evaluate(self, pattern, obs_seq, dictionary=None):
    """Perform a 'peer review' validation against current observations.

    Args:
        pattern: HierarchicalPattern to evaluate.
        obs_seq: Sequence of observation class IDs (0=letter, 1=digit, 2=space, etc.)
        dictionary: Optional DictionaryValidator. If provided, adds a small lexical
            bonus proportional to word-boundary count in obs_seq. Note: obs_seq
            contains char class IDs, not actual chars, so only boundary-level
            statistics are used (not actual word lookup).
    """
    base_score = 0.0

    if pattern.complexity >= 2:
        ll = pattern.log_likelihood(obs_seq)

        if ll > -len(obs_seq) * 0.6:
            self.replication_history[pattern.id].append(1)
        else:
            self.replication_history[pattern.id].append(0)

        if len(self.replication_history[pattern.id]) >= 3:
            success_rate = np.mean(self.replication_history[pattern.id][-5:])
            if success_rate > 0.8:
                base_score = 0.5
            elif success_rate < 0.3:
                base_score = -0.3

    if dictionary is not None:
        # Count word-boundary events: class-0 (letter) run followed by class-2 (space)
        word_boundaries = sum(
            1 for i in range(1, len(obs_seq))
            if obs_seq[i] == 2 and obs_seq[i - 1] == 0
        )
        lexical_bonus = 0.05 * min(word_boundaries, 4)  # cap at 0.20
        base_score += lexical_bonus

    return base_score
```

### Step 3d — Run (expect all pass)

```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_dictionary.py -v
```

### Step 3e — Commit

```bash
git add hpm_ai_v4/field.py hpm_ai_v4/tests/test_dictionary.py
git commit -m "feat: add lexical_bonus to InstitutionalField via DictionaryValidator"
```

---

## Full test run

After all three tasks:

```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_dictionary.py -v
```

All tests should pass. Also verify existing tests are unbroken:

```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/ -v
```

---

## Constraints summary

- Dictionary is always optional — `None` by default everywhere
- No changes to EM updates, replicator dynamics, or `meta_pattern_update`
- Char-class limitation documented in docstrings and this plan
- `HPMAgent.__init__` is not touched — callers wire the dictionary when constructing `Reasoner`
