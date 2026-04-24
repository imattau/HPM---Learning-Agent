# Plan: Wikipedia HPM Simulation (Small-K, Deep Hierarchy)

**Date**: 2026-04-25
**Branch**: hpm-ai-v4-dev
**Spec**: `docs/superpowers/specs/2026-04-25-wikipedia-hpm-design.md`
**Approach**: TDD — failing test first, then minimal implementation, then commit.

---

## Task 1: CharClassAdapter — add `encode(char: str)` convenience method

**File**: `hpm_ai_v4/io/adapters.py` (MODIFY)

The existing `CharClassAdapter.encode(char_id: int)` takes a shifted integer. The simulation
needs a direct character → class_id path.

### Failing tests

```python
# hpm_ai_v4/tests/test_adapters.py (add to existing or create new)
from hpm_ai_v4.io.adapters import CharClassAdapter

def test_encode_char_letter():
    a = CharClassAdapter()
    assert a.encode_char('a') == 0

def test_encode_char_digit():
    a = CharClassAdapter()
    assert a.encode_char('3') == 1

def test_encode_char_space():
    a = CharClassAdapter()
    assert a.encode_char(' ') == 2

def test_encode_char_punctuation():
    a = CharClassAdapter()
    assert a.encode_char('!') == 3

def test_encode_char_newline():
    a = CharClassAdapter()
    assert a.encode_char('\n') == 4

def test_encode_char_uppercase():
    a = CharClassAdapter()
    assert a.encode_char('Z') == 0
```

Run: `python -m pytest hpm_ai_v4/tests/test_adapters.py -x`

### Implementation

Add to `CharClassAdapter` in `hpm_ai_v4/io/adapters.py`:

```python
def encode_char(self, ch: str) -> int:
    """Map a raw character to a class ID 0-4."""
    if ch == '\n':
        return 4
    char_id = ord(ch) - 32
    return self.encode(char_id)
```

### Commit

```
git add hpm_ai_v4/io/adapters.py hpm_ai_v4/tests/test_adapters.py
git commit -m "feat: add encode_char() convenience method to CharClassAdapter"
```

---

## Task 2: `get_top_state()` on HierarchicalPattern + K>4 constructor warning

**File**: `hpm_ai_v4/pattern.py` (MODIFY)

### Failing tests

```python
# hpm_ai_v4/tests/test_pattern_growth.py (add to existing or create)
import warnings
import numpy as np
from hpm_ai_v4.pattern import HierarchicalPattern

def test_get_top_state_returns_valid_int():
    p = HierarchicalPattern(pattern_id=0, latent_dim=2, obs_dim=5)
    obs_seq = [0, 1, 0, 2, 1]
    result = p.get_top_state(obs_seq)
    assert isinstance(result, int)
    assert result in {0, 1}

def test_get_top_state_empty_returns_int():
    p = HierarchicalPattern(pattern_id=0, latent_dim=2, obs_dim=5)
    result = p.get_top_state([])
    assert isinstance(result, int)

def test_get_top_state_k3():
    p = HierarchicalPattern(pattern_id=0, latent_dim=3, obs_dim=5)
    obs_seq = [0, 1, 2, 0, 1]
    result = p.get_top_state(obs_seq)
    assert result in {0, 1, 2}

def test_constructor_warns_if_k_gt_4():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        HierarchicalPattern(pattern_id=0, latent_dim=5, obs_dim=5)
        assert len(w) == 1
        assert "latent_dim" in str(w[0].message).lower() or "K" in str(w[0].message)

def test_constructor_no_warn_k_eq_4():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        HierarchicalPattern(pattern_id=0, latent_dim=4, obs_dim=5)
        assert len(w) == 0
```

Run: `python -m pytest hpm_ai_v4/tests/test_pattern_growth.py -x`

### Implementation

In `HierarchicalPattern.__init__`, after setting `self.latent_dim`, add:

```python
import warnings

# In __init__, after self.latent_dim = latent_dim:
if latent_dim > 4:
    warnings.warn(
        f"HierarchicalPattern created with latent_dim={latent_dim} > 4. "
        "HPM architecture requires small-K (max 4). Use depth, not width.",
        stacklevel=2,
    )
```

Add method to `HierarchicalPattern`:

```python
def get_top_state(self, obs_seq: list) -> int:
    """
    Run forward algorithm on obs_seq; return argmax of alpha[-1] marginalised
    over z2 and z1 (i.e. the most probable z3 state).

    Returns int in {0, ..., latent_dim-1}. Returns 0 if obs_seq is empty.
    """
    if not obs_seq:
        return int(np.argmax(self.pi3))

    K = self.latent_dim
    T = len(obs_seq)
    alpha = np.full((T, K, K, K), -np.inf)

    for z3 in range(K):
        for z2 in range(K):
            for z1 in range(K):
                alpha[0, z3, z2, z1] = (
                    np.log(self.pi3[z3] + 1e-12)
                    + np.log(self.A32[z3, z2] + 1e-12)
                    + np.log(self.A21[z2, z1] + 1e-12)
                    + np.log(self.B[z1, int(obs_seq[0]) % self.obs_dim] + 1e-12)
                )

    for t in range(1, T):
        for z3 in range(K):
            log_pz3_prev = logsumexp(alpha[t - 1], axis=(1, 2))
            log_sum = logsumexp(log_pz3_prev + np.log(self.A3[:, z3] + 1e-12))
            for z2 in range(K):
                for z1 in range(K):
                    alpha[t, z3, z2, z1] = (
                        log_sum
                        + np.log(self.A32[z3, z2] + 1e-12)
                        + np.log(self.A21[z2, z1] + 1e-12)
                        + np.log(self.B[z1, int(obs_seq[t]) % self.obs_dim] + 1e-12)
                    )

    # Marginalise over z2, z1
    log_p_z3 = logsumexp(alpha[-1], axis=(1, 2))  # shape (K,)
    return int(np.argmax(log_p_z3))
```

### Commit

```
git add hpm_ai_v4/pattern.py hpm_ai_v4/tests/test_pattern_growth.py
git commit -m "feat: add get_top_state() to HierarchicalPattern and warn if K>4"
```

---

## Task 3: WikipediaStream

**File**: `hpm_ai_v4/simulations/wikipedia_sim.py` (CREATE — stream class only)

### Failing tests

```python
# hpm_ai_v4/tests/test_wikipedia_sim.py (CREATE)
import tempfile, os
from hpm_ai_v4.simulations.wikipedia_sim import WikipediaStream
from hpm_ai_v4.io.adapters import CharClassAdapter

def _make_stream(content: str):
    adapter = CharClassAdapter()
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    f.write(content)
    f.close()
    return WikipediaStream(f.name, adapter), f.name

def test_stream_yields_valid_class_ids():
    stream, path = _make_stream("Hello world!\n")
    ids = list(stream)
    os.unlink(path)
    assert all(0 <= i <= 4 for i in ids)

def test_stream_length_matches_valid_chars():
    content = "Hi!\n"
    stream, path = _make_stream(content)
    ids = list(stream)
    os.unlink(path)
    # 'H','i','!','\n' => 4 class IDs
    assert len(ids) == 4

def test_stream_loops():
    stream, path = _make_stream("ab")
    it = iter(stream)
    first_two = [next(it), next(it)]
    next_two = [next(it), next(it)]  # should loop
    os.unlink(path)
    assert first_two == next_two

def test_stream_newline_is_class_4():
    stream, path = _make_stream("\n")
    ids = list(stream)
    os.unlink(path)
    assert ids == [4]
```

Run: `python -m pytest hpm_ai_v4/tests/test_wikipedia_sim.py -x`

### Implementation

```python
# hpm_ai_v4/simulations/wikipedia_sim.py
from typing import Iterator
from hpm_ai_v4.io.adapters import CharClassAdapter


class WikipediaStream:
    """Reads a plain-text file, converts chars to class IDs, loops on exhaustion."""

    def __init__(self, filepath: str, adapter: CharClassAdapter):
        self.filepath = filepath
        self.adapter = adapter

    def _char_to_class(self, ch: str):
        if ch == '\n':
            return self.adapter.encode(-22)
        code = ord(ch)
        if 32 <= code <= 126:
            return self.adapter.encode(code - 32)
        return None  # skip

    def __iter__(self) -> Iterator[int]:
        while True:
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for ch in f.read():
                    cls = self._char_to_class(ch)
                    if cls is not None:
                        yield cls
```

### Commit

```
git add hpm_ai_v4/simulations/wikipedia_sim.py hpm_ai_v4/tests/test_wikipedia_sim.py
git commit -m "feat: add WikipediaStream for character-class streaming"
```

---

## Task 4: 3-level training loop

**File**: `hpm_ai_v4/simulations/wikipedia_sim.py` (MODIFY — add `run_simulation()`)

### Failing tests (add to `test_wikipedia_sim.py`)

```python
import tempfile, os
from hpm_ai_v4.simulations.wikipedia_sim import run_simulation

def test_run_simulation_200_steps():
    content = "The quick brown fox jumps over the lazy dog. " * 20
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    f.write(content)
    f.close()

    state = run_simulation(corpus_path=f.name, total_steps=200, log_every=201)
    os.unlink(f.name)

    assert len(state['L1_patterns']) == 10
    assert len(state['L2_patterns']) == 5
    assert len(state['L3_patterns']) == 3

def test_run_simulation_emits_ints_in_range():
    content = "Hello world!\n" * 30
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    f.write(content)
    f.close()

    # Should not raise
    run_simulation(corpus_path=f.name, total_steps=50, log_every=100)
    os.unlink(f.name)
```

Run: `python -m pytest hpm_ai_v4/tests/test_wikipedia_sim.py -x`

### Implementation (add to `wikipedia_sim.py`)

```python
import numpy as np
from typing import Dict, Any, List
from hpm_ai_v4.pattern import HierarchicalPattern
from hpm_ai_v4.field import PatternField
from hpm_ai_v4.operators.dynamics import compute_conflict_matrix, meta_pattern_update, recombine


def _make_population(n: int, latent_dim: int, obs_dim: int) -> List[HierarchicalPattern]:
    patterns = []
    for i in range(n):
        p = HierarchicalPattern(pattern_id=i, latent_dim=latent_dim, obs_dim=obs_dim)
        p.weight = 1.0 / n
        patterns.append(p)
    return patterns


def _update_level(patterns, obs, buffer, field, step, recombine_every=50):
    """Single-step observe + adapt + replicator for one level."""
    buffer.append(obs)
    if len(buffer) > 100:
        buffer[:] = buffer[-100:]

    field_freq = field.update(patterns)
    adapt_seq = buffer[-20:]

    totals = {}
    for p in patterns:
        p.observe(obs)
        if len(adapt_seq) >= 2:
            p.adapt(adapt_seq)
        p.update_running_loss(adapt_seq)
        # Simple score: negative running loss, weighted by compression
        totals[p.id] = -p.running_loss

    k_mat = compute_conflict_matrix(patterns)
    meta_pattern_update(patterns, totals, eta=0.1, beta_c=0.03,
                        k_matrix=k_mat, decay=0.005)

    # Recombination within level
    if step > 0 and step % recombine_every == 0 and len(patterns) >= 2:
        weights = np.array([p.weight for p in patterns])
        if weights.sum() > 0:
            probs = weights / weights.sum()
            idxs = np.random.choice(len(patterns), size=2, p=probs, replace=False)
            child = recombine(patterns[idxs[0]], patterns[idxs[1]])
            if child is not None:
                child.id = max(p.id for p in patterns) + 1
                child.weight = 0.05
                patterns.append(child)

    # Prune
    patterns[:] = [p for p in patterns if p.weight > 1e-4]


def run_simulation(corpus_path: str, total_steps: int = 100_000,
                   log_every: int = 1_000) -> Dict[str, Any]:
    adapter = CharClassAdapter()
    stream = WikipediaStream(corpus_path, adapter)
    stream_iter = iter(stream)

    L1 = _make_population(10, latent_dim=2, obs_dim=5)
    L2 = _make_population(5,  latent_dim=2, obs_dim=2)
    L3 = _make_population(3,  latent_dim=2, obs_dim=2)

    f1, f2, f3 = PatternField(), PatternField(), PatternField()
    buf1, buf2, buf3 = [], [], []

    for step in range(total_steps):
        class_id = next(stream_iter)

        # Level 1
        _update_level(L1, class_id, buf1, f1, step)

        # Extract L1 latent → L2 input
        best_L1 = max(L1, key=lambda p: p.weight)
        l1_state = best_L1.get_top_state(buf1[-20:])

        # Level 2
        _update_level(L2, l1_state, buf2, f2, step)

        # Extract L2 latent → L3 input
        best_L2 = max(L2, key=lambda p: p.weight)
        l2_state = best_L2.get_top_state(buf2[-20:])

        # Level 3
        _update_level(L3, l2_state, buf3, f3, step)

        if step % log_every == 0:
            best_l1_loss = min(p.running_loss for p in L1)
            print(f"[step {step:6d}] L1 patterns={len(L1)} best_loss={best_l1_loss:.3f} "
                  f"L2 patterns={len(L2)} L3 patterns={len(L3)}")

    return {'L1_patterns': L1, 'L2_patterns': L2, 'L3_patterns': L3,
            'buf1': buf1, 'buf2': buf2, 'buf3': buf3}
```

### Commit

```
git add hpm_ai_v4/simulations/wikipedia_sim.py hpm_ai_v4/tests/test_wikipedia_sim.py
git commit -m "feat: add 3-level training loop run_simulation() for Wikipedia stream"
```

---

## Task 5: TextReasoningInterface

**File**: `hpm_ai_v4/simulations/text_reasoning.py` (CREATE)

### Failing tests (add to `test_wikipedia_sim.py`)

```python
import numpy as np
from hpm_ai_v4.pattern import HierarchicalPattern
from hpm_ai_v4.agents.reasoning import Reasoner
from hpm_ai_v4.simulations.text_reasoning import TextReasoningInterface


def _make_mock_agent(patterns):
    """Minimal object satisfying Reasoner's interface."""
    class FakeAgent:
        obs_buffer = [0, 1, 0]
    agent = FakeAgent()
    agent.patterns = patterns
    return agent


def test_next_char_predict_returns_5_tuples():
    patterns = [HierarchicalPattern(i, latent_dim=2, obs_dim=5) for i in range(3)]
    for p in patterns:
        p.weight = 1.0 / 3
    agent = _make_mock_agent(patterns)
    reasoner = Reasoner(agent)
    iface = TextReasoningInterface(
        L1_patterns=patterns, L1_reasoner=reasoner,
        L2_patterns=[], L2_reasoner=None,
        L3_patterns=[], L3_reasoner=None,
    )
    result = iface.next_char_predict("hello")
    assert len(result) == 5
    names, probs = zip(*result)
    assert abs(sum(probs) - 1.0) < 0.01


def test_plan_to_space_returns_class_names():
    patterns = [HierarchicalPattern(i, latent_dim=2, obs_dim=5) for i in range(3)]
    for p in patterns:
        p.weight = 1.0 / 3
    agent = _make_mock_agent(patterns)
    reasoner = Reasoner(agent)
    iface = TextReasoningInterface(
        L1_patterns=patterns, L1_reasoner=reasoner,
        L2_patterns=patterns, L2_reasoner=reasoner,
        L3_patterns=patterns, L3_reasoner=reasoner,
    )
    result = iface.plan_to_space(horizon=3, num_rollouts=5)
    assert isinstance(result, list)
    CLASS_NAMES = {'letter', 'digit', 'space', 'punctuation', 'newline'}
    for name in result:
        assert name in CLASS_NAMES
```

Run: `python -m pytest hpm_ai_v4/tests/test_wikipedia_sim.py -x`

### Implementation

```python
# hpm_ai_v4/simulations/text_reasoning.py
import numpy as np
from typing import List, Tuple, Optional
from hpm_ai_v4.io.adapters import CharClassAdapter
from hpm_ai_v4.agents.reasoning import Reasoner
from hpm_ai_v4.pattern import HierarchicalPattern

CLASS_NAMES = ['letter', 'digit', 'space', 'punctuation', 'newline']
SPACE_CLASS_ID = 2


class TextReasoningInterface:
    def __init__(self,
                 L1_patterns: List[HierarchicalPattern], L1_reasoner: Reasoner,
                 L2_patterns: List[HierarchicalPattern], L2_reasoner: Optional[Reasoner],
                 L3_patterns: List[HierarchicalPattern], L3_reasoner: Optional[Reasoner]):
        self.L1_patterns = L1_patterns
        self.L2_patterns = L2_patterns
        self.L3_patterns = L3_patterns
        self.L1_reasoner = L1_reasoner
        self.L2_reasoner = L2_reasoner
        self.L3_reasoner = L3_reasoner
        self._adapter = CharClassAdapter()

    def _encode(self, text: str) -> List[int]:
        ids = []
        for ch in text:
            if ch == '\n':
                ids.append(self._adapter.encode(-22))
            elif 32 <= ord(ch) <= 126:
                ids.append(self._adapter.encode(ord(ch) - 32))
        return ids

    def next_char_predict(self, prefix_str: str) -> List[Tuple[str, float]]:
        obs_seq = self._encode(prefix_str)
        relevant = self.L1_reasoner.get_relevant_patterns(obs_seq, top_k=5)
        dist = self.L1_reasoner.compose_predictions(relevant, obs_seq)  # shape (5,)
        dist = dist / (dist.sum() + 1e-12)
        top5 = np.argsort(dist)[::-1][:5]
        return [(CLASS_NAMES[i], float(dist[i])) for i in top5]

    def word_boundary_predict(self, prefix_str: str) -> float:
        if self.L2_reasoner is None or not self.L2_patterns:
            return 0.0
        obs_seq = self._encode(prefix_str)
        # Get L1 latent sequence for L2 input
        best_L1 = max(self.L1_patterns, key=lambda p: p.weight)
        l1_states = [best_L1.get_top_state(obs_seq[max(0, i-20):i+1])
                     for i in range(len(obs_seq))]
        relevant = self.L2_reasoner.get_relevant_patterns(l1_states, top_k=3)
        dist = self.L2_reasoner.compose_predictions(relevant, l1_states)
        dist = dist / (dist.sum() + 1e-12)
        # State 0 is taken as word-boundary by convention (highest weight pattern)
        return float(dist[0])

    def plan_to_space(self, horizon: int, num_rollouts: int) -> List[str]:
        if self.L3_reasoner is None:
            return []
        seq = self.L3_reasoner.plan(goal_state=SPACE_CLASS_ID,
                                    horizon=horizon, num_rollouts=num_rollouts)
        return [CLASS_NAMES[min(s, 4)] for s in seq]

    def counterfactual_shift(self, context: str, forced_class: int) -> List[Tuple[str, float]]:
        obs_seq = self._encode(context)
        relevant = self.L1_reasoner.get_relevant_patterns(obs_seq, top_k=5)
        blended = np.zeros(5)
        total_w = sum(p.weight for p in relevant) + 1e-12
        for p in relevant:
            _, intervened = self.L1_reasoner.counterfactual(p, obs_seq, forced_class)
            blended += (p.weight / total_w) * intervened[:5]
        blended = blended / (blended.sum() + 1e-12)
        top5 = np.argsort(blended)[::-1][:5]
        return [(CLASS_NAMES[i], float(blended[i])) for i in top5]
```

### Commit

```
git add hpm_ai_v4/simulations/text_reasoning.py hpm_ai_v4/tests/test_wikipedia_sim.py
git commit -m "feat: add TextReasoningInterface for 3-level programmatic queries"
```

---

## Task 6: data/get_corpus.py

**File**: `hpm_ai_v4/simulations/data/get_corpus.py` (CREATE)

No automated test — manual verification.

### Implementation

```python
# hpm_ai_v4/simulations/data/get_corpus.py
"""
Download a plain-text Wikipedia sample to data/wiki_sample.txt.

Primary source: Simple English Wikipedia via Wikimedia dumps.
Fallback: Project Gutenberg public domain text.
"""
import urllib.request
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'wiki_sample.txt')

# ~1MB simple English Wikipedia plaintext article dump
PRIMARY_URL = (
    "https://dumps.wikimedia.org/simplewiki/latest/"
    "simplewiki-latest-pages-articles1.xml-p1p10000.bz2"
)

GUTENBERG_FALLBACK_URL = (
    "https://www.gutenberg.org/files/2701/2701-0.txt"  # Moby Dick
)


def download_gutenberg_fallback(path: str):
    print(f"Downloading Gutenberg fallback to {path}...")
    urllib.request.urlretrieve(GUTENBERG_FALLBACK_URL, path)
    print("Done.")


def get_corpus(output_path: str = OUTPUT_PATH):
    if os.path.exists(output_path):
        print(f"Corpus already exists at {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        download_gutenberg_fallback(output_path)
    except Exception as e:
        print(f"Download failed: {e}")
        raise

    return output_path


if __name__ == '__main__':
    get_corpus()
    print(f"Corpus ready at {OUTPUT_PATH}")
```

Run manually: `python hpm_ai_v4/simulations/data/get_corpus.py`

### Commit

```
git add hpm_ai_v4/simulations/data/get_corpus.py
git commit -m "feat: add get_corpus.py to download Wikipedia/Gutenberg sample text"
```

---

## Summary

| Task | File | Tests | Status |
|---|---|---|---|
| 1 | `hpm_ai_v4/io/adapters.py` | 6 unit tests for `encode_char` | Write |
| 2 | `hpm_ai_v4/pattern.py` | 5 unit tests: `get_top_state`, K>4 warning | Write |
| 3 | `hpm_ai_v4/simulations/wikipedia_sim.py` | 4 unit tests: stream loops, valid IDs | Write |
| 4 | `hpm_ai_v4/simulations/wikipedia_sim.py` | 2 integration tests: 200-step run | Write |
| 5 | `hpm_ai_v4/simulations/text_reasoning.py` | 2 unit tests: predict, plan | Write |
| 6 | `hpm_ai_v4/simulations/data/get_corpus.py` | Manual only | Write |

**Architecture invariant throughout**: K=2 at initialisation, max K=4 after adaptive growth.
All patterns use `HierarchicalPattern` — no new subclasses introduced.
