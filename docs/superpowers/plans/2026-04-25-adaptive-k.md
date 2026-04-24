# Plan: Adaptive-K Implementation (CharClassAdapter + grow_latent)

**Date:** 2026-04-25  
**Spec:** docs/superpowers/specs/2026-04-25-adaptive-k-design.md  
**Approach:** TDD — failing test first, then minimal implementation, then commit.

---

## Task 1: CharClassAdapter

### Files
- Modify: `hpm_ai_v4/io/adapters.py`
- Create: `hpm_ai_v4/tests/test_adapters.py`

### Step 1a: Write failing tests

Create `hpm_ai_v4/tests/test_adapters.py`:

```python
import pytest
from hpm_ai_v4.io.adapters import CharClassAdapter

@pytest.fixture
def adapter():
    return CharClassAdapter()

def test_letter_lowercase(adapter):
    # 'a' = ord('a') - 32 = 65
    assert adapter.encode(65) == 0

def test_letter_uppercase(adapter):
    # 'A' = ord('A') - 32 = 33
    assert adapter.encode(33) == 0

def test_digit(adapter):
    # '5' = ord('5') - 32 = 21
    assert adapter.encode(21) == 1

def test_space(adapter):
    # ' ' = ord(' ') - 32 = 0
    assert adapter.encode(0) == 2

def test_newline(adapter):
    # '\n' = ord('\n') - 32 = -22
    assert adapter.encode(-22) == 4

def test_punctuation(adapter):
    # '!' = ord('!') - 32 = 1
    assert adapter.encode(1) == 3

def test_decode_class_letter(adapter):
    assert adapter.decode_class(0) == 'letter'

def test_decode_class_digit(adapter):
    assert adapter.decode_class(1) == 'digit'

def test_decode_class_space(adapter):
    assert adapter.decode_class(2) == 'space'

def test_decode_class_punctuation(adapter):
    assert adapter.decode_class(3) == 'punctuation'

def test_decode_class_newline(adapter):
    assert adapter.decode_class(4) == 'newline'

def test_round_trip_letter(adapter):
    char_id = 65  # 'a'
    assert adapter.decode_class(adapter.encode(char_id)) == 'letter'

def test_round_trip_digit(adapter):
    char_id = 16  # '0'
    assert adapter.decode_class(adapter.encode(char_id)) == 'digit'

def test_obs_dim(adapter):
    assert adapter.obs_dim == 5

def test_decode_invalid_raises(adapter):
    with pytest.raises(ValueError):
        adapter.decode_class(5)
```

### Run (expect failure)

```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_adapters.py -v
```

Expected failure:
```
ImportError: cannot import name 'CharClassAdapter' from 'hpm_ai_v4.io.adapters'
```

### Step 1b: Implement CharClassAdapter

Add to the end of `hpm_ai_v4/io/adapters.py`, after `DiscreteInputAdapter`:

```python
class CharClassAdapter:
    """
    Maps 95 printable ASCII character IDs (ord(ch)-32 for ch in range(32,127))
    to 5 coarse character classes, reducing obs_dim from 95 to 5.

    Classes:
        0 = letter      (A-Z: IDs 33-58, a-z: IDs 65-90)
        1 = digit       (0-9: IDs 16-25)
        2 = space       (ID 0)
        3 = punctuation (all other printable ASCII)
        4 = newline     (ord('\\n')-32 = -22, passed as sentinel)
    """

    CLASS_NAMES = ['letter', 'digit', 'space', 'punctuation', 'newline']
    NEWLINE_ID = -22  # ord('\n') - 32

    def __init__(self):
        # Build lookup table for IDs 0-94
        self._table = {}
        for char_id in range(95):
            ch = chr(char_id + 32)
            if ch == ' ':
                self._table[char_id] = 2
            elif ch.isdigit():
                self._table[char_id] = 1
            elif ch.isalpha():
                self._table[char_id] = 0
            else:
                self._table[char_id] = 3

    @property
    def obs_dim(self) -> int:
        return 5

    def encode(self, char_id: int) -> int:
        """Map a character ID (ord(ch)-32) to a class ID 0-4."""
        if char_id == self.NEWLINE_ID:
            return 4
        return self._table.get(char_id, 3)  # default to punctuation

    def decode_class(self, class_id: int) -> str:
        """Return the name of the character class."""
        if class_id < 0 or class_id >= len(self.CLASS_NAMES):
            raise ValueError(f"class_id {class_id} out of range [0, 4]")
        return self.CLASS_NAMES[class_id]
```

### Run (expect pass)

```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_adapters.py -v
```

### Commit

```bash
git add hpm_ai_v4/io/adapters.py hpm_ai_v4/tests/test_adapters.py
git commit -m "feat: add CharClassAdapter to reduce obs_dim from 95 to 5"
```

---

## Task 2: grow_latent() method on HierarchicalPattern

### Files
- Modify: `hpm_ai_v4/pattern.py`
- Create: `hpm_ai_v4/tests/test_pattern_growth.py`

### Step 2a: Write failing tests

Create `hpm_ai_v4/tests/test_pattern_growth.py`:

```python
import pytest
import numpy as np
from hpm_ai_v4.pattern import HierarchicalPattern

@pytest.fixture
def pattern():
    np.random.seed(42)
    p = HierarchicalPattern(pattern_id=0, latent_dim=2, obs_dim=5)
    return p

def test_latent_dim_incremented(pattern):
    old_K = pattern.latent_dim
    pattern.grow_latent()
    assert pattern.latent_dim == old_K + 1

def test_A3_shape(pattern):
    old_K = pattern.latent_dim
    pattern.grow_latent()
    assert pattern.A3.shape == (old_K + 1, old_K + 1)

def test_A3_rows_sum_to_one(pattern):
    pattern.grow_latent()
    row_sums = pattern.A3.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(pattern.latent_dim), atol=1e-6)

def test_A32_shape(pattern):
    old_K = pattern.latent_dim
    pattern.grow_latent()
    assert pattern.A32.shape == (old_K + 1, old_K + 1)

def test_A32_rows_sum_to_one(pattern):
    pattern.grow_latent()
    row_sums = pattern.A32.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(pattern.latent_dim), atol=1e-6)

def test_A21_shape(pattern):
    old_K = pattern.latent_dim
    pattern.grow_latent()
    assert pattern.A21.shape == (old_K + 1, old_K + 1)

def test_B_shape(pattern):
    old_K = pattern.latent_dim
    obs_dim = pattern.obs_dim
    pattern.grow_latent()
    assert pattern.B.shape == (old_K + 1, obs_dim)

def test_B_rows_sum_to_one(pattern):
    pattern.grow_latent()
    row_sums = pattern.B.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(pattern.latent_dim), atol=1e-6)

def test_pi3_shape(pattern):
    old_K = pattern.latent_dim
    pattern.grow_latent()
    assert pattern.pi3.shape == (old_K + 1,)

def test_pi3_sums_to_one(pattern):
    pattern.grow_latent()
    assert abs(pattern.pi3.sum() - 1.0) < 1e-6

def test_old_A3_values_preserved(pattern):
    old_A3 = pattern.A3.copy()
    pattern.grow_latent(noise_scale=0.0)
    # Old block should be preserved (up to re-normalisation)
    K = old_A3.shape[0]
    # Check ratios are consistent (rows of old block still normalise same way)
    for i in range(K):
        old_row = old_A3[i]
        # After expansion with noise_scale=0, new col K is 0 so old rows
        # re-normalise to same relative values
        new_row = pattern.A3[i, :K]
        old_norm = old_row / (old_row.sum() + 1e-12)
        new_norm = new_row / (new_row.sum() + 1e-12)
        np.testing.assert_allclose(old_norm, new_norm, atol=1e-5)

def test_log_likelihood_finite_after_growth(pattern):
    obs_seq = [0, 1, 2, 3, 4, 0, 1]
    pattern.grow_latent()
    ll = pattern.log_likelihood(obs_seq)
    assert np.isfinite(ll)

def test_ss_shapes_updated(pattern):
    old_K = pattern.latent_dim
    pattern.grow_latent()
    K1 = old_K + 1
    assert pattern.SS_A3.shape == (K1, K1)
    assert pattern.SS_B.shape == (K1, pattern.obs_dim)

def test_grow_twice(pattern):
    pattern.grow_latent()
    pattern.grow_latent()
    assert pattern.latent_dim == 4
    assert pattern.A3.shape == (4, 4)
```

### Run (expect failure)

```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_pattern_growth.py -v
```

Expected failure:
```
AttributeError: 'HierarchicalPattern' object has no attribute 'grow_latent'
```

### Step 2b: Implement grow_latent()

Add the following method to the `HierarchicalPattern` class in `hpm_ai_v4/pattern.py`, after `update_running_loss`:

```python
def grow_latent(self, noise_scale: float = 0.01) -> None:
    """
    Expand latent dimension from K to K+1 in-place.

    Existing parameter rows are preserved; the new row/column is
    initialised near uniform with small noise. All rows are
    re-normalised after expansion.
    """
    K = self.latent_dim
    K1 = K + 1
    rng = np.random.default_rng()

    def expand_transition(A):
        new_A = np.zeros((K1, K1))
        new_A[:K, :K] = A
        new_A[K, :] = 1.0 / K1 + rng.normal(0, noise_scale, K1)
        new_A[:K, K] = np.abs(rng.normal(0, noise_scale, K))
        new_A = np.abs(new_A)
        row_sums = new_A.sum(axis=1, keepdims=True)
        return new_A / (row_sums + 1e-12)

    def expand_emission(B):
        new_B = np.zeros((K1, self.obs_dim))
        new_B[:K, :] = B
        new_B[K, :] = B.mean(axis=0) + rng.normal(0, noise_scale, self.obs_dim)
        new_B = np.abs(new_B)
        row_sums = new_B.sum(axis=1, keepdims=True)
        return new_B / (row_sums + 1e-12)

    def expand_ss_transition(SS):
        new_SS = np.ones((K1, K1)) * 0.1
        new_SS[:K, :K] = SS
        return new_SS

    def expand_ss_emission(SS):
        new_SS = np.ones((K1, self.obs_dim)) * 0.1
        new_SS[:K, :] = SS
        return new_SS

    self.A3  = expand_transition(self.A3)
    self.A32 = expand_transition(self.A32)
    self.A21 = expand_transition(self.A21)
    self.B   = expand_emission(self.B)

    new_pi = np.append(self.pi3, 1.0 / K1)
    new_pi = np.abs(new_pi)
    self.pi3 = new_pi / (new_pi.sum() + 1e-12)

    self.SS_A3  = expand_ss_transition(self.SS_A3)
    self.SS_A32 = expand_ss_transition(self.SS_A32)
    self.SS_A21 = expand_ss_transition(self.SS_A21)
    self.SS_B   = expand_ss_emission(self.SS_B)

    self.latent_dim = K1
```

### Run (expect pass)

```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_pattern_growth.py -v
```

### Commit

```bash
git add hpm_ai_v4/pattern.py hpm_ai_v4/tests/test_pattern_growth.py
git commit -m "feat: add grow_latent() method to HierarchicalPattern"
```

---

## Task 3: Growth trigger in HPMAgent

### Files
- Modify: `hpm_ai_v4/agents/agent.py`
- Create: `hpm_ai_v4/tests/test_growth_trigger.py`

### Step 3a: Write failing tests

Create `hpm_ai_v4/tests/test_growth_trigger.py`:

```python
import pytest
from unittest.mock import MagicMock, patch, call
import numpy as np

from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.pattern import HierarchicalPattern


def make_mock_pattern(compression_val, running_loss_val, latent_dim=2):
    p = MagicMock(spec=HierarchicalPattern)
    p.compression.return_value = compression_val
    p.running_loss = running_loss_val
    p.latent_dim = latent_dim
    p.weight = 0.5
    p.id = 0
    p.complexity = 3
    return p


def test_grow_called_when_conditions_met():
    """compression high + loss high + below max_K => grow_latent() called."""
    agent = HPMAgent.__new__(HPMAgent)
    agent.obs_buffer = list(range(20))
    agent.step_counter = 0

    pattern = make_mock_pattern(compression_val=0.5, running_loss_val=2.0, latent_dim=2)
    agent.patterns = [pattern]

    agent._maybe_grow_patterns(max_K=4, loss_threshold=1.0)

    pattern.grow_latent.assert_called_once()


def test_grow_not_called_when_compression_low():
    """compression low => grow_latent() NOT called."""
    agent = HPMAgent.__new__(HPMAgent)
    agent.obs_buffer = list(range(20))
    agent.step_counter = 0

    pattern = make_mock_pattern(compression_val=0.1, running_loss_val=2.0, latent_dim=2)
    agent.patterns = [pattern]

    agent._maybe_grow_patterns(max_K=4, loss_threshold=1.0)

    pattern.grow_latent.assert_not_called()


def test_grow_not_called_when_loss_low():
    """loss low => grow_latent() NOT called."""
    agent = HPMAgent.__new__(HPMAgent)
    agent.obs_buffer = list(range(20))
    agent.step_counter = 0

    pattern = make_mock_pattern(compression_val=0.5, running_loss_val=0.5, latent_dim=2)
    agent.patterns = [pattern]

    agent._maybe_grow_patterns(max_K=4, loss_threshold=1.0)

    pattern.grow_latent.assert_not_called()


def test_grow_not_called_at_max_k():
    """pattern at max_K => grow_latent() NOT called."""
    agent = HPMAgent.__new__(HPMAgent)
    agent.obs_buffer = list(range(20))
    agent.step_counter = 0

    pattern = make_mock_pattern(compression_val=0.5, running_loss_val=2.0, latent_dim=8)
    agent.patterns = [pattern]

    agent._maybe_grow_patterns(max_K=4, loss_threshold=1.0)

    pattern.grow_latent.assert_not_called()


def test_growth_checked_every_500_steps(monkeypatch):
    """_maybe_grow_patterns is only called at step multiples of 500."""
    called_at = []

    original_maybe_grow = HPMAgent._maybe_grow_patterns

    def tracking_maybe_grow(self, **kwargs):
        called_at.append(self.step_counter)

    monkeypatch.setattr(HPMAgent, '_maybe_grow_patterns', tracking_maybe_grow)

    # Build a minimal agent that won't error on perceive_and_learn internals
    agent = HPMAgent(num_initial_patterns=1, obs_dim=2)

    # Run 1001 steps, track which steps trigger growth check
    # We patch the expensive parts
    with patch.object(agent._pool, 'map_patterns', return_value=[
        {'pattern_id': p.id, 'A3': p.A3, 'A32': p.A32, 'A21': p.A21,
         'B': p.B, 'pi3': p.pi3, 'SS_A3': p.SS_A3, 'SS_A32': p.SS_A32,
         'SS_A21': p.SS_A21, 'SS_B': p.SS_B, 'running_loss': 0.0,
         'total_score': 0.0}
        for p in agent.patterns
    ]):
        for i in range(1001):
            agent.perceive_and_learn(i % 2)

    # Growth should have been checked at steps 500 and 1000
    assert 500 in called_at
    assert 1000 in called_at
    # Should NOT be checked at step 1
    assert 1 not in called_at
```

### Run (expect failure)

```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_growth_trigger.py -v
```

Expected failure:
```
AttributeError: 'HPMAgent' object has no attribute '_maybe_grow_patterns'
```

### Step 3b: Implement _maybe_grow_patterns and trigger

Add the private method to `HPMAgent` in `hpm_ai_v4/agents/agent.py`, after the `gossip_with_substrate` method:

```python
def _maybe_grow_patterns(self, max_K: int = 4, loss_threshold: float = 1.0) -> None:
    """
    Check each pattern and grow its latent dimension if:
      - compression() > 0.3  (pattern has learned structure)
      - running_loss > loss_threshold  (still has significant error)
      - latent_dim < max_K  (not already at cap)
    """
    for p in self.patterns:
        if p.complexity < 2:
            continue  # FlatPattern: not eligible
        if p.latent_dim >= max_K:
            continue
        if p.running_loss <= loss_threshold:
            continue
        compression = p.compression(self.obs_buffer)
        if compression > 0.3:
            p.grow_latent()
            print(f"[HPMAgent] Pattern {p.id} grew to K={p.latent_dim} "
                  f"(compression={compression:.3f}, loss={p.running_loss:.3f})")
```

Then add the trigger call inside `perceive_and_learn`, at the end of the method just before `self.step_counter += 1`:

```python
        # 9. Adaptive K growth (checked every 500 steps)
        if self.step_counter > 0 and self.step_counter % 500 == 0:
            self._maybe_grow_patterns()
```

### Run (expect pass)

```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/test_growth_trigger.py -v
```

### Run full test suite to verify no regressions

```bash
PYTHONPATH=. pytest hpm_ai_v4/tests/ -v
```

### Commit

```bash
git add hpm_ai_v4/agents/agent.py hpm_ai_v4/tests/test_growth_trigger.py
git commit -m "feat: add adaptive K growth trigger in HPMAgent"
```

---

## Summary of Changes

| File | Change |
|------|--------|
| `hpm_ai_v4/io/adapters.py` | Add `CharClassAdapter` class |
| `hpm_ai_v4/pattern.py` | Add `grow_latent()` method to `HierarchicalPattern` |
| `hpm_ai_v4/agents/agent.py` | Add `_maybe_grow_patterns()` method; call every 500 steps in `perceive_and_learn` |
| `hpm_ai_v4/tests/test_adapters.py` | New: 15 tests for `CharClassAdapter` |
| `hpm_ai_v4/tests/test_pattern_growth.py` | New: 13 tests for `grow_latent()` |
| `hpm_ai_v4/tests/test_growth_trigger.py` | New: 5 tests for growth trigger logic |
