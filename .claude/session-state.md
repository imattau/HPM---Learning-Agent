# Session State — 2026-03-30

execution_mode: unattended
auto_continue: true

## Current Task Objective

Implement graduated prior protection in `hfn/observer.py`, replacing the binary `protected_ids` mechanism with a density-based plasticity system grounded in the HPM framework.

## Background

The HPM paper (Section 2.6 pattern density, Section 2.5.2 dynamics) confirms that priors should be refinable, not permanently frozen. High-density patterns resist revision; low-density patterns that keep missing should drift toward new observations.

## Changes to make in `hfn/observer.py`

### 1. Add new parameters to `__init__` signature (after `vocab` param, before closing `)`):

```python
prior_plasticity: bool = False,        # enable density-based prior revision
prior_drift_rate: float = 0.01,        # mu drift step when revision triggered
prior_revision_threshold: int = 200,   # misses before eligible for drift
```

### 2. Add new instance variables after `self.protected_ids` line (~line 165):

```python
# Prior plasticity (graduated protection based on density)
self.prior_plasticity: bool = prior_plasticity
self.prior_drift_rate: float = prior_drift_rate
self.prior_revision_threshold: int = prior_revision_threshold
self._prior_miss_counts: dict[str, int] = defaultdict(int)
self._prior_hit_counts: dict[str, int] = defaultdict(int)
```

### 3. Modify `_update_weights` — replace the early-continue for protected nodes:

FIND this block (around line 281):
```python
            if nid in self.protected_ids:
                continue
```

REPLACE with:
```python
            if nid in self.protected_ids:
                if self.prior_plasticity:
                    if nid in explaining:
                        self._prior_hit_counts[nid] += 1
                        self._prior_miss_counts[nid] = 0
                    else:
                        self._prior_miss_counts[nid] += 1
                continue
```

### 4. Add new method `_check_prior_plasticity` after `_update_weights`, before `_update_scores`:

```python
    def _check_prior_plasticity(self, x: np.ndarray) -> None:
        """Drift low-density priors toward observations they keep missing (HPM Section 2.6).

        A prior is eligible for revision when:
        - miss_count > prior_revision_threshold (consistently missing)
        - hit_rate < 0.5 (less than half encounters explain the observation)

        Drift: mu += prior_drift_rate * (x - mu)
        Counts reset after drift so prior gets fresh chance to stabilise.
        """
        if not self.prior_plasticity:
            return
        for nid in list(self.protected_ids):
            miss = self._prior_miss_counts[nid]
            if miss < self.prior_revision_threshold:
                continue
            hit = self._prior_hit_counts[nid]
            total = hit + miss
            if total == 0:
                continue
            hit_rate = hit / total
            if hit_rate >= 0.5:
                continue
            node = self.forest.get(nid)
            if node is None:
                continue
            node.mu = node.mu + self.prior_drift_rate * (x - node.mu)
            self._prior_miss_counts[nid] = 0
            self._prior_hit_counts[nid] = 0
```

### 5. In `observe()` method, call `_check_prior_plasticity(x)` after `_update_weights` call.

## After implementing

1. Run tests: `PYTHONPATH=. python -m pytest tests/ -x -q 2>&1 | head -40`
2. Smoke test: `PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_math.py 2>&1 | tail -20`
3. Commit: `git add hfn/observer.py && git commit -m "feat: graduated prior protection — density-based plasticity for HFN priors"`

## Notes
- `prior_plasticity=False` by default — no behaviour change for existing code
- Do NOT remove nids from `protected_ids` — priors stay protected from absorption; only mu drifts
- `defaultdict` is already imported at top of observer.py
