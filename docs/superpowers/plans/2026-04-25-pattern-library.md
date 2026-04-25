# Pattern Library — Implementation Plan

**Date:** 2026-04-25
**Spec:** docs/superpowers/specs/2026-04-25-pattern-library-design.md
**Approach:** TDD — write failing tests first, then implement to pass.

---

## Task 1: PatternSerializer (save/load pickle + JSON)

### Files

- `hpm_ai_v4/tools/__init__.py` — CREATE if missing
- `hpm_ai_v4/tools/serializer.py` — CREATE
- `hpm_ai_v4/tests/test_serializer.py` — CREATE

### Failing tests

```python
# hpm_ai_v4/tests/test_serializer.py
import tempfile, os
import numpy as np
from hpm_ai_v4.pattern import HierarchicalPattern
from hpm_ai_v4.tools.serializer import PatternSerializer


def test_save_load_roundtrip(tmp_path):
    p = HierarchicalPattern(42, latent_dim=2, obs_dim=6)
    path = str(tmp_path / "test.pkl")
    PatternSerializer.save([p], path)
    loaded = PatternSerializer.load(path)
    assert len(loaded) == 1
    assert loaded[0].id == 42
    assert loaded[0].A.shape == (2, 2)
    assert np.allclose(loaded[0].A, p.A, atol=1e-5)


def test_load_preserves_weights(tmp_path):
    p = HierarchicalPattern(1, latent_dim=2, obs_dim=6)
    p.weight = 0.75
    path = str(tmp_path / "w.pkl")
    PatternSerializer.save([p], path)
    loaded = PatternSerializer.load(path)
    assert abs(loaded[0].weight - 0.75) < 1e-5


def test_save_multiple(tmp_path):
    patterns = [HierarchicalPattern(i, latent_dim=2, obs_dim=6) for i in range(5)]
    path = str(tmp_path / "multi.pkl")
    PatternSerializer.save(patterns, path)
    loaded = PatternSerializer.load(path)
    assert len(loaded) == 5
    assert [p.id for p in loaded] == list(range(5))


def test_json_roundtrip(tmp_path):
    p = HierarchicalPattern(7, latent_dim=2, obs_dim=6)
    path = str(tmp_path / "test.json")
    PatternSerializer.save_json([p], path)
    loaded = PatternSerializer.load_json(path)
    assert loaded[0].id == 7
    assert np.allclose(loaded[0].B, p.B, atol=1e-5)
```

### Run (expect ModuleNotFoundError)

```
PYTHONPATH=. pytest hpm_ai_v4/tests/test_serializer.py -v
```

### Implementation

```python
# hpm_ai_v4/tools/serializer.py
import pickle
import json
import numpy as np
from hpm_ai_v4.pattern import HierarchicalPattern

FIELDS = ['id', 'A', 'B', 'pi', 'latent_dim', 'obs_dim',
          'running_loss', 'weight', 'creation_step']
OPTIONAL = ['source_corpus', 'density_at_save']


class PatternSerializer:

    @staticmethod
    def _to_dict(p):
        d = {f: getattr(p, f) for f in FIELDS}
        for f in OPTIONAL:
            if hasattr(p, f):
                d[f] = getattr(p, f)
        return d

    @staticmethod
    def _from_dict(d):
        p = HierarchicalPattern(d['id'], latent_dim=d['latent_dim'], obs_dim=d['obs_dim'])
        p.A = np.array(d['A'], dtype=np.float32)
        p.B = np.array(d['B'], dtype=np.float32)
        p.pi = np.array(d['pi'], dtype=np.float32)
        p.running_loss = d['running_loss']
        p.weight = d['weight']
        p.creation_step = d.get('creation_step', 0)
        for f in OPTIONAL:
            if f in d:
                setattr(p, f, d[f])
        p._refresh_log_cache()
        return p

    @staticmethod
    def save(patterns, path):
        with open(path, 'wb') as f:
            pickle.dump([PatternSerializer._to_dict(p) for p in patterns], f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return [PatternSerializer._from_dict(d) for d in pickle.load(f)]

    @staticmethod
    def save_json(patterns, path):
        def convert(d):
            return {k: (v.tolist() if hasattr(v, 'tolist') else v)
                    for k, v in d.items()}
        with open(path, 'w') as f:
            json.dump([convert(PatternSerializer._to_dict(p)) for p in patterns], f)

    @staticmethod
    def load_json(path):
        with open(path) as f:
            return [PatternSerializer._from_dict(d) for d in json.load(f)]
```

Also create `hpm_ai_v4/tools/__init__.py` if missing (empty file is fine).

### Run (expect all pass)

```
PYTHONPATH=. pytest hpm_ai_v4/tests/test_serializer.py -v
```

### Commit

```
feat: add PatternSerializer for pickle and JSON pattern library I/O
```

---

## Task 2: HPMAgent.load_library()

### Files

- `hpm_ai_v4/agents/agent.py` — MODIFY
- `hpm_ai_v4/tests/test_serializer.py` — ADD tests

### Failing tests (add to test_serializer.py)

```python
from hpm_ai_v4.agents.agent import HPMAgent


def test_load_library_replaces_patterns(tmp_path):
    patterns = [HierarchicalPattern(i, latent_dim=2, obs_dim=6) for i in range(3)]
    path = str(tmp_path / "lib.pkl")
    PatternSerializer.save(patterns, path)
    agent = HPMAgent()
    n = agent.load_library(path)
    assert n == 3
    assert len(agent.patterns) == 3


def test_load_library_resets_weights(tmp_path):
    patterns = [HierarchicalPattern(i, latent_dim=2, obs_dim=6) for i in range(4)]
    for p in patterns:
        p.weight = 0.99
    path = str(tmp_path / "lib2.pkl")
    PatternSerializer.save(patterns, path)
    agent = HPMAgent()
    agent.load_library(path, reset_weights=True)
    for p in agent.patterns:
        assert abs(p.weight - 0.25) < 1e-5
```

### Run (expect AttributeError: 'HPMAgent' has no attribute 'load_library')

```
PYTHONPATH=. pytest hpm_ai_v4/tests/test_serializer.py -v
```

### Implementation

Add to `HPMAgent` class in `hpm_ai_v4/agents/agent.py`:

```python
def load_library(self, path: str, reset_weights: bool = True) -> int:
    """Load patterns from a serialised library file.

    Replaces self.patterns with the loaded population.
    If reset_weights=True, normalises all weights to 1/N so no single
    prior pattern dominates at the start of the new learning session.
    Returns N (number of patterns loaded).
    """
    from hpm_ai_v4.tools.serializer import PatternSerializer
    self.patterns = PatternSerializer.load(path)
    if reset_weights and self.patterns:
        w = 1.0 / len(self.patterns)
        for p in self.patterns:
            p.weight = w
    return len(self.patterns)
```

### Run (expect all pass)

```
PYTHONPATH=. pytest hpm_ai_v4/tests/test_serializer.py -v
```

### Commit

```
feat: add HPMAgent.load_library() for initialising from pattern library
```

---

## Task 3: build_library.py script

### File

- `hpm_ai_v4/simulations/build_library.py` — CREATE

### No unit test — integration script. Verify manually.

### Implementation

```python
#!/usr/bin/env python3
"""
Offline pattern library creation script.

Usage:
    PYTHONPATH=. python3 hpm_ai_v4/simulations/build_library.py \
        --corpus wiki_sample.txt \
        --output wiki_patterns.pkl \
        --steps 100000 \
        --min-density 0.3
"""
import argparse
import sys
import numpy as np

from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.io.adapters import CharClassAdapter
from hpm_ai_v4.evaluators.metrics import pattern_density, affective_score, social_score
from hpm_ai_v4.tools.serializer import PatternSerializer


SYNTHETIC_CORPUS = (
    "the cat sat on the mat. the cat ate the rat. "
    "a bat sat on a flat mat. the fat cat and the rat. "
) * 200


def load_corpus(path: str) -> str:
    if path == '/dev/stdin':
        return sys.stdin.read()
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except FileNotFoundError:
        print(f"[warn] corpus file not found: {path!r} — using synthetic fallback")
        return SYNTHETIC_CORPUS


def compute_density(p, obs_buffer, field_freq):
    aff = affective_score(p, obs_buffer)
    soc = social_score(p, field_freq)
    return pattern_density(p, obs_buffer, [aff, soc])


def main():
    parser = argparse.ArgumentParser(description="Build HPM pattern library from corpus")
    parser.add_argument('--corpus', required=True, help='Path to text corpus')
    parser.add_argument('--output', required=True, help='Output .pkl path')
    parser.add_argument('--steps', type=int, default=100_000)
    parser.add_argument('--min-density', type=float, default=0.3)
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    adapter = CharClassAdapter()

    # Build flat token stream from corpus, cycling if needed
    tokens = [adapter.encode_char(ch) for ch in corpus]
    if not tokens:
        print("[error] Empty corpus after encoding")
        sys.exit(1)

    agent = HPMAgent(num_initial_patterns=20, obs_dim=5)
    print(f"[init] {len(agent.patterns)} patterns, running {args.steps} steps")

    for step in range(args.steps):
        obs = tokens[step % len(tokens)]
        agent.perceive_and_learn(obs)

        if (step + 1) % 10_000 == 0:
            weights = [p.weight for p in agent.patterns]
            field_freq = {p.id: p.weight for p in agent.patterns}
            densities = [
                compute_density(p, agent.obs_buffer, field_freq)
                for p in agent.patterns
            ]
            avg_d = float(np.mean(densities)) if densities else 0.0
            best_w = float(max(weights)) if weights else 0.0
            print(f"[step={step+1}] pop_size={len(agent.patterns)} "
                  f"best_weight={best_w:.3f} avg_density={avg_d:.3f}")

    # Filter
    field_freq = {p.id: p.weight for p in agent.patterns}
    kept = []
    for p in agent.patterns:
        d = compute_density(p, agent.obs_buffer, field_freq)
        if d > args.min_density and p.weight > 0.01:
            p.source_corpus = args.corpus
            p.density_at_save = d
            kept.append(p)

    if not kept:
        print("[warn] No patterns passed the density filter. Lowering --min-density may help.")
    else:
        PatternSerializer.save(kept, args.output)
        densities = [p.density_at_save for p in kept]
        print(f"\n[done] {len(kept)} patterns saved to {args.output}")
        print(f"       density: min={min(densities):.3f} "
              f"mean={float(np.mean(densities)):.3f} max={max(densities):.3f}")


if __name__ == '__main__':
    main()
```

### Manual verification

```bash
PYTHONPATH=. python3 hpm_ai_v4/simulations/build_library.py \
    --corpus /dev/stdin \
    --steps 1000 \
    --output /tmp/test_lib.pkl \
    <<< "the cat sat on the mat the cat sat"
```

Expected: runs without error, prints step log, saves `/tmp/test_lib.pkl` (may be 0 patterns at low step count — that is acceptable; decrease `--min-density` to 0.0 to confirm saving works).

### Commit

```
feat: add build_library.py offline pattern library creation script
```

---

## Task order summary

| # | Task | Files changed | Test command |
|---|---|---|---|
| 1 | PatternSerializer | `tools/serializer.py`, `tests/test_serializer.py` | `pytest hpm_ai_v4/tests/test_serializer.py -v` |
| 2 | HPMAgent.load_library | `agents/agent.py`, `tests/test_serializer.py` | `pytest hpm_ai_v4/tests/test_serializer.py -v` |
| 3 | build_library.py | `simulations/build_library.py` | manual run |
