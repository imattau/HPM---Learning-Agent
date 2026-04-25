# Pattern Library System — Design Spec

**Date:** 2026-04-25
**Branch:** hpm-ai-v4-dev
**Status:** Draft

---

## Goal

A two-stage system for reusing learned patterns across HPM agents:

1. **Offline library creation** — `build_library.py` reads a large corpus, runs HPM learning, exports high-density patterns to a `.pkl` file (the pattern library).
2. **Online agent loading** — `HPMAgent.load_library()` loads the library as its initial pattern population and continues learning from new data.

This directly implements HPM's concept of cultural transmission and institutional pattern fields (Section 2.5.4, 5.3): the library is an artefact that encodes regularities discovered from a large prior corpus and makes them available as "innate priors" for new agents (Appendix A.7.4).

---

## HPM Paper Alignment

| Component | HPM concept |
|---|---|
| Offline corpus run | Pattern field operating over a large environment |
| Density filter | Pattern evaluator / gatekeeper (Section 3.4) |
| Serialised library | Institutional artefact / cultural transmission (Section 5.3) |
| `load_library()` | Innate priors — agents do not start as blank slates (Section 1.3) |
| `source_corpus` metadata | Provenance tracking for pattern fields (Appendix A.7.4) |

---

## Source files consulted

- `hpm_ai_v4/pattern.py` — `HierarchicalPattern`, `FlatPattern`
- `hpm_ai_v4/agents/agent.py` — `HPMAgent`
- `hpm_ai_v4/repository.py` — `PatternRepository` (existing, harvests across agents; distinct from library)
- `hpm_ai_v4/evaluators/metrics.py` — `pattern_density()`, `total_score()`
- `hpm_ai_v4/io/adapters.py` — `CharClassAdapter` (obs_dim=5, used in corpus pipeline)

---

## Attributes serialised per pattern

`HierarchicalPattern` fields written to the library:

| Field | Type | Notes |
|---|---|---|
| `id` | int | Pattern identifier |
| `A` | np.ndarray (KxK) | Transition matrix |
| `B` | np.ndarray (Kxobs_dim) | Emission matrix |
| `pi` | np.ndarray (K,) | Initial state distribution |
| `latent_dim` | int | K |
| `obs_dim` | int | Observation alphabet size |
| `running_loss` | float | Accumulated prediction loss |
| `weight` | float | Replicator weight at save time |
| `creation_step` | int | Step at which pattern was created |
| `source_corpus` | str | Optional: corpus path/label |
| `density_at_save` | float | Optional: density score at save time |

`source_corpus` and `density_at_save` are optional fields added to `HierarchicalPattern` with defaults `""` and `0.0`. PatternSerializer saves and restores them if present.

---

## Stage 1 — PatternSerializer

**File:** `hpm_ai_v4/tools/serializer.py` (CREATE)

### Interface

```python
class PatternSerializer:
    @staticmethod
    def save(patterns: List[HierarchicalPattern], path: str) -> None: ...
    @staticmethod
    def load(path: str) -> List[HierarchicalPattern]: ...
    @staticmethod
    def save_json(patterns: List[HierarchicalPattern], path: str) -> None: ...
    @staticmethod
    def load_json(path: str) -> List[HierarchicalPattern]: ...
```

### Pickle format

- Serialises each pattern as a plain dict (all numpy arrays are picklable).
- Saves list of dicts via `pickle.dump`.
- Load reconstructs `HierarchicalPattern(id, latent_dim, obs_dim)`, sets all attributes from dict, calls `_refresh_log_cache()`.

### JSON format

- Human-readable alternative.
- Numpy arrays converted via `.tolist()` on save; reconstructed via `np.array(..., dtype=np.float32)` on load.
- Useful for inspection and version control diffing of small libraries.

### Internal helpers

```python
FIELDS = ['id', 'A', 'B', 'pi', 'latent_dim', 'obs_dim',
          'running_loss', 'weight', 'creation_step']
OPTIONAL = ['source_corpus', 'density_at_save']

_to_dict(p) -> dict
_from_dict(d) -> HierarchicalPattern
```

---

## Stage 2 — Library creation script

**File:** `hpm_ai_v4/simulations/build_library.py` (CREATE)

### CLI

```
python build_library.py \
    --corpus wiki_sample.txt \
    --output wiki_patterns.pkl \
    --steps 100000 \
    --min-density 0.3
```

### Algorithm

1. Load corpus text from `--corpus` (supports plain `.txt`; uses `WikipediaStream` if available, otherwise reads file directly). Falls back to synthetic text if file not found (for testing).
2. Encode each character via `CharClassAdapter.encode_char()` to obs tokens in [0, 4].
3. Create `HPMAgent(num_initial_patterns=20, obs_dim=5)` with K=2 hierarchical patterns.
4. Feed tokens one at a time via `perceive_and_learn()` for `--steps` steps.
5. Filter: retain patterns where `pattern_density() > --min-density` AND `weight > 0.01`.
6. Stamp `source_corpus` and `density_at_save` on each kept pattern.
7. Save filtered patterns via `PatternSerializer.save()`.
8. Print summary: N patterns saved, density min/mean/max.

### Progress logging

Every 10 000 steps, log:
```
[step=10000] pop_size=23 best_weight=0.412 avg_density=0.187
```

---

## Stage 3 — HPMAgent.load_library()

**File:** `hpm_ai_v4/agents/agent.py` (MODIFY)

### Interface

```python
def load_library(self, path: str, reset_weights: bool = True) -> int:
    """
    Replace self.patterns with patterns loaded from path.
    If reset_weights=True, set all weights to uniform 1/N.
    Returns N (number of patterns loaded).
    """
```

### Behaviour

- Imports `PatternSerializer` lazily to avoid circular imports.
- If `reset_weights=True`: each pattern weight set to `1 / N` so no single prior pattern dominates at the start.
- After loading, the agent's `perceive_and_learn()` loop continues normally; loaded patterns compete via replicator dynamics exactly like freshly initialised ones.

### Usage pattern

```python
agent = HPMAgent(num_initial_patterns=0, obs_dim=5)
agent.load_library("wiki_patterns.pkl")
for obs in stream:
    agent.perceive_and_learn(obs)
```

---

## Distinction from PatternRepository

`PatternRepository` (existing, `hpm_ai_v4/repository.py`) harvests patterns across a live multi-agent pool at runtime. `PatternSerializer` + `build_library.py` is an offline batch pipeline producing a static file. The two are complementary: a library provides the starting population; the repository maintains a shared pool during a run.

---

## File structure

| Path | Action |
|---|---|
| `hpm_ai_v4/tools/serializer.py` | CREATE |
| `hpm_ai_v4/tools/__init__.py` | CREATE if missing |
| `hpm_ai_v4/agents/agent.py` | MODIFY — add `load_library()` |
| `hpm_ai_v4/simulations/build_library.py` | CREATE |
| `hpm_ai_v4/tests/test_serializer.py` | CREATE |

---

## Non-goals

- No streaming or incremental serialisation (libraries are small enough for full pickle).
- No compression of numpy arrays (standard pickle size is acceptable).
- No versioning/schema migration (patterns have a stable attribute set in v4).
