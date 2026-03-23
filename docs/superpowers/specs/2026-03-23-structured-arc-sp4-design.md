# Sub-project 4: Structured ARC Benchmark — HPM-Faithful Multi-Level Architecture

## Goal

Replace the random-projection encoder in `hierarchical_arc.py` with a three-level structured encoding pipeline where each level processes a genuinely different abstraction (pixel deltas → object anatomy → relational rules). Implement a domain-agnostic `StructuredOrchestrator` and ARC-specific `LevelEncoder` implementations. Validate with a new benchmark (`structured_arc.py`) that measures per-level accuracy contribution.

## Architecture

### Domain-Agnostic Framework (`hpm/`)

**`hpm/encoders/base.py`** — `LevelEncoder` protocol:
```python
class LevelEncoder(Protocol):
    feature_dim: int
    max_steps_per_obs: int | None  # None = variable (e.g. L2 per-object); 1 = fixed (L1, L3)
    def encode(self, observation, epistemic: tuple[float, float] | None) -> list[np.ndarray]:
        ...
```
Returns a **list** of numpy arrays. L2 returns one vector per matched object pair (variable length). L1 and L3 always return a list of length 1. `max_steps_per_obs` allows the orchestrator to handle variable-length L2 uniformly without special-casing.

**`hpm/agents/structured.py`** — `StructuredOrchestrator`:
- Accepts `encoders: list[LevelEncoder]` (one per level) — knows nothing about ARC
- Accepts `orches: list[MultiAgentOrchestrator]` (one per level)
- **Cadence**: L3 fires every K *calls to `step()`* (i.e., training-pair steps), not object-pair ticks. L2 is always stepped for every matched object pair on every `step()` call. This matches `StackedOrchestrator` semantics and is invariant to per-task object count.
- Extracts epistemic state `(weight, epistemic_loss)` from level i's agents after each step:
  - `weight`: mean of all pattern weights in `agent.store.query(agent.agent_id)` for the primary agent (index 0); 0.0 if store empty
  - `epistemic_loss`: value of key `"epistemic_loss"` in the agent step result dict; 0.0 if absent
- Threads epistemic state as `epistemic=(weight, epistemic_loss)` into the next level's encoder
- Extension points: `generative_head=None`, `meta_monitor=None` — `None` is valid (no-op); non-`None` raises `NotImplementedError` until L4/L5 are implemented
- `step(observation, l1_obs_dict: dict | None = None) -> dict` — observation is domain-specific; `l1_obs_dict` optionally overrides L1 routing for partitioned training

### ARC-Specific Encoders (`benchmarks/arc_encoders.py`)

**ArcL1Encoder** (`feature_dim=64`, `max_steps_per_obs=1`):
- Input: `(input_grid, output_grid)` as lists of lists
- Output: `[delta @ _PROJ]` — pixel delta projected to 64-dim
- Projection matrix: `np.random.default_rng(0).standard_normal((400, 64)) / np.sqrt(64)`, same seed as `multi_agent_arc.py`
- Returns list of length 1

**ObjectParser** (internal utility, not part of encoder protocol):
- 4-connected component labelling per colour, background=0 ignored
- Per object: `{id, color, bbox=(min_r, min_c, max_r, max_c), area, perimeter, centroid}`
- `area`: count of cells; `perimeter`: count of boundary cell edges adjacent to non-same-colour
- All spatial values normalised by `MAX_GRID_DIM=20` (fixed denominator for cross-task consistency)

**ObjectMatcher** (internal utility):
- Match input objects to output objects: colour-first (unique colour → direct match), then greedy by centroid proximity (iteration order: largest area first, then row-major centroid)
- Returns: `matched_pairs: list[(in_obj, out_obj)]`, `appeared: list[out_obj]`, `disappeared: list[in_obj]`

**ArcL2Encoder** (`feature_dim=9`, `max_steps_per_obs=None`):
- Input: `(input_grid, output_grid)`, epistemic `(l1_weight, l1_loss)`
- Per **matched pair** only → 9-dim vector:
  ```
  [min_r, min_c, max_r, max_c,   # output bounding box (normalised by MAX_GRID_DIM=20)
   area, perimeter,               # output object morphology (area/MAX_GRID_DIM², perimeter/MAX_GRID_DIM)
   color_id / 9.0,                # output object colour (0–9, normalised to 0–1)
   l1_weight, l1_loss]            # epistemic thread from L1
  ```
- **Appeared and disappeared objects are excluded** from the returned list and from the L2 score denominator. They contribute no signal (the agent has not trained on object creation/deletion patterns in per-task reset mode). This prevents zero-vector bias against correct answers that involve structural novelty.
- Returns list of N vectors (matched pairs only; length 0 if no objects match)

**ArcL3Encoder** (`feature_dim=14`, `max_steps_per_obs=1`):
- Input: `(input_grid, output_grid)`, epistemic `(l2_weight, l2_loss)`
- One 14-dim relational summary:
  ```
  [n_objects_in / MAX_OBJ,        # normalised object counts (MAX_OBJ=20)
   n_objects_out / MAX_OBJ,
   n_moved / MAX_OBJ,             # objects whose centroid shifted > 1 cell
   n_recolored / MAX_OBJ,         # objects whose colour changed
   n_appeared / MAX_OBJ,          # objects present in output only
   n_disappeared / MAX_OBJ,       # objects present in input only
   mean_dr, mean_dc,              # mean centroid delta (normalised by MAX_GRID_DIM)
   color_map_consistent,          # 1.0 if all recolourings follow one src→dst mapping (1.0 if no recolourings)
   count_preserved,               # 1.0 if n_objects_in == n_objects_out
   area_preserved,                # 1.0 if mean input area ≈ mean output area (within 10%)
   task_complexity,               # (n_moved + n_recolored) / max(n_objects_in, 1)
   l2_weight, l2_loss]            # epistemic thread from L2
  ```
- Returns list of length 1

### Benchmark (`benchmarks/structured_arc.py`)

**Discrimination protocol** — identical to `multi_agent_arc.py`:
- 5-way discrimination: correct output vs 4 distractors, all sharing the same `test_input`
- Distractors: `encode_pair(test_input, tasks[di]["train"][0]["output"])` for di in chosen
- TRAIN_REPS=10, N_DISTRACTORS=4, MAX_GRID_DIM=20, rng seed=42

**Training loop per task:**
```
pairs_a = train_pairs[0::2] or train_pairs  # L1 agent 0
pairs_b = train_pairs[1::2] or train_pairs  # L1 agents 1+
n = max(len(pairs_a), len(pairs_b))
for rep in range(TRAIN_REPS):
    for k in range(n):
        obs_a = (pairs_a[k%len(pairs_a)]["input"], pairs_a[k%len(pairs_a)]["output"])
        obs_b = (pairs_b[k%len(pairs_b)]["input"], pairs_b[k%len(pairs_b)]["output"])
        # l1_enc, l2_enc, l3_enc are pre-constructed encoder instances
        l1_obs_dict = {l1_agent_ids[0]: l1_enc.encode(obs_a, epistemic=None)[0],
                       l1_agent_ids[1]: l1_enc.encode(obs_b, epistemic=None)[0]}
        structured_orch.step(obs_a, l1_obs_dict=l1_obs_dict)
        # L2: all agents see all matched object pairs from obs_a (no L2 partitioning)
        # L3: fires every K=3 calls to step() (cadence on training-pair steps)
```

**Scoring at test time:**

For each candidate `(test_input, candidate_output_grid)`:
```python
# L1
pixel_vec = ArcL1Encoder.encode((test_input, candidate_output))[0]
L1_score = ensemble_score(l1_agents, pixel_vec)

# L2: run ArcL2Encoder on (test_input, candidate_output) to get object-pair vecs
obj_vecs = ArcL2Encoder.encode((test_input, candidate_output), epistemic=(l1_w, l1_l))
L2_score = mean(ensemble_score(l2_agents, v) for v in obj_vecs) if obj_vecs else 0.0
# mean normalises by N matched pairs — unmatched objects excluded

# L3
rel_vec = ArcL3Encoder.encode((test_input, candidate_output), epistemic=(l2_w, l2_l))[0]
L3_score = ensemble_score(l3_agents, rel_vec)

total = L1_score + L2_score + L3_score  # lower = more probable = preferred
```
Epistemic state `(l1_w, l1_l)` and `(l2_w, l2_l)` are extracted from the trained agents at the end of training (not re-run during scoring).

**Four baselines in the same run:**

| Baseline | Agents | Training | Scoring |
|----------|--------|----------|---------|
| `flat` | 2, pixel delta | Partitioned pixel delta | L1 only (matches multi_agent_arc.py) |
| `l1_only` | 2+2+1 structured | Full structured training | L1 score only |
| `l2_only` | 2+2+1 structured | Full structured training | L2 score only (mean over pairs) |
| `full_structured` | 2+2+1 structured | Full structured training | L1 + L2/N + L3 |

The `flat` baseline uses 2 agents with partitioned training to match `multi_agent_arc.py` exactly. This is the correct reference: any benefit from `full_structured` over `flat` reflects the structured hierarchy, not agent count.

**Success criteria:**
- `full_structured` accuracy > `flat` accuracy
- `l1_only` vs `flat`: does structured training improve even pixel-level discrimination?
- At least one of L2-only or L3-only contributes measurable accuracy above chance (>20%) — confirms the encoders capture discriminative signal at each level

## Agent Configuration

| Level | Agents | feature_dim | pattern_type | Notes |
|-------|--------|-------------|--------------|-------|
| L1 | 2 | 64 | gaussian | Partitioned training pairs |
| L2 | 2 | 9 | gaussian | All agents see all matched object pairs per step |
| L3 | 1 | 14 | gaussian | Fires every K=3 `step()` calls |

All levels: `gamma_soc=0.5`, `init_sigma=2.0`, `with_monitor=False`, `InMemoryStore`, per-task reset.

## File Structure

```
hpm/encoders/__init__.py              — re-exports LevelEncoder
hpm/encoders/base.py                  — LevelEncoder Protocol
hpm/agents/structured.py              — StructuredOrchestrator
benchmarks/arc_encoders.py            — ArcL1Encoder, ArcL2Encoder, ArcL3Encoder, ObjectParser, ObjectMatcher
benchmarks/structured_arc.py          — benchmark script
tests/encoders/test_arc_encoders.py   — unit tests for encoders and object parser
tests/agents/test_structured.py       — unit tests for StructuredOrchestrator
```

## L4/L5 Extension Points

`StructuredOrchestrator.__init__` accepts:
- `generative_head=None`: at L4, will reverse encoders to synthesise output grid from L3 rule
- `meta_monitor=None`: at L5, will monitor cross-level coherence and bias L3 attention

Passing `None` is valid (no-op). Passing a non-`None` value raises `NotImplementedError`.

## Non-Goals (this sub-project)

- No persistent/cross-task learning (per-task reset only)
- No L4 (generative output synthesis) or L5 (meta-monitoring)
- No modification of existing benchmarks (`hierarchical_arc.py`, `multi_agent_arc.py`)
- No changes to core `hpm/` modules (agents, store, field, dynamics, patterns)
