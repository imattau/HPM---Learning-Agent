# Sub-project 4: Structured ARC Benchmark — HPM-Faithful Multi-Level Architecture

## Goal

Replace the random-projection encoder in `hierarchical_arc.py` with a three-level structured encoding pipeline where each level processes a genuinely different abstraction (pixel deltas → object anatomy → relational rules). Implement a domain-agnostic `StructuredOrchestrator` and ARC-specific `LevelEncoder` implementations. Validate with a new benchmark (`structured_arc.py`) that measures per-level accuracy contribution.

## Architecture

### Domain-Agnostic Framework (`hpm/`)

**`hpm/encoders/base.py`** — `LevelEncoder` protocol:
```python
class LevelEncoder(Protocol):
    feature_dim: int
    def encode(self, observation, epistemic: tuple[float, float] | None) -> list[np.ndarray]:
        ...
```
Returns a **list** of numpy arrays — one observation may produce multiple L2 vectors (one per object pair). L1 and L3 always return a list of length 1.

**`hpm/agents/structured.py`** — `StructuredOrchestrator`:
- Accepts `encoders: list[LevelEncoder]` (one per level) — knows nothing about ARC
- Accepts `orches: list[MultiAgentOrchestrator]` (one per level)
- Manages cadence: L3 fires every `K` L2 fires (L2 tick counter, not L1)
- Extracts epistemic state `(weight, epistemic_loss)` from each level's agents after each step
- Threads epistemic state down to the next encoder
- Extension points: `generative_head=None`, `meta_monitor=None` (L4/L5, not implemented)
- `step(observation) -> dict` — observation type is domain-specific, passed through opaquely

### ARC-Specific Encoders (`benchmarks/arc_encoders.py`)

**ArcL1Encoder** (`feature_dim=64`):
- Input: `(input_grid, output_grid)` as lists of lists
- Output: `[delta @ _PROJ]` — pixel delta (output_flat − input_flat) projected to 64-dim
- Projection matrix seeded with `np.random.default_rng(0)` for reproducibility
- Returns list of length 1

**ObjectParser** (internal utility):
- 4-connected component labelling per colour, background=0 ignored
- Per object: `{id, color, bbox=(min_r,min_c,max_r,max_c), area, perimeter, centroid}`
- Perimeter: count of non-background neighbours on object boundary

**ObjectMatcher** (internal utility):
- Match input objects to output objects: colour-first (unique colour → direct match), then greedy centroid proximity
- Returns: `matched_pairs: list[(in_obj, out_obj)]`, `appeared: list[out_obj]`, `disappeared: list[in_obj]`

**ArcL2Encoder** (`feature_dim=9`):
- Input: `(input_grid, output_grid)`, epistemic `(l1_weight, l1_loss)`
- Per matched pair → 9-dim vector:
  ```
  [min_r, min_c, max_r, max_c,   # output bounding box (normalised 0–1)
   area, perimeter,               # output object morphology (normalised)
   color_id,                      # output object colour (0–9, normalised)
   l1_weight, l1_loss]            # epistemic thread from L1
  ```
- Appeared objects: zero-vector (maximum entropy — unexpected novelty)
- Disappeared objects: zero-vector (absence encoded as unknown)
- Returns list of N vectors (one per matched pair + unmatched)

**ArcL3Encoder** (`feature_dim=14`):
- Input: `(input_grid, output_grid)`, epistemic `(l2_weight, l2_loss)`
- One 14-dim relational summary:
  ```
  [n_objects_in, n_objects_out,           # object counts (normalised)
   n_moved, n_recolored,                  # transformation type indicators
   n_appeared, n_disappeared,             # topology changes
   mean_dr, mean_dc,                      # mean positional delta (normalised)
   color_map_consistent,                  # 1.0 if all recolourings follow one mapping
   count_preserved,                       # 1.0 if |in| == |out|
   area_preserved,                        # 1.0 if mean area unchanged
   task_complexity,                       # normalised: (n_moved + n_recolored) / max(n_in, 1)
   l2_weight, l2_loss]                    # epistemic thread from L2
  ```
- Returns list of length 1

### Benchmark (`benchmarks/structured_arc.py`)

**Same discrimination protocol** as `multi_agent_arc.py`:
- 5-way discrimination: correct output vs 4 distractors sharing the same `test_input`
- Distractors: `encode_pair(test_input, tasks[di]["train"][0]["output"])` for di in chosen
- TRAIN_REPS=10, N_DISTRACTORS=4, MAX_GRID_DIM=20

**Training loop per task:**
```
For each training pair (input_grid, output_grid):
    For each of TRAIN_REPS repetitions:
        structured_orch.step((input_grid, output_grid))
        → L1 stepped once (pixel delta)
        → L2 stepped N times (once per matched object pair)
        → L3 stepped if l2_tick % K == 0 (K=3)
```

**Partitioned training across L1 agents** (same as multi_agent_arc):
- L1 agent 0 sees even-indexed training pairs; remaining L1 agents see odd-indexed pairs
- `StructuredOrchestrator.step()` accepts optional `agent_partition: dict` to route observations

**Scoring:**
```python
L1_score = ensemble_score(l1_agents, pixel_delta_vec)
L2_score = mean(ensemble_score(l2_agents, obj_vec) for obj_vec in object_pairs)  # normalised by N pairs
L3_score = ensemble_score(l3_agents, relation_vec)
total = L1_score + L2_score + L3_score
```
L2 normalised by number of matched pairs so a 10-object task does not dominate a 1-object task.

**Three baselines in the same run:**
| Baseline | Training | Scoring |
|----------|----------|---------|
| `flat` | 1 agent, pixel delta only | L1 ensemble_score |
| `l1_only` | Full structured training | L1 score only |
| `full_structured` | Full structured training | L1 + L2/N + L3 |

**Success criteria:**
- `full_structured` accuracy > `flat` accuracy (hierarchy adds value over baseline)
- `l1_only` vs `flat`: determines whether structured training improves even pixel-level discrimination
- L2 and L3 each contribute measurably when removed from scoring (ablation printed in results table)

## Agent Configuration

| Level | Agents | feature_dim | pattern_type | Notes |
|-------|--------|-------------|--------------|-------|
| L1 | 2 | 64 | gaussian | Partitioned training pairs |
| L2 | 2 | 9 | gaussian | One step per object pair |
| L3 | 1 | 14 | gaussian | Fires every K=3 L2 steps |

All levels: `gamma_soc=0.5`, `init_sigma=2.0`, `with_monitor=False` (speed), `InMemoryStore` (per-task reset).

## File Structure

```
hpm/encoders/__init__.py          — re-exports LevelEncoder
hpm/encoders/base.py              — LevelEncoder Protocol
hpm/agents/structured.py          — StructuredOrchestrator
benchmarks/arc_encoders.py        — ArcL1Encoder, ArcL2Encoder, ArcL3Encoder, ObjectParser, ObjectMatcher
benchmarks/structured_arc.py      — benchmark script
tests/encoders/test_arc_encoders.py   — unit tests for encoders
tests/agents/test_structured.py       — unit tests for StructuredOrchestrator
```

## L4/L5 Extension Points

`StructuredOrchestrator.__init__` accepts:
- `generative_head=None`: at L4, will reverse encoders to synthesise output grid from L3 rule
- `meta_monitor=None`: at L5, will monitor cross-level coherence and bias attention

These parameters are validated at construction (`if not None: raise NotImplementedError`) until implemented.

## Non-Goals (this sub-project)

- No persistent/cross-task learning (per-task reset only; persistent mode is a follow-on)
- No L4 (generative output synthesis) or L5 (meta-monitoring)
- No modification of existing benchmarks (`hierarchical_arc.py`, `multi_agent_arc.py`)
- No changes to `hpm/` agent, store, field, or dynamics modules
