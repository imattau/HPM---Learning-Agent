# Hierarchical Abstraction Stack — Sub-project 2 Design Spec

## Overview

This spec covers Sub-project 2 of the hierarchical extension to the HPM Learning Agent. It extends the 2-level `HierarchicalOrchestrator` (Sub-project 1) to an **N-level `StackedOrchestrator`** supporting arbitrary depth, and adds a **separate hierarchical ARC benchmark** to evaluate whether stacked abstraction improves discrimination accuracy over the flat single-agent baseline.

Sub-projects:
- **Sub-project 1 (done):** Inter-level encoding protocol + `HierarchicalOrchestrator` (2-level stack)
- **Sub-project 2 (this spec):** N-level `StackedOrchestrator` + hierarchical ARC benchmark
- **Sub-project 3:** Confidence-gated stepping (epistemic routing)

---

## Motivation

`HierarchicalOrchestrator` is hardwired to exactly 2 levels. HPM predicts that hierarchy depth should scale with task complexity — ARC tasks require at minimum 3 abstraction levels (feature extraction → relational modelling → global strategy). The `StackedOrchestrator` makes depth a hyperparameter, not an architectural constraint.

The hierarchical ARC benchmark tests whether the additional abstraction levels produce a measurable signal: can L3 agents, trained on L2 bundles derived from L1 features, discriminate ARC transformations better than a flat agent?

---

## Design Decisions

### N-level stack with natural dimension growth

Each level's `feature_dim` is computed automatically by the factory:

| Level | Input Source | Dimension | Cognitive Role |
|---|---|---|---|
| L1 | Raw substrate | D | Feature extraction |
| L2 | L1 bundles | D + 2 | Relational modelling |
| L3 | L2 bundles | D + 4 | Global strategy |
| L_n | L_{n-1} bundles | D + 2(n-1) | ... |

`LevelConfig` specifies only the "biological" parameters: `n_agents`, `pattern_type`, `K`. No `feature_dim` — the factory computes it.

This keeps dimensions growing naturally (each `encode_bundle` adds `[weight, epistemic_loss]`). Projection/compression at level boundaries is deferred to Sub-project 3.

### Per-level cadence (K relative to level below)

Each level has its own `K` — the number of times the level below must fire before this level fires. A 3-level stack with K=3 at each level means:
- L2 fires every 3 L1 steps
- L3 fires every 3 L2 fires = every 9 L1 steps

Per-level tick counters (`_level_ticks: list[int]`) track how many times each level has been stepped. Level `i+1` fires when `_level_ticks[i] % K[i+1] == 0`.

### Witness model preserved (same as Sub-project 1)

When level `i+1` fires, the orchestrator extracts one bundle per level-`i` agent and delivers N separate `step()` calls to the level-`i+1` orchestrator. Level `i+1` sees N "witnesses" per cadence tick. This is identical to the witness model in Sub-project 1 and scales to arbitrary depth.

### Direct agent exposure via `level_agents`

`StackedOrchestrator` exposes `self.level_agents: list[list[Agent]]` — one list of agents per level. This allows `StructuralLawMonitor` or diagnostic code to walk the hierarchy and inspect which level is most confident at any point during a run.

### Separate hierarchical ARC benchmark

The existing `benchmarks/arc_benchmark.py` and `benchmarks/multi_agent_arc.py` are unchanged. A new `benchmarks/hierarchical_arc.py` runs the stacked orchestrator on the same ARC encoding (64-dim random projection of grid delta, same seed, same 342-task subset). L3 agents score the final discrimination. Output compares L3 accuracy vs flat single-agent baseline.

---

## Architecture

### New file: `hpm/agents/stacked.py`

```
LevelConfig                   dataclass
  n_agents: int
  pattern_type: str = "gaussian"
  K: int = 1
  agent_ids: list[str] | None = None

StackedOrchestrator
  level_orches: list[MultiAgentOrchestrator]
  level_agents: list[list[Agent]]      # public — one list per level
  level_Ks: list[int]                  # K[0] unused (L1 always steps); K[i] = cadence for level i
  _level_ticks: list[int]              # how many times each level has been stepped

  step(obs: np.ndarray) -> dict
    — steps level 0 on obs, increments _level_ticks[0]
    — for i in range(1, n_levels):
        if _level_ticks[i-1] % level_Ks[i] == 0:
          for each agent in level_agents[i-1]:
            bundle = extract_bundle(agent)
            encoded = encode_bundle(bundle)
            obs_dict = {a.agent_id: encoded for a in level_agents[i]}
            result_i = level_orches[i].step(obs_dict)
          increments _level_ticks[i]
    — returns {"level1": ..., "level2": ..., ..., "t": _level_ticks[0]}
      keys are "level1" through "level{n}" (1-indexed)
      deeper levels return {} on non-cadence steps

  Edge cases:
    K[i] > total steps: level i never fires. Valid, no error.
    n_levels=2: degrades to HierarchicalOrchestrator behaviour.

make_stacked_orchestrator(
    l1_feature_dim: int,
    level_configs: list[LevelConfig],
) -> tuple[StackedOrchestrator, list[list[Agent]]]
  — for each config at index i:
      feature_dim = l1_feature_dim + 2 * i
      agent_ids = config.agent_ids or [f"l{i+1}_{j}" for j in range(config.n_agents)]
      build orchestrator via make_orchestrator(
          n_agents=config.n_agents,
          feature_dim=feature_dim,
          agent_ids=agent_ids,
          pattern_types=[config.pattern_type] * config.n_agents,
      )
  — level_Ks = [1] + [cfg.K for cfg in level_configs[1:]]
    (L1 always steps; K[i] is cadence for level i relative to level i-1)
  — returns (StackedOrchestrator, [[l1_agents], [l2_agents], ...])
  — lazy import of make_orchestrator from benchmarks/ inside function body
    (same pattern as make_hierarchical_orchestrator in Sub-project 1)
```

### Modified files

- `hpm/agents/__init__.py` — re-export `LevelConfig`, `StackedOrchestrator`, `make_stacked_orchestrator`

### New test file: `tests/agents/test_stacked.py`

Tests:
- `LevelConfig` defaults: `pattern_type="gaussian"`, `K=1`, `agent_ids=None`
- `make_stacked_orchestrator` computes correct dims: l1_feature_dim=64 → [64, 66, 68] for 3 levels
- `StackedOrchestrator.level_agents` shape: `len(level_agents) == n_levels`, `len(level_agents[i]) == config.n_agents`
- K-cadence: 3-level stack with K=3 — L2 fires at t=3,6,9..., L3 fires at t=9,18,27...
- Non-cadence steps: deeper levels return `{}` in result dict
- `_level_ticks[0]` increments on every step; `_level_ticks[1]` increments only on L2 cadence steps
- K larger than n_steps: deeper levels never fire, no error, no exception
- 2-level degenerate case: `make_stacked_orchestrator` with 2 configs behaves like `HierarchicalOrchestrator`

### New benchmark: `benchmarks/hierarchical_arc.py`

Stack configuration:
- `l1_feature_dim = 64` (same random projection as existing ARC benchmarks)
- L1: 2 agents, GaussianPattern, K=1
- L2: 2 agents, GaussianPattern, K=3
- L3: 1 agent, GaussianPattern, K=3 (fires every 9 L1 steps)

Training: same ARC train split, same 342-task subset (grids ≤ 20×20), same random projection seed. Each training pair stepped through `stacked_orch.step()`. L3 agents used for final discrimination scoring via `ensemble_score`.

Output table: task count, L3 accuracy, vs chance (20%), vs flat single-agent baseline (from `arc_benchmark.py`).

---

## Data flow summary

```
Raw ARC delta (np.ndarray, shape 64)
    │
    ▼
L1 Orchestrator (2 agents, GaussianPattern, dim=64)
    ├── Agent l1_0  →  GaussianPattern.update(obs)
    └── Agent l1_1  →  GaussianPattern.update(obs)
    │
    │  [every 3 steps]
    │  extract_bundle(l1_0) → encode → shape (66,)
    │  extract_bundle(l1_1) → encode → shape (66,)
    ▼
L2 Orchestrator (2 agents, GaussianPattern, dim=66)
    ├── Agent l2_0  →  GaussianPattern.update(bundle_l1_0)
    │                   GaussianPattern.update(bundle_l1_1)
    └── Agent l2_1  →  GaussianPattern.update(bundle_l1_0)
                        GaussianPattern.update(bundle_l1_1)
    │
    │  [every 3 L2 steps = every 9 L1 steps]
    │  extract_bundle(l2_0) → encode → shape (68,)
    │  extract_bundle(l2_1) → encode → shape (68,)
    ▼
L3 Orchestrator (1 agent, GaussianPattern, dim=68)
    └── Agent l3_0  →  GaussianPattern.update(bundle_l2_0)
                        GaussianPattern.update(bundle_l2_1)
                        → final discrimination score
```

---

## What is NOT in this spec

- Projection/compression at level boundaries (Sub-project 3)
- Confidence-gated / variable cadence stepping (Sub-project 3)
- Top-down prediction across levels (Sub-project 3)
- Normalisation of epistemic_loss scale (Sub-project 3)
- Backward compatibility shim for `HierarchicalOrchestrator` (it stays as-is; `StackedOrchestrator` is the general replacement)

---

## Success criteria

- All unit tests in `test_stacked.py` pass
- `hierarchical_arc.py` runs without error and reports a finite L3 accuracy score
- L3 accuracy is reported alongside flat single-agent baseline for comparison
- Setting n_levels=2 with K=1 produces the same structural behaviour as `HierarchicalOrchestrator`
