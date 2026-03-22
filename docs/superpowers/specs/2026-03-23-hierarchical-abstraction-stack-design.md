# Hierarchical Abstraction Stack — Design Spec

## Overview

This spec covers Sub-project 1 of a planned 3-phase hierarchical extension to the HPM Learning Agent. The goal is to implement a 2-level abstraction stack where the output of Level 1 agents becomes the substrate for Level 2 agents — rather than all agents competing over the same raw observations.

Sub-projects:
- **Sub-project 1 (this spec):** Inter-level encoding protocol + `HierarchicalOrchestrator` (2-level stack)
- **Sub-project 2:** 3-level stack + ARC benchmark integration
- **Sub-project 3:** Confidence-gated stepping (epistemic routing)

---

## Motivation

The existing `MultiAgentOrchestrator` places all agents at the same abstraction level — they all see the same raw observations and compete over a shared `PatternField`. This is effective for learning within a single abstraction level but cannot implement the HPM principle that patterns at higher levels of abstraction are built from regularities detected at lower levels.

The hierarchical stack addresses this by making Level 1 agent outputs the inputs to Level 2 agents. Each level "compresses" the data: raw pixels become object features, object features become relational laws. This mirrors the HPM multi-level abstraction hierarchy and the structure of the human visual cortex (V1 → V4 → IT cortex).

---

## Design Decisions

### Inter-level encoding: Structured Bundle (Option D)

Each Level 1 agent's output to Level 2 is a **structured bundle**: `[μ, w, L]` concatenated into a single numpy vector.

- `μ` (mu): shape `(D,)` — the mean of the top-weighted pattern in the agent's store. Represents what the agent currently "believes" about the signal.
- `w` (weight): scalar float — the field weight of that pattern. Represents how well-established the pattern is.
- `L` (epistemic_loss): scalar float — the agent's current epistemic loss. Represents how confident the agent is in its current belief.

This encoding preserves **epistemic integrity**: Level 2 knows not just what Level 1 thinks it saw, but how much to trust it. A high-loss bundle is naturally down-weighted by Level 2's existing `MetaPatternRule` and `PatternField` machinery without any special handling.

If a Level 1 agent's store is empty, it emits a zero bundle with `weight=0.0` and `epistemic_loss=1.0` (maximum uncertainty).

Level 2's `feature_dim = level1_feature_dim + 2`. Level 2 agents use `GaussianPattern` — no new pattern type required.

### Stepping cadence: K-cadence (Option B)

Level 2 steps every K Level 1 steps. K is a constructor parameter, default K=1 (synchronous, degrades to Option A behaviour for debugging). At K=10, Level 1 accumulates 10 observations before Level 2 sees a bundle — Level 1's epistemic noise is averaged out, and Level 2 sees a more stable signal.

This implements HPM's timescale hierarchy: higher abstraction levels operate on slower timescales.

Future Sub-project 3 will extend this to confidence-gated stepping (variable cadence based on epistemic loss threshold).

### Bundle delivery: N separate bundles — Witness Model (Option A)

When the cadence fires, the orchestrator extracts one bundle per Level 1 agent and delivers them as N separate `step()` calls to the Level 2 orchestrator. Level 2 sees N "witnesses" per cadence tick.

This is scale-invariant (Level 2 doesn't care if there are 3 or 300 Level 1 agents), allows the Level 2 `PatternField` to naturally weight high-confidence witnesses over noisy ones, and enables future cross-level recombination by the `RecombinationStrategist`.

---

## Architecture

### New file: `hpm/agents/hierarchical.py`

```
LevelBundle                   dataclass
  agent_id: str
  mu: np.ndarray              shape (D,)
  weight: float
  epistemic_loss: float

extract_bundle(agent) -> LevelBundle
  — pulls top pattern mu from agent's store (zeros if empty)
  — pulls field weight via agent.field.get_weight(pattern.id), or 0.0
  — pulls epistemic_loss from agent.epistemic.last_loss

encode_bundle(bundle: LevelBundle) -> np.ndarray
  — np.concatenate([bundle.mu, [bundle.weight, bundle.epistemic_loss]])
  — output shape: (D + 2,)

make_hierarchical_orchestrator(
    n_l1_agents: int,
    n_l2_agents: int,
    l1_feature_dim: int,
    K: int = 1,
    l1_pattern_type: str = "gaussian",
    l2_pattern_type: str = "gaussian",
    **kwargs
) -> tuple[HierarchicalOrchestrator, list[Agent], list[Agent]]
  — builds level1_orch via make_orchestrator(n_l1_agents, l1_feature_dim, ...)
  — builds level2_orch via make_orchestrator(n_l2_agents, l1_feature_dim + 2, ...)
  — returns HierarchicalOrchestrator + both agent lists

HierarchicalOrchestrator
  level1_orch: MultiAgentOrchestrator
  level2_orch: MultiAgentOrchestrator
  level1_agents: list[Agent]
  level2_agents: list[Agent]
  K: int
  _t: int

  step(obs: np.ndarray) -> dict
    — broadcasts obs to all Level 1 agents, calls level1_orch.step()
    — increments _t
    — if _t % K == 0:
        for each Level 1 agent, extract_bundle() → encode_bundle()
        for each bundle: calls level2_orch.step({l2_agent_id: encoded_bundle})
    — returns {"level1": l1_result, "level2": l2_result, "t": _t}
```

### Modified files

- `hpm/agents/__init__.py` — re-export `HierarchicalOrchestrator`, `LevelBundle`, `make_hierarchical_orchestrator`

### New test file: `tests/agents/test_hierarchical.py`

Tests:
- `encode_bundle` produces shape `(D+2,)` with correct values
- `extract_bundle` on empty-store agent returns zero mu, w=0.0, L=1.0
- `extract_bundle` on populated agent returns top pattern's mu and field weight
- `HierarchicalOrchestrator.step()` calls Level 2 only on cadence ticks
- `HierarchicalOrchestrator.step()` at K=1 calls Level 2 on every step
- `HierarchicalOrchestrator.step()` at K=5 calls Level 2 on steps 5, 10, 15...
- Level 2 agents receive encoded bundles of correct shape

### New benchmark: `benchmarks/hierarchical_smoke.py`

Synthetic validation (not ARC). Protocol:
- Level 1: 2 agents, Gaussian signal in 16-dim space, 100 steps
- Level 2: 1 agent, GaussianPattern with feature_dim=18, K=5
- Assert: Level 2 receives bundles at correct cadence
- Assert: Level 2 mean accuracy is finite and non-NaN
- Assert: Level 2 step count = floor(100 / 5) * 2 (N bundles per tick)

---

## Data flow summary

```
Raw observation (np.ndarray, shape D)
    │
    ▼
Level 1 Orchestrator
    ├── Agent A  →  GaussianPattern.update(obs)
    └── Agent B  →  GaussianPattern.update(obs)
    │
    │  [every K steps]
    │  extract_bundle(Agent A) → encode → np.ndarray shape (D+2,)
    │  extract_bundle(Agent B) → encode → np.ndarray shape (D+2,)
    ▼
Level 2 Orchestrator
    └── Agent X  →  GaussianPattern.update(bundle_A)
                    GaussianPattern.update(bundle_B)
```

---

## What is NOT in this spec

- 3-level stack (Sub-project 2)
- ARC benchmark integration (Sub-project 2)
- Confidence-gated / variable cadence stepping (Sub-project 3)
- Top-down prediction / Forecaster integration across levels (Sub-project 3)
- Cross-level recombination (Sub-project 3)

---

## Success criteria

- `hierarchical_smoke.py` runs without error and Level 2 receives bundles at the correct cadence
- All unit tests in `test_hierarchical.py` pass
- Setting K=1 produces identical behaviour to a flat `MultiAgentOrchestrator` run with feature_dim = D+2 (regression check)
