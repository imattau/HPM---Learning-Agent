# HPM v4 Reasoning Layer — Design Spec

**Date**: 2026-04-24
**Branch**: hpm-ai-v4-dev
**Status**: Approved for implementation

---

## 1. Problem

HPM v4 learns patterns via online EM, replicator dynamics, and recombination, but the learned population is never used for deliberative action. The current cognitive cycle is:

```
Observe → Learn → (emit best_pattern.predict_next()) → done
```

This is not a cognitive cycle — it is one-step reactive prediction. The HPM framework requires:

```
Observe → Learn → Deliberate → Act → Observe (repeat)
```

The `Reasoner` class exists in `hpm_ai_v4/agents/reasoning.py` but is not wired into `HPMAgent` or `TotalHPMSystem`.

---

## 2. Architecture Decision

**Reasoner is a separate object** — not embedded in HPMAgent.

```
HPMAgent                       Reasoner(agent)
├── patterns: List             ├── get_relevant_patterns()
├── obs_buffer: List[int]      ├── compose_predictions()
├── perceive_and_learn(obs)    ├── simulate_future()
└── act(goal) ──delegates──►  ├── plan()
                               ├── counterfactual()
                               └── explain()
```

This preserves HPM's separation of:
- **Substrate** (HPMAgent — pattern population, weights, EM)
- **Evaluator/deliberation** (Reasoner — queries population for action)

`TotalHPMSystem.step()` calls `agent.act()` instead of `best_pattern.predict_next()`.

---

## 3. HPM Framework Mapping

| Reasoning mode | HPM level | Description |
|---|---|---|
| `compose_predictions` | L2 — Latent structural | Blend top-K pattern predictive distributions by replicator weight |
| `simulate_future` | L4 — Generative rules | Unroll pattern as generative model to produce imagined futures |
| `plan` | L4 — Generative rules | Stochastic rollout search to reach goal state |
| `counterfactual` | L4 — Generative rules | Intervene on latent emissions, observe divergence |
| `explain` | L3 — Relational rules | Translate pattern structure into human-readable description |

---

## 4. Interface Contracts

### 4.1 HierarchicalPattern (existing, unchanged)

```python
pattern.predict_next(obs_seq: List[int]) -> int
pattern.predict_next_distribution(obs_seq: List[int]) -> np.ndarray  # shape (obs_dim,)
pattern.log_likelihood(obs_seq: List[int]) -> float
pattern.get_belief(obs_seq: List[int]) -> np.ndarray  # shape (K, K, K)
pattern.weight: float
pattern.complexity: int  # 1=flat, 3=hierarchical
pattern.latent_dim: int
pattern.B: np.ndarray   # shape (latent_dim, obs_dim)
pattern.A3: np.ndarray  # shape (K, K)
pattern.A32: np.ndarray # shape (K, K)
pattern.A21: np.ndarray # shape (K, K)
```

### 4.2 Reasoner (target state)

```python
class Reasoner:
    def __init__(self, agent: HPMAgent) -> None: ...

    def get_relevant_patterns(
        self, context_obs: List[int], top_k: int = 5
    ) -> List[HierarchicalPattern]:
        """Return top_k patterns by combined likelihood+weight score."""

    def compose_predictions(
        self, patterns: List[HierarchicalPattern], obs_seq: List[int]
    ) -> np.ndarray:
        """Weighted blend of predictive distributions. Returns shape (obs_dim,)."""

    def simulate_future(
        self, steps: int = 10, top_k: int = 3
    ) -> List[int]:
        """Generate imagined future sequence using best patterns. Returns obs list."""

    def plan(
        self, goal_state: int, horizon: int = 5, num_rollouts: int = 10
    ) -> List[int]:
        """Stochastic rollout search. Returns best action sequence found."""

    def counterfactual(
        self,
        pattern: HierarchicalPattern,
        obs_seq: List[int],
        intervention_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (original_dist, intervened_dist), both shape (obs_dim,)."""

    def explain(self, pattern: HierarchicalPattern) -> str:
        """Human-readable description of what the pattern has learned."""
```

### 4.3 HPMAgent additions (target state)

```python
class HPMAgent:
    def __init__(self, ...):
        ...
        self.reasoner = Reasoner(self)   # already present in current codebase

    def act(self, goal: Optional[int] = None) -> int:
        """Deliberative action selection via self.reasoner."""
```

`act()` already exists in the current codebase. No signature change required.

### 4.4 TotalHPMSystem.step() target state

```python
def step(self, raw_input: Any) -> Optional[int]:
    obs_seq = self.input_adapter.to_observations(raw_input)
    for obs in obs_seq:
        self.meta_layer.run_step(obs)

    # Use primary agent's deliberative act(), not raw predict_next()
    primary_agent = self.meta_layer.agent_pool.agents[0]
    prediction = primary_agent.act()

    self.output_adapter.act(prediction, context=None)
    return prediction
```

---

## 5. Reasoning Mode Specs

### 5.1 compose_predictions

**Input**: `patterns: List[HierarchicalPattern]`, `obs_seq: List[int]`
**Output**: `np.ndarray` shape `(obs_dim,)`, sums to 1.0

Algorithm:
1. For each pattern, call `predict_next_distribution(obs_seq)`.
2. Weight by `pattern.weight`.
3. Normalise by sum of weights.
4. Return blended distribution.

Edge cases:
- Empty patterns list → return uniform `[0.5, 0.5]`.
- All weights zero → fall back to equal weighting.

### 5.2 simulate_future

**Input**: `steps: int = 10`, `top_k: int = 3`
**Output**: `List[int]` of length `steps`

Algorithm:
1. Get top_k relevant patterns from `self.agent.obs_buffer`.
2. Blend their predictive distributions via `compose_predictions`.
3. Sample from blended distribution at each step.
4. Append sampled observation to running context for next step.

This is the population-level simulation, distinct from the existing `simulate()` which uses a single pattern.

### 5.3 plan

**Input**: `goal_state: int`, `horizon: int = 5`, `num_rollouts: int = 10`
**Output**: `List[int]` — best action sequence found

Algorithm (already implemented, verify correctness):
1. Start from `agent.obs_buffer[-20:]`.
2. For each rollout: step through horizon using `get_relevant_patterns` + sample from `predict_next_distribution`.
3. Score each rollout by `-abs(final_obs - goal_state)`.
4. Return sequence with highest score.

### 5.4 counterfactual

**Input**: `pattern: HierarchicalPattern`, `obs_seq: List[int]`, `intervention_idx: int`
**Output**: `Tuple[np.ndarray, np.ndarray]` — `(original_dist, intervened_dist)`

Algorithm (already implemented):
1. Compute `original_dist = pattern.predict_next_distribution(obs_seq)`.
2. Temporarily force all emission rows to emit `intervention_idx` with probability 1.
3. Compute `intervened_dist`.
4. Restore `pattern.B`.
5. Return both.

**Important**: Must be side-effect free — `pattern.B` must be restored exactly.

### 5.5 explain

**Input**: `pattern: HierarchicalPattern`
**Output**: `str`

Algorithm (already implemented):
- If `pattern.complexity >= 2`: report most likely emission path through latent hierarchy.
- Else (flat): report `theta` (emission probability).

---

## 6. Test Strategy

All tests in `hpm_ai_v4/tests/test_reasoning.py`.

Run command: `PYTHONPATH=. pytest hpm_ai_v4/tests/ -v`

Test categories:
1. **Unit — Reasoner methods**: each method tested in isolation with a minimal HPMAgent fixture.
2. **Unit — HPMAgent.act()**: verify `act()` delegates to reasoner and returns `int`.
3. **Integration — TotalHPMSystem.step()**: verify `step()` calls `agent.act()`, not `best_pattern.predict_next()`.
4. **Regression — no side effects**: counterfactual does not mutate `pattern.B`.

---

## 7. What Is Not Changing

- `HierarchicalPattern` — no modifications.
- `perceive_and_learn()` — no modifications.
- Replicator dynamics, EM, recombination — no modifications.
- `DevelopmentalStage` — no modifications.
- All existing tests must continue to pass.

---

## 8. Open Questions (resolved)

| Question | Decision |
|---|---|
| Should Reasoner be embedded in HPMAgent? | No — separate object with reference to agent |
| Which agent is "primary" in TotalHPMSystem? | agents[0] — same as before for obs_buffer history |
| Should plan() use all agents or just one? | One agent's reasoner — multi-agent planning is out of scope |
| Does simulate_future replace existing simulate()? | No — simulate() stays on Reasoner, simulate_future() is population-level wrapper |
