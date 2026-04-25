# Spec: MetaCognitivePattern (L5 Strategic Oversight)

Date: 2026-04-22
Status: Draft

---

## 1. Purpose and HPM Position

MetaCognitivePattern is the **Level 5 meta-pattern** in the HPM hierarchy. It does not encode domain knowledge or compile substrates — it observes the learning process itself and issues strategic directives to reshape how lower-level patterns are selected, explored, and organised.

HPM hierarchy as implemented in hpm_ai_v3:
- L1: Sensory regularities (CausalPattern neural substrate)
- L2: Latent structural representations (CausalPattern with compiled structure)
- L3: Relational/symbolic rules (SymbolicPattern via SubstrateCompiler)
- L4: Generative / action sequences (ActionPattern pipelines, MotorPattern)
- **L5: Meta-patterns — strategic oversight of the learning process** ← this spec

This is distinct from SubstrateCompiler, which performs L2→L3 distillation (structural compilation). MetaCognitivePattern operates on a slower timescale (every N episodes) and issues directives about exploration, exploitation, phase advancement, and population health.

---

## 2. Inputs: Meta-Feature Vector (64-dim)

At the end of every N=10 episodes, the meta-pattern receives a 64-dimensional observation vector computed from population and curriculum state.

### 2.1 Core Features (8 raw scalars)

| Feature | Source | Description |
|---|---|---|
| `success_rate` | trailing 10-episode mean reward | Overall task performance |
| `phase` | `CurriculumManager.phase` (normalised 0–1) | Current curriculum phase |
| `entropy` | Shannon entropy of `population.patterns` weights | Exploration diversity of the population |
| `top_pattern_dominance` | weight of highest-weight pattern | Risk of premature convergence |
| `exploration_rate` | mean `exploration_temperature` across patterns | Tendency to explore vs. exploit |
| `steps_since_advance` | episodes since last `ADVANCE_PHASE` | Curriculum stagnation signal |
| `avg_reward` | mean reward over trailing window | Smoothed performance |
| `diversity` | mean pairwise structural distance across population | Structural diversity |

### 2.2 Expansion to 64-dim

The 8 raw scalars are passed through a learned linear projection layer (`nn.Linear(8, 64)`) with ReLU activation. This embedding allows the policy to learn non-linear combinations of the raw signals without hardcoding feature engineering.

---

## 3. Policy Network Architecture

```
meta_features (64-dim)
    ↓
nn.Linear(64, 128) + ReLU
    ↓
nn.Linear(128, 8)
    ↓
Softmax → directive probabilities (8-dim)
```

- Input: 64-dim meta-feature embedding
- Hidden: 128 units, ReLU
- Output: 8 directive logits → softmax probabilities
- At inference: argmax over probabilities (or sample during training)
- Weights initialised with Xavier uniform; biases zero

---

## 4. The 8 Meta-Directives

| Index | Name | Trigger condition | Effect in codebase |
|---|---|---|---|
| 0 | `CONTINUE` | Performance stable and improving | No state change — continue current trajectory |
| 1 | `INJECT_EXPLORATION` | Low entropy, high dominance | Increase `exploration_temperature` on all ActionPatterns; add random low-weight ActionPattern to population |
| 2 | `RESET_STUCK_PATTERNS` | Stagnant success_rate for >20 episodes | Set weight of bottom-25% patterns to `pruning_threshold`; triggers natural extinction next population step |
| 3 | `ADVANCE_PHASE` | High success_rate (>0.8), low steps_since_advance | Call `CurriculumManager.advance_phase()` directly, bypassing the moving-average gate |
| 4 | `SPAWN_SPECIALIST` | High diversity + declining avg_reward | Call `SubstrateCompiler.spawn_agent_from_composite()` on top-3 patterns to create a specialist sub-agent |
| 5 | `INCREASE_DIFFICULTY` | High success_rate but slow phase advance | Call `CurriculumManager.set_difficulty(+0.1)` |
| 6 | `DECREASE_DIFFICULTY` | Low success_rate (<0.2) | Call `CurriculumManager.set_difficulty(-0.2)` |
| 7 | `CONSOLIDATE` | High diversity, low coherence across population | Zero-out weights below median; normalise remaining weights — prune noise, strengthen dominant patterns |

---

## 5. Meta-Reward Signal

The meta-reward is computed at the end of each meta-step (every N=10 episodes) and is distinct from the base-level reward used by EvaluatorManager.

```
r_meta = w_phase * phase_advance_bonus
       + w_perf  * (success_rate_now - success_rate_prev)
       + w_eff   * sample_efficiency_bonus
       + w_div   * diversity_bonus
```

### Component definitions

| Component | Weight | Computation |
|---|---|---|
| `phase_advance_bonus` | 2.0 | +2.0 if phase advanced during this window, else 0 |
| `success_rate_delta` | 1.0 | `success_rate_now - success_rate_prev` (signed) |
| `sample_efficiency_bonus` | 0.5 | `success_rate_now / max(1, steps_since_advance)` scaled to [0,1] |
| `diversity_bonus` | 0.3 | `diversity` (mean pairwise structural distance, already in [0,1]) |

The meta-reward drives the policy to advance phases quickly, improve performance, do so with fewer episodes, and maintain structural diversity.

---

## 6. Learning Algorithm: REINFORCE with Eligibility Traces

MetaCognitivePattern learns via REINFORCE (Williams, 1992) with eligibility traces to handle delayed credit assignment.

### Update rule

At each meta-step t:
1. Record `(s_t, a_t, r_t)` — state (meta-features), action (directive), reward
2. Accumulate eligibility trace: `e_t = γλ * e_{t-1} + ∇_θ log π(a_t | s_t)`
3. Policy gradient update: `θ ← θ + α * r_t * e_t`

### Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `meta_lr` | 1e-3 | Adam optimiser |
| `gamma` | 0.99 | Discount factor |
| `lambda_trace` | 0.8 | Eligibility trace decay |
| `N` | 10 | Episodes between meta-steps |
| `baseline` | Running mean of r_meta | Variance reduction |

---

## 7. Integration Points

### 7.1 base_discovery.py — PureAgnosticDiscoveryAgent

Required additions:
- `exploration_temperature: float = 1.0` field on `ActionPattern` — modulates softmax temperature in `act()` weight computation
- `recent_use_count: int = 0` field on `ActionPattern` — tracks use in trailing N-episode window (reset each meta-step)
- `run_episode()` method on `PureAgnosticDiscoveryAgent` — wraps a single episode and returns `(reward, steps)` for meta-feature computation
- Hook at episode boundary: after every N episodes, call `meta_pattern.observe_and_act(agent)` if a meta-pattern is attached

### 7.2 population.py — PatternPopulation

- `get_top_patterns(k)` already exists (line 168–170) — no changes needed
- New method `get_population_entropy() -> float` — returns Shannon entropy of weight distribution
- New method `get_diversity() -> float` — returns mean pairwise structural distance

### 7.3 curriculum.py — CurriculumManager

Required additions:
- `advance_phase()` — directly advances `active_pattern_idx` and resets window (extracted from the existing conditional in `update()`)
- `set_difficulty(delta: float)` — clamps `self.difficulty = max(0.0, min(1.0, self.difficulty + delta))`

### 7.4 meta_cognitive_pattern.py — new file

Primary class `MetaCognitivePattern(HPMPattern)` with:
- `observe(agent) -> torch.Tensor` — compute 64-dim meta-feature vector
- `act(meta_features) -> int` — run policy network, return directive index
- `execute_directive(directive, agent, curriculum)` — dispatch to the 8 directive implementations
- `record_transition(meta_features, directive, reward)` — store for REINFORCE update
- `update_policy()` — run REINFORCE update with eligibility traces
- `observe_and_act(agent, curriculum)` — convenience entry point called from training loop

### 7.5 meta_training.py — new file

`MetaTrainingLoop` that:
- Wraps existing training loop
- Calls `agent.run_episode()` N times
- Computes meta-reward
- Calls `meta_pattern.observe_and_act()`
- Calls `meta_pattern.update_policy()`

---

## 8. HPMPattern Abstract Method Implementations

MetaCognitivePattern inherits HPMPattern. Required abstract methods are satisfied as follows:

| Method | Implementation |
|---|---|
| `log_prob(observations)` | Returns log probability of the chosen directive given meta-features extracted from `observations` |
| `sample(context, num_samples)` | Returns sampled directive index and probabilities |
| `intervene(intervention, context)` | Forces a specific directive (for external override / testing) |
| `update_parameters(observations, lr)` | Delegates to `update_policy()` — REINFORCE step |
| `structural_distance(other)` | Returns 0.0 if same class, 1.0 otherwise (L5 patterns are singleton by design) |
| `extract_causal_graph()` | Returns a single-node DiGraph labelled "meta_cognitive" |

---

## 9. Constraints and Design Invariants

1. MetaCognitivePattern operates at episode-level granularity only — it does not intervene within a single episode.
2. There is at most one MetaCognitivePattern per agent (singleton role).
3. Directives are executed deterministically — the meta-pattern selects a directive and it is executed in full before the next episode begins.
4. The meta-pattern's own weight in the population is fixed at 1.0 — it is never subject to replicator dynamics.
5. The meta-pattern never modifies its own policy parameters during directive execution — only during `update_policy()` which runs after the meta-reward is observed.
6. SPAWN_SPECIALIST creates a new agent but does not modify the calling agent's population.

---

## 10. Out of Scope

- Multi-agent meta-coordination (one meta-pattern per agent)
- Hierarchical meta-patterns (no L6)
- Online fine-tuning of the underlying LLM/tool selector
- Replacing SubstrateCompiler — structural compilation remains L2→L3 distillation, separate concern
