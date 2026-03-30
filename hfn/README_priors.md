# Priors in HFN

## What Is a Prior?

A prior is an HFN node that is **registered in the Forest before any observations begin** and **protected from absorption and deregistration** for the lifetime of the agent. It represents structural knowledge the agent arrives with — the equivalent of evolved neural architecture, innate reflexes, or species-level pattern endowment in HPM terms.

Priors are ordinary `HFN` instances (Gaussian identity + DAG polygraph body). What makes them priors is not their type but their status: they are registered with the Forest and their IDs are added to the Observer's `protected_ids` set.

By default (`prior_plasticity=False`) priors are completely static — their geometry never changes. When `prior_plasticity=True`, low-density priors can slowly drift their `mu` toward observations they keep missing, while remaining permanently protected from absorption. See [Graduated Prior Protection](#graduated-prior-protection-prior-plasticity) below.

---

## Protection Mechanics

Once a node's ID is in `protected_ids`, it is exempt from:

- **Weight decay** — the Observer never reduces its weight
- **Miss-count absorption** — consecutive misses do not accumulate against it
- **Hausdorff geometric absorption** — it cannot be absorbed by proximity to another node
- **Deregistration** — `TieredForest.deregister()` silently ignores protected IDs; they are never evicted to cold storage or removed

A prior node therefore has **permanent presence** in the Forest. Every `observe()` call includes it as a candidate explainer, regardless of how many observations it has or hasn't explained.

**Note:** when `prior_plasticity=True`, a prior's `mu` can slowly drift (see below), but absorption protection is **never** lifted. The prior remains in `protected_ids` for its entire lifetime regardless of plasticity settings.

---

## Graduated Prior Protection (Prior Plasticity)

### The Default: Binary Protection

By default (`prior_plasticity=False`), protection is fully binary. A prior is either protected or it is not. Protected priors are exempt from all dynamics forever — their geometry (`mu`, `sigma`) never changes, regardless of how poorly they explain incoming observations. This is appropriate when priors encode truly invariant structural knowledge that should anchor the agent's world model permanently.

### The Plasticity Model (`prior_plasticity=True`)

When `prior_plasticity=True`, priors can undergo **density-guided mu drift**. The core idea comes from HPM Section 2.6: high-density patterns (those that frequently explain observations) resist revision, while low-density patterns (those that keep missing) are eligible for gradual update. Section 2.5.2 further motivates this: forgetting and decay apply to all patterns, priors included — only the rate and mechanism differ.

The mechanism works as follows:

1. During each `observe()` call, the Observer tracks per-prior **hit counts** (times a prior was in the explanation tree) and **miss counts** (times it was not).
2. After weight updates, `_check_prior_plasticity()` evaluates each prior:
   - If `miss_count < prior_revision_threshold`: prior is stable — no drift.
   - If `miss_count >= prior_revision_threshold` AND `hit_rate >= 0.5`: prior explains enough observations — no drift (high density protects it).
   - If `miss_count >= prior_revision_threshold` AND `hit_rate < 0.5`: prior is low-density and persistently missing — **drift is triggered**.
3. Drift: `mu += prior_drift_rate * (x - mu)` — the prior's centre nudges toward the current observation.
4. Hit and miss counts reset after drift, giving the prior a fresh chance to stabilise at its new position.

Absorption protection is **never** affected. A drifting prior remains in `protected_ids` and cannot be absorbed, merged, or deregistered.

### Three Conceptual Tiers

When plasticity is enabled, priors fall into three informal tiers based on their observed density:

| Tier | Condition | Behaviour |
|------|-----------|-----------|
| **Anchored** | High hit rate (≥ 0.5) | Stable — geometry does not change |
| **Plastic** | Low hit rate, misses above threshold | Eligible for drift — mu nudges toward observations |
| **Fragile** | New / recently reset | Accumulating counts — tier not yet determined |

These tiers are dynamic. A prior that drifts to a better-fitting region will accumulate hits and stabilise as Anchored. A prior placed in a region that diverges from the data distribution will remain Plastic and keep drifting slowly.

### Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `prior_plasticity` | `False` | Enable/disable density-guided drift entirely |
| `prior_drift_rate` | `0.01` | Step size for mu update: larger = faster drift, less stable |
| `prior_revision_threshold` | `200` | Consecutive misses before eligibility check: larger = more patient before drifting |

### When to Use Prior Plasticity

Use `prior_plasticity=True` when:
- The agent will operate across **domain shift** — the observation distribution changes over the agent's lifetime and innate priors may no longer be well-placed.
- Priors were seeded with approximate geometry (e.g. random initialisation within a region) and should self-correct during early exposure.
- You want priors to act as **soft attractors** rather than fixed landmarks.

Keep `prior_plasticity=False` (the default) when:
- Priors encode truly invariant structure that must never shift (e.g. hardwired sensory filters, safety constraints).
- The observation distribution is stationary and priors were carefully placed to match it.
- You want fully deterministic, reproducible prior behaviour.

### Code Example

```python
from hfn.tiered_forest import TieredForest
from hfn.observer import Observer

forest = TieredForest()
prior_ids = set()
for node in my_prior_nodes:
    forest.register(node)
    prior_ids.add(node.id)
forest.set_protected(prior_ids)

# Default: fully static priors
obs_static = Observer(forest, tau=tau, protected_ids=prior_ids)

# Plasticity enabled: low-density priors can drift
obs_plastic = Observer(
    forest,
    tau=tau,
    protected_ids=prior_ids,
    prior_plasticity=True,
    prior_drift_rate=0.01,        # small step — slow, stable drift
    prior_revision_threshold=200, # patience before eligibility
)
```

---

### How to Register

```python
from hfn.tiered_forest import TieredForest

forest = TieredForest()

# Register nodes into the forest
prior_ids = set()
for node in my_prior_nodes:
    forest.register(node)
    prior_ids.add(node.id)

# Mark them as protected
forest.set_protected(prior_ids)

# Pass to Observer
obs = Observer(forest, tau=tau, protected_ids=prior_ids, ...)
```

---

## What Priors Encode

Each prior is a Gaussian `N(μ, Σ)` in the observation space. Its **geometry is its knowledge**:

- The mean `μ` defines where in observation space the prior is centred
- The covariance `Σ` defines how broadly it covers that region
- The DAG children (polygraph body) encode any compositional structure

Priors do not carry semantic labels at the HFN level. Any label (e.g. `obj_food`, `gram_root`) is a naming convention in the ID string — the Observer and Forest are indifferent to it. What matters is where `μ` sits relative to observed data.

**Implication:** a prior named `word_animal` will explain food observations if its `μ` is geometrically closer to food vectors than any food prior is. Priors encode geometry, not intent.

---

## Priors as Attractors

Because priors are never absorbed, they function as **permanent attractors** in the agent's observation space:

1. An incoming observation `x` is compared against all active nodes
2. Whichever node best explains `x` (highest accuracy score) wins
3. If a prior wins, no new learned node is created for that observation
4. If no node wins (gap), the query pathway fires and may create a new learned node

The density and placement of priors therefore directly controls:

- **Coverage** — how much of observation space is explained without learning
- **Novelty signal** — how often the gap query fires (sparser priors → more gaps → more learning)
- **Abstraction level** — learned nodes form in the gaps *between* priors; a richer prior library pushes learned structure to higher abstraction levels

---

## Prior Libraries and Learning Rate

An agent with a small prior library (sparse coverage) will:
- Create many learned nodes early (gaps are frequent)
- Show high early learning rate
- Risk noise accumulation (many unstable learned nodes)

An agent with a large prior library (dense coverage) will:
- Create fewer learned nodes (gaps are rare)
- Show lower early learning rate but higher precision
- Form learned nodes at higher abstraction levels (combinations of priors rather than raw observations)

This mirrors the HPM account of how innate priors shape individual learning: a richer endowment produces faster high-level abstraction at the cost of reduced sensitivity to genuinely novel structure.

---

## Empirical Observations (NLP Experiment)

From `hpm_fractal_node/experiments/experiment_nlp.py` with 195 priors over 2500 observations:

- 195 protected priors explained ~50% of all observations
- 2700+ learned nodes explained ~71% (with overlap — one observation can be attributed to multiple nodes)
- Hausdorff distance between learned nodes and the prior cloud: ~1.04 (learned nodes find gaps between priors, not clustering around any single one)
- Several priors drifted from their intended semantic role due to geometric proximity effects (e.g. `word_<start>` became a `family` category predictor because family sentences statistically began sentences most often in the training data)

Key finding: **prior geometry determines prior behaviour**. Naming a node `gram_preposition` does not make it fire on prepositions — its `μ` must actually be close to preposition observation vectors.

---

## Designing a Prior Library

For a domain-specific HFN agent, a prior library should:

1. **Tile the expected observation space** — ensure `μ` vectors cover the regions where real observations will land
2. **Vary abstraction levels** — include priors at multiple levels: fine-grained (individual tokens, specific objects) and coarse-grained (categories, roles, schemas)
3. **Set covariance deliberately** — wide `Σ` = broad attractor, low precision; narrow `Σ` = tight attractor, high selectivity
4. **Leave intentional gaps** — regions without prior coverage become the agent's learning frontier

The NLP world model (`hpm_fractal_node/nlp/nlp_world_model.py`) is an example: it builds priors for individual words, grammatical roles, sentence templates, and semantic object categories — four levels of abstraction — and seeds them with geometry derived from the vocabulary encoding.

---

## Relationship to HPM

In HPM terms, a prior library pre-populates the **pattern field** — the structured environment that selects which patterns get reinforced. Priors are not merely starting weights; they are the fixed landscape against which all learned patterns are measured. An observation that fits a prior well requires no new structure. An observation that fits no prior well demands structural response: the agent must create, absorb, or restructure to restore explanatory coverage.

This makes the prior library the single most important architectural decision for an HFN-based HPM agent — more so than any hyperparameter tuning.
