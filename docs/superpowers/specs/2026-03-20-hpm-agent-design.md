# HPM Learning Agent — Design Specification

**Date:** 2026-03-20
**Author:** Matt Thomson
**Status:** Draft v3 (added PatternStore + ExternalSubstrate)

---

## 1. Overview

This document specifies the architecture for a multi-agent learning system grounded in the Hierarchical Pattern Modelling (HPM) framework (Thomson, v1.25). The system implements HPM's mathematical formulation faithfully, using a modular architecture that starts with concept learning and is designed to extend to sequence prediction and reinforcement learning domains.

**Success criteria:**
1. Agents demonstrably learn and improve over time — patterns stabilise, accuracy increases, hierarchical compression grows
2. Agent behaviour validates HPM's testable predictions: §9.1 (deep vs surface sensitivity), §9.4 (curiosity at intermediate complexity), §9.5 (social field convergence)

---

## 2. Architecture

The system has five layers:

```
+---------------------------+   +---------------------------+
|     External Substrate    |   |       Domain Plugin        |
|  (internet, web, APIs)    |   | (Concept Learning, RL...) |
|  pattern field + source   |   |  generates x_{1:T} streams|
+-------------+-------------+   +-------------+-------------+
              |                               |
              +---------------+---------------+
                              |
+-----------------------------v-----------------------------+
|                           Agent                           |
|   PatternLibrary --> EvaluatorPipeline --> Dynamics       |
|         |          (selectors)        (Meta Pattern Rule) |
|         v                                                 |
|   PatternStore  (in-memory | SQLite | PostgreSQL)         |
+-----------------------------+-----------------------------+
                              |
+-----------------------------v-----------------------------+
|                       Pattern Field                       |
|       (shared social/env structure across agents)         |
|   Modes: Observational | Communicative | Competitive      |
|   Sources: agent population + ExternalSubstrate           |
+-----------------------------+-----------------------------+
                              |
+-----------------------------v-----------------------------+
|                 Metrics & Instrumentation                 |
|            (tracks HPM predictions 9.1, 9.4, 9.5)       |
+-----------------------------------------------------------+
```

**Package structure:**
```
hpm/
  patterns/     # Pattern interface + Gaussian implementation
  evaluators/   # Epistemic, affective, social evaluators
  dynamics/     # Meta Pattern Rule (D5), recombination (App. E)
  field/        # Pattern field (D6), interaction modes
  agents/       # Single agent + multi-agent orchestrator
  domains/      # Domain interface + concept learning
  store/        # PatternStore interface + backends (memory, SQLite, PostgreSQL)
  substrate/    # ExternalSubstrate interface + web/API implementations
  metrics/      # HPM prediction validators
```

---

## 3. Core HPM Engine

### 3.1 Pattern Interface

**Copy semantics (B1):** Patterns are value types. They are never mutated in place after creation. `update()` returns a new Pattern instance. This prevents aliasing when patterns are shared across agents via the PatternField in communicative mode.

Every pattern `h` implements the following Protocol:

```python
class Pattern(Protocol):
    id: str                                           # canonical UUID, assigned at creation

    def log_prob(self, x: np.ndarray) -> float        # -log p(x|h), epistemic loss
    def description_length(self) -> float             # MDL/parameter count for A1 loss term
    def connectivity(self) -> float                   # structural connectivity for density D(h)
    def compress(self) -> float                       # Comp(h), MI between latent layers (A3)
    def update(self, x: np.ndarray) -> 'Pattern'      # returns new updated instance (immutable)
    def recombine(self, other: 'Pattern') -> 'Pattern' # Appendix E operator R
    def is_structurally_valid(self) -> bool           # feasibility constraint for recombination
```

**Symbol disambiguation (B4, M4):**
- `description_length()` maps to `C(h)` in A1 loss term: `L(h|x) = -log p(x|h) + lambda * C(h)`
- `connectivity()` maps to structural connectivity in density: `C_struct(h) = (1/n) sum_{i,j} w_{ij}`
- `is_structurally_valid()` is the feasibility gate for recombination (replaces ambiguous `C(h*)=1` from E6)

**Default implementation — GaussianPattern:**
- Parameterised as `(mu, Sigma, id)` over feature space; `id` is a UUID assigned at construction
- Two-level hierarchy: z^(2) = abstract cluster centroid, z^(1) = instance-level variation
- `description_length()` = number of non-zero parameters (MDL approximation)
- `connectivity()` = mean absolute correlation among feature dimensions (proxy for subpattern linkage)
- `compress()` = variance explained by z^(2) relative to total variance (mutual information proxy)
- `recombine(other)` = convex combination of parent parameters; `is_structurally_valid()` checks resulting Sigma is positive-definite

### 3.2 Evaluator Pipeline

Three evaluators computed per pattern per timestep, mapping directly to Appendix D.

**Epistemic evaluator (D2, D3):**

```
ell_i(t) = -log p_i(x_t | h_i)                      # instantaneous loss
L_i(t)  = (1 - lambda_L) L_i(t-1) + lambda_L ell_i(t)  # EMA running loss, L_i(0) = 0
A_i(t)  = -L_i(t)                                    # accuracy (<=0 always)
E_epi(t) = A_i(t)
```

**Sign convention (B3):** `A_i(t) <= 0` always. The replicator term `eta*(Total_i(t) - Total_bar(t))` operates on *relative* differences — patterns with above-average total score gain weight, those below-average lose weight. This is correct regardless of absolute sign. `Total_bar(t)` acts as a moving baseline. No normalisation needed.

**Affective evaluator (D3, M6):**

Curiosity signal rewarding intermediate complexity per §9.4:

```
Delta_A_i(t) = A_i(t) - A_i(t-1)                    # improvement rate
c_i(t)       = description_length(h_i)               # complexity proxy
novelty(t)   = sigmoid(k * Delta_A_i(t))             # sigmoid of improvement rate
capacity(t)  = 1 - novelty(t)                         # diminishes as mastered
E_aff_i(t)  = novelty(t) * capacity(t) * g(c_i)
```

Where `g(c) = exp(-(c - c_opt)^2 / (2 * sigma_c^2))` is a Gaussian centred on optimal complexity `c_opt`. Peaks at intermediate complexity, falls at extremes — produces the inverted-U prediction of §9.4. Parameters `c_opt`, `sigma_c`, `k` are configurable.

Configurable external reward signal for RL domain:
```
E_aff_i(t) += alpha_r * r_t       # alpha_r = 0 by default
```

**Social evaluator (D3, D6):**
- Observational mode: `E_soc_i(t) = rho * freq_i(t)` (freq defined in §5.1)
- Communicative mode: direct pattern weight initialisation at transfer (§5.1)
- Competitive mode: `E_soc_i(t)` suppressed for out-group patterns, amplified for in-group

**Combined non-epistemic evaluator (D3):**
```
J_i(t) = beta_aff * E_aff_i(t) + gamma_soc * E_soc_i(t)
```
Epistemic term excluded to prevent double-counting (required by D3).

**Total score (D4):**
```
Total_i(t) = A_i(t) + J_i(t)
```

### 3.3 Meta Pattern Rule Dynamics (D5)

Discrete-time replicator dynamics with conflict inhibition:

```
w_i(t+1) = w_i(t)
          + eta*(Total_i(t) - Total_bar(t)) * w_i(t)      # replicator term
          - beta_c * sum_{j!=i} kappa_{ij} * w_i(t) * w_j(t)  # conflict inhibition
```

Where:
- `Total_bar(t) = sum_j w_j(t) * Total_j(t)` — population-average total score
- `eta` — learning rate
- `beta_c` — conflict scale
- `kappa_{ij}` — incompatibility: symmetrised KL divergence between pattern distributions, normalised to [0,1]. Not specified in paper; derived from framework logic.

Post-update: weights renormalised to sum to 1.

**Edge case — empty library (M2):** If all weights would fall below pruning threshold `epsilon`, the single pattern with highest `Total_i(t)` is retained with weight reset to 1.0. Library never drops below 1 pattern.

### 3.4 Pattern Density and Stability (A8)

Distinct symbols used throughout to avoid collision with evaluator weights (M7):

```
C_struct(h) = connectivity(h)   = (1/n) sum_{i,j} w_{ij}   # structural connectivity
E_reinf(h)  = sum_k e_k(h)                                  # evaluator reinforcement
F_field(h)  = lambda_h                                       # field amplification

D(h) = alpha_D * C_struct(h) + beta_D * E_reinf(h) + gamma_D * F_field(h)
S(h) = sigmoid(eta_S * D(h) - delta * L(h | x_{1:T}))
```

High density compensates for moderate epistemic loss — models sticky but inaccurate patterns. `alpha_D`, `beta_D`, `gamma_D`, `eta_S`, `delta` are all distinct from the evaluator weights `beta_aff`, `gamma_soc`.

### 3.5 Structural Recombination (Appendix E) — Phase 3

Recombination is a Phase 3 feature. The Phase 1/2 data flow (§7) excludes it.

```
h* = R(h_a, h_b)
```

**Trigger conditions (B5):** Attempted once every `T_recomb` timesteps (default 100). Candidate pairs sampled proportional to `w_a * w_b` with `kappa_{ab} < kappa_max`. Maximum `N_recomb` attempts per trigger (default 3).

**Insight evaluator (E4, M5):**
```
I(h*) = beta_orig * (alpha_nov * Nov(h*) + alpha_eff * Eff(h*))
```
Where:
- `beta_orig` — scaling constant controlling magnitude of insight boost
- `Nov(h*)` = `1 - max(kappa_{h*,ha}, kappa_{h*,hb})` — structural novelty vs parents
- `Eff(h*)` = `-L_{h*}(t)` on held-out probe set — performance signal

**Entry weight (E5):** `w_{h*}(t_0) = kappa_0 * I(h*)`

**Feasibility gate (E6):** Recombined pattern discarded if `h*.is_structurally_valid()` is False.

**Hierarchical total score (D7):**
```
Total_i_hier(t) = -L_i_hier(t) + beta_comp * Comp_i(t) + J_i(t)
```
Replaces `Total_i(t)` when hierarchical patterns are active (Phase 3+).

### 3.6 Key Hyperparameters

| Parameter | Symbol | Role |
|-----------|--------|------|
| Learning rate | `eta` | Replicator step size (D5) |
| Conflict scale | `beta_c` | Pattern inhibition strength (D5) |
| Affective weight | `beta_aff` | Weight of curiosity/reward in J_i |
| Social weight | `gamma_soc` | Weight of field influence in J_i |
| EMA decay | `lambda_L` | Running loss smoothing (D2) |
| Pruning threshold | `epsilon` | Minimum weight to retain pattern |
| Field influence | `rho` | Frequency amplification scale (D6) |
| Compression weight | `beta_comp` | Weight of compression in hierarchical score (D7) |
| Curiosity sharpness | `k` | Sigmoid sharpness for novelty signal |
| Optimal complexity | `c_opt` | Peak of curiosity Gaussian (§9.4) |
| Complexity bandwidth | `sigma_c` | Width of curiosity Gaussian |
| External reward weight | `alpha_r` | Weight of external reward in E_aff (RL mode) |
| Density weights | `alpha_D, beta_D, gamma_D` | Components of D(h) (A8) |
| Stability weights | `eta_S, delta` | Parameters of S(h) (A8) |
| Recombination period | `T_recomb` | Timesteps between recombination triggers |
| Max recombination attempts | `N_recomb` | Attempts per trigger |
| Insight scale | `beta_orig` | Scaling constant for insight evaluator (E4) |
| Insight novelty weight | `alpha_nov` | Weight of novelty in I(h*) |
| Insight efficacy weight | `alpha_eff` | Weight of performance in I(h*) |
| Entry weight scale | `kappa_0` | Initial weight of recombined pattern (E5) |

### 3.7 PatternStore — Persistent Pattern Substrate

The paper (§2.5.1) treats external symbolic systems as legitimate pattern substrates. A database-backed PatternStore implements this: patterns are not only held in working memory but persist across sessions, accumulate over time, and can be shared across agents.

**Interface:**

```python
class PatternStore(Protocol):
    def save(self, pattern: Pattern, weight: float, agent_id: str) -> None
    def load(self, pattern_id: str) -> tuple[Pattern, float]          # (pattern, weight)
    def query(self, agent_id: str) -> list[tuple[Pattern, float]]     # all patterns for agent
    def query_all(self) -> list[tuple[Pattern, float, str]]           # all agents (for field)
    def delete(self, pattern_id: str) -> None
    def update_weight(self, pattern_id: str, weight: float) -> None
```

Patterns must implement serialisation for persistence:
```python
class Pattern(Protocol):
    ...
    def to_dict(self) -> dict        # serialise to JSON-compatible dict
    @classmethod
    def from_dict(cls, d: dict) -> 'Pattern'   # deserialise
```

**Backends (selectable at runtime):**

| Backend | Use case |
|---------|----------|
| `InMemoryStore` | Default; ephemeral; fast for experiments |
| `SQLiteStore` | Single-machine persistence; no infrastructure required |
| `PostgreSQLStore` | Multi-agent production deployments; concurrent writes |

**PatternLibrary** uses `PatternStore` as its backing store. On agent initialisation, it loads existing patterns from the store (resuming prior learning). On each dynamics step, updated weights are flushed to the store.

**Cross-agent pattern pool:** In communicative and observational modes, a shared `PatternStore` instance acts as a communal substrate — agents can query patterns learned by other agents in prior sessions, not just the current run. This models the paper's notion of patterns persisting in external substrates beyond any individual learner.

### 3.8 ExternalSubstrate — Internet and External Sources

Maps directly to §2.5.1: "external symbolic systems such as spoken and written language, diagrams, maps, mathematical notation and code" and "artefacts and tools, from notebooks and smartphones to software systems."

The ExternalSubstrate plays two roles:

1. **As a Domain plugin** — fetches real-world content as observation streams `x_{1:T}`, grounding agent learning in actual external structure rather than synthetic data
2. **As a PatternField component** — the internet acts as a social pattern field: how frequently a pattern appears in external content influences its `freq_i(t)` score and therefore its social evaluator signal

**Interface:**

```python
class ExternalSubstrate(Protocol):
    def fetch(self, query: str) -> list[np.ndarray]    # retrieve observations matching query
    def field_frequency(self, pattern: Pattern) -> float  # how common is this pattern externally?
    def stream(self) -> Iterator[np.ndarray]            # continuous observation stream
```

**Implementations:**

| Implementation | Source | Use |
|----------------|--------|-----|
| `WebSearchSubstrate` | Search API (e.g. SerpAPI, Brave) | Fetch concept examples from web |
| `WikipediaSubstrate` | Wikipedia API | Structured knowledge as pattern source |
| `RSSSubstrate` | RSS/Atom feeds | Streaming real-world observation source |
| `LocalFileSubstrate` | Local files/dirs | Offline external substrate for testing |

**Field frequency computation:** `field_frequency(pattern)` queries the external substrate for content matching the pattern's generative model and returns a normalised frequency score. This augments the agent-population-based `freq_i(t)` with an external signal:

```
freq_i_total(t) = alpha_int * freq_i_agents(t) + (1 - alpha_int) * field_frequency(h_i)
```

Where `alpha_int` (internal weight, default 0.8) controls the balance between social learning from agents vs external pattern availability. Setting `alpha_int = 1.0` disables external substrate influence on the field.

**Rate limiting and caching:** All external substrate implementations must implement a response cache and configurable rate limits to prevent excessive API calls during training loops.

---

## 4. Concept Learning Domain

### 4.1 Structure

Each concept `c` is defined by:
- **Deep features** — structural invariants (e.g. "has bilateral symmetry", "odd count", "transitive relation")
- **Surface features** — vary across instances (colour, size, orientation, label format)

Observations `x_t = (surface_features, deep_features)` with configurable noise and surface variation.

Directly supports §9.1: perturbations to deep structure should produce larger accuracy drops than matched surface perturbations.

### 4.2 Transfer Probes

Domain generates two test sets automatically:
- **Near transfer**: same deep structure, similar surface
- **Far transfer**: same deep structure, novel surface (measures genuine hierarchical learning, §9.2)

Expertise measured as compression + far-transfer accuracy, not just near-transfer accuracy.

### 4.3 Domain Interface

```python
class Domain(Protocol):
    def observe(self) -> np.ndarray                                    # emit next x_t
    def feature_dim(self) -> int                                       # dimensionality of x
    def deep_perturb(self) -> 'Domain'                                 # structurally altered copy
    def surface_perturb(self) -> 'Domain'                              # surface-altered copy
    def transfer_probe(self, near: bool) -> list[tuple[np.ndarray, int]]  # (x, label) pairs (M8)
```

---

## 5. Multi-Agent Pattern Field

### 5.1 Three Interaction Modes

**freq_i(t) definition (B2):** Patterns identified across agents by canonical UUID. In observational mode:

```
freq_i(t) = sum_{agents a} w_i^a(t)    # sum of weights for pattern uuid_i across all agents
```

Normalised by total weight mass. Only UUID + weight are broadcast — pattern objects not shared between agents.

**Single-agent edge case (M3):** When `n_agents == 1`, `gamma_soc` is automatically set to 0 — social evaluator gated off.

| Mode | Mechanism | Paper mapping |
|------|-----------|---------------|
| **Observational** | `F_i(t) = rho * freq_i(t)` across agent population | D6 exactly |
| **Communicative** | Agent shares copy of `h_i` (new UUID, source ID recorded); recipient initialises with `w = kappa_0 * I(h*)` | App. E5 |
| **Competitive** | Agents have divergent `beta_aff`/`gamma_soc` per group; field amplifies in-group patterns | §9.5 |

### 5.2 Field Quality Metrics

- **Diversity** — entropy of pattern distribution across agents
- **Redundancy** — mean pairwise overlap (`kappa_{ij}`) of stabilised patterns
- **Conflict** — proportion of high-weight incompatible pattern pairs

---

## 6. Metrics & Instrumentation

**§9.1 — Deep vs Surface Sensitivity:**
```
sensitivity_ratio = Delta_accuracy(deep_perturbation) / Delta_accuracy(surface_perturbation)
```
HPM predicts ratio > 1.

**§9.4 — Curiosity at Intermediate Complexity:**
Track `E_aff` vs `description_length(h_i)` across timesteps. HPM predicts inverted-U.

**§9.5 — Social Field Convergence:**
Compare convergence speed and pattern divergence across agent groups under high vs low field quality conditions.

All metrics emitted as structured JSON logs per timestep.

---

## 7. Data Flow — Phase 1/2 (Single Timestep)

1. Domain (or ExternalSubstrate stream) emits `x_t`
2. Each pattern `h_i` computes `ell_i(t) = -log p(x_t | h_i)`
3. EMA updates `L_i(t)` → `A_i(t) = -L_i(t)`
4. AffectiveEvaluator computes `E_aff_i(t)` from `Delta_A_i(t)` and `description_length(h_i)`
5. PatternField provides `freq_i_total(t)` = blend of agent population + ExternalSubstrate frequencies (0 if single-agent, no external substrate) → SocialEvaluator computes `E_soc_i(t)`
6. `J_i(t) = beta_aff * E_aff_i(t) + gamma_soc * E_soc_i(t)`
7. `Total_i(t) = A_i(t) + J_i(t)`
8. Meta Pattern Rule updates `w_i(t+1)` (D5)
9. Floor check: if all `w_i < epsilon`, retain highest-scoring pattern at weight 1.0
10. Prune patterns with `w_i < epsilon`
11. PatternStore flushed with updated weights
12. PatternField updated (broadcast UUID + weight per pattern per agent)
13. Metrics recorded (JSON)

---

## 8. Implementation Phases

**Phase 1 — Single agent, concept learning, in-memory:**
- GaussianPattern, EpistemicEvaluator, AffectiveEvaluator
- Meta Pattern Rule dynamics (no recombination)
- Concept learning domain with deep/surface features and transfer probes
- InMemoryStore (PatternStore default)
- Metrics for §9.1 and §9.4

**Phase 2 — Persistence + external substrate:**
- SQLiteStore backend for PatternStore (cross-session pattern accumulation)
- LocalFileSubstrate for offline external substrate testing
- WebSearchSubstrate or WikipediaSubstrate for live external pattern field
- `alpha_int` blending of agent vs external field frequencies

**Phase 3 — Multi-agent, pattern field:**
- SocialEvaluator + PatternField (observational mode first)
- Multi-agent orchestrator with shared PatternStore (PostgreSQL for concurrent writes)
- Metrics for §9.5
- Communicative and competitive modes

**Phase 4 — Recombination + hierarchical patterns:**
- Full recombination operator (Appendix E) with defined trigger conditions
- Hierarchical total score (D7)
- Pattern density and stability (A8)
- Additional domain plugins (sequence prediction)

---

## 9. Open Questions (paper §6, A.6, D8)

- Evaluator dynamics (how `beta_aff`/`gamma_soc` evolve over time) not yet modelled
- Forgetting and interference not yet modelled
- Developmental staging of representational capacity not yet modelled
- `kappa_{ij}` incompatibility uses symmetrised KL divergence — derived from framework logic, not specified in paper
- Dynamic pattern fields deferred to Phase 3+
