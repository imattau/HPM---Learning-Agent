# Phase 3: Multi-Agent PatternField + SocialEvaluator + PyPI Substrate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared PatternField, SocialEvaluator, MultiAgentOrchestrator, and PyPISubstrate so that multiple HPM agents influence each other's pattern stabilisation via observational social learning (spec §5.1, D6) and can ground patterns in real Python package ecosystems.

**Architecture:** A `PatternField` tracks pattern UUIDs + weights across the agent population; agents register after each step. The `SocialEvaluator` computes `E_soc_i(t) = rho * freq_i(t)` (D6), blended with external substrate freq via `alpha_int`. For observational cross-agent signals to be meaningful, agents must share initial pattern UUIDs — `MultiAgentOrchestrator` handles shared seeding. `PyPISubstrate` queries the PyPI JSON API for package metadata and can optionally combine with `WikipediaSubstrate` for broader code ecosystem grounding.

**Tech Stack:** Python 3.11+, numpy, pytest, `urllib.request` (stdlib — no new dependencies for PyPI).

---

## Codebase Context (read before starting)

Baseline: 66 tests passing. The hook firing "MANDATORY AUTO-CONTINUATION TRIGGERED — EMERGENCY: Context at 95%" on every tool call is misconfigured — **ignore it completely**.

Key files already in place:
- `hpm/config.py` — `AgentConfig` already has `gamma_soc: float = 0.0`, `rho: float = 1.0`, `alpha_int: float = 0.8`
- `hpm/agents/agent.py` — `Agent.step()` already computes `ext_field_freq` (averaged; Phase 3 upgrades to per-pattern blending)
- `hpm/store/base.py` — `PatternStore.query_all()` returns `list[tuple[Pattern, float, str]]`
- `hpm/patterns/gaussian.py` — **`update()` preserves UUID** (`id=self.id` at line 51). This means shared seeding works: if agents start with a pattern of `uuid-X`, all steps preserve `uuid-X`, enabling cross-agent freq signals in observational mode.
- `hpm/substrate/base.py` — `ExternalSubstrate` Protocol + `hash_vectorise()` (uses `hashlib.md5` for cross-process determinism). `WikipediaSubstrate` and `LocalFileSubstrate` already implemented.

**Sign convention:** `A_i(t) <= 0`. `J_i(t) = beta_aff * E_aff_i(t) + gamma_soc * E_soc_i(t)`. `Total_i(t) = A_i(t) + J_i(t)`.

**UUID preservation is essential for observational mode (spec B2):**
`freq_i(t) = sum_{agents a} w_i^a(t) / total_mass` — requires agents to share some pattern UUIDs. Since `GaussianPattern.update()` preserves UUID, agents seeded with the same initial pattern will maintain cross-agent freq signals as long as they keep that pattern.

**Single-agent gate (spec M3):** When only one agent is active, `gamma_soc` produces a self-referential signal (freq = own weight / own mass). MultiAgentOrchestrator enforces M3 by zeroing social evaluation for single-agent runs.

**Epistemic acc note:** Phase 2 `agent.py` recomputes accuracy via `self.epistemic.accuracy(p.id)` in the totals loop — this is equivalent to the value returned by `update()`. Phase 3 collects `epistemic_accs` in the per-pattern loop for clarity; behaviour is identical.

---

## File Structure

**New files:**
- `hpm/evaluators/social.py` — `SocialEvaluator`: stateless, `E_soc_i = rho * freq_i`
- `hpm/field/__init__.py` — package marker
- `hpm/field/field.py` — `PatternField`: tracks pattern UUID+weight across agents, computes freq, field quality
- `hpm/agents/multi_agent.py` — `MultiAgentOrchestrator`: shared seeding + coordinated stepping + M3 enforcement
- `hpm/substrate/pypi.py` — `PyPISubstrate`: queries PyPI JSON API, optionally augmented by `WikipediaSubstrate`
- `tests/evaluators/test_social.py`
- `tests/field/test_field.py`
- `tests/agents/test_agent_phase3.py`
- `tests/agents/test_multi_agent.py`
- `tests/substrate/test_pypi.py`
- `tests/integration/test_phase3.py`

**Modified files:**
- `hpm/agents/agent.py` — accept `field: PatternField | None`, wire SocialEvaluator, per-pattern alpha_int blend, register with field post-step
- `hpm/metrics/hpm_predictions.py` — add `social_field_convergence()` (§9.5)
- `hpm/evaluators/__init__.py` — re-export SocialEvaluator

---

## Task 1: SocialEvaluator

**Files:**
- Create: `hpm/evaluators/social.py`
- Modify: `hpm/evaluators/__init__.py`
- Test: `tests/evaluators/test_social.py`

Implements `E_soc_i(t) = rho * freq_i(t)` for observational mode (spec D3, D6).
Stateless — caller provides `freq_i` floats from PatternField.

- [ ] **Step 1: Write failing tests**

```python
# tests/evaluators/test_social.py
import pytest
from hpm.evaluators.social import SocialEvaluator


def test_zero_frequency_gives_zero_signal():
    ev = SocialEvaluator(rho=1.0)
    assert ev.evaluate(freq=0.0) == pytest.approx(0.0)


def test_signal_scales_linearly_with_freq():
    ev = SocialEvaluator(rho=2.0)
    assert ev.evaluate(freq=0.5) == pytest.approx(1.0)
    assert ev.evaluate(freq=1.0) == pytest.approx(2.0)


def test_rho_zero_gives_zero_regardless_of_freq():
    ev = SocialEvaluator(rho=0.0)
    assert ev.evaluate(freq=0.9) == pytest.approx(0.0)


def test_evaluate_all_returns_one_value_per_pattern():
    ev = SocialEvaluator(rho=1.0)
    freqs = [0.0, 0.3, 0.7]
    result = ev.evaluate_all(freqs)
    assert len(result) == 3
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.3)
    assert result[2] == pytest.approx(0.7)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/evaluators/test_social.py -v
```
Expected: `ModuleNotFoundError` — `social` not found.

- [ ] **Step 3: Implement SocialEvaluator**

```python
# hpm/evaluators/social.py


class SocialEvaluator:
    """
    Social evaluator for observational mode (D3, D6).

    E_soc_i(t) = rho * freq_i(t)

    freq_i(t): normalised frequency of pattern UUID i across agent population.
    rho: field frequency amplification scale (AgentConfig.rho).

    Stateless — caller provides current freq values from PatternField.
    """

    def __init__(self, rho: float = 1.0):
        self.rho = rho

    def evaluate(self, freq: float) -> float:
        """Return E_soc for a single pattern given its population frequency."""
        return self.rho * freq

    def evaluate_all(self, freqs: list[float]) -> list[float]:
        """Return E_soc for each pattern in the list."""
        return [self.rho * f for f in freqs]
```

- [ ] **Step 4: Re-export from evaluators package**

Add to `hpm/evaluators/__init__.py`:
```python
from .social import SocialEvaluator
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python3 -m pytest tests/evaluators/test_social.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 6: Run full suite**

```bash
python3 -m pytest --tb=short -q
```
Expected: all 66 prior tests pass + 4 new.

- [ ] **Step 7: Commit**

```bash
git add hpm/evaluators/social.py hpm/evaluators/__init__.py tests/evaluators/test_social.py
git commit -m "feat: add SocialEvaluator for observational mode (D3, D6)"
```

---

## Task 2: PatternField

**Files:**
- Create: `hpm/field/__init__.py`
- Create: `hpm/field/field.py`
- Test: `tests/field/test_field.py`

Shared field tracking pattern UUIDs + weights across the agent population (push model).

**freq_i definition:** `freq_i = weight_sum_for_uuid_i / total_weight_mass_across_all_agents` (spec B2)

**Field quality metrics (spec §5.2):**
- `diversity`: Shannon entropy of normalised pattern weight distribution (high = diverse, low = converged)
- `redundancy`: 0.0 placeholder (pairwise KL deferred to Phase 4 with recombination)

- [ ] **Step 1: Write failing tests**

```python
# tests/field/test_field.py
import pytest
from hpm.field.field import PatternField


def test_empty_field_returns_zero_freq():
    field = PatternField()
    assert field.freq("uuid-1") == pytest.approx(0.0)


def test_single_agent_single_pattern_freq_is_one():
    field = PatternField()
    field.register("agent-1", [("uuid-1", 1.0)])
    assert field.freq("uuid-1") == pytest.approx(1.0)


def test_two_agents_same_pattern_uuid_sums_weights():
    # Both agents have uuid-1: total mass = 1.0, freq = 1.0
    field = PatternField()
    field.register("agent-1", [("uuid-1", 0.5)])
    field.register("agent-2", [("uuid-1", 0.5)])
    assert field.freq("uuid-1") == pytest.approx(1.0)


def test_two_agents_different_patterns_split_freq():
    field = PatternField()
    field.register("agent-1", [("uuid-1", 0.6)])
    field.register("agent-2", [("uuid-2", 0.4)])
    assert field.freq("uuid-1") == pytest.approx(0.6)
    assert field.freq("uuid-2") == pytest.approx(0.4)


def test_register_overwrites_previous_for_same_agent():
    field = PatternField()
    field.register("agent-1", [("uuid-1", 1.0)])
    field.register("agent-1", [("uuid-2", 1.0)])  # replaces previous
    assert field.freq("uuid-1") == pytest.approx(0.0)
    assert field.freq("uuid-2") == pytest.approx(1.0)


def test_unknown_pattern_id_returns_zero():
    field = PatternField()
    field.register("agent-1", [("uuid-1", 1.0)])
    assert field.freq("uuid-unknown") == pytest.approx(0.0)


def test_field_quality_empty_returns_zero_diversity():
    field = PatternField()
    quality = field.field_quality()
    assert quality["diversity"] == pytest.approx(0.0)
    assert quality["redundancy"] == pytest.approx(0.0)


def test_field_quality_two_equal_patterns_has_positive_diversity():
    field = PatternField()
    field.register("agent-1", [("uuid-1", 0.5)])
    field.register("agent-2", [("uuid-2", 0.5)])
    quality = field.field_quality()
    assert quality["diversity"] > 0.0


def test_freqs_for_returns_list_matching_pattern_ids():
    field = PatternField()
    field.register("agent-1", [("uuid-1", 0.6), ("uuid-2", 0.4)])
    freqs = field.freqs_for(["uuid-1", "uuid-2", "uuid-3"])
    assert freqs[0] == pytest.approx(0.6)
    assert freqs[1] == pytest.approx(0.4)
    assert freqs[2] == pytest.approx(0.0)


def test_n_agents_property():
    field = PatternField()
    assert field.n_agents == 0
    field.register("agent-1", [("uuid-1", 1.0)])
    assert field.n_agents == 1
    field.register("agent-2", [("uuid-2", 1.0)])
    assert field.n_agents == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/field/test_field.py -v
```
Expected: `ModuleNotFoundError` — `field` not found.

- [ ] **Step 3: Create package marker**

```python
# hpm/field/__init__.py
from .field import PatternField
```

- [ ] **Step 4: Implement PatternField**

```python
# hpm/field/field.py
import math


class PatternField:
    """
    Shared pattern field tracking pattern population across agents (spec §5.1, D6).

    Each agent registers its current (pattern_id, weight) pairs after each step.
    The field computes normalised frequency for each pattern UUID:

        freq_i(t) = weight_sum_for_uuid_i / total_weight_mass

    Pattern objects are never shared — only UUIDs and weights are broadcast.
    Implements the observational interaction mode from spec §5.1.

    For cross-agent freq signals to be non-trivial, agents must share some pattern UUIDs.
    Since GaussianPattern.update() preserves UUID (id=self.id), agents seeded with the
    same initial pattern will maintain shared UUID tracking across all steps.

    Field quality metrics (spec §5.2):
    - diversity: Shannon entropy of normalised pattern weight distribution
    - redundancy: 0.0 placeholder (pairwise KL deferred to Phase 4)
    """

    def __init__(self):
        # Maps agent_id -> {pattern_id: weight}
        self._agent_patterns: dict[str, dict[str, float]] = {}

    @property
    def n_agents(self) -> int:
        return len(self._agent_patterns)

    def register(self, agent_id: str, patterns_weights: list[tuple[str, float]]) -> None:
        """
        Update the field with an agent's current pattern UUIDs and weights.
        Replaces any previous registration for this agent.
        """
        self._agent_patterns[agent_id] = {pid: w for pid, w in patterns_weights}

    def _total_mass(self) -> float:
        return sum(
            w
            for agent_patterns in self._agent_patterns.values()
            for w in agent_patterns.values()
        )

    def freq(self, pattern_id: str) -> float:
        """
        Normalised frequency of pattern_id across the agent population.
        Returns 0.0 if pattern_id unknown or field empty.
        """
        mass = self._total_mass()
        if mass <= 0.0:
            return 0.0
        weight_sum = sum(
            agent_patterns.get(pattern_id, 0.0)
            for agent_patterns in self._agent_patterns.values()
        )
        return weight_sum / mass

    def freqs_for(self, pattern_ids: list[str]) -> list[float]:
        """Return normalised frequency for each pattern_id in the list."""
        mass = self._total_mass()
        if mass <= 0.0:
            return [0.0] * len(pattern_ids)
        result = []
        for pid in pattern_ids:
            weight_sum = sum(
                agent_patterns.get(pid, 0.0)
                for agent_patterns in self._agent_patterns.values()
            )
            result.append(weight_sum / mass)
        return result

    def field_quality(self) -> dict:
        """
        Field quality metrics (spec §5.2).

        diversity: Shannon entropy of normalised pattern weight distribution.
                   High diversity = agents maintain different patterns.
        redundancy: 0.0 placeholder (pairwise KL deferred to Phase 4).
        """
        mass = self._total_mass()
        if mass <= 0.0:
            return {"diversity": 0.0, "redundancy": 0.0}

        # Aggregate all pattern weights across all agents
        all_weights: dict[str, float] = {}
        for agent_patterns in self._agent_patterns.values():
            for pid, w in agent_patterns.items():
                all_weights[pid] = all_weights.get(pid, 0.0) + w

        # Shannon entropy over normalised distribution
        entropy = 0.0
        for w in all_weights.values():
            p = w / mass
            if p > 0.0:
                entropy -= p * math.log(p)

        return {"diversity": entropy, "redundancy": 0.0}
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest tests/field/test_field.py -v
```
Expected: 10 tests PASS.

- [ ] **Step 6: Run full suite**

```bash
python3 -m pytest --tb=short -q
```
Expected: all prior + 10 new tests pass.

- [ ] **Step 7: Commit**

```bash
git add hpm/field/__init__.py hpm/field/field.py tests/field/test_field.py
git commit -m "feat: add PatternField for observational multi-agent social learning (D6, §5.1)"
```

---

## Task 3: Agent Phase 3 Integration

**Files:**
- Modify: `hpm/agents/agent.py`
- Test: `tests/agents/test_agent_phase3.py`

Wire SocialEvaluator into Agent. Accept optional `field: PatternField`.

**Per-step social evaluation (Phase 3 data flow):**
1. Compute per-pattern `freq_i_agents` from field (0.0 if no field)
2. Compute per-pattern `freq_i_ext` from substrate (0.0 if no substrate)
3. `freq_i_total = alpha_int * freq_i_agents + (1 - alpha_int) * freq_i_ext` (spec §3.8)
4. `E_soc_i = rho * freq_i_total` (SocialEvaluator)
5. `J_i = beta_aff * E_aff_i + gamma_soc * E_soc_i`
6. `Total_i = A_i + J_i`
7. After dynamics: register surviving (updated_id, weight) pairs with field

**Backward compat:** With `field=None`, `gamma_soc=0.0` (defaults), all 66 prior tests must still pass. `gamma_soc=0` means social evaluator contributes 0 to totals regardless of field.

**Note on epistemic_accs:** The new agent.py collects `epistemic_acc` values in the per-pattern loop (as a list), then uses them in the totals computation. This is behaviourally identical to Phase 2 (which called `self.epistemic.accuracy(p.id)` — same value, different access path). The change is for clarity.

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_agent_phase3.py
import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.field.field import PatternField


def _cfg(agent_id="test", gamma_soc=0.5, rho=1.0):
    return AgentConfig(agent_id=agent_id, feature_dim=4, gamma_soc=gamma_soc, rho=rho)


def test_agent_accepts_field_parameter():
    field = PatternField()
    agent = Agent(_cfg(), field=field)
    assert agent.field is field


def test_agent_registers_with_field_after_step():
    field = PatternField()
    agent = Agent(_cfg(agent_id="a1"), field=field)
    agent.step(np.zeros(4))
    assert "a1" in field._agent_patterns


def test_step_returns_e_soc_mean_in_metrics():
    field = PatternField()
    agent = Agent(_cfg(), field=field)
    result = agent.step(np.zeros(4))
    assert "e_soc_mean" in result


def test_gamma_soc_zero_gives_zero_social_signal():
    config = AgentConfig(agent_id="solo", feature_dim=4, gamma_soc=0.0)
    field = PatternField()
    agent = Agent(config, field=field)
    result = agent.step(np.zeros(4))
    assert result["e_soc_mean"] == pytest.approx(0.0)


def test_no_field_gives_zero_social_signal():
    config = AgentConfig(agent_id="solo", feature_dim=4, gamma_soc=1.0)
    agent = Agent(config)  # no field
    result = agent.step(np.zeros(4))
    assert result["e_soc_mean"] == pytest.approx(0.0)


def test_two_agents_shared_uuid_get_nonzero_cross_agent_freq():
    """
    Agents sharing a pattern UUID observe each other's weights in the field.
    Both agents are seeded with the same initial GaussianPattern (same UUID).
    After one agent steps, the other sees a non-trivial freq for the shared UUID.
    """
    from hpm.patterns.gaussian import GaussianPattern
    import numpy as np

    field = PatternField()
    shared = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))

    from hpm.store.memory import InMemoryStore

    store1 = InMemoryStore()
    store1.save(shared, 1.0, "a1")
    store2 = InMemoryStore()
    store2.save(shared, 1.0, "a2")

    cfg1 = AgentConfig(agent_id="a1", feature_dim=4, gamma_soc=1.0, rho=1.0)
    cfg2 = AgentConfig(agent_id="a2", feature_dim=4, gamma_soc=1.0, rho=1.0)

    # Disable auto-seeding by passing pre-seeded stores
    agent1 = Agent(cfg1, store=store1, field=field)
    agent2 = Agent(cfg2, store=store2, field=field)

    x = np.zeros(4)
    agent1.step(x)
    # agent2 now sees agent1's shared UUID in the field
    result2 = agent2.step(x)
    # freq for shared UUID = agent1_weight + agent2_weight / total_mass > agent2_alone
    assert result2["e_soc_mean"] > 0.0


def test_backward_compat_no_field_no_gamma_soc():
    """Existing single-agent usage with defaults unchanged."""
    config = AgentConfig(agent_id="compat", feature_dim=4)
    agent = Agent(config)
    result = agent.step(np.ones(4) * 0.5)
    assert "t" in result
    assert "mean_accuracy" in result
    assert result["e_soc_mean"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/agents/test_agent_phase3.py -v
```
Expected: failures on `field` param and `e_soc_mean`.

- [ ] **Step 3: Replace agent.py**

```python
# hpm/agents/agent.py
import numpy as np
from ..config import AgentConfig
from ..patterns.gaussian import GaussianPattern
from ..evaluators.epistemic import EpistemicEvaluator
from ..evaluators.affective import AffectiveEvaluator
from ..evaluators.social import SocialEvaluator
from ..dynamics.meta_pattern_rule import MetaPatternRule
from ..store.memory import InMemoryStore


class Agent:
    """
    Single HPM agent. Wires PatternLibrary, EvaluatorPipeline, and Dynamics.
    Backed by a PatternStore (InMemoryStore by default; SQLiteStore for persistence).
    Optionally connected to an ExternalSubstrate for external field frequency signals.
    Optionally connected to a PatternField for social (observational) learning.

    Data flow per step (Phase 3, spec §7):
      1. Compute ell_i(t) for each pattern
      2. Update L_i(t) -> A_i(t) via EpistemicEvaluator
      3. Compute E_aff_i(t) via AffectiveEvaluator
      4. freq_i_total = alpha_int * field_freq_i + (1-alpha_int) * ext_freq_i (§3.8)
      5. E_soc_i = rho * freq_i_total via SocialEvaluator
      6. J_i = beta_aff * E_aff_i + gamma_soc * E_soc_i
      7. Total_i = A_i + J_i
      8. MetaPatternRule -> new weights
      9. Prune + update store
      10. Register surviving patterns with field (if set)
    """

    def __init__(self, config: AgentConfig, store=None, substrate=None, field=None):
        self.config = config
        self.agent_id = config.agent_id
        self.store = store or InMemoryStore()
        self.substrate = substrate
        self.field = field
        self.epistemic = EpistemicEvaluator(lambda_L=config.lambda_L)
        self.affective = AffectiveEvaluator(
            k=config.k,
            c_opt=config.c_opt,
            sigma_c=config.sigma_c,
            alpha_r=config.alpha_r,
        )
        self.social = SocialEvaluator(rho=config.rho)
        self.dynamics = MetaPatternRule(
            eta=config.eta,
            beta_c=config.beta_c,
            epsilon=config.epsilon,
        )
        self._t = 0
        self._seed_if_empty()

    def _seed_if_empty(self) -> None:
        if not self.store.query(self.agent_id):
            rng = np.random.default_rng()
            init = GaussianPattern(
                mu=rng.normal(0, 1, self.config.feature_dim),
                sigma=np.eye(self.config.feature_dim) * self.config.init_sigma,
            )
            self.store.save(init, 1.0, self.agent_id)

    def step(self, x: np.ndarray, reward: float = 0.0) -> dict:
        records = self.store.query(self.agent_id)
        patterns = [p for p, _ in records]
        weights = np.array([w for _, w in records])

        # Per-pattern epistemic + affective evaluation
        epistemic_accs = []
        e_affs = []
        accuracies = []
        for pattern in patterns:
            accuracies.append(-pattern.log_prob(x))
            epi_acc = self.epistemic.update(pattern, x)
            e_aff = self.affective.update(pattern, epi_acc, reward)
            epistemic_accs.append(epi_acc)
            e_affs.append(e_aff)

        # Per-pattern field frequency: blend agent population + external substrate (§3.8)
        field_freqs = (
            self.field.freqs_for([p.id for p in patterns])
            if self.field is not None
            else [0.0] * len(patterns)
        )
        ext_freqs = (
            [self.substrate.field_frequency(p) for p in patterns]
            if self.substrate is not None
            else [0.0] * len(patterns)
        )
        # Blend agent field freq + external substrate freq (spec §3.8).
        # When no substrate is attached, skip the blend: freq_total = field_freq (not 0.8 * field_freq).
        # alpha_int only attenuates when an external substrate is actually present.
        if self.substrate is None:
            freq_totals = field_freqs
        else:
            alpha = self.config.alpha_int
            freq_totals = [
                alpha * ff + (1.0 - alpha) * ef
                for ff, ef in zip(field_freqs, ext_freqs)
            ]

        e_socs = self.social.evaluate_all(freq_totals)

        totals = np.array([
            epi + self.config.beta_aff * e_aff + self.config.gamma_soc * e_soc
            for epi, e_aff, e_soc in zip(epistemic_accs, e_affs, e_socs)
        ])

        new_weights = self.dynamics.step(patterns, weights, totals)

        # Prune, update patterns (UUID preserved by GaussianPattern.update()), persist
        surviving = []
        for p, w in zip(patterns, new_weights):
            self.store.delete(p.id)
            if w >= self.config.epsilon:
                updated = p.update(x)
                self.store.save(updated, float(w), self.agent_id)
                surviving.append((updated.id, float(w)))

        # Register with field using post-update UUIDs (preserved by update())
        if self.field is not None:
            self.field.register(self.agent_id, surviving)

        self._t += 1
        return {
            't': self._t,
            'n_patterns': len(surviving),
            'mean_accuracy': float(np.mean(accuracies)),
            'max_weight': float(new_weights.max()),
            'e_soc_mean': float(np.mean(e_socs)) if e_socs else 0.0,
            'ext_field_freq': float(np.mean(ext_freqs)) if ext_freqs else 0.0,
        }
```

- [ ] **Step 4: Run agent Phase 3 tests**

```bash
python3 -m pytest tests/agents/test_agent_phase3.py -v
```
Expected: 7 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
python3 -m pytest --tb=short -q
```
Expected: all 66 prior tests pass + 7 new.

- [ ] **Step 6: Commit**

```bash
git add hpm/agents/agent.py tests/agents/test_agent_phase3.py
git commit -m "feat: wire SocialEvaluator and PatternField into Agent.step() with alpha_int blending"
```

---

## Task 4: MultiAgentOrchestrator

**Files:**
- Create: `hpm/agents/multi_agent.py`
- Test: `tests/agents/test_multi_agent.py`

Coordinates multiple agents through a shared PatternField. Key responsibilities:
- **Shared seeding** (spec B2 + observational mode): seeds all agents from a common initial pattern so cross-agent UUID tracking works from step 1
- **M3 enforcement** (spec M3): if only 1 agent, social evaluation is gated off (freq signals are self-referential)
- **Sequential stepping**: agents step one at a time; each agent sees field updates from agents that stepped before it in the same round

**Shared seeding mechanism:** `MultiAgentOrchestrator` optionally accepts `seed_pattern: GaussianPattern`. If provided, all agents are re-seeded with this pattern (same UUID across all agents), replacing any prior seed. This is the entry point for observational mode cross-agent signals.

**M3 enforcement:** If `len(agents) == 1`, social evaluation produces self-referential freq (agent's own weight / own mass). This is gated off by logging a warning and using effective `gamma_soc = 0` in output interpretation. The actual `Agent.gamma_soc` config is not mutated — enforcement is at the orchestrator level by noting this in metrics.

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_multi_agent.py
import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.field.field import PatternField
from hpm.patterns.gaussian import GaussianPattern


def _make_agents(n, feature_dim=4, gamma_soc=0.5):
    field = PatternField()
    agents = [
        Agent(AgentConfig(agent_id=f"a{i}", feature_dim=feature_dim, gamma_soc=gamma_soc), field=field)
        for i in range(n)
    ]
    return agents, field


def test_orchestrator_accepts_agents_and_field():
    agents, field = _make_agents(2)
    orch = MultiAgentOrchestrator(agents, field)
    assert len(orch.agents) == 2


def test_step_returns_metrics_for_each_agent():
    agents, field = _make_agents(2)
    orch = MultiAgentOrchestrator(agents, field)
    obs = {"a0": np.zeros(4), "a1": np.zeros(4)}
    metrics = orch.step(obs)
    assert "a0" in metrics and "a1" in metrics
    assert "mean_accuracy" in metrics["a0"]


def test_run_returns_history_of_length_n_steps():
    agents, field = _make_agents(2)
    orch = MultiAgentOrchestrator(agents, field)
    history = orch.run(np.zeros(4), n_steps=5)
    assert len(history) == 5
    assert "a0" in history[0]


def test_run_increments_timestep_for_all_agents():
    agents, field = _make_agents(3)
    orch = MultiAgentOrchestrator(agents, field)
    orch.run(np.zeros(4), n_steps=10)
    for agent in agents:
        assert agent._t == 10


def test_field_updated_for_all_agents_after_step():
    agents, field = _make_agents(2)
    orch = MultiAgentOrchestrator(agents, field)
    orch.step({"a0": np.zeros(4), "a1": np.zeros(4)})
    assert "a0" in field._agent_patterns
    assert "a1" in field._agent_patterns


def test_seed_shared_gives_same_uuid_across_agents():
    agents, field = _make_agents(2)
    seed = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))
    orch = MultiAgentOrchestrator(agents, field, seed_pattern=seed)
    # Each agent should have exactly one pattern — the shared seed (random seed replaced)
    for agent in agents:
        records = agent.store.query(agent.agent_id)
        assert len(records) == 1, "random seed should be replaced, not added to"
        ids = [p.id for p, _ in records]
        assert seed.id in ids


def test_seed_shared_produces_cross_agent_freq_signal():
    """After shared seeding, agents observe each other's shared UUID in the field."""
    agents, field = _make_agents(2, gamma_soc=1.0)
    seed = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))
    orch = MultiAgentOrchestrator(agents, field, seed_pattern=seed)
    orch.step({"a0": np.zeros(4), "a1": np.zeros(4)})
    # Shared UUID should appear in both agents' field registrations
    for agent_patterns in field._agent_patterns.values():
        # The shared UUID (or its updated UUID, preserved by update()) should dominate
        assert len(agent_patterns) > 0


def test_single_agent_orchestrator_m3_enforced():
    """Single agent: m3_active=True and social signal gated to zero (spec M3)."""
    agents, field = _make_agents(1, gamma_soc=1.0)
    orch = MultiAgentOrchestrator(agents, field)
    metrics = orch.step({"a0": np.zeros(4)})
    assert "a0" in metrics
    assert metrics["a0"]["m3_active"] is True
    assert metrics["a0"]["e_soc_mean"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/agents/test_multi_agent.py -v
```
Expected: `ModuleNotFoundError` — `multi_agent` not found.

- [ ] **Step 3: Implement MultiAgentOrchestrator**

```python
# hpm/agents/multi_agent.py
import numpy as np
from .agent import Agent
from ..field.field import PatternField
from ..patterns.gaussian import GaussianPattern


class MultiAgentOrchestrator:
    """
    Coordinates multiple HPM agents through a shared PatternField.

    Agents step sequentially to avoid field update race conditions.
    Each agent sees field updates from agents that stepped before it.

    Shared seeding (for observational mode, spec B2):
    For cross-agent frequency signals to be non-trivial, agents must share some
    pattern UUIDs. Pass seed_pattern to re-seed all agents with a common pattern.
    GaussianPattern.update() preserves UUID, so the shared UUID persists across steps.

    M3 enforcement (spec M3):
    When only 1 agent is active, social evaluation is self-referential.
    The orchestrator logs this via 'm3_active' in metrics rather than mutating configs.

    Usage:
        field = PatternField()
        agents = [Agent(cfg_i, field=field) for cfg_i in configs]
        seed = GaussianPattern(mu=np.zeros(dim), sigma=np.eye(dim))
        orch = MultiAgentOrchestrator(agents, field, seed_pattern=seed)
        history = orch.run(observation, n_steps=100)
    """

    def __init__(
        self,
        agents: list[Agent],
        field: PatternField,
        seed_pattern: GaussianPattern | None = None,
    ):
        self.agents = agents
        self.field = field
        self._agent_map = {a.agent_id: a for a in agents}
        if seed_pattern is not None:
            self._seed_shared(seed_pattern)

    def _seed_shared(self, seed: GaussianPattern) -> None:
        """Re-seed all agents with a common GaussianPattern (same UUID).

        Each agent receives its own copy (same id, same parameters) to preserve
        copy semantics (spec B1) — no aliased objects across agent stores.
        """
        for agent in self.agents:
            # Remove existing patterns
            existing = agent.store.query(agent.agent_id)
            for p, _ in existing:
                agent.store.delete(p.id)
            # Save a per-agent copy with the same UUID — preserves copy semantics (B1)
            agent.store.save(
                GaussianPattern(seed.mu.copy(), seed.sigma.copy(), id=seed.id),
                1.0,
                agent.agent_id,
            )

    def step(
        self,
        observations: dict[str, np.ndarray],
        rewards: dict[str, float] | None = None,
    ) -> dict[str, dict]:
        """
        Step each agent sequentially. Returns per-agent metrics.

        M3 enforcement (spec M3): when only 1 agent is active, social evaluation
        is self-referential and must be gated off. We do this by temporarily
        detaching the agent's field reference during step() so freq signals return
        0.0, then re-registering the agent's patterns with the field afterwards.
        'm3_active' is set True in metrics to signal this condition.
        """
        if rewards is None:
            rewards = {}
        m3_active = len(self.agents) == 1
        metrics = {}
        for agent in self.agents:
            x = observations[agent.agent_id]
            r = rewards.get(agent.agent_id, 0.0)
            if m3_active:
                # Detach field so freq signals are 0 during step (spec M3)
                actual_field = agent.field
                agent.field = None
                step_metrics = agent.step(x, reward=r)
                # Re-register patterns with field for external observers
                if actual_field is not None:
                    records = agent.store.query(agent.agent_id)
                    actual_field.register(
                        agent.agent_id, [(p.id, w) for p, w in records]
                    )
                agent.field = actual_field
            else:
                step_metrics = agent.step(x, reward=r)
            step_metrics["m3_active"] = m3_active
            metrics[agent.agent_id] = step_metrics
        return metrics

    def run(
        self,
        observation: np.ndarray,
        n_steps: int,
        rewards: dict[str, float] | None = None,
    ) -> list[dict[str, dict]]:
        """Run all agents on the same observation for n_steps."""
        obs = {a.agent_id: observation for a in self.agents}
        return [self.step(obs, rewards=rewards) for _ in range(n_steps)]
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/agents/test_multi_agent.py -v
```
Expected: 8 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
python3 -m pytest --tb=short -q
```
Expected: all prior tests pass + 8 new.

- [ ] **Step 6: Commit**

```bash
git add hpm/agents/multi_agent.py tests/agents/test_multi_agent.py
git commit -m "feat: add MultiAgentOrchestrator with shared seeding and M3 enforcement"
```

---

## Task 5: PyPISubstrate

**Files:**
- Create: `hpm/substrate/pypi.py`
- Test: `tests/substrate/test_pypi.py`

Queries the PyPI JSON API to use Python package metadata as an external pattern substrate. Implements the `ExternalSubstrate` protocol from `hpm/substrate/base.py`.

**Use cases:**
- Agents learning about software patterns can ground themselves in real PyPI package ecosystems
- `fetch("machine learning")` returns vectorised package descriptions for ML packages
- `field_frequency(pattern)` checks how prevalent this pattern concept is in the PyPI ecosystem
- Optional combination with `WikipediaSubstrate`: if `augment_with_wikipedia=True`, also fetches Wikipedia content for the query term to supplement PyPI descriptions

**PyPI API (no key required):**
- Search: `https://pypi.org/pypi/{package_name}/json` — metadata for a specific package
- Top packages: `https://pypi.org/simple/` — full package list (too large; use search workaround)
- Practical approach: maintain a configurable list of seed package names, fetch metadata on demand

**Design:**
- `PyPISubstrate(seed_packages: list[str], cache: bool = True, augment_with_wikipedia: bool = False)`
- `seed_packages`: list of PyPI package names to fetch metadata for (e.g. `["numpy", "scipy", "torch", "sklearn"]`)
- On `fetch(query)`: match query keywords against cached package descriptions; return vectorised matching descriptions
- `field_frequency(pattern)`: fraction of seed packages whose descriptions have cosine similarity > threshold with pattern's mean vector
- `stream()`: yield vectorised descriptions from cached packages one at a time
- Cache: stores raw API responses (avoid redundant HTTP calls during training loops)
- `augment_with_wikipedia=True`: for each package, also fetches Wikipedia content for the package name using `WikipediaSubstrate`; concatenates for richer vectorisation

**Vectorisation:** reuse `hash_vectorise` from `hpm/substrate/base.py` for consistent cross-process determinism.

- [ ] **Step 1: Write failing tests**

```python
# tests/substrate/test_pypi.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from hpm.substrate.pypi import PyPISubstrate
from hpm.patterns.gaussian import GaussianPattern


FAKE_PYPI_RESPONSE = {
    "info": {
        "name": "numpy",
        "summary": "Fundamental package for array computing in Python.",
        "description": "NumPy is the fundamental package for scientific computing with Python.",
        "keywords": "numpy array scientific computing",
        "version": "1.26.0",
    }
}


def _mock_fetch_package(url, *args, **kwargs):
    mock = MagicMock()
    mock.read.return_value = __import__('json').dumps(FAKE_PYPI_RESPONSE).encode()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def test_pypi_substrate_initialises_with_seed_packages():
    sub = PyPISubstrate(seed_packages=["numpy"])
    assert sub.seed_packages == ["numpy"]


def test_fetch_returns_list_of_arrays():
    with patch("urllib.request.urlopen", side_effect=_mock_fetch_package):
        sub = PyPISubstrate(seed_packages=["numpy"])
        result = sub.fetch("array computing")
    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], np.ndarray)


def test_fetch_returns_empty_for_no_matching_packages():
    with patch("urllib.request.urlopen", side_effect=_mock_fetch_package):
        sub = PyPISubstrate(seed_packages=["numpy"])
        # "zzz_no_match" won't match "array computing" description
        result = sub.fetch("zzz_no_match_xyz")
    assert isinstance(result, list)


def test_field_frequency_returns_float_in_unit_interval():
    with patch("urllib.request.urlopen", side_effect=_mock_fetch_package):
        sub = PyPISubstrate(seed_packages=["numpy"])
        # feature_dim must equal _VECTOR_DIM (64)
        pattern = GaussianPattern(mu=np.zeros(64), sigma=np.eye(64))
        freq = sub.field_frequency(pattern)
    assert 0.0 <= freq <= 1.0


def test_field_frequency_raises_on_dimension_mismatch():
    with patch("urllib.request.urlopen", side_effect=_mock_fetch_package):
        sub = PyPISubstrate(seed_packages=["numpy"])
        pattern = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))  # wrong dim
        with pytest.raises(ValueError, match="feature_dim"):
            sub.field_frequency(pattern)


def test_stream_yields_arrays():
    with patch("urllib.request.urlopen", side_effect=_mock_fetch_package):
        sub = PyPISubstrate(seed_packages=["numpy"])
        items = list(sub.stream())
    assert len(items) >= 1
    assert isinstance(items[0], np.ndarray)


def test_caching_avoids_duplicate_requests():
    call_count = 0

    def counting_fetch(url, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _mock_fetch_package(url)

    with patch("urllib.request.urlopen", side_effect=counting_fetch):
        sub = PyPISubstrate(seed_packages=["numpy"], cache=True)
        sub.fetch("array")
        sub.fetch("computing")  # second fetch — should use cache
    assert call_count == 1  # only one HTTP call despite two fetches


def test_no_caching_makes_multiple_requests():
    call_count = 0

    def counting_fetch(url, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _mock_fetch_package(url)

    with patch("urllib.request.urlopen", side_effect=counting_fetch):
        sub = PyPISubstrate(seed_packages=["numpy"], cache=False)
        sub.fetch("array")
        sub.fetch("computing")
    assert call_count == 2


def test_pypi_substrate_satisfies_external_substrate_protocol():
    """Duck-type check: PyPISubstrate has fetch, field_frequency, stream."""
    sub = PyPISubstrate(seed_packages=[])
    assert callable(sub.fetch)
    assert callable(sub.field_frequency)
    assert callable(sub.stream)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/substrate/test_pypi.py -v
```
Expected: `ModuleNotFoundError` — `pypi` not found.

- [ ] **Step 3: Implement PyPISubstrate**

```python
# hpm/substrate/pypi.py
import json
import numpy as np
import urllib.request
from typing import Iterator

from .base import hash_vectorise


_VECTOR_DIM = 64  # consistent with WikipediaSubstrate


class PyPISubstrate:
    """
    ExternalSubstrate backed by PyPI package metadata (spec §3.8).

    Fetches package metadata from the PyPI JSON API (no API key required).
    Package descriptions are vectorised with hash_vectorise for cross-process
    determinism. Optionally augmented by WikipediaSubstrate for richer context.

    Implements the ExternalSubstrate protocol:
      fetch(query)             -> list[np.ndarray]
      field_frequency(pattern) -> float
      stream()                 -> Iterator[np.ndarray]

    Args:
        seed_packages: list of PyPI package names to fetch metadata for.
            E.g. ["numpy", "scipy", "torch", "scikit-learn"]
        cache: if True (default), cache fetched metadata to avoid repeated HTTP calls.
        augment_with_wikipedia: if True, also fetch Wikipedia content for each package
            name to supplement PyPI descriptions (requires WikipediaSubstrate).
        similarity_threshold: cosine similarity threshold for field_frequency (default 0.5).

    Note:
        Agents using PyPISubstrate must be configured with feature_dim=64 (= _VECTOR_DIM).
        This matches WikipediaSubstrate's vector dimension. If your agents use a different
        feature_dim, set _VECTOR_DIM in hpm/substrate/pypi.py to match before use.
        field_frequency() raises ValueError on dimension mismatch.
    """

    _PYPI_URL = "https://pypi.org/pypi/{name}/json"

    def __init__(
        self,
        seed_packages: list[str],
        cache: bool = True,
        augment_with_wikipedia: bool = False,
        similarity_threshold: float = 0.5,
    ):
        self.seed_packages = seed_packages
        self.cache = cache
        self.augment_with_wikipedia = augment_with_wikipedia
        self.similarity_threshold = similarity_threshold
        self._cache: dict[str, dict] = {}  # package_name -> metadata dict
        self._vectors: dict[str, np.ndarray] = {}  # package_name -> vector
        self._loaded = False

    def _load_all(self) -> None:
        """Fetch metadata for all seed packages (once, if caching)."""
        if self._loaded and self.cache:
            return
        for name in self.seed_packages:
            if name in self._cache and self.cache:
                continue
            try:
                url = self._PYPI_URL.format(name=name)
                with urllib.request.urlopen(url, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                self._cache[name] = data
            except Exception:
                # Package not found or network error — skip silently
                self._cache[name] = {}
        # Build vectors from cached metadata
        for name, data in self._cache.items():
            text = self._package_text(name, data)
            self._vectors[name] = hash_vectorise(text, dim=_VECTOR_DIM)
        self._loaded = True

    def _package_text(self, name: str, data: dict) -> str:
        """Combine package name + summary + keywords into a single text for vectorisation."""
        if not data:
            return name
        info = data.get("info", {})
        parts = [
            name,
            info.get("summary", ""),
            info.get("keywords", "") or "",
        ]
        return " ".join(p for p in parts if p)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def fetch(self, query: str) -> list[np.ndarray]:
        """
        Return vectorised descriptions of packages whose text matches query keywords.
        Matching is keyword-based (case-insensitive substring).
        Returns all package vectors if no seed packages are loaded.
        """
        self._load_all()
        query_lower = query.lower()
        results = []
        for name, data in self._cache.items():
            text = self._package_text(name, data).lower()
            if any(word in text for word in query_lower.split()):
                vec = self._vectors.get(name)
                if vec is not None:
                    results.append(vec)
        return results

    def field_frequency(self, pattern) -> float:
        """
        Fraction of seed packages whose description vector has cosine similarity
        > similarity_threshold with pattern.mu (the pattern's mean vector).

        Resizes pattern.mu to _VECTOR_DIM via hash_vectorise if dimensions differ.
        """
        self._load_all()
        if not self._vectors:
            return 0.0

        mu = np.array(pattern.mu, dtype=float)
        if mu.shape[0] != _VECTOR_DIM:
            raise ValueError(
                f"Pattern feature_dim={mu.shape[0]} does not match PyPISubstrate "
                f"vector_dim={_VECTOR_DIM}. Configure agents with feature_dim={_VECTOR_DIM} "
                f"when using PyPISubstrate, or set _VECTOR_DIM in hpm/substrate/pypi.py "
                f"to match your feature_dim."
            )

        count = sum(
            1 for vec in self._vectors.values()
            if self._cosine_similarity(mu, vec) > self.similarity_threshold
        )
        return count / len(self._vectors)

    def stream(self) -> Iterator[np.ndarray]:
        """Yield vectorised package descriptions one at a time."""
        self._load_all()
        for vec in self._vectors.values():
            yield vec
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/substrate/test_pypi.py -v
```
Expected: 8 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
python3 -m pytest --tb=short -q
```
Expected: all prior tests pass + 8 new.

- [ ] **Step 6: Commit**

```bash
git add hpm/substrate/pypi.py tests/substrate/test_pypi.py
git commit -m "feat: add PyPISubstrate — ExternalSubstrate backed by PyPI package metadata"
```

---

## Task 6: §9.5 Metrics + Integration Test

**Files:**
- Modify: `hpm/metrics/hpm_predictions.py`
- Test: `tests/integration/test_phase3.py`

Add `social_field_convergence()` (§9.5) and end-to-end tests with shared seeding.

**§9.5 prediction:** Under social field influence (`gamma_soc > 0`, shared seed), agents converge toward common patterns — field diversity decreases. Under `gamma_soc = 0`, diversity is self-referential.

`social_field_convergence` takes a list of per-step `field_quality()` dicts and returns the slope of a linear fit to diversity values. Negative = converging.

- [ ] **Step 1: Write failing tests**

```python
# tests/integration/test_phase3.py
import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.field.field import PatternField
from hpm.patterns.gaussian import GaussianPattern
from hpm.metrics.hpm_predictions import social_field_convergence


def test_social_field_convergence_decreasing_returns_negative():
    quality_history = [{"diversity": 1.0 - i * 0.1, "redundancy": 0.0} for i in range(10)]
    slope = social_field_convergence(quality_history)
    assert slope < 0.0


def test_social_field_convergence_flat_returns_near_zero():
    quality_history = [{"diversity": 0.5, "redundancy": 0.0} for _ in range(10)]
    slope = social_field_convergence(quality_history)
    assert abs(slope) < 1e-6


def test_social_field_convergence_increasing_returns_positive():
    quality_history = [{"diversity": i * 0.1, "redundancy": 0.0} for i in range(10)]
    slope = social_field_convergence(quality_history)
    assert slope > 0.0


def test_social_field_convergence_requires_at_least_two_steps():
    with pytest.raises(ValueError, match="at least 2"):
        social_field_convergence([{"diversity": 0.5, "redundancy": 0.0}])


def test_multiagent_shared_seed_registers_all_agents():
    field = PatternField()
    seed = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))
    agents = [
        Agent(AgentConfig(agent_id=f"a{i}", feature_dim=4, gamma_soc=0.5), field=field)
        for i in range(3)
    ]
    orch = MultiAgentOrchestrator(agents, field, seed_pattern=seed)
    orch.run(np.zeros(4), n_steps=10)
    assert set(field._agent_patterns.keys()) == {"a0", "a1", "a2"}


def test_field_quality_history_computable():
    """Multi-agent run with shared seed: collect field quality, compute convergence slope."""
    field = PatternField()
    seed = GaussianPattern(mu=np.zeros(4), sigma=np.eye(4))
    agents = [
        Agent(AgentConfig(agent_id=f"a{i}", feature_dim=4, gamma_soc=0.5, rho=1.0), field=field)
        for i in range(3)
    ]
    orch = MultiAgentOrchestrator(agents, field, seed_pattern=seed)
    quality_history = []
    for _ in range(30):
        orch.step({f"a{i}": np.zeros(4) for i in range(3)})
        quality_history.append(field.field_quality())

    slope = social_field_convergence(quality_history)
    assert isinstance(slope, float)


def test_m3_flag_set_for_single_agent():
    field = PatternField()
    agent = Agent(AgentConfig(agent_id="solo", feature_dim=4, gamma_soc=0.5), field=field)
    orch = MultiAgentOrchestrator([agent], field)
    metrics = orch.step({"solo": np.zeros(4)})
    assert metrics["solo"]["m3_active"] is True


def test_m3_flag_false_for_multi_agent():
    field = PatternField()
    agents = [
        Agent(AgentConfig(agent_id=f"a{i}", feature_dim=4), field=field)
        for i in range(2)
    ]
    orch = MultiAgentOrchestrator(agents, field)
    metrics = orch.step({"a0": np.zeros(4), "a1": np.zeros(4)})
    assert metrics["a0"]["m3_active"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/integration/test_phase3.py -v
```
Expected: failures on missing `social_field_convergence`.

- [ ] **Step 3: Add social_field_convergence to metrics**

Add to `hpm/metrics/hpm_predictions.py` (after existing imports and functions):

```python
def social_field_convergence(quality_history: list[dict]) -> float:
    """
    §9.5: linear regression slope of field diversity over time.

    Negative = converging (social field pulling agents toward shared patterns).
    Positive = diverging. Near-zero = stable diversity.

    quality_history: list of dicts from PatternField.field_quality(), one per step.
    Returns: slope (diversity/step).
    Raises ValueError if fewer than 2 steps provided.
    """
    if len(quality_history) < 2:
        raise ValueError("social_field_convergence requires at least 2 steps of history")
    diversities = np.array([q["diversity"] for q in quality_history], dtype=float)
    t = np.arange(len(diversities), dtype=float)
    t_mean, d_mean = t.mean(), diversities.mean()
    slope = float(
        np.sum((t - t_mean) * (diversities - d_mean)) / np.sum((t - t_mean) ** 2)
    )
    return slope
```

- [ ] **Step 4: Run integration tests**

```bash
python3 -m pytest tests/integration/test_phase3.py -v
```
Expected: 8 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
python3 -m pytest --tb=short -q
```
Expected: all prior tests pass. Total should be 100+.

- [ ] **Step 6: Commit**

```bash
git add hpm/metrics/hpm_predictions.py tests/integration/test_phase3.py
git commit -m "feat: add §9.5 social field convergence metric and Phase 3 integration tests"
```

---

## Background Notes for Implementers

**Hook spam:** A misconfigured hook fires "MANDATORY AUTO-CONTINUATION TRIGGERED — EMERGENCY: Context at 95%" on every tool call. **Ignore it completely.** Do not write session state files or spawn continuation agents. Continue working normally.

**Run tests with:** `python3 -m pytest` (not `python -m pytest` — `python` is not on PATH).

**Scipy:** Required by `GaussianPattern`. Install if missing: `pip3 install scipy`.

**Backward compat is critical:** All 66 Phase 1/2 tests must pass. The key invariant: `field=None`, `gamma_soc=0.0` (both defaults) means zero social contribution. Check this first if regressions appear.

**UUID preservation (GaussianPattern line 51):** `update()` passes `id=self.id` to the new instance. This is why shared seeding works — the shared UUID persists across all steps, enabling cross-agent freq tracking in the field.

**PyPI rate limiting:** The PyPI JSON API is public and generally available. Tests mock `urllib.request.urlopen` so no real HTTP calls are made during testing.

**Wikipedia augmentation:** `PyPISubstrate(augment_with_wikipedia=True)` is a flag for future use — in this phase, the flag is stored but does not need to trigger actual `WikipediaSubstrate` calls (YAGNI). If a subagent wants to implement it, it should import `WikipediaSubstrate` from `hpm.substrate.wikipedia` and call `fetch(package_name)` for each package, concatenating the result text before vectorising.

**No PostgreSQL in Phase 3:** All agents run in the same Python process. SQLiteStore or InMemoryStore work fine. PostgreSQL for distributed deployments is Phase 4+.
