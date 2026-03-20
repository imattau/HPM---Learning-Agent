import numpy as np
from ..config import AgentConfig
from ..patterns.gaussian import GaussianPattern
from ..evaluators.epistemic import EpistemicEvaluator
from ..evaluators.affective import AffectiveEvaluator
from ..dynamics.meta_pattern_rule import MetaPatternRule
from ..store.memory import InMemoryStore


class Agent:
    """
    Single HPM agent. Wires PatternLibrary, EvaluatorPipeline, and Dynamics.
    Backed by a PatternStore (InMemoryStore by default; SQLiteStore for persistence).
    Optionally connected to an ExternalSubstrate for external field frequency signals.

    Data flow per step (Phase 1/2, §7):
      1. Compute ell_i(t) for each pattern
      2. Update L_i(t) -> A_i(t) via EpistemicEvaluator
      3. Compute E_aff_i(t) via AffectiveEvaluator
      4. Total_i(t) = A_i(t) + beta_aff * E_aff_i(t)
      5. MetaPatternRule -> new weights
      6. Prune + update store
      7. If substrate set: compute ext_field_freq (logged; blending into totals in Phase 3)
    """

    def __init__(self, config: AgentConfig, store=None, substrate=None):
        self.config = config
        self.agent_id = config.agent_id
        self.store = store or InMemoryStore()
        self.substrate = substrate
        self.epistemic = EpistemicEvaluator(lambda_L=config.lambda_L)
        self.affective = AffectiveEvaluator(
            k=config.k,
            c_opt=config.c_opt,
            sigma_c=config.sigma_c,
            alpha_r=config.alpha_r,
        )
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

        accuracies = []
        e_affs = []
        for pattern in patterns:
            instant_acc = -pattern.log_prob(x)
            epistemic_acc = self.epistemic.update(pattern, x)
            e_aff = self.affective.update(pattern, epistemic_acc, reward)
            accuracies.append(instant_acc)
            e_affs.append(e_aff)

        totals = np.array([
            epistemic_acc + self.config.beta_aff * e_aff
            for epistemic_acc, e_aff in zip([self.epistemic.accuracy(p.id) for p in patterns], e_affs)
        ])

        new_weights = self.dynamics.step(patterns, weights, totals)

        # Prune and persist
        for p in patterns:
            self.store.delete(p.id)
        for p, w in zip(patterns, new_weights):
            if w >= self.config.epsilon:
                updated = p.update(x)
                self.store.save(updated, float(w), self.agent_id)

        # External substrate: compute field frequencies (logged; not yet in totals — Phase 3)
        ext_field_freq = 0.0
        if self.substrate is not None:
            freqs = [self.substrate.field_frequency(p) for p in patterns]
            ext_field_freq = float(np.mean(freqs)) if freqs else 0.0

        self._t += 1
        return {
            't': self._t,
            'n_patterns': int(np.sum(new_weights >= self.config.epsilon)),
            'mean_accuracy': float(np.mean(accuracies)),
            'max_weight': float(new_weights.max()),
            'ext_field_freq': ext_field_freq,
        }
