import numpy as np
from collections import deque
from ..config import AgentConfig
from ..patterns.gaussian import GaussianPattern
from ..evaluators.epistemic import EpistemicEvaluator
from ..evaluators.affective import AffectiveEvaluator
from ..evaluators.social import SocialEvaluator
from ..evaluators.resource_cost import ResourceCostEvaluator
from ..dynamics.meta_pattern_rule import MetaPatternRule
from ..dynamics.density import PatternDensity
from ..dynamics.recombination import RecombinationOperator
from ..patterns.classifier import HPMLevelClassifier
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
         (if no substrate: freq_i_total = field_freq_i, no attenuation)
      5. E_soc_i = rho * freq_i_total via SocialEvaluator
      6. J_i = beta_aff * E_aff_i + gamma_soc * E_soc_i
      7. Total_i = A_i + J_i
      8. MetaPatternRule -> new weights
      9. Prune + update store (UUID preserved by GaussianPattern.update())
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
        self.resource_cost = ResourceCostEvaluator(
            lambda_cost=config.lambda_cost,
            w_mem=config.w_mem,
            w_cpu=config.w_cpu,
        )
        self.pattern_density = PatternDensity(
            alpha_conn=config.alpha_conn,
            alpha_sat=config.alpha_sat,
            alpha_amp=config.alpha_amp,
        )
        self.level_classifier = HPMLevelClassifier(
            l5_density=config.l5_density,
            l5_conn=config.l5_conn,
            l5_comp=config.l5_comp,
            l4_conn=config.l4_conn,
            l4_comp=config.l4_comp,
            l3_conn=config.l3_conn,
            l3_comp=config.l3_comp,
            l2_conn=config.l2_conn,
        )
        self.dynamics = MetaPatternRule(
            eta=config.eta,
            beta_c=config.beta_c,
            epsilon=config.epsilon,
            kappa_D=config.kappa_D,
        )
        self._t = 0
        self._obs_buffer: deque = deque(maxlen=config.obs_buffer_size)
        self._last_recomb_t: int = -config.recomb_cooldown
        self._recomb_op = RecombinationOperator(rng=np.random.default_rng(0))
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
        self._obs_buffer.append(x)
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

        # Compute ext_freqs once (reused for blend and metric)
        ext_freqs = (
            [self.substrate.field_frequency(p) for p in patterns]
            if self.substrate is not None
            else [0.0] * len(patterns)
        )

        if self.substrate is None:
            # No external substrate: use field freq directly (no alpha_int attenuation)
            freq_totals = field_freqs
        elif self.field is None:
            # No field (single-agent substrate use): use ext_freqs directly
            freq_totals = ext_freqs
        else:
            alpha = self.config.alpha_int
            freq_totals = [
                alpha * ff + (1.0 - alpha) * ef
                for ff, ef in zip(field_freqs, ext_freqs)
            ]

        e_socs = self.social.evaluate_all(freq_totals)

        # Compute per-pattern density D(h_i) using current evaluator state.
        # Uses field_freqs (agent population signal), not freq_totals (blended with
        # external substrate), because density tracks pattern prevalence in the social
        # field specifically — separate from the E_soc evaluator signal.
        densities = [
            self.pattern_density.compute(
                p,
                loss=-epi,          # loss L_i = -A_i (non-negative)
                capacity=self.affective.last_capacity(p.id),
                field_freq=ff,
            )
            for p, epi, ff in zip(patterns, epistemic_accs, field_freqs)
        ]

        # Classify HPM level for each pattern and build per-pattern kappa_D list
        for p, d in zip(patterns, densities):
            p.level = self.level_classifier.compute_level(p, d)
        kappa_d_per_pattern = [self.config.kappa_d_levels[p.level - 1] for p in patterns]

        # Guard: only compute e_costs if delta_cost is non-zero (avoids psutil import for default agents)
        if self.config.delta_cost != 0.0:
            e_costs = [self.resource_cost.evaluate(p) for p in patterns]
        else:
            e_costs = [0.0] * len(patterns)

        totals = np.array([
            epi + self.config.beta_aff * e_aff + self.config.gamma_soc * e_soc
            + self.config.delta_cost * e_cost
            for epi, e_aff, e_soc, e_cost in zip(epistemic_accs, e_affs, e_socs, e_costs)
        ])

        step_result = self.dynamics.step(
            patterns, weights, totals,
            densities=densities,
            kappa_d_per_pattern=kappa_d_per_pattern,
        )
        new_weights = step_result.weights
        total_conflict = step_result.total_conflict

        # Prune, update patterns (UUID preserved by GaussianPattern.update()), persist
        surviving = []
        surviving_patterns = []
        for p, w in zip(patterns, new_weights):
            self.store.delete(p.id)
            if w >= self.config.epsilon:
                updated = p.update(x)
                self.store.save(updated, float(w), self.agent_id)
                surviving.append((updated.id, float(w)))
                surviving_patterns.append(updated)

        # Register with field using post-update UUIDs (preserved by update())
        if self.field is not None:
            self.field.register(self.agent_id, surviving)

        self._t += 1

        recomb_result = None
        recomb_attempted = False
        recomb_trigger = None

        time_trigger = (self._t % self.config.T_recomb == 0)
        conflict_trigger = (total_conflict > self.config.conflict_threshold)
        cooldown_ok = (self._t - self._last_recomb_t >= self.config.recomb_cooldown)

        if (time_trigger or conflict_trigger) and cooldown_ok:
            # conflict_trigger takes priority when both fire simultaneously,
            # so the return dict reflects the stronger signal (spec §Trigger Logic)
            recomb_trigger = "conflict" if conflict_trigger else "time"
            recomb_attempted = True
            post_prune_records = self.store.query(self.agent_id)
            post_prune_patterns = [p for p, _ in post_prune_records]
            post_prune_weights = np.array([w for _, w in post_prune_records])
            recomb_result = self._recomb_op.attempt(
                post_prune_patterns, post_prune_weights,
                list(self._obs_buffer), self.config, recomb_trigger,
                total_conflict=total_conflict,
            )
            if recomb_result is not None:
                entry_weight = self.config.kappa_0 * recomb_result.insight_score
                self.store.save(recomb_result.pattern, entry_weight, self.agent_id)
                all_records = self.store.query(self.agent_id)
                total_w = sum(w for _, w in all_records)
                if total_w > 0:
                    for p, w in all_records:
                        self.store.update_weight(p.id, w / total_w)
            # Cooldown resets unconditionally whether attempt() accepted or not.
            # Intentional: prevents thrashing on an incompatible pattern population.
            self._last_recomb_t = self._t

        # If recombination was accepted, re-query the store so level metrics include the new pattern
        if recomb_result is not None:
            final_records = self.store.query(self.agent_id)
            report_patterns = [p for p, _ in final_records]
        else:
            report_patterns = surviving_patterns

        return {
            't': self._t,
            'n_patterns': len(report_patterns),
            'mean_accuracy': float(np.mean(accuracies)),
            'max_weight': float(new_weights.max()) if len(new_weights) > 0 else 0.0,
            'e_soc_mean': float(np.mean(e_socs)) if len(e_socs) > 0 else 0.0,
            'ext_field_freq': float(np.mean(ext_freqs)),
            'e_cost_mean': float(np.mean(e_costs)) if len(e_costs) > 0 else 0.0,
            'density_mean': float(np.mean(densities)) if len(densities) > 0 else 0.0,
            'level_mean': float(np.mean([p.level for p in report_patterns])) if report_patterns else 0.0,
            'level_distribution': {lvl: sum(1 for p in report_patterns if p.level == lvl) for lvl in range(1, 6)},
            'total_conflict': float(total_conflict),
            'recombination_attempted': recomb_attempted,
            'recombination_accepted': recomb_result is not None,
            'recombination_trigger': recomb_trigger,
            'insight_score': recomb_result.insight_score if recomb_result else None,
            'recomb_parent_ids': (
                (recomb_result.parent_a_id, recomb_result.parent_b_id)
                if recomb_result else None
            ),
        }
