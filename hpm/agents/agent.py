import hashlib
from collections import deque

import numpy as np
from ..config import AgentConfig
from ..patterns.gaussian import GaussianPattern
from ..patterns.factory import make_pattern, pattern_from_dict
from ..evaluators.epistemic import EpistemicEvaluator
from ..evaluators.affective import AffectiveEvaluator
from ..evaluators.social import SocialEvaluator
from ..evaluators.resource_cost import ResourceCostEvaluator
from ..dynamics.meta_pattern_rule import MetaPatternRule
from ..dynamics.density import PatternDensity
from ..dynamics.recombination import RecombinationOperator
from ..patterns.classifier import HPMLevelClassifier
from ..store.memory import InMemoryStore
from .relational import StructuralMessage
from .completion import DecisionTrace, EvaluatorArbitrator, FieldConstraint, PatternLifecycleTracker, EvaluatorVector


def _stable_seed(agent_id: str, feature_dim: int, pattern_type: str) -> int:
    payload = f"{agent_id}:{feature_dim}:{pattern_type}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


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
        self._recomb_seed = (
            config.seed
            if config.seed is not None
            else _stable_seed(config.agent_id, config.feature_dim, f"{config.pattern_type}:recomb")
        )
        self._recomb_op = RecombinationOperator(seed=self._recomb_seed)
        self._shared_ids: set[str] = set()
        self._structural_inbox: list[tuple[str, object]] = []
        self._decision_traces: deque = deque(maxlen=512)
        self._outcome_history: deque = deque(maxlen=getattr(config, "obs_buffer_size", 50))
        self._structural_message_eta = float(getattr(config, "structural_message_eta", 0.05))
        self._arbitrator = EvaluatorArbitrator(
            mode=getattr(config, "evaluator_arbitration_mode", "fixed"),
            learning_rate=getattr(config, "meta_evaluator_learning_rate", 0.1),
        )
        self._lifecycle = PatternLifecycleTracker(
            consolidation_window=getattr(config, "lifecycle_consolidation_window", 3),
            stable_weight_threshold=getattr(config, "lifecycle_stable_weight_threshold", 0.25),
            retire_weight_threshold=getattr(config, "lifecycle_retire_weight_threshold", 0.05),
            absence_window=getattr(config, "lifecycle_absence_window", 3),
            decay_rate=getattr(config, "lifecycle_decay_rate", 0.1),
        )
        self._init_seed = (
            config.seed
            if config.seed is not None
            else _stable_seed(config.agent_id, config.feature_dim, config.pattern_type)
        )
        self._seed_if_empty()
        self._prime_lifecycle_state()

    def _share_pending(self, field, patterns: list) -> int:
        """
        Broadcast patterns that have newly reached Level 4+ to the field.
        Precondition: field must not be None (caller's responsibility).
        Returns count of patterns newly shared this call.
        """
        count = 0
        for p in patterns:
            if p.level >= 4 and p.id not in self._shared_ids:
                d = p.to_dict()
                d['id'] = None      # fresh UUID (constructor does: id or str(uuid.uuid4()))
                d['source_id'] = p.id
                shared_copy = pattern_from_dict(d)
                field.broadcast(self.agent_id, shared_copy)
                self._shared_ids.add(p.id)
                count += 1
        return count

    def _accept_communicated(self, pattern, source_agent_id: str) -> bool:
        """
        Evaluate an incoming communicated pattern and admit it if I(h*) > 0.
        Returns True if admitted.

        Sign convention: log_prob(x) returns NLL (positive). -log_prob(x) gives
        log-likelihood (<= 0). Eff is therefore non-positive; I(h*) > 0 requires
        novelty to offset negative efficacy.

        Nov = max(sym_kl_normalised(pattern, p) for p in library)  [1.0 if empty]
        Eff = mean(-pattern.log_prob(x) for x in obs_buffer)  <= 0  [0.0 if empty]
        I   = beta_orig * (alpha_nov * Nov + alpha_eff * Eff)
        """
        from ..dynamics.meta_pattern_rule import sym_kl_normalised
        records = self.store.query(self.agent_id)
        existing = [p for p, _ in records]

        nov = (
            max(sym_kl_normalised(pattern, p) for p in existing)
            if existing else 1.0
        )
        obs = list(self._obs_buffer)
        eff = float(np.mean([-pattern.log_prob(x) for x in obs])) if obs else 0.0
        insight = self.config.beta_orig * (
            self.config.alpha_nov * nov + self.config.alpha_eff * eff
        )
        if insight <= 0:
            return False

        entry_weight = self.config.kappa_0 * insight
        self.store.save(pattern, entry_weight, self.agent_id)
        all_records = self.store.query(self.agent_id)
        total_w = sum(w for _, w in all_records)
        if total_w > 0:
            for p, w in all_records:
                self.store.update_weight(p.id, self.agent_id, w / total_w)
        return True

    def emit_structural_message(self):
        """Emit a compact structural message summarizing the current top pattern."""
        try:
            from .hierarchical import extract_relational_bundle, bundle_to_structural_message
            bundle = extract_relational_bundle(self)
            if not bundle.relations:
                return None
            return bundle_to_structural_message(bundle)
        except Exception:
            return None

    def _pattern_id_from_reference(self, reference: str) -> str | None:
        if not isinstance(reference, str):
            return None
        prefix = "pattern:"
        if not reference.startswith(prefix):
            return None
        return reference[len(prefix):]

    def _renormalize_weights(self) -> None:
        records = self.store.query(self.agent_id)
        total_w = sum(w for _, w in records)
        if total_w <= 0:
            return
        for p, w in records:
            self.store.update_weight(p.id, self.agent_id, w / total_w)

    def _apply_structural_message(self, message: StructuralMessage) -> bool:
        if message.confidence <= 0.0:
            return False

        records = self.store.query(self.agent_id)
        if not records:
            return False

        record_map = {p.id: (p, w) for p, w in records}
        touched = False
        message_strength = float(np.clip(message.confidence, 0.0, 1.0))

        for edge in message.relations:
            if edge.relation != "tracks_pattern":
                continue
            pattern_id = self._pattern_id_from_reference(edge.target)
            if pattern_id is None or pattern_id not in record_map:
                continue
            pattern, weight = record_map[pattern_id]
            boost = self._structural_message_eta * message_strength * float(np.clip(edge.confidence, 0.0, 1.0))
            if boost <= 0.0:
                continue
            self.store.update_weight(pattern.id, self.agent_id, weight * (1.0 + boost))
            touched = True

        if touched:
            self._renormalize_weights()
        return touched

    def accept_structural_message(self, message, source_agent_id: str) -> bool:
        """Record an incoming structural message for higher-level inspection."""
        self._structural_inbox.append((source_agent_id, message))
        if source_agent_id == self.agent_id:
            return True
        if isinstance(message, StructuralMessage):
            self._apply_structural_message(message)
        return True

    def consume_structural_inbox(self, clear: bool = True) -> list[tuple[str, object]]:
        """Return structural inbox contents, optionally clearing after read."""
        items = list(self._structural_inbox)
        if clear:
            self._structural_inbox.clear()
        return items

    def consume_decision_traces(self, clear: bool = True) -> list[dict]:
        """Return captured decision traces, optionally clearing after read."""
        items = [trace for trace in self._decision_traces]
        if clear:
            self._decision_traces.clear()
        return items

    def _seed_if_empty(self) -> None:
        if not self.store.query(self.agent_id):
            rng = np.random.default_rng(self._init_seed)
            scale = (np.eye(self.config.feature_dim) * self.config.init_sigma
                     if self.config.pattern_type == "gaussian"
                     else self.config.init_sigma)
            init = make_pattern(
                mu=rng.normal(0, 1, self.config.feature_dim),
                scale=scale,
                pattern_type=self.config.pattern_type,
                alphabet_size=self.config.alphabet_size,
            )
            self.store.save(init, 1.0, self.agent_id)

    def _prime_lifecycle_state(self) -> None:
        records = self.store.query(self.agent_id)
        for pattern, weight in records:
            self._lifecycle.observe(pattern, weight, step=0)

    def _field_constraints(self) -> list[FieldConstraint]:
        if self.field is None or not hasattr(self.field, "constraints_for"):
            return []
        return list(self.field.constraints_for(self.agent_id))

    def _constraint_adjustment(self, pattern, constraints: list[FieldConstraint]) -> float:
        if not constraints:
            return 0.0
        complexity = pattern.description_length() / max(1.0, float(self.config.feature_dim))
        level_score = (float(getattr(pattern, "level", 1)) - 1.0) / 4.0
        adjustment = 0.0
        for constraint in constraints:
            strength = float(np.clip(constraint.strength, 0.0, 1.0))
            ctype = constraint.constraint_type
            if ctype == "penalize_complexity":
                adjustment -= strength * complexity
            elif ctype == "prefer_simple":
                adjustment += strength * (1.0 - complexity)
            elif ctype == "prefer_high_level":
                adjustment += strength * level_score
            elif ctype == "prefer_low_level":
                adjustment -= strength * level_score
            else:
                adjustment += 0.1 * strength
        return float(adjustment)

    def _evaluator_vector(self, predictive: float, coherence: float, cost: float, horizon: float) -> EvaluatorVector:
        aggregate = self._arbitrator.aggregate(predictive, coherence, cost, horizon)
        return EvaluatorVector(
            predictive=float(predictive),
            coherence=float(coherence),
            cost=float(cost),
            horizon=float(horizon),
            aggregate=float(aggregate),
            arbitration_mode=self._arbitrator.mode,
        )

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

        field_constraints = self._field_constraints()
        predictive_terms = np.asarray(epistemic_accs, dtype=float)
        coherence_terms = np.asarray(e_affs, dtype=float)
        cost_terms = np.asarray(e_costs, dtype=float)
        horizon_terms = np.asarray(e_socs, dtype=float)
        evaluator_vector = self._evaluator_vector(
            predictive=float(np.mean(predictive_terms)) if len(predictive_terms) else 0.0,
            coherence=float(np.mean(coherence_terms)) if len(coherence_terms) else 0.0,
            cost=float(np.mean(cost_terms)) if len(cost_terms) else 0.0,
            horizon=float(np.mean(horizon_terms)) if len(horizon_terms) else 0.0,
        )

        totals = np.array([
            epi
            + (self.config.beta_comp * p.compress() if self.config.beta_comp != 0.0 else 0.0)
            + self.config.beta_aff * e_aff
            + self.config.gamma_soc * e_soc
            + self.config.delta_cost * e_cost
            + self._constraint_adjustment(p, field_constraints)
            for p, epi, e_aff, e_soc, e_cost in zip(patterns, epistemic_accs, e_affs, e_socs, e_costs)
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
            self.store.delete(p.id, self.agent_id)
            if w >= self.config.epsilon:
                updated = p.update(x)
                self.store.save(updated, float(w), self.agent_id)
                surviving.append((updated, float(w)))
                surviving_patterns.append(updated)


        # Register with field using post-update UUIDs (preserved by update())
        if self.field is not None:
            self.field.register(self.agent_id, [(p.id, w) for p, w in surviving])

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
                setattr(recomb_result.pattern, "parent_ids", (recomb_result.parent_a_id, recomb_result.parent_b_id))
                setattr(recomb_result.pattern, "source_id", recomb_result.parent_a_id)
                setattr(recomb_result.pattern, "lineage_kind", "recombined")
                setattr(recomb_result.pattern, "source_step", self._t)
                entry_weight = self.config.kappa_0 * recomb_result.insight_score
                self.store.save(recomb_result.pattern, entry_weight, self.agent_id)
                all_records = self.store.query(self.agent_id)
                total_w = sum(w for _, w in all_records)
                if total_w > 0:
                    for p, w in all_records:
                        self.store.update_weight(p.id, self.agent_id, w / total_w)
            # Cooldown resets unconditionally whether attempt() accepted or not.
            # Intentional: prevents thrashing on an incompatible pattern population.
            self._last_recomb_t = self._t

        # If recombination was accepted, re-query the store so level metrics include the new pattern
        if recomb_result is not None:
            final_records = self.store.query(self.agent_id)
            report_patterns = [p for p, _ in final_records]
        else:
            report_patterns = surviving_patterns

        for pattern, weight in self.store.query(self.agent_id):
            self._lifecycle.observe(pattern, weight, step=self._t)
        active_ids = {p.id for p in report_patterns}
        self._lifecycle.finalize(active_ids, step=self._t)
        lifecycle_summary = self._lifecycle.summary().to_dict()

        communicated_out = 0
        if self.field is not None:
            communicated_out = self._share_pending(self.field, report_patterns)

        # Inhibitory channel: Step A — pull negative patterns from other agents
        if self.field is not None and hasattr(self.store, '_tier2_negative'):
            neg_incoming = self.field.pull_negative(self.agent_id, self.config.gamma_neg)
            for pattern, weight in neg_incoming:
                current_neg = self.store._tier2_negative.query_all()
                if len(current_neg) < self.config.max_tier2_negative:
                    self.store._tier2_negative.save(pattern, weight, self.agent_id)

        # Inhibitory channel: Step B — broadcast own negative patterns to field
        if self.field is not None and hasattr(self.store, 'query_negative'):
            self.field._negative[self.agent_id] = []   # reset before re-broadcasting
            for pattern, weight in self.store.query_negative(self.agent_id):
                self.field.broadcast_negative(pattern, weight, self.agent_id)

        selected_pattern_ids = tuple(p.id for p in report_patterns[: min(3, len(report_patterns))])
        selected_parent_ids = tuple(
            sorted(
                {
                    parent_id
                    for pattern in report_patterns[: min(3, len(report_patterns))]
                    for parent_id in tuple(getattr(pattern, "parent_ids", ()) or (getattr(pattern, "source_id", None),))
                    if parent_id is not None
                }
            )
        )
        action = (
            "recombination:accepted"
            if recomb_result is not None
            else ("recombination:attempted" if recomb_attempted else "update")
        )
        step_outcome = float(reward - 0.25 * total_conflict + (0.1 if recomb_result is not None else 0.0))
        self._outcome_history.append(step_outcome)
        meta_signal = float(np.mean(self._outcome_history)) if self._outcome_history else step_outcome
        meta_signal_source = "rolling_outcome"
        meta_evaluator_state = self._arbitrator.update(
            meta_signal,
            predictive=float(np.mean(predictive_terms)) if len(predictive_terms) else 0.0,
            coherence=float(np.mean(coherence_terms)) if len(coherence_terms) else 0.0,
            cost=float(np.mean(cost_terms)) if len(cost_terms) else 0.0,
            horizon=float(np.mean(horizon_terms)) if len(horizon_terms) else 0.0,
            signal_source=meta_signal_source,
        )
        decision_trace = DecisionTrace(
            trace_id=f"{self.agent_id}:{self._t}",
            selected_pattern_ids=selected_pattern_ids,
            selected_parent_ids=selected_parent_ids,
            evaluator_vector=evaluator_vector,
            constraint_ids=tuple(
                f"{c.scope}:{c.constraint_type}:{c.timestamp}" for c in field_constraints
            ),
            meta_evaluator_state=meta_evaluator_state,
            signal_source=meta_signal_source,
            action=action,
        )
        self._decision_traces.append(decision_trace.to_dict())

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
            'compress_mean': (
                float(np.mean([p.compress() for p in report_patterns]))
                if (report_patterns and self.config.beta_comp != 0.0)
                else 0.0
            ),
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
            'communicated_out': communicated_out,
            'lifecycle_summary': lifecycle_summary,
            'lifecycle_snapshot': self._lifecycle.snapshot(),
            'field_constraint_count': len(field_constraints),
            'field_constraint_strength': float(np.mean([c.strength for c in field_constraints])) if field_constraints else 0.0,
            'evaluator_vector': evaluator_vector.to_dict(),
            'meta_evaluator_state': meta_evaluator_state.to_dict(),
            'decision_trace': decision_trace.to_dict(),
            'decision_trace_count': len(self._decision_traces),
            'outcome_signal': step_outcome,
        }
