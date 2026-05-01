import numpy as np
import copy
from collections import deque
from typing import List, Dict, Any, Optional

from hpm_ai_v4.pattern import HierarchicalPattern, FlatPattern
from hpm_ai_v4.operators.parallel import ParallelPatternPool, update_pattern_resident
from hpm_ai_v4.evaluators.metrics import total_score, epistemic_score, compression_gate
from hpm_ai_v4.operators.dynamics import compute_conflict_matrix, meta_pattern_update, recombine
from hpm_ai_v4.field import PatternField, InstitutionalField
from hpm_ai_v4.tools.substrate import ExternalSubstrate
from hpm_ai_v4.agents.reasoning import Reasoner
from hpm_ai_v4.tools.dictionary import DictionaryValidator
from hpm_ai_v4.tools.grammar import GrammarValidator

class DevelopmentalStage:
    """Modulates evaluator focus based on current population complexity."""
    LEVELS = ['surface', 'local', 'relational', 'abstract', 'generative']
    
    def __init__(self, agent: 'HPMAgent'):
        self.agent = agent
        self.level_idx = 0   # start at surface level
        self._recent_signals = deque(maxlen=8)
        self._min_observation_steps = 20
        self._min_hier_patterns = 5
        self._stage_targets = [
            {"mean_ep": -0.30, "var_ep": 0.02, "mean_gate": 0.35, "mean_comp": 0.02},
            {"mean_ep": -0.25, "var_ep": 0.02, "mean_gate": 0.40, "mean_comp": 0.03},
            {"mean_ep": -0.22, "var_ep": 0.018, "mean_gate": 0.45, "mean_comp": 0.04},
            {"mean_ep": -0.18, "var_ep": 0.015, "mean_gate": 0.50, "mean_comp": 0.05},
            {"mean_ep": -0.15, "var_ep": 0.012, "mean_gate": 0.55, "mean_comp": 0.06},
        ]

    @property
    def level(self) -> str:
        return self.LEVELS[self.level_idx]

    def _hierarchical_signals(self, patterns: List[HierarchicalPattern]) -> Optional[Dict[str, float]]:
        hier = [p for p in patterns if p.latent_dim > 1]
        if len(hier) < self._min_hier_patterns:
            return None

        ep_scores = np.array([epistemic_score(p) for p in hier], dtype=np.float32)
        gates = np.array([compression_gate(p) for p in hier], dtype=np.float32)
        compressions = np.array([p.compression() for p in hier], dtype=np.float32)
        weights = np.array([max(0.0, float(p.weight)) for p in hier], dtype=np.float32)
        if float(weights.sum()) > 0.0:
            weights = weights / (weights.sum() + 1e-12)
            mean_ep = float(np.sum(weights * ep_scores))
            mean_gate = float(np.sum(weights * gates))
            mean_comp = float(np.sum(weights * compressions))
        else:
            mean_ep = float(np.mean(ep_scores))
            mean_gate = float(np.mean(gates))
            mean_comp = float(np.mean(compressions))

        return {
            "mean_ep": mean_ep,
            "var_ep": float(np.var(ep_scores)),
            "mean_gate": mean_gate,
            "mean_comp": mean_comp,
            "count": float(len(hier)),
        }

    def _should_advance_stage(self, patterns: List[HierarchicalPattern], global_step: int) -> bool:
        if global_step < self._min_observation_steps:
            return False

        signal = self._hierarchical_signals(patterns)
        if signal is None:
            return False

        self._recent_signals.append(signal)
        if len(self._recent_signals) < 4:
            return False

        window = list(self._recent_signals)[-4:]
        mean_ep = float(np.mean([s["mean_ep"] for s in window]))
        var_ep = float(np.var([s["mean_ep"] for s in window]))
        mean_gate = float(np.mean([s["mean_gate"] for s in window]))
        mean_comp = float(np.mean([s["mean_comp"] for s in window]))

        target_idx = min(self.level_idx, len(self._stage_targets) - 1)
        target = self._stage_targets[target_idx]
        return (
            mean_ep >= target["mean_ep"]
            and var_ep <= target["var_ep"]
            and mean_gate >= target["mean_gate"]
            and mean_comp >= target["mean_comp"]
            and signal["var_ep"] <= target["var_ep"] * 1.5
        )

    def update(self, patterns: List[HierarchicalPattern], global_step: int):
        if not patterns:
            return

        # Progression logic: advance only when the current evaluator signal has
        # stabilised, not when a hard-coded complexity threshold is crossed.
        if self.level_idx < len(self.LEVELS) - 1 and self._should_advance_stage(patterns, global_step):
            self.level_idx += 1
       
        # Modulate evaluator weightings based on developmental stage
        # (Following Section 7.4 of the framework)
        if self.level_idx == 0:
            self.agent.beta_aff = 0.2   # focus on surface feedback (social)
            self.agent.gamma_soc = 0.6
        elif self.level_idx == 1:
            self.agent.beta_aff = 0.4
            self.agent.gamma_soc = 0.4
        elif self.level_idx == 2:
            self.agent.beta_aff = 0.6
            self.agent.gamma_soc = 0.2
        else:
            self.agent.beta_aff = 0.8   # focus on curiosity and generative discovery
            self.agent.gamma_soc = 0.1

class HPMAgent:
    """The central HPM learner, integrating patterns, evaluators, and fields."""
    def __init__(self, num_initial_patterns: int = 5, external_substrate: Optional[ExternalSubstrate] = None,
                 obs_dim: int = 2, num_workers: int = 1, 
                 dictionary: Optional[DictionaryValidator] = None,
                 grammar: Optional[GrammarValidator] = None):
        self.obs_dim = obs_dim
        self.dictionary = dictionary
        self.grammar = grammar
        self.patterns = []
        for i in range(num_initial_patterns):
            p = HierarchicalPattern(pattern_id=i, obs_dim=obs_dim)
            p.weight = 0.01  # Initial hierarchical patterns start as weak hypotheses
            self.patterns.append(p)
            
        # Include one flat pattern for comparative baseline (strong initial weight)
        flat_p = FlatPattern.flat(num_initial_patterns, obs_dim=obs_dim)
        flat_p.weight = 0.95
        self.patterns.append(flat_p)

        self.external = external_substrate if external_substrate else ExternalSubstrate()
        self.field = PatternField()
        self.development = DevelopmentalStage(self)
        self.reasoner = Reasoner(self, dictionary=dictionary, grammar=grammar)
        
        self.step_counter = 0
        self.obs_buffer = []
        self.external_social_scores = {} # pattern_id -> reliability score [0, 1]
        self._density_state: Dict[str, Any] = {"density_weight": 0.1}
        
        # Default evaluator weights (will be modulated by development)
        self.beta_aff = 0.4
        self.gamma_soc = 0.3
        
        self._pending_feedback: Dict[str, Any] = {}
        self._pool = ParallelPatternPool(num_workers=num_workers)

    def gossip_with_substrate(self, substrate: ExternalSubstrate):
        """Retrieve a random pattern from the collective substrate and inject it into the local population."""
        other = substrate.get_random_pattern()
        if other is not None and other.id not in [p.id for p in self.patterns]:
            # Incorporate as a low-weight hypothesis to avoid population destabilization
            new_p = copy.deepcopy(other)
            # Re-id to avoid local collisions if necessary, or just keep global ID
            new_p.weight = 0.05
            self.patterns.append(new_p)

    def _feedback_worker_params(self, base_params: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Adjust low-level learning pressure from higher-level feedback.

        The feedback channel is intentionally soft: it nudges update cadence and
        evaluator weights rather than overriding the core learner.
        """
        params = dict(base_params)
        if not feedback:
            return params

        meta = dict(feedback)
        control_strength = float(meta.get("control_strength", meta.get("meta_structural_score", 0.0)))
        topdown_gate = float(meta.get("topdown_gate", meta.get("topdown_confidence", 0.0)))
        mode_prior = meta.get("control_mode_prior") or meta.get("mode_prior") or {}
        dominant_mode = str(meta.get("control_dominant_mode") or meta.get("mode") or meta.get("desired_mode") or "")
        reward_hint = float(meta.get("reward", meta.get("quality", 0.0)))
        structure_hint = float(meta.get("meta_structural_score", meta.get("structural_score", 0.0)))
        certainty = max(control_strength, structure_hint, reward_hint, topdown_gate)
        prefer_repair = dominant_mode == "repair" or float(mode_prior.get("repair", 0.0)) >= 0.5
        gate = float(np.clip(max(certainty, topdown_gate), 0.0, 1.0))

        # The lower stack should update more eagerly when higher-level control is confident.
        params["learning_rate"] = float(params.get("learning_rate", 0.02)) * (0.80 + 0.45 * gate)
        params["lambda_l"] = float(params.get("lambda_l", 0.1)) * (0.85 + 0.30 * min(1.0, max(control_strength, topdown_gate)))
        params["adapt_window"] = int(round(float(params.get("adapt_window", 20)) * (1.0 + 0.35 * max(0.0, gate - 0.35))))

        beta_aff = float(params.get("beta_aff", self.beta_aff))
        gamma_soc = float(params.get("gamma_soc", self.gamma_soc))
        if prefer_repair:
            beta_aff += 0.10 * min(1.0, gate + 0.2)
            gamma_soc -= 0.04 * min(1.0, gate)
        else:
            beta_aff += 0.04 * max(0.0, gate - 0.45)
            gamma_soc += 0.03 * max(0.0, gate - 0.25)
        params["beta_aff"] = float(np.clip(beta_aff, 0.05, 0.95))
        params["gamma_soc"] = float(np.clip(gamma_soc, 0.05, 0.95))

        # Strong high-level confidence should encourage actual parameter updates.
        if gate > 0.75:
            params["do_param_update"] = True
        elif gate < 0.20:
            params["do_param_update"] = False

        return params

    def _apply_topdown_suppression(
        self,
        totals: Dict[int, float],
        results: Dict[int, Dict[str, Any]],
        feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, float]:
        """Down-weight weak lower-level patterns when higher-level confidence is high.

        This is the selection-side complement to `_feedback_worker_params`: instead of
        only speeding up the winning pattern, we also suppress patterns whose recent
        loss suggests they are inconsistent with the current higher-level context.
        """
        if not totals:
            return totals

        meta = dict(feedback or {})
        control_strength = float(meta.get("control_strength", meta.get("meta_structural_score", 0.0)))
        topdown_gate = float(meta.get("topdown_gate", meta.get("topdown_confidence", 0.0)))
        pattern_pressure = float(meta.get("topdown_pattern_suppression", meta.get("topdown_pattern_pressure", 0.0)))
        gate = float(np.clip(max(control_strength, topdown_gate, pattern_pressure), 0.0, 1.0))
        if gate <= 0.2:
            return totals

        losses = []
        ids = []
        for pattern_id, total in totals.items():
            result = results.get(pattern_id, {})
            loss = result.get("running_loss")
            if loss is None or not np.isfinite(float(loss)):
                continue
            ids.append(pattern_id)
            losses.append(float(loss))

        if len(losses) < 2:
            return totals

        loss_arr = np.asarray(losses, dtype=np.float32)
        median = float(np.median(loss_arr))
        spread = float(np.std(loss_arr) + 1e-12)
        pressure = np.maximum(0.0, (loss_arr - median) / spread)
        suppress_strength = 0.12 + 0.55 * gate

        adjusted = dict(totals)
        for pattern_id, loss_pressure in zip(ids, pressure):
            adjusted[pattern_id] = float(adjusted[pattern_id] - suppress_strength * float(loss_pressure))
        return adjusted

    def perceive_and_learn(self, obs: int, feedback: Optional[Dict[str, Any]] = None):
        """Update patterns based on a new observation."""
        if feedback is None:
            feedback = {}
        if hasattr(self, "_pending_feedback") and self._pending_feedback:
            feedback.update(self._pending_feedback)
            self._pending_feedback = {}

        context_before = list(self.obs_buffer[-20:])
        self.obs_buffer.append(obs)
        cap = self.reasoner.context_window
        if len(self.obs_buffer) > cap:
            self.obs_buffer = self.obs_buffer[-cap:]

        # Reasoning feedback: update per-pattern loss based on prediction error
        self.reasoner.observe_outcome(obs, context_before, metadata=feedback)

        # 1. Update Social Context (Pattern Field) - Move up for workers
        field_freq = self.field.update(self.patterns)

        # 2. Parallel per-pattern update + score computation
        worker_params = {
            'learning_rate': 0.02,
            'lambda_l': 0.1,
            'adapt_window': 20,
            'beta_aff': self.beta_aff,
            'gamma_soc': self.gamma_soc,
            'external_soc_map': self.external_social_scores,
            'do_param_update': (self.step_counter % 5 == 0),
        }
        worker_params = self._feedback_worker_params(worker_params, feedback)

        if self._pool.num_workers == 1:
            results = []
            for p in self.patterns:
                result = update_pattern_resident(p, self.obs_buffer, field_freq, worker_params)
                result['pattern_id'] = p.id
                result['complexity'] = p.complexity
                result['latent_dim'] = p.latent_dim
                result['obs_dim'] = p.obs_dim
                result['B'] = p.B.copy()
                result['running_loss'] = float(p.running_loss)
                result['weight'] = float(p.weight)
                if p.complexity >= 2 or p.latent_dim > 1:
                    result['A'] = p.A.copy()
                    result['pi'] = p.pi.copy()
                results.append(result)
        else:
            results = self._pool.map_patterns(
                self.patterns, self.obs_buffer, field_freq, worker_params
            )

        # 3. Write updated state back into pattern objects and collect totals
        result_by_id = {r['pattern_id']: r for r in results}
        totals = {}
        mean_running_loss = None
        running_losses = []
        for p in self.patterns:
            r = result_by_id[p.id]
            if p.complexity >= 2:
                p.A = r['A']
                p.B = r['B']
                p.pi = r['pi']
            else:
                p.B = r['B']
            p.running_loss = r['running_loss']
            running_losses.append(float(p.running_loss))
            totals[p.id] = r['total_score']
        if running_losses:
            mean_running_loss = float(np.mean(running_losses))

        totals = self._apply_topdown_suppression(totals, result_by_id, feedback)

        # 4. Meta Pattern Update (Replicator Dynamics with Conflict)
        k_mat = compute_conflict_matrix(self.patterns)
        meta_pattern_update(self.patterns, totals, eta=0.1, beta_c=0.03,
                            k_matrix=k_mat, decay=0.005,
                            density_state=self._density_state,
                            mean_running_loss=mean_running_loss)

        # 5. Population Pruning
        self.patterns = [p for p in self.patterns if p.weight > 1e-4]

        # 6. Recombination / Innovation
        if self.step_counter > 0 and self.step_counter % 100 == 0 and len(self.patterns) >= 2:
            weights = np.array([p.weight for p in self.patterns])
            if np.sum(weights) > 0:
                probs = weights / np.sum(weights)
                parents_idx = np.random.choice(len(self.patterns), size=2, p=probs, replace=False)
                
                child = recombine(self.patterns[parents_idx[0]], self.patterns[parents_idx[1]])
                if child is not None:
                    # Assign new id
                    current_ids = [p.id for p in self.patterns]
                    child.id = max(current_ids) + 1 if current_ids else 0
                    child.weight = 0.05
                    self.patterns.append(child)

        # 7. Persistence / Substrate Sharing & Gossip
        if self.step_counter % 20 == 0:
            self.external.broadcast(self.patterns)
            self.gossip_with_substrate(self.external)
        elif self.step_counter % 10 == 0:
            self.external.broadcast(self.patterns)

        # 8. Developmental Update
        self.development.update(self.patterns, self.step_counter)

        self.step_counter += 1

    def load_library(self, path: str, reset_weights: bool = True) -> int:
        """Load patterns from a serialised library file.

        Replaces self.patterns with the loaded population.
        If reset_weights=True, normalises all weights to 1/N so no single
        prior pattern dominates at the start of the new learning session.
        Returns N (number of patterns loaded).
        """
        from hpm_ai_v4.tools.serializer import PatternSerializer
        self.patterns = PatternSerializer.load(path)
        if reset_weights and self.patterns:
            w = 1.0 / len(self.patterns)
            for p in self.patterns:
                p.weight = w
        return len(self.patterns)

    def act(self, goal: Optional[int] = None) -> int:
        """Select an action (next observation to aim for) using the reasoning layer."""
        if not self.obs_buffer:
            return 0
            
        relevant = self.reasoner.get_relevant_patterns(self.obs_buffer, top_k=3)
        
        if goal is not None:
            plan = self.reasoner.plan(goal_state=goal, horizon=3)
            if plan:
                return plan[0]
                
        # Fallback to compositional predictive inference
        blended = self.reasoner.compose_predictions(relevant, self.obs_buffer)
        return int(np.argmax(blended))
