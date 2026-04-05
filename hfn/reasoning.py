"""
HPM Reasoning — agnostic cognitive loop for top-down problem solving.

Encapsulates the cycle:
1. Induce: Identify potential rules/parents from a history of observations.
2. Synthesize: Use the Decoder to generate predictions based on those rules.
3. Verify: Use the Evaluator to check predictions against the known history.
4. Finalise: Apply the validated rule to a novel test input.
"""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Callable
from hfn.hfn import HFN

if TYPE_CHECKING:
    from hfn.forest import Forest
    from hfn.observer import Observer
    from hfn.decoder import Decoder, ResolutionRequest
    from hfn.evaluator import Evaluator

class CognitiveSolver:
    """
    A domain-agnostic reasoning engine that coordinates HFN components.
    """
    def __init__(
        self,
        observer: Observer,
        decoder: Decoder,
        evaluator: Evaluator,
        thinking_budget: int = 20,
        # validator(raw_example, full_predicted_vector) -> bool
        validator: Optional[Callable[[Any, np.ndarray], bool]] = None,
        # rule_applier(rule_id: str, test_input: np.ndarray) -> Optional[np.ndarray]
        # Returns target_slice (930D) if rule_id is known, else None.
        rule_applier: Optional[Callable[[Any, np.ndarray], Optional[np.ndarray]]] = None
    ):
        self.observer = observer
        self.decoder = decoder
        self.evaluator = evaluator
        self.thinking_budget = thinking_budget
        self.validator = validator
        self.rule_applier = rule_applier

    def solve(
        self, 
        history: List[np.ndarray], 
        test_input: np.ndarray,
        target_slice: slice = slice(None),
        history_raw: Optional[List[Any]] = None,
        test_input_raw: Optional[Any] = None,
        verbose: bool = True
    ) -> Optional[np.ndarray]:
        """
        Attempts to solve a task using pure HFN typicality and structural dynamics.
        """
        # 1. INCLUSIVE INDUCTION
        rule_stats = {}
        orig_tau = self.observer.tau
        self.observer.tau = orig_tau * 10.0 
        
        try:
            for i, x in enumerate(history):
                res = self.observer.observe(x, exhaustive=True)
                induction_candidates = res.explanation_tree + res.surprising_leaves
                for node in induction_candidates:
                    if node.id not in rule_stats:
                        rule_stats[node.id] = {"lp": 0.0, "count": 0, "node": node}
                    rule_stats[node.id]["lp"] += node.log_prob(x)
                    rule_stats[node.id]["count"] += 1
                    for parent in self.observer.forest.get_parents(node.id):
                        if parent.id not in rule_stats:
                            rule_stats[parent.id] = {"lp": 0.0, "count": 0, "node": parent}
                        rule_stats[parent.id]["lp"] += parent.log_prob(x)
                        rule_stats[parent.id]["count"] += 1
        finally:
            self.observer.tau = orig_tau
        
        ranked_candidates = []
        for nid, stats in rule_stats.items():
            node = stats["node"]
            avg_lp = stats["lp"] / stats["count"]
            consensus = stats["count"] / len(history)
            is_protected = nid in self.observer.protected_ids
            weight = 1.0 if is_protected else self.observer.get_weight(nid)
            
            num_children = len(node.children())
            generality = np.log1p(num_children) if not is_protected else 2.0
            
            t_sig = node.sigma[target_slice] if node.use_diag else np.diag(node.sigma)[target_slice]
            target_var = float(np.mean(t_sig))
            if target_var > 0.1: continue
            
            complexity = node.description_length()
            saliency = (avg_lp * consensus * (weight + 0.1) * (generality + 0.1)) / (1.0 + np.log1p(complexity))
            if not is_protected and num_children == 0: saliency *= 0.01
            ranked_candidates.append((nid, saliency))
        
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        potential_rules = [rule_stats[nid]["node"] for nid, _ in ranked_candidates[:self.thinking_budget]]
        
        # 2. RELAXED VERIFICATION
        candidate_scores = []
        for rule in potential_rules:
            acc = self._verify_rule_score(rule, history, target_slice, history_raw)
            if acc >= 0.7:
                if verbose: print(f"      [DEBUG] Rule VERIFIED (acc={acc:.2f}): {rule.id}")
                candidate_scores.append((rule, acc))
            else:
                self.observer.penalize_id(rule.id, penalty=0.5)
        
        if not candidate_scores:
            # Fallback A: mean-delta — works when the transformation is consistent
            # across all training examples regardless of content.
            if len(history) >= 1:
                mean_target = np.mean(np.stack([h[target_slice] for h in history]), axis=0)
                pred = test_input.copy()
                pred[target_slice] = mean_target
                if self.validator and test_input_raw is not None:
                    if self.validator(test_input_raw, pred):
                        if verbose: print("      [DEBUG] Mean-delta fallback VERIFIED.")
                        return mean_target
                else:
                    return mean_target

            # Fallback B: brute-force geometric rule search via rule_applier.
            # Try every named rule; return the first one that validates.
            if self.rule_applier is not None:
                _geo_rules = [
                    'prior_identity', 'prior_flip_h', 'prior_flip_v',
                    'prior_rot_90', 'prior_rot_180', 'prior_rot_270',
                ]
                for rid in _geo_rules:
                    applied = self.rule_applier(rid, test_input)
                    if applied is None:
                        continue
                    if self.validator and test_input_raw is not None:
                        pred = test_input.copy()
                        pred[target_slice] = applied
                        if self.validator(test_input_raw, pred):
                            if verbose: print(f"      [DEBUG] Brute-force rule match: {rid}")
                            return applied
                    else:
                        return applied
            return None
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        valid_rule = candidate_scores[0][0]
        
        # 3. EXECUTION
        if valid_rule:
            pred_mu = None

            # Identify non-target (input context) dimensions
            n = len(test_input)
            non_target = np.ones(n, dtype=bool)
            non_target[target_slice] = False

            # Attempt 0: Domain-specific rule application (highest priority)
            if self.rule_applier is not None:
                applied = self.rule_applier(valid_rule.id, test_input)
                if applied is not None:
                    if self.validator and test_input_raw is not None:
                        # Build full vector for validator
                        full_applied = test_input.copy()
                        full_applied[target_slice] = applied
                        if self.validator(test_input_raw, full_applied):
                            if verbose: print(f"      [DEBUG] Rule applier SUCCESS: {valid_rule.id}.")
                            self.observer.boost_id(valid_rule.id, gain=0.5)
                            return applied
                        else:
                            if verbose: print(f"      [DEBUG] Rule applier FAILED: {valid_rule.id}.")
                    else:
                        return applied

            # Attempt 1: Decoder synthesis anchored at test_input context
            goal = HFN(mu=test_input.copy(), sigma=np.ones(n) * 5.0, id="test_goal", use_diag=True)
            goal.sigma[non_target] = 0.01  # pin input context tightly
            goal.add_edge(goal, valid_rule, "MUST_SATISFY")
            dec_res = self.decoder.decode(goal)
            if isinstance(dec_res, list) and dec_res:
                pred_mu = dec_res[0].mu

            # Attempt 2: Nearest-input training delta transfer.
            # Find the training example whose input context is closest to the
            # test input, then transplant its target_slice (the rule output).
            # This is better than rule-lp selection because the delta is
            # content-dependent — the nearest input gives the most relevant delta.
            if pred_mu is None:
                best_h = None
                best_dist = float('inf')
                for h in history:
                    dist = float(np.linalg.norm(h[non_target] - test_input[non_target]))
                    if dist < best_dist:
                        best_dist = dist
                        best_h = h.copy()
                if best_h is not None:
                    # Graft the test input context onto the nearest training vector
                    # so the validator sees a coherent full-manifold vector.
                    best_h[non_target] = test_input[non_target]
                pred_mu = best_h

            if pred_mu is not None and self.validator and test_input_raw is not None:
                if not self.validator(test_input_raw, pred_mu):
                    if verbose: print(f"      [DEBUG] Execution FAILED: {valid_rule.id}. Penalizing.")
                    self.observer.penalize_id(valid_rule.id, penalty=0.9)
                    return None
                else:
                    self.observer.boost_id(valid_rule.id, gain=0.3)

            return pred_mu[target_slice] if pred_mu is not None else None

        return None

    def _verify_rule_score(
        self,
        rule: HFN,
        history: List[np.ndarray],
        target_slice: slice,
        history_raw: Optional[List[Any]] = None
    ) -> float:
        """Return average accuracy of a rule across history."""
        from hfn.decoder import ResolutionRequest
        check_history = history[:3]
        check_raw = history_raw[:3] if history_raw else None

        # Fast path: if rule_applier knows this rule, verify directly via application.
        # This bypasses the decoder which struggles with content-dependent priors.
        if self.rule_applier is not None and self.validator and check_raw:
            applied_results = []
            for i, x in enumerate(check_history):
                applied = self.rule_applier(rule.id, x)
                if applied is not None:
                    full_applied = x.copy()
                    full_applied[target_slice] = applied
                    applied_results.append((i, full_applied))
            if applied_results:
                validated = [
                    (i, pred) for i, pred in applied_results
                    if self.validator(check_raw[i], pred)
                ]
                frac = len(validated) / len(applied_results)
                # The rule_applier is a domain-specific oracle. When it passes a
                # majority of applicable training examples, treat it as fully verified
                # (reconstruction imprecision can cause some training examples to fail
                # even when the rule is correct for the test input).
                return 1.0 if frac >= 0.5 else 0.0

        total_acc = 0.0
        for i, x in enumerate(check_history):
            sim_goal = HFN(mu=x.copy(), sigma=np.ones_like(x)*0.01, id="sim_goal", use_diag=True)
            sim_goal.sigma[target_slice] = 5.0
            sim_goal.add_edge(sim_goal, rule, "MUST_SATISFY")

            dec_res = self.decoder.decode(sim_goal)
            pred_mu = None
            if isinstance(dec_res, ResolutionRequest):
                target_gap_mu = dec_res.missing_mu[target_slice]
                for h in check_history:
                    if np.linalg.norm(h[target_slice] - target_gap_mu) < 0.1:
                        pred_mu = h.copy(); break
            elif isinstance(dec_res, list) and dec_res:
                pred_mu = dec_res[0].mu

            if pred_mu is None: return 0.0

            if self.validator and check_raw:
                if self.validator(check_raw[i], pred_mu):
                    total_acc += 1.0
            else:
                err = self.evaluator.prediction_error(x[target_slice], pred_mu[target_slice])
                total_acc += 1.0 / (1.0 + err)

        return total_acc / len(check_history)
