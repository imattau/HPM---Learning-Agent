import numpy as np
from typing import List, Optional
from hpm_ai_v4.pattern import HierarchicalPattern

class Reasoner:
    """Deliberative reasoning layer sitting above the HPM core."""
    def __init__(self, agent):
        self.agent = agent

    def get_relevant_patterns(self, context_obs: List[int], top_k: int = 5) -> List[HierarchicalPattern]:
        """Return patterns with highest predictive likelihood for given context."""
        if not context_obs:
            return sorted(self.agent.patterns, key=lambda p: p.weight, reverse=True)[:top_k]
            
        scores = []
        for p in self.agent.patterns:
            ll = p.log_likelihood(context_obs[-20:])  # Check against recent history
            # Combine current likelihood with established replicator weight
            score = p.weight * np.exp(ll / max(1, len(context_obs[-20:])))
            scores.append((p, score))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scores[:top_k]]

    def compose_predictions(self, patterns: List[HierarchicalPattern], obs_seq: List[int]) -> np.ndarray:
        """Combine multiple patterns by weighted averaging of their predictive distributions."""
        if not patterns:
            return np.array([0.5, 0.5])
            
        preds = []
        for p in patterns:
            dist = p.predict_next_distribution(obs_seq)
            preds.append((p.weight, dist))
            
        total_weight = sum(w for w, _ in preds) + 1e-12
        blended = np.zeros_like(preds[0][1])
        for w, d in preds:
            blended += (w / total_weight) * d
        return blended

    def counterfactual(self, pattern: HierarchicalPattern, obs_seq: List[int], intervention_idx: int):
        """
        Force a latent state or emission and observe the change in prediction.
        Returns (original_dist, intervened_dist).
        """
        orig_dist = pattern.predict_next_distribution(obs_seq)
        
        # Temporary intervention: force emission to the intervention_idx
        B_orig = pattern.B.copy()
        for z1 in range(pattern.latent_dim):
            pattern.B[z1] = 0.0
            pattern.B[z1, intervention_idx] = 1.0
            
        intervened_dist = pattern.predict_next_distribution(obs_seq)
        pattern.B = B_orig # Restore
        
        return orig_dist, intervened_dist

    def simulate(self, pattern: HierarchicalPattern, initial_obs_seq: List[int], steps: int = 10) -> List[int]:
        """Generate a possible future sequence using the pattern as a generative model."""
        belief = pattern.get_belief(initial_obs_seq)
        simulated = []
        
        K = pattern.latent_dim
        for _ in range(steps):
            # Sample current latent state from belief
            flat_belief = belief.flatten()
            sample_idx = np.random.choice(len(flat_belief), p=flat_belief / (flat_belief.sum() + 1e-12))
            z3_idx, z2_idx, z1_idx = np.unravel_index(sample_idx, belief.shape)
            
            # Sample next observation
            next_obs = np.random.choice([0, 1], p=pattern.B[z1_idx])
            simulated.append(next_obs)
            
            # Transition latent state
            new_z3 = np.random.choice(K, p=pattern.A3[z3_idx])
            new_z2 = np.random.choice(K, p=pattern.A32[new_z3])
            new_z1 = np.random.choice(K, p=pattern.A21[new_z2])
            
            # Update belief to delta for next step simulation
            belief = np.zeros_like(belief)
            belief[new_z3, new_z2, new_z1] = 1.0
            
        return simulated

    def plan(self, goal_state: int, horizon: int = 5, num_rollouts: int = 10) -> List[int]:
        """Search for a sequence of observations that reaches the goal state."""
        best_seq = []
        best_score = -np.inf
        
        curr_obs_base = self.agent.obs_buffer[-20:] if self.agent.obs_buffer else []
        
        for _ in range(num_rollouts):
            seq = []
            curr_obs = list(curr_obs_base)
            for _ in range(horizon):
                relevant = self.get_relevant_patterns(curr_obs, top_k=1)
                if not relevant: break
                
                dist = relevant[0].predict_next_distribution(curr_obs)
                action = np.argmax(dist) # In this simple sim, 'action' is targeting next obs
                seq.append(int(action))
                curr_obs.append(int(action))
                
            if seq:
                # Score based on goal proximity
                score = -abs(seq[-1] - goal_state)
                if score > best_score:
                    best_score = score
                    best_seq = seq
                    
        return best_seq

    def explain(self, pattern: HierarchicalPattern) -> str:
        """Translate pattern structure into human-readable description."""
        if pattern.complexity >= 2:
            most_common_z1 = np.argmax(np.sum(pattern.B, axis=1)) # Heuristic
            likely_obs = np.argmax(pattern.B[most_common_z1])
            return (f"Hierarchical pattern (ID:{pattern.id}) predicts observation {likely_obs} "
                    f"via {pattern.complexity} levels of latent abstraction.")
        else:
            theta = getattr(pattern, 'theta', 0.5)
            return f"Flat pattern (ID:{pattern.id}) predicts state 1 with probability {theta:.2f}."
