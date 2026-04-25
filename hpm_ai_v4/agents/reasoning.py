import numpy as np
from typing import List, Optional
from hpm_ai_v4.pattern import HierarchicalPattern

from hpm_ai_v4.tools.dictionary import DictionaryValidator
from hpm_ai_v4.tools.grammar import GrammarValidator

class Reasoner:
    """Deliberative reasoning layer sitting above the HPM core."""
    def __init__(self, agent, dictionary: Optional[DictionaryValidator] = None,
                 grammar: Optional[GrammarValidator] = None):
        self.agent = agent
        self.dictionary = dictionary
        self.grammar = grammar

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
        
        # Determine next state distribution from context
        if not obs_seq:
            next_state = pattern.pi @ pattern.A
        else:
            alpha, _ = pattern._forward(obs_seq[-20:])
            next_state = alpha[-1] @ pattern.A
            
        # Intervened distribution: force next observation to be intervention_idx
        intervened_dist = np.zeros(pattern.obs_dim, dtype=np.float32)
        intervened_dist[intervention_idx % pattern.obs_dim] = 1.0
        
        return orig_dist, intervened_dist

    def simulate(self, pattern: HierarchicalPattern, initial_obs_seq: List[int], steps: int = 10) -> List[int]:
        """Generate a possible future sequence using the pattern as a generative model."""
        if initial_obs_seq:
            alpha, _ = pattern._forward(initial_obs_seq[-20:])
            state_dist = alpha[-1]
        else:
            state_dist = pattern.pi.copy()

        simulated = []
        K = pattern.latent_dim
        for _ in range(steps):
            # Sample current state
            z = np.random.choice(K, p=state_dist / (state_dist.sum() + 1e-12))
            # Sample observation
            obs_probs = pattern.B[z]
            next_obs = np.random.choice(pattern.obs_dim, p=obs_probs / (obs_probs.sum() + 1e-12))
            simulated.append(int(next_obs))
            # Transition
            state_dist = pattern.A[z]
        return simulated

    def simulate_future(self, steps: int = 10, top_k: int = 3) -> List[int]:
        """Generate imagined future sequence by sampling from blended population prediction."""
        context = list(self.agent.obs_buffer[-20:]) if self.agent.obs_buffer else []
        simulated: List[int] = []

        for _ in range(steps):
            relevant = self.get_relevant_patterns(context, top_k=top_k)
            if not relevant:
                simulated.append(0)
                continue
            dist = self.compose_predictions(relevant, context)
            dist = dist / (dist.sum() + 1e-12)
            obs = int(np.random.choice(len(dist), p=dist))
            simulated.append(obs)
            context.append(obs)
            if len(context) > 40:
                context = context[-40:]

        return simulated

    def plan(self, goal_state: int, horizon: int = 5, num_rollouts: int = 10,
             require_valid_words: bool = True, require_grammatical: bool = True) -> List[int]:
        """
        Search for a sequence of observations that reaches the goal state using stochastic rollouts.
        Prunes invalid word completions and ungrammatical POS transitions.
        """
        best_seq = []
        best_score = -np.inf
        
        curr_obs_base = self.agent.obs_buffer[-20:] if self.agent.obs_buffer else []
        
        for _ in range(num_rollouts):
            seq = []
            curr_obs = list(curr_obs_base)
            for _ in range(horizon):
                relevant = self.get_relevant_patterns(curr_obs, top_k=3)
                if not relevant: break
                
                # Get blended prediction
                dist = self.compose_predictions(relevant, curr_obs)
                dist = dist / (dist.sum() + 1e-12)
                
                # Sample a candidate
                action = np.random.choice(len(dist), p=dist)
                
                # Linguistic guidance
                if self._is_word_boundary(curr_obs, action):
                    # Check grammar when finishing a word
                    if self.grammar and require_grammatical:
                        prev_word = self._last_word(curr_obs)
                        curr_word = self._partial_word(curr_obs, action)
                        if prev_word and curr_word:
                            if not self.grammar.is_valid_transition(prev_word, curr_word):
                                # Penalise by sampling again
                                action = np.random.choice(len(dist), p=dist)
                else:
                    # Check dictionary prefix
                    if self.dictionary and require_valid_words:
                        word_so_far = self._partial_word(curr_obs, action)
                        if word_so_far and not self.dictionary.is_prefix(word_so_far):
                            action = np.random.choice(len(dist), p=dist)
                
                seq.append(int(action))
                curr_obs.append(int(action))
                
            if seq:
                # Score based on goal proximity
                score = -abs(seq[-1] - goal_state)
                
                # Linguistic bonus
                if self._word_completed(curr_obs):
                    last_word = self._last_word(curr_obs)
                    # Lexical
                    if self.dictionary:
                        score += 0.5 * self.dictionary.score_word(last_word)
                    # Syntactic
                    if self.grammar:
                        prev_word = self._word_before_last(curr_obs)
                        if prev_word and last_word:
                            if self.grammar.is_valid_transition(prev_word, last_word):
                                score += 0.3
                    
                if score > best_score:
                    best_score = score
                    best_seq = seq
                    
        return best_seq

    def _is_word_boundary(self, obs_seq, next_obs) -> bool:
        """Space (approx token 0 in TextAdapter or 2 in CharClass) indicates boundary."""
        if not obs_seq: return True
        # Space followed by letter/digit
        return obs_seq[-1] in (0, 2) and next_obs not in (0, 2)

    def _partial_word(self, obs_seq, next_obs) -> str:
        """Reconstruct partial word from recent tokens."""
        # Find last space
        tokens = obs_seq + [next_obs]
        last_space = -1
        for i in range(len(tokens)-1, -1, -1):
            if tokens[i] in (0, 2): # Space token
                last_space = i
                break
        
        word_tokens = tokens[last_space+1:]
        if not word_tokens: return ""
        
        # Convert tokens to chars. Assuming TextAdapter (token + 32 = ord)
        chars = []
        for t in word_tokens:
            if t < 256:
                chars.append(chr(t + 32))
            else:
                chars.append('?')
        return "".join(chars)

    def _word_completed(self, obs_seq) -> bool:
        """True if last token was a space or punctuation."""
        if not obs_seq: return False
        return obs_seq[-1] in (0, 2)

    def _last_word(self, obs_seq) -> str:
        if not obs_seq: return ""
        # Find the segment before the last space
        idx = -1
        for i in range(len(obs_seq)-2, -1, -1):
            if obs_seq[i] in (0, 2):
                idx = i
                break
        return self._partial_word(obs_seq[:idx+1], obs_seq[idx+1] if idx+1 < len(obs_seq) else 0)

    def _word_before_last(self, obs_seq) -> str:
        """Extract the word before the last completed word."""
        if not obs_seq: return ""
        # Find last space
        spaces = [i for i, t in enumerate(obs_seq) if t in (0, 2)]
        if len(spaces) < 2: return ""
        
        last_space = spaces[-1]
        second_last_space = spaces[-2]
        
        word_tokens = obs_seq[second_last_space+1 : last_space]
        if not word_tokens: return ""
        
        chars = [chr(t + 32) if t < 256 else '?' for t in word_tokens]
        return "".join(chars).lower()

    def explain(self, pattern: HierarchicalPattern) -> str:
        """Translate pattern structure into human-readable description."""
        if pattern.complexity >= 1:
            likely_obs = np.argmax(pattern.B[np.argmax(pattern.pi)])
            return (f"Hierarchical pattern (ID:{pattern.id}) predicts observation {likely_obs} "
                    f"via {pattern.latent_dim} latent states.")
        else:
            return f"Pattern (ID:{pattern.id}) is unknown."
