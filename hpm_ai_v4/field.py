import numpy as np
from collections import defaultdict, deque
from typing import Dict, Any, Optional

class PatternField:
    """Represents the social/population field: frequencies of patterns in the environment."""
    def __init__(self):
        self.frequencies = {}
        self.community_affinity: Dict[int, float] = {}
        self.generalization_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=8))
        self.generalization_margin: Dict[int, float] = {}

    def record_generalization(self, pattern_id: int, train_ll: float, holdout_ll: float, chunk_size: Optional[int] = None) -> float:
        """Record whether a pattern generalised beyond its training chunk."""
        margin = float(holdout_ll) - float(train_ll)
        scale = max(1.0, float(chunk_size or 1))
        score = float(1.0 / (1.0 + np.exp(-(margin / np.sqrt(scale)))))
        history = self.generalization_history[int(pattern_id)]
        history.append(score)
        if history:
            affinity = float(np.mean(history))
        else:
            affinity = 0.5
        self.community_affinity[int(pattern_id)] = affinity
        self.generalization_margin[int(pattern_id)] = margin
        return affinity

    def affinity_for(self, pattern_or_id: Any) -> float:
        pattern_id = int(getattr(pattern_or_id, "id", pattern_or_id))
        return float(self.community_affinity.get(pattern_id, 0.5))

    def update(self, patterns, episode_stats: Optional[Dict[int, Dict[str, float]]] = None):
        if episode_stats:
            for pattern_id, stats in episode_stats.items():
                train_ll = stats.get("train_ll")
                holdout_ll = stats.get("holdout_ll")
                if train_ll is None or holdout_ll is None:
                    continue
                self.record_generalization(
                    int(pattern_id),
                    train_ll=float(train_ll),
                    holdout_ll=float(holdout_ll),
                    chunk_size=stats.get("chunk_size"),
                )

        weighted = []
        for p in patterns:
            affinity = self.affinity_for(p)
            weighted.append((p.id, float(p.weight) * (0.5 + affinity)))

        total = sum(weight for _, weight in weighted) + 1e-12
        self.frequencies = {pattern_id: weight / total for pattern_id, weight in weighted}
        return self.frequencies

from typing import Optional, List
from hpm_ai_v4.tools.dictionary import DictionaryValidator
from hpm_ai_v4.tools.grammar import GrammarValidator

class InstitutionalField:
    """Implements replication, validation, and social peer review."""
    def __init__(self, dictionary: Optional[DictionaryValidator] = None, 
                 grammar: Optional[GrammarValidator] = None):
        # Tracking replication success per pattern id
        self.replication_history = defaultdict(list)
        self.dictionary = dictionary
        self.grammar = grammar

    def evaluate(self, pattern, obs_seq) -> float:
        """Perform a 'peer review' validation against current observations."""
        score = 0.0
        if pattern.complexity >= 2:
            # 1. Replication check
            ll = pattern.log_likelihood(obs_seq)
            if ll > -len(obs_seq) * 0.6:   # better than pure noise
                self.replication_history[pattern.id].append(1)
            else:
                self.replication_history[pattern.id].append(0)
            
            # Reputation boost/penalty
            if len(self.replication_history[pattern.id]) >= 3:
                success_rate = np.mean(self.replication_history[pattern.id][-5:])
                if success_rate > 0.8:
                    score += 0.3
                elif success_rate < 0.3:
                    score -= 0.3
                    
            # 2. Lexical & Syntactic Validation
            if self.dictionary or self.grammar:
                # Use pattern to predict the next few characters, form words
                predicted_words = self._simulate_words(pattern, obs_seq, num_words=3)
                
                # Lexical bonus
                if self.dictionary and predicted_words:
                    word = predicted_words[0]
                    if len(word) > 2:
                        if self.dictionary.contains(word):
                            score += 0.3
                        elif self.dictionary.is_prefix(word):
                            score += 0.1
                        else:
                            score -= 0.1
                            
                # Grammar bonus
                if self.grammar and len(predicted_words) >= 2:
                    g_score = self.grammar.score_sequence(predicted_words)
                    score += 0.4 * (g_score - 0.5) # Scale to [-0.2, 0.2] bonus
                        
        return score

    def _simulate_words(self, pattern, obs_seq, num_words: int = 3) -> List[str]:
        """Stochastically simulate next tokens to form multiple words."""
        words = []
        curr_obs = list(obs_seq[-10:])
        
        for _ in range(num_words):
            sim_tokens = []
            for _ in range(15): # Max chars per word
                dist = pattern.predict_next_distribution(curr_obs)
                token = np.random.choice(len(dist), p=dist / (dist.sum() + 1e-12))
                if token in (0, 2): # Space/boundary
                    curr_obs.append(int(token))
                    break
                sim_tokens.append(int(token))
                curr_obs.append(int(token))
            
            if sim_tokens:
                chars = [chr(t + 32) if t < 256 else '?' for t in sim_tokens]
                words.append("".join(chars).lower())
            else:
                break
        return words

    def _simulate_word(self, pattern, obs_seq) -> str:
        words = self._simulate_words(pattern, obs_seq, num_words=1)
        return words[0] if words else ""
