import numpy as np
from collections import defaultdict

class PatternField:
    """Represents the social/population field: frequencies of patterns in the environment."""
    def __init__(self):
        self.frequencies = {}

    def update(self, patterns):
        total = sum(p.weight for p in patterns) + 1e-12
        self.frequencies = {p.id: p.weight / total for p in patterns}
        return self.frequencies

class InstitutionalField:
    """Implements replication, validation, and social peer review."""
    def __init__(self):
        # Tracking replication success per pattern id
        self.replication_history = defaultdict(list)

    def evaluate(self, pattern, obs_seq):
        """Perform a 'peer review' validation against current observations."""
        if pattern.complexity >= 2:
            # Replicate: does the pattern predict current data better than chance?
            ll = pattern.log_likelihood(obs_seq)
            
            # Heuristic threshold for 'replication success'
            if ll > -len(obs_seq) * 0.6:   # better than pure noise (approx -0.69)
                self.replication_history[pattern.id].append(1)
            else:
                self.replication_history[pattern.id].append(0)
            
            # Institutional boost/penalty based on recent history
            if len(self.replication_history[pattern.id]) >= 3:
                success_rate = np.mean(self.replication_history[pattern.id][-5:])
                if success_rate > 0.8:
                    return 0.5   # social validation boost
                elif success_rate < 0.3:
                    return -0.3  # reputation penalty
                    
        return 0.0
