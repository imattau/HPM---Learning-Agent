import numpy as np

class PatternInspector:
    """Advanced analysis of a pattern's internal generative states."""
    def __init__(self, pattern):
        self.pattern = pattern

    def summary(self):
        """Print a summary of the pattern's current properties."""
        print(f"Pattern ID: {self.pattern.id}")
        print(f"Complexity: {self.pattern.complexity} levels")
        print(f"Running loss: {self.pattern.running_loss:.4f}")
        print(f"Current weight: {self.pattern.weight:.4f}")

    def transition_entropy(self):
        """Compute the average entropy across the hierarchy's transition matrices."""
        if self.pattern.complexity < 2:
            return {"A": 0.0} # Flat pattern
            
        entropies = {}
        # H(P) = -sum p(x)log(p(x))
        for name, mat in [('A3', self.pattern.A3), ('A32', self.pattern.A32), ('A21', self.pattern.A21)]:
            # Compute row-wise entropy
            ent = -np.sum(mat * np.log(mat + 1e-12), axis=1)
            entropies[name] = np.mean(ent)
            
        return entropies
