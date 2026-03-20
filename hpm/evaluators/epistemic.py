import numpy as np


class EpistemicEvaluator:
    """
    Implements D2-D3 epistemic evaluator.
    Maintains EMA running loss L_i(t) per pattern.
    Returns accuracy A_i(t) = -L_i(t).
    """

    def __init__(self, lambda_L: float = 0.1):
        self.lambda_L = lambda_L
        self._running_loss: dict[str, float] = {}

    def update(self, pattern, x: np.ndarray) -> float:
        """
        Update running loss for pattern given observation x.
        Returns A_i(t) = -L_i(t).
        """
        ell = pattern.log_prob(x)   # instantaneous loss (D2)
        prev = self._running_loss.get(pattern.id, 0.0)  # L_i(0) = 0 per spec §3.2
        L = (1.0 - self.lambda_L) * prev + self.lambda_L * ell
        self._running_loss[pattern.id] = L
        return -L   # A_i(t)

    def accuracy(self, pattern_id: str) -> float:
        return -self._running_loss.get(pattern_id, 0.0)
