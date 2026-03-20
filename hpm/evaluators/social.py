import numpy as np


class SocialEvaluator:
    """
    Social evaluator for observational mode (D3, D6).

    E_soc_i(t) = rho * freq_i(t)

    freq_i(t): normalised frequency of pattern UUID i across agent population [0.0, 1.0].
    rho: field frequency amplification scale (AgentConfig.rho), a positive real value.

    Stateless — caller provides current freq values from PatternField.
    """

    def __init__(self, rho: float = 1.0):
        self.rho = rho

    def evaluate(self, freq: float) -> float:
        """Return E_soc for a single pattern given its population frequency."""
        return self.rho * freq

    def evaluate_all(self, freqs: np.ndarray | list[float]) -> np.ndarray:
        """Return E_soc for each pattern. Accepts list or array of freq values."""
        return np.asarray(freqs, dtype=float) * self.rho
