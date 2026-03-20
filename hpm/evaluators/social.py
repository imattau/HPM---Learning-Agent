class SocialEvaluator:
    """
    Social evaluator for observational mode (D3, D6).

    E_soc_i(t) = rho * freq_i(t)

    freq_i(t): normalised frequency of pattern UUID i across agent population.
    rho: field frequency amplification scale (AgentConfig.rho).

    Stateless — caller provides current freq values from PatternField.
    """

    def __init__(self, rho: float = 1.0):
        self.rho = rho

    def evaluate(self, freq: float) -> float:
        """Return E_soc for a single pattern given its population frequency."""
        return self.rho * freq

    def evaluate_all(self, freqs: list[float]) -> list[float]:
        """Return E_soc for each pattern in the list."""
        return [self.rho * f for f in freqs]
