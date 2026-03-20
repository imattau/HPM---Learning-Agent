class PatternDensity:
    """
    Computes pattern density D(h_i) in [0, 1].

    D(h_i) = alpha_conn * structural_i
            + alpha_sat * saturation_i
            + alpha_amp * field_freq_i

    structural_i  = (connectivity() + compress()) / 2
    saturation_i  = (1 - loss_norm) * capacity_i
    loss_norm     = loss / (1 + loss)   maps [0, inf) -> [0, 1)

    Stateless: call compute() once per pattern per step.
    """

    def __init__(
        self,
        alpha_conn: float = 0.33,
        alpha_sat: float = 0.33,
        alpha_amp: float = 0.34,
    ):
        self.alpha_conn = alpha_conn
        self.alpha_sat = alpha_sat
        self.alpha_amp = alpha_amp

    def compute(
        self,
        pattern,
        loss: float,
        capacity: float,
        field_freq: float,
    ) -> float:
        """
        Args:
            pattern:    object with connectivity() and compress() methods
            loss:       Running loss L_i = -A_i (non-negative). Defensively clamped to >= 0.
            capacity:   1 - novelty from AffectiveEvaluator.last_capacity()
            field_freq: Normalised field frequency in [0, 1]

        Returns:
            D(h_i) in [0, 1]
        """
        loss = max(loss, 0.0)

        structural = (pattern.connectivity() + pattern.compress()) / 2.0
        loss_norm = loss / (1.0 + loss)
        saturation = (1.0 - loss_norm) * capacity

        D = (
            self.alpha_conn * structural
            + self.alpha_sat * saturation
            + self.alpha_amp * field_freq
        )
        return float(max(0.0, min(1.0, D)))
