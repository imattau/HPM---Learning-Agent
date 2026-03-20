import numpy as np


class AffectiveEvaluator:
    """
    Affective evaluator (D3, §9.4).
    Curiosity signal: peaks at intermediate complexity patterns
    that are currently improving — models the Goldilocks effect.

    E_aff_i(t) = novelty(t) * capacity(t) * g(c_i) + alpha_r * r_t

    where:
      novelty(t)  = sigmoid(k * Delta_A_i(t))
      capacity(t) = 1 - novelty(t)
      g(c)        = exp(-(c - c_opt)^2 / (2 * sigma_c^2))
    """

    def __init__(
        self,
        k: float = 1.0,
        c_opt: float = 10.0,
        sigma_c: float = 5.0,
        alpha_r: float = 0.0,
    ):
        self.k = k
        self.c_opt = c_opt
        self.sigma_c = sigma_c
        self.alpha_r = alpha_r
        self._prev_accuracy: dict[str, float] = {}

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def _g(self, c: float) -> float:
        return float(np.exp(-((c - self.c_opt) ** 2) / (2.0 * self.sigma_c ** 2)))

    def update(self, pattern, current_accuracy: float, reward: float = 0.0) -> float:
        prev = self._prev_accuracy.get(pattern.id, current_accuracy)
        delta_A = current_accuracy - prev
        self._prev_accuracy[pattern.id] = current_accuracy

        novelty = self._sigmoid(self.k * delta_A)
        capacity = 1.0 - novelty
        c = pattern.description_length()

        e_aff = novelty * capacity * self._g(c)
        e_aff += self.alpha_r * reward
        return float(max(e_aff, 0.0))
