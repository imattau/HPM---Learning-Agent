from collections import deque

import numpy as np


def _empty_report() -> dict:
    """Return the canonical empty forecast_report with all 7 keys."""
    return {
        "prediction": None,
        "prediction_error": None,
        "fragility_score": None,
        "delta_nll": None,
        "fragility_flag": False,
        "top_pattern_level": None,
        "top_pattern_id": None,
    }


class PredictiveSynthesisAgent:
    """
    HPM Phase 5 — Predictive Synthesis Agent ("The Forecaster").

    Uses the highest-weight Level 4+ GaussianPattern mu as the system's
    primary structural prediction, then measures robustness via a Far
    Transfer probe (additive noise + NLL scoring).

    Called every orchestrator step; returns a forecast_report dict.
    """

    def __init__(
        self,
        store,
        probe_k: int = 10,
        probe_n: int = 5,
        probe_sigma_scale: float = 0.5,
        fragility_threshold: float = 1.0,
        min_bridge_level: int = 4,
    ):
        self._store = store
        self.probe_k = probe_k
        self.probe_n = probe_n
        self.probe_sigma_scale = probe_sigma_scale
        self.fragility_threshold = fragility_threshold
        self.min_bridge_level = min_bridge_level
        self._obs_history: deque = deque(maxlen=probe_k)

    def step(self, step_t: int, current_obs: dict, field_quality: dict) -> dict:
        """
        Called by MultiAgentOrchestrator every step.

        Args:
            step_t:        Current orchestrator step counter.
            current_obs:   Dict {agent_id: np.ndarray} of current observations.
            field_quality: Dict from StructuralLawMonitor.step() (or {}).

        Returns:
            forecast_report dict — all 7 keys always present.
        """
        # --- 3.1 Fast gate ---
        if field_quality.get("level4plus_count", 0) == 0:
            return _empty_report()

        # --- 3.2 Query store and select top pattern ---
        all_records = self._store.query_all()  # list of (pattern, weight, agent_id)

        # Filter to min_bridge_level (default 4)
        eligible = [
            (p, w) for p, w, _ in all_records
            if p.level >= self.min_bridge_level
        ]

        # Fallback to level 3 if no level 4+ found (timing discrepancy)
        if not eligible:
            eligible = [
                (p, w) for p, w, _ in all_records
                if p.level >= 3
            ]

        if not eligible:
            return _empty_report()

        # Select the highest-weight pattern
        top_pattern, _ = max(eligible, key=lambda pw: pw[1])
        top_level = top_pattern.level

        # --- 3.3 Extract observation ---
        x_obs = next(iter(current_obs.values()))
        # Defensive trim/pad to match pattern dimension
        pat_dim = top_pattern.mu.shape[0]
        if x_obs.shape[0] > pat_dim:
            x_obs = x_obs[:pat_dim]
        elif x_obs.shape[0] < pat_dim:
            x_obs = np.concatenate([x_obs, np.zeros(pat_dim - x_obs.shape[0])])

        # --- 3.4 Predict and score ---
        prediction = top_pattern.mu.copy()
        prediction_error = float(top_pattern.log_prob(x_obs))

        # --- 3.5 Update obs history ---
        self._obs_history.append(x_obs.copy())

        # --- 3.6 Far Transfer probe ---
        fragility_score = None
        delta_nll = None
        fragility_flag = False

        if len(self._obs_history) >= self.probe_k:
            sigma_diag_mean = float(np.mean(np.diag(top_pattern.sigma)))
            sigma_probe = self.probe_sigma_scale * np.sqrt(max(sigma_diag_mean, 1e-9))

            rng = np.random.default_rng()
            nlls = []
            for x_hist in self._obs_history:
                for _ in range(self.probe_n):
                    noise = rng.normal(0.0, sigma_probe, size=x_hist.shape)
                    x_perturbed = x_hist + noise
                    nlls.append(float(top_pattern.log_prob(x_perturbed)))

            fragility_score = float(np.mean(nlls))
            delta_nll = fragility_score - prediction_error
            fragility_flag = bool(delta_nll > self.fragility_threshold)

        return {
            "prediction": prediction,
            "prediction_error": prediction_error,
            "fragility_score": fragility_score,
            "delta_nll": delta_nll,
            "fragility_flag": fragility_flag,
            "top_pattern_level": top_level,
            "top_pattern_id": top_pattern.id,
        }
