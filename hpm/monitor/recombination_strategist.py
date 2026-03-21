"""
hpm/monitor/recombination_strategist.py — Recombination Strategist ("The Innovator")

Population-level governor that mutates per-agent config parameters to prevent
premature convergence and stimulate structural novelty.

Three intervention modes:
  - Recombination Burst: temporarily lowers conflict_threshold on all agents
  - Adoption Scaling: adjusts kappa_0 based on diversity trend (EMA)
  - Conflict Scale Damping: reduces beta_c when conflict persists (one-way ratchet)
"""


class RecombinationStrategist:
    """
    Reads field_quality from StructuralLawMonitor and mutates agent configs.

    Composed into MultiAgentOrchestrator as an optional strategist=None kwarg.
    Degrades gracefully: if field_quality["diversity"] is None, all interventions
    that require heavy metrics are skipped.
    """

    def __init__(
        self,
        diversity_low: float = 0.5,
        conflict_high: float = 0.3,
        stagnation_window: int = 3,
        burst_conflict_threshold: float = 0.01,
        burst_duration: int = 50,
        burst_cooldown: int = 100,
        kappa_0_min: float = 0.05,
        kappa_0_max: float = 0.3,
        kappa_0_ema_alpha: float = 0.2,
        beta_c_min: float = 0.1,
        beta_c_decay: float = 0.9,
    ):
        self.diversity_low = diversity_low
        self.conflict_high = conflict_high
        self.stagnation_window = stagnation_window
        self.burst_conflict_threshold = burst_conflict_threshold
        self.burst_duration = burst_duration
        self.burst_cooldown = burst_cooldown
        self.kappa_0_min = kappa_0_min
        self.kappa_0_max = kappa_0_max
        self.kappa_0_ema_alpha = kappa_0_ema_alpha
        self.beta_c_min = beta_c_min
        self.beta_c_decay = beta_c_decay

        # Internal state
        self._stagnation_count: int = 0
        self._burst_steps_remaining: int = 0
        self._cooldown_steps_remaining: int = 0
        self._original_conflict_thresholds: dict = {}
        self._diversity_ema: float | None = None
        self._conflict_persistent_cycles: int = 0

    def step(self, step_t: int, field_quality: dict, agents: list) -> dict:
        """
        Called by MultiAgentOrchestrator after monitor.step().

        Args:
            step_t:        Current orchestrator step counter.
            field_quality: Dict from StructuralLawMonitor.step() (or {}).
            agents:        List of Agent instances (mutable config).

        Returns:
            interventions dict with keys:
              burst_active, kappa_0, beta_c_scaled, stagnation_count, cooldown_remaining
        """
        # Step 1: Tick down burst / cooldown (before stagnation check)
        if self._burst_steps_remaining > 0:
            self._burst_steps_remaining -= 1
            if self._burst_steps_remaining == 0:
                self._restore_conflict_thresholds(agents)
                self._cooldown_steps_remaining = self.burst_cooldown
        elif self._cooldown_steps_remaining > 0:
            self._cooldown_steps_remaining -= 1

        diversity = field_quality.get("diversity")  # None on light steps
        conflict = float(field_quality.get("conflict", 0.0))

        kappa_0_applied = None
        beta_c_scaled = False

        if diversity is not None:
            # --- Stagnation check + burst fire (only outside burst/cooldown) ---
            if self._burst_steps_remaining == 0 and self._cooldown_steps_remaining == 0:
                if diversity < self.diversity_low and conflict > self.conflict_high:
                    self._stagnation_count += 1
                else:
                    self._stagnation_count = 0

                if self._stagnation_count >= self.stagnation_window:
                    self._fire_burst(agents)

            # --- Adoption Scaling (kappa_0) ---
            kappa_0_applied = self._update_kappa_0(diversity, agents)

            # --- Conflict Scale Damping (beta_c) ---
            beta_c_scaled = self._update_beta_c(conflict, agents)

        return {
            "burst_active": self._burst_steps_remaining > 0,
            "kappa_0": kappa_0_applied,
            "beta_c_scaled": beta_c_scaled,
            "stagnation_count": self._stagnation_count,
            "cooldown_remaining": self._cooldown_steps_remaining,
        }

    # ------------------------------------------------------------------
    # Burst
    # ------------------------------------------------------------------

    def _fire_burst(self, agents: list) -> None:
        for agent in agents:
            self._original_conflict_thresholds[agent.agent_id] = agent.config.conflict_threshold
            agent.config.conflict_threshold = self.burst_conflict_threshold
        self._burst_steps_remaining = self.burst_duration
        self._stagnation_count = 0

    def _restore_conflict_thresholds(self, agents: list) -> None:
        for agent in agents:
            if agent.agent_id in self._original_conflict_thresholds:
                agent.config.conflict_threshold = self._original_conflict_thresholds[agent.agent_id]
        self._original_conflict_thresholds = {}

    # ------------------------------------------------------------------
    # Adoption Scaling
    # ------------------------------------------------------------------

    def _update_kappa_0(self, diversity: float, agents: list) -> float | None:
        if self._diversity_ema is None:
            self._diversity_ema = diversity
            return None  # No nudge on first heavy step

        self._diversity_ema = (
            self.kappa_0_ema_alpha * diversity
            + (1 - self.kappa_0_ema_alpha) * self._diversity_ema
        )

        if diversity == self._diversity_ema:
            return None

        kappa_0_result = None
        for agent in agents:
            current = agent.config.kappa_0
            if diversity > self._diversity_ema:
                new_k0 = current + self.kappa_0_ema_alpha * (self.kappa_0_max - current)
            else:
                new_k0 = current - self.kappa_0_ema_alpha * (current - self.kappa_0_min)
            agent.config.kappa_0 = max(self.kappa_0_min, min(self.kappa_0_max, new_k0))
            kappa_0_result = agent.config.kappa_0

        return kappa_0_result

    # ------------------------------------------------------------------
    # Conflict Scale Damping
    # ------------------------------------------------------------------

    def _update_beta_c(self, conflict: float, agents: list) -> bool:
        if conflict > self.conflict_high:
            self._conflict_persistent_cycles += 1
        else:
            self._conflict_persistent_cycles = 0

        if self._conflict_persistent_cycles >= self.stagnation_window:
            for agent in agents:
                if hasattr(agent.config, "beta_c"):
                    agent.config.beta_c = max(self.beta_c_min, agent.config.beta_c * self.beta_c_decay)
            return True

        return False
