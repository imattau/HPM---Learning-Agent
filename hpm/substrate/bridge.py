"""
hpm/substrate/bridge.py — Substrate Bridge Agent ("The Translator")

Anchors internal GaussianPattern weights to external symbolic systems.
Prevents echo-chamber effects by boosting externally-grounded patterns
and penalising ungrounded ones when the Librarian reports high redundancy.

Two-pass weight adjustment (every T_substrate steps):
  Pass 1 (Standard): w *= (1 + alpha * f_freq)        — boosts all Level 3+ patterns
  Pass 2 (Echo):     w *= (1 - gamma)                  — penalises low-freq patterns
                     (only when redundancy > threshold)

Weights are renormalised per agent after each full pass.
"""

import numpy as np


class SubstrateBridgeAgent:
    """
    Cadence-gated post-step processor that adjusts pattern weights based on
    field_frequency() scores from a connected ExternalSubstrate.

    Args:
        substrate:               Any ExternalSubstrate instance.
        store:                   Shared SQLiteStore (held as self._store).
        T_substrate:             Steps between substrate query passes.
        min_bridge_level:        Minimum pattern level included in frequency checks.
        alpha:                   Alignment boost scale.
        gamma:                   Echo-chamber grounding penalty.
        redundancy_threshold:    Redundancy level above which penalty pass activates.
        frequency_low_threshold: f_freq below which a pattern is "ungrounded".
        cache_distance_threshold: L2 norm below which cached f_freq is reused.
    """

    def __init__(
        self,
        substrate,
        store,
        T_substrate: int = 20,
        min_bridge_level: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.2,
        redundancy_threshold: float = 0.3,
        frequency_low_threshold: float = 0.2,
        cache_distance_threshold: float = 0.05,
    ):
        self._substrate = substrate
        self._store = store
        self.T_substrate = T_substrate
        self.min_bridge_level = min_bridge_level
        self.alpha = alpha
        self.gamma = gamma
        self.redundancy_threshold = redundancy_threshold
        self.frequency_low_threshold = frequency_low_threshold
        self.cache_distance_threshold = cache_distance_threshold
        self._t = 0
        self._freq_cache: dict = {}  # pattern_id -> (cached_mu, f_freq)

    def step(self, step_t: int, field_quality: dict) -> dict:
        """
        Called by MultiAgentOrchestrator after strategist.step().

        Returns {} on non-cadence steps.
        Returns bridge_report dict on cadence steps.
        """
        self._t += 1
        if self._t % self.T_substrate != 0:
            return {}

        # --- Snapshot ---
        all_records = self._store.query_all()  # (pattern, weight, agent_id)
        candidates = [
            (p, w, aid) for p, w, aid in all_records
            if getattr(p, "level", 1) >= self.min_bridge_level
        ]

        if not candidates:
            return {
                "patterns_checked": 0,
                "cache_hits": 0,
                "echo_chamber_penalty_applied": False,
                "mean_field_frequency": 0.0,
            }

        # --- Frequency cache + compute f_freq ---
        freq_map: dict = {}  # pattern_id -> f_freq
        cache_hits = 0
        for pattern, weight, agent_id in candidates:
            pid = pattern.id
            if pid in self._freq_cache:
                cached_mu, cached_freq = self._freq_cache[pid]
                if np.linalg.norm(pattern.mu - cached_mu) < self.cache_distance_threshold:
                    freq_map[pid] = cached_freq
                    cache_hits += 1
                    continue
            f_freq = float(self._substrate.field_frequency(pattern))
            self._freq_cache[pid] = (pattern.mu.copy(), f_freq)
            freq_map[pid] = f_freq

        # --- Standard Alignment Pass ---
        # Switch to ADDITIVE boost: w = w + alpha * f_freq
        # This allows rescuing near-zero patterns.
        updated_weights: dict = {}  # (pattern_id, agent_id) -> new weight after boost
        for pattern, weight, agent_id in candidates:
            pid = pattern.id
            f_freq = freq_map[pid]
            new_weight = weight + self.alpha * f_freq
            self._store.update_weight(pid, agent_id, new_weight)
            updated_weights[(pid, agent_id)] = new_weight

        # --- Echo-Chamber Audit ---
        # Switch to ADDITIVE penalty: w = max(0, w - gamma)
        echo_penalty_applied = False
        redundancy = field_quality.get("redundancy")
        if redundancy is not None and redundancy > self.redundancy_threshold:
            for pattern, weight, agent_id in candidates:
                pid = pattern.id
                if freq_map[pid] < self.frequency_low_threshold:
                    penalised = max(0.0, updated_weights[(pid, agent_id)] - self.gamma)
                    self._store.update_weight(pid, agent_id, penalised)
                    updated_weights[(pid, agent_id)] = penalised
            echo_penalty_applied = True

        # --- Per-Agent Normalisation ---
        agent_ids = {aid for _, _, aid in candidates}
        for aid in agent_ids:
            records = self._store.query(aid)
            total = sum(w for _, w in records)
            if total > 0:
                for p, w in records:
                    self._store.update_weight(p.id, aid, w / total)

        mean_freq = float(np.mean(list(freq_map.values()))) if freq_map else 0.0

        return {
            "patterns_checked": len(candidates),
            "cache_hits": cache_hits,
            "echo_chamber_penalty_applied": echo_penalty_applied,
            "mean_field_frequency": mean_freq,
        }
