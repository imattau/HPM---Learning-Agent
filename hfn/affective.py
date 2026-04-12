"""
Affective Evaluator (HPM Sections 2.5.3, 9.3, 9.4)

Implements an evaluator with affective state stored as HFN nodes:
- Global state: arousal, valence, emotional state
- Per‑pattern affect: bias, usage count, recency

Provides:
- Anxiety‑driven persistence of spurious patterns (even with high epistemic loss)
- Curiosity‑driven exploration (Goldilocks peak for intermediate learnability)
- Affective bonus that modulates weight updates
"""

from __future__ import annotations

import tempfile
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest
from hfn.evaluator import Evaluator


class AffectiveState(IntEnum):
    """Emotional states mapped to integer codes for HFN storage."""
    NEUTRAL = 0
    CURIOUS = 1
    FRUSTRATED = 2
    CONFIDENT = 3
    ANXIOUS = 4


class AffectiveEvaluator(Evaluator):
    """
    Affective evaluator with all state stored as HFN nodes.

    Global state node: mu = [arousal, valence, state_code, step_counter]
    Per‑pattern nodes: mu = [affect_score, usage_count, persistence_bias, last_step]
    """

    GLOBAL_DIM = 4
    PATTERN_DIM = 4

    def __init__(self, cold_dir: Optional[Path] = None):
        super().__init__()
        if cold_dir is None:
            cold_dir = Path(tempfile.mkdtemp(prefix="hfn_affective_"))
        self._forest = TieredForest(
            D=self.GLOBAL_DIM,
            cold_dir=cold_dir,
            forest_id="affective_evaluator"
        )
        # Global state node
        self._global_node = HFN(
            mu=np.array([0.5, 0.5, float(AffectiveState.NEUTRAL), 0.0]),
            sigma=np.ones(self.GLOBAL_DIM),
            id="affective:global",
            use_diag=True
        )
        self._forest.register(self._global_node)
        self._step_counter = 0

    # ------------------------------------------------------------------
    # Internal state access
    # ------------------------------------------------------------------

    def _get_pattern_node(self, pattern_id: str) -> HFN:
        node_id = f"affective:{pattern_id}"
        node = self._forest.get(node_id)
        if node is None:
            node = HFN(
                mu=np.array([0.5, 0.0, 0.0, 0.0]),
                sigma=np.ones(self.PATTERN_DIM),
                id=node_id,
                use_diag=True
            )
            self._forest.register(node)
        return node

    def _get_global_state(self) -> Tuple[float, float, AffectiveState]:
        arousal = float(self._global_node.mu[0])
        valence = float(self._global_node.mu[1])
        state_code = int(self._global_node.mu[2])
        state = AffectiveState(state_code) if 0 <= state_code < len(AffectiveState) else AffectiveState.NEUTRAL
        return arousal, valence, state

    def _set_global_state(self, arousal: float, valence: float, state: AffectiveState) -> None:
        self._global_node.mu[0] = max(0.0, min(1.0, arousal))
        self._global_node.mu[1] = max(0.0, min(1.0, valence))
        self._global_node.mu[2] = float(state.value)
        self._global_node.mu[3] = float(self._step_counter)

    # ------------------------------------------------------------------
    # Public API – update from outcomes
    # ------------------------------------------------------------------

    def update_from_outcome(self, pattern_id: str, success: bool, surprise: float) -> None:
        """
        Update affective state based on task outcome.

        Parameters
        ----------
        pattern_id : str
            The pattern (macro or node) involved.
        success : bool
            Whether the outcome was successful.
        surprise : float
            How unexpected the outcome was (0..1, higher = more surprising).
        """
        self._step_counter += 1
        arousal, valence, state = self._get_global_state()

        # Valence update
        valence_delta = 0.1 if success else -0.1
        valence = max(0.0, min(1.0, valence + valence_delta))

        # Arousal: surprise increases, then decays
        arousal_delta = 0.15 * min(1.0, surprise / 2.0)
        arousal = max(0.0, min(1.0, arousal + arousal_delta))
        arousal *= 0.95  # decay

        # Determine new emotional state
        if arousal < 0.3:
            state = AffectiveState.NEUTRAL
        elif arousal > 0.7 and valence > 0.6:
            state = AffectiveState.CONFIDENT
        elif arousal > 0.7 and valence < 0.4:
            state = AffectiveState.FRUSTRATED
        elif arousal > 0.6:
            state = AffectiveState.ANXIOUS
        else:
            state = AffectiveState.CURIOUS if valence > 0.5 else AffectiveState.NEUTRAL

        self._set_global_state(arousal, valence, state)

        # Update per‑pattern affect
        pnode = self._get_pattern_node(pattern_id)
        current_affect = pnode.mu[0]
        affect_delta = 0.1 if success else -0.05
        new_affect = max(0.0, min(1.0, current_affect + affect_delta))
        pnode.mu[0] = new_affect
        pnode.mu[1] += 1  # usage count
        pnode.mu[3] = float(self._step_counter)

    # ------------------------------------------------------------------
    # Affective signals
    # ------------------------------------------------------------------

    def get_affective_bonus(self, pattern_id: str) -> float:
        """
        Return E_aff – the affective evaluator signal (0..1).

        Higher bonus makes a pattern more likely to be selected/retained.
        """
        arousal, valence, state = self._get_global_state()
        pnode = self._get_pattern_node(pattern_id)
        base_affect = pnode.mu[0]
        usage = pnode.mu[1]

        if state == AffectiveState.ANXIOUS:
            # Familiarity bonus: well‑used patterns get extra boost
            familiarity = min(0.3, usage * 0.05 / 10.0)  # normalised usage
            return min(1.0, base_affect + familiarity + 0.3)
        elif state == AffectiveState.CURIOUS:
            # Explore intermediate patterns (0.3 < affect < 0.7)
            if 0.3 < base_affect < 0.7:
                return base_affect + 0.15
            return base_affect
        elif state == AffectiveState.CONFIDENT:
            # Boost successful patterns
            if base_affect > 0.6:
                return base_affect + 0.2
            return base_affect
        else:
            return base_affect

    def should_persist(self, pattern_id: str, epistemic_loss: float) -> bool:
        """
        HPM §9.3: Under anxiety, patterns persist even with poor epistemic fit.

        Parameters
        ----------
        pattern_id : str
            Pattern to evaluate.
        epistemic_loss : float in [0,1]
            Higher = worse fit.

        Returns
        -------
        bool
            True if the pattern should be kept despite high loss.
        """
        arousal, _, state = self._get_global_state()
        if state != AffectiveState.ANXIOUS:
            return False
        threshold = 0.8 if arousal > 0.7 else 0.6
        return epistemic_loss < threshold

    def curiosity_exploration_probability(self, learnability: float) -> float:
        """
        HPM §9.4: Probability of exploring a domain given its learnability.

        learnability : float in [0,1]
            Estimated structure detectability (0 = random, 1 = trivial).
        Returns
        -------
        float
            Exploration probability, peaking near learnability = 0.5.
        """
        if learnability < 0.2 or learnability > 0.8:
            return 0.1
        peak = 4 * learnability * (1 - learnability)
        _, _, state = self._get_global_state()
        if state == AffectiveState.CURIOUS:
            return min(0.9, peak * 1.5)
        return peak * 0.5

    # ------------------------------------------------------------------
    # Override Evaluator.reinforcement_signal
    # ------------------------------------------------------------------

    def reinforcement_signal(self, node_id: str) -> float:
        """
        Override to include affective bonus with external reinforcement.
        """
        external = super().reinforcement_signal(node_id)
        affective = self.get_affective_bonus(node_id)
        return 0.5 * external + 0.5 * affective

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> str:
        """Return human‑readable affective state summary."""
        arousal, valence, state = self._get_global_state()
        lines = [
            f"Affective State: {state.name} (arousal={arousal:.2f}, valence={valence:.2f})",
            "Per‑pattern affect (top 5):"
        ]
        items = []
        for node in self._forest.active_nodes():
            if node.id.startswith("affective:") and node.id != "affective:global":
                pat_id = node.id[len("affective:"):]
                items.append((pat_id, node.mu[0], node.mu[1]))
        items.sort(key=lambda x: x[1], reverse=True)
        for pat_id, affect, usage in items[:5]:
            lines.append(f"  {pat_id[:20]}: affect={affect:.2f}, usage={int(usage)}")
        return "\n".join(lines)
