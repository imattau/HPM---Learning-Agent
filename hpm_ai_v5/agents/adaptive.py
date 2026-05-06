"""Self-adaptive HPM agent that tracks environment shifts and adjusts strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import BaseAgent
from .scoring import ScoringWeightAdaptationAgent
from ..adapter.changepoint import ChangepointAdapter
from ..adapter.packet import AdapterPacket
from ..pipeline import HPMPipeline, PipelineResult


@dataclass
class SelfAdaptiveAgent(BaseAgent):
    """
    An agent that uses Changepoint detection to sense regime shifts
    and SWA to optimize its scoring weights online.
    """

    swa: ScoringWeightAdaptationAgent = field(default_factory=lambda: ScoringWeightAdaptationAgent())
    changepoint: ChangepointAdapter = field(default_factory=lambda: ChangepointAdapter(window_size=16, threshold=1.5))
    adaptation_epsilon: float = 0.1
    min_epsilon: float = 0.01
    epsilon_decay: float = 0.98

    def step_adaptive(self, raw: Any, *, env_name: str = "default", context: dict[str, Any] | None = None) -> PipelineResult:
        """
        Run one step of the adaptive loop.
        1. Select weights via SWA.
        2. Execute HPM pipeline.
        3. Detect shifts via Changepoint (using Polygraph signal).
        4. Adapt if necessary.
        """
        if self.pipeline is None:
            raise ValueError("SelfAdaptiveAgent requires an initialized pipeline")
            
        ctx = dict(context or {})
        
        # 1. Weights from SWA
        weights = self.swa._weights_for_env(env_name)
        
        # 2. Pipeline Step
        res = self.pipeline.step(raw, goal=weights, context=ctx)
        
        # 3. Regime Detection (using Polygraph sum as shift signal)
        pg_sum = sum(s.score for s in res.polygraph_scores.values()) if res.polygraph_scores else 0.5
        cp_packet = AdapterPacket(raw=float(pg_sum))
        self.changepoint.run(cp_packet)
        
        # Merge changepoint context into result carry_context
        res.carry_context.update({
            "regime_changed": cp_packet.context.get("regime_changed"),
            "shift_score": cp_packet.context.get("shift_score"),
        })
        
        if cp_packet.context.get("regime_changed"):
            # Shift detected: increase exploration and decay old patterns
            self.adaptation_epsilon = 0.3
            for p in self.pipeline.engine.store.patterns:
                p.utility *= 0.6
        else:
            # Gradually converge
            self.adaptation_epsilon = max(self.min_epsilon, self.adaptation_epsilon * self.epsilon_decay)
            
        return res

    def observe_reward(self, env_name: str, res: PipelineResult, reward: float) -> None:
        """Provide external feedback to the SWA layer."""
        if res.action and res.action.selected_pattern:
            self.swa.observe_reward(
                env_name, 
                res.action.selected_pattern, 
                res.input.state, 
                float(reward)
            )
