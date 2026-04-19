"""
LearningAgent: base class for HFN agents that modify the forest through observation.
Enforces a unified semantic encoding space and observer-driven learning dynamics.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hfn.hfn import HFN

if TYPE_CHECKING:
    from hfn.observer import ExplanationResult

class LearningAgent(BaseHFNAgent):
    """
    Agents that 'learn' by observing new signals and modifying forest structure/weights.
    Ensures all learners operate in the same representational space.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_enabled = True

    def observe(self, x: np.ndarray, exhaustive: bool = False) -> ExplanationResult:
        """Observe a signal and update forest structure/weights."""
        return self.observer.observe(x, exhaustive=exhaustive)

    def predict(self, context: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate a prediction from current state/context.
        [HPM CORE] Prediction before observation.
        """
        if context is not None:
            # Simple heuristic: retrieve closest node and use its 'forward' link
            # For now, we return current context as 'expected' state (identity prediction)
            return context
        return np.zeros(self.m_dim)

    def predict_and_observe(self, x: np.ndarray) -> ExplanationResult:
        """
        Full HPM Cycle: Predict -> Observe -> Compute Error -> Update.
        Ensures prediction error (surprise) drives reinforcement and creation.
        """
        # 1. Prediction (L4)
        # In simple ingestion, we might predict from the last observed passage
        last_mu = getattr(self, "_last_mu", None)
        prediction = self.predict(last_mu)
        
        # 2. Observation (L1-L3)
        result = self.observe(x)
        
        # 3. Prediction Error (Surprise)
        # result.residual_surprise already contains the -log_prob error
        
        # 4. State Update
        self._last_mu = x
        
        return result
