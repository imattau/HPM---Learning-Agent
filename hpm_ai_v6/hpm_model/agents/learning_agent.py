from typing import List, Dict, Any
import numpy as np
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.agents.base_agent import BaseHPMAgent

class LearningAgent(BaseHPMAgent):
    """
    Standard HPM Learning Agent.
    Updates weights and embeddings based on the full HPM learning cycle.
    """
    def perceive(self, 
                 observation_seq: List[Cell], 
                 population: List[Cell], 
                 context: Dict[str, Any]):
        """
        Processes an observation sequence and executes a learning step.
        context keys: consensus_vec, field_amplifications
        """
        # Execute learning step (weight update + embedding gradient ascent)
        scores = self.learner.step(
            observation_seq=observation_seq,
            population=population,
            context=context
        )
        return scores
