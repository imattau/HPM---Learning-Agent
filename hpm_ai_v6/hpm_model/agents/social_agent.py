from typing import List, Dict, Any, Optional
import numpy as np
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.agents.learning_agent import LearningAgent
from hpm_ai_v6.hpm_model.fields.pattern_field import DynamicPatternField

class SocialAgent(LearningAgent):
    """
    Specialized HPM Agent that interacts with a shared Pattern Field.
    Automatically retrieves field amplifications before perception.
    """
    def __init__(self, 
                 patterns: List[Cell],
                 shared_field: Optional[DynamicPatternField] = None,
                 **kwargs):
        super().__init__(patterns, **kwargs)
        self.shared_field = shared_field

    def perceive(self, 
                 observation_seq: List[Cell], 
                 population: List[Cell], 
                 context: Dict[str, Any]):
        """
        Overrides perceive to inject shared field amplifications into the context.
        """
        # Inject field amplifications if shared field is available
        if self.shared_field:
            field_amps = self.shared_field.get_amplifications(self.patterns)
            # Merge with existing field_amplifications in context if any
            existing_amps = context.get("field_amplifications", {})
            for k, v in field_amps.items():
                existing_amps[k] = existing_amps.get(k, 0.0) + v
            context["field_amplifications"] = existing_amps
            
        return super().perceive(observation_seq, population, context)
