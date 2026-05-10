import numpy as np
from typing import Dict
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.evaluators.base import BaseEvaluator, EvaluatorResult

class SocialEvaluator(BaseEvaluator):
    """
    Measures alignment with social consensus and field amplification.
    """
    def evaluate(self, cell: Cell, context: Dict) -> EvaluatorResult:
        consensus_vec = context.get("consensus_vec", np.zeros_like(cell.embedding))
        field_amp = context.get("field_amplification", 0.0)
        
        # Consensus: cosine similarity to group vector
        consensus_score = cell.similarity(consensus_vec)
        
        total_social = consensus_score + field_amp
        return EvaluatorResult(score=total_social, metadata={"consensus": float(consensus_score), "field": float(field_amp)})
