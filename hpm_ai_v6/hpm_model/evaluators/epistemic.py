import numpy as np
from typing import Dict, List
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.evaluators.base import BaseEvaluator, EvaluatorResult

class EpistemicEvaluator(BaseEvaluator):
    """
    Measures prediction accuracy (Negative Log Likelihood).
    Higher score = better prediction.
    """
    def evaluate(self, cell: Cell, context: Dict) -> EvaluatorResult:
        obs_seq = context.get("observation_seq", [])
        population = context.get("population", [])
        temperature = context.get("temperature", 1.0)
        
        if not obs_seq or cell.dim == 0:
            return EvaluatorResult(score=0.0)
            
        nll = 0.0
        count = 0
        target_dim = cell.dim - 1
        
        probs = cell.predict_probs(population, temperature=temperature)
        for i in range(len(obs_seq) - 1):
            curr, next_obj = obs_seq[i], obs_seq[i+1]
            
            # Context-aware: only score if source matches (for dim=1)
            if cell.dim == 1:
                if cell.source and curr.name == cell.source.name:
                    try:
                        idx = population.index(next_obj)
                        nll -= np.log(probs[idx] + 1e-9)
                        count += 1
                    except (ValueError, IndexError):
                        nll += 5.0 # Penalty for unseen
                        count += 1
            else:
                # Generalized for higher dims
                if next_obj.dim == target_dim:
                    try:
                        idx = population.index(next_obj)
                        nll -= np.log(probs[idx] + 1e-9)
                        count += 1
                    except (ValueError, IndexError):
                        nll += 5.0
                        count += 1

        avg_nll = nll / count if count > 0 else 2.0
        return EvaluatorResult(score=-avg_nll, metadata={"nll": avg_nll, "count": count})
