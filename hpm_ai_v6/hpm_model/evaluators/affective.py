import numpy as np
from typing import Dict
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.evaluators.base import BaseEvaluator, EvaluatorResult

class AffectiveEvaluator(BaseEvaluator):
    """
    Measures curiosity/complexity based on predictive entropy.
    Peaks at intermediate complexity (Prediction 9.4).
    """
    def __init__(self, target_entropy_ratio: float = 0.5):
        self.target_entropy_ratio = target_entropy_ratio

    def evaluate(self, cell: Cell, context: Dict) -> EvaluatorResult:
        population = context.get("population", [])
        if cell.dim == 0 or not population:
            return EvaluatorResult(score=0.0)
            
        probs = cell.predict_probs(population)
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        max_entropy = np.log(len(population))
        target_entropy = self.target_entropy_ratio * max_entropy
        
        curiosity_score = -abs(entropy - target_entropy)
        return EvaluatorResult(score=curiosity_score, metadata={"entropy": float(entropy), "target": float(target_entropy)})
