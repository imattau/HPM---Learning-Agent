from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict
from hpm_ai_v6.core.cell import Cell

class EvaluatorResult(BaseModel):
    score: float
    metadata: Dict[str, float] = {}

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, cell: Cell, context: Dict) -> EvaluatorResult:
        pass

class EpistemicEvaluator(BaseEvaluator):
    """
    Measures prediction accuracy (Negative Log Likelihood).
    Higher score = better prediction.
    """
    def evaluate(self, cell: Cell, context: Dict) -> EvaluatorResult:
        obs_seq = context.get("observation_seq", [])
        population = context.get("population", [])
        
        if not obs_seq or cell.dim == 0:
            return EvaluatorResult(score=0.0)
            
        # L_hier implementation from toy
        nll = 0.0
        count = 0
        target_dim = cell.dim - 1
        
        probs = cell.predict_probs(population)
        for i in range(len(obs_seq) - 1):
            curr, next_obj = obs_seq[i], obs_seq[i+1]
            # Context-aware: only score if source matches (for dim=1)
            if cell.dim == 1 and curr == cell.source:
                try:
                    idx = population.index(next_obj)
                    nll -= np.log(probs[idx] + 1e-9)
                    count += 1
                except (ValueError, IndexError):
                    nll += 5.0
                    count += 1
            elif cell.dim > 1:
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
        return EvaluatorResult(score=-avg_nll, metadata={"nll": avg_nll})

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
        return EvaluatorResult(score=curiosity_score, metadata={"entropy": entropy, "target": target_entropy})

class SocialEvaluator(BaseEvaluator):
    """
    Measures alignment with social consensus and field amplification.
    """
    def evaluate(self, cell: Cell, context: Dict) -> EvaluatorResult:
        consensus_vec = context.get("consensus_vec", np.zeros_like(cell.embedding))
        field_amp = context.get("field_amplification", 0.0)
        
        # Consensus: dot product similarity
        consensus_score = cell.similarity(consensus_vec)
        
        total_social = consensus_score + field_amp
        return EvaluatorResult(score=total_social, metadata={"consensus": consensus_score, "field": field_amp})

class HPMUtilityAggregator(BaseModel):
    """
    Aggregates Epistemic, Affective, and Social scores into a single Utility value.
    Utility = beta_e * Epistemic + beta_a * Affective + beta_s * Social
    """
    beta_e: float = 1.0
    beta_a: float = 0.5
    beta_s: float = 0.5

    def compute_utility(self, cell: Cell, context: Dict) -> EvaluatorResult:
        e_res = EpistemicEvaluator().evaluate(cell, context)
        a_res = AffectiveEvaluator().evaluate(cell, context)
        s_res = SocialEvaluator().evaluate(cell, context)
        
        total_utility = (self.beta_e * e_res.score + 
                         self.beta_a * a_res.score + 
                         self.beta_s * s_res.score)
                         
        return EvaluatorResult(
            score=total_utility,
            metadata={
                "epistemic": e_res.score,
                "affective": a_res.score,
                "social": s_res.score
            }
        )
