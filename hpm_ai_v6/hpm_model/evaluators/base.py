from abc import ABC, abstractmethod
from typing import Dict
from pydantic import BaseModel
from hpm_ai_v6.hpm_model.core.cell import Cell

class EvaluatorResult(BaseModel):
    score: float
    metadata: Dict[str, float] = {}

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, cell: Cell, context: Dict) -> EvaluatorResult:
        pass
