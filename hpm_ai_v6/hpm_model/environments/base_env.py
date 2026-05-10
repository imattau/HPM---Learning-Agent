from abc import ABC, abstractmethod
from typing import List
from hpm_ai_v6.hpm_model.core.cell import Cell

class BaseEnvironment(ABC):
    """
    Abstract Base Class for HPM Environments.
    Provides sequences of observation cells.
    """
    @abstractmethod
    def generate_episode(self, length: int) -> List[Cell]:
        """Generate a sequence of cells (observations)."""
        pass
