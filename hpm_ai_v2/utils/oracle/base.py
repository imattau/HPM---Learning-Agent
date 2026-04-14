"""
BaseOracle interface and CountingOracle wrapper.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, List, Optional

class BaseOracle(ABC):
    """Abstract base class for all oracles."""
    
    @abstractmethod
    def compute_state(self, outputs: List[Any], errors: List[Optional[str]], code: str = "") -> np.ndarray:
        pass

class CountingOracle(BaseOracle):
    """Wraps any oracle with a per-task call counter."""
    
    def __init__(self, wrapped: BaseOracle) -> None:
        self.wrapped = wrapped
        self.call_count = 0

    @property
    def config(self) -> Any:
        return getattr(self.wrapped, 'config', None)

    def compute_state(self, outputs: List[Any], errors: List[Optional[str]], code: str = "") -> np.ndarray:
        self.call_count += 1
        return self.wrapped.compute_state(outputs, errors, code)
