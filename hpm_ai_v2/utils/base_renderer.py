"""
Base Renderer class for HPM agents.
Defines the abstract interface for turning HFN nodes into executable or representable strings.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hfn.hfn import HFN

class Renderer(ABC):
    """
    Abstract Base Class for HPM renderers.
    Responsible for converting pattern representations (HFN nodes) into strings.
    """
    
    @abstractmethod
    def render(self, node: HFN) -> str:
        """
        Convert an HFN node (or composite tree) into a generic string representation.
        (e.g., Python code source, LaTeX formula, etc.)
        """
        pass

    @abstractmethod
    def render_function(self, node: HFN, func_name: str = "macro_func") -> str:
        """
        Convert an HFN node into a standalone function or macro representation.
        """
        pass
