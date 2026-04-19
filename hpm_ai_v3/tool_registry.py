"""
tool_registry.py - Global registry of available tools for HPM agents.
"""

from typing import Dict, Callable, List, Optional, Any
from .tool_pattern import ToolPattern


class ToolRegistry:
    """Global registry of available tools that HPM agents can discover and use."""
    _tools: Dict[str, Dict] = {}
    
    @classmethod
    def register(cls, 
                 name: str, 
                 tool_fn: Callable, 
                 input_keys: List[str], 
                 output_key: str, 
                 cost: float = 0.1,
                 description: str = ""):
        """
        Register a tool function that can be wrapped as a ToolPattern.
        """
        cls._tools[name] = {
            'fn': tool_fn,
            'input_keys': input_keys,
            'output_key': output_key,
            'cost': cost,
            'description': description
        }
    
    @classmethod
    def create_pattern(cls, name: str) -> Optional[ToolPattern]:
        """Create a ToolPattern instance for a registered tool."""
        if name not in cls._tools:
            return None
        info = cls._tools[name]
        return ToolPattern(
            tool_fn=info['fn'],
            input_keys=info['input_keys'],
            output_key=info['output_key'],
            tool_name=name,
            cost=info['cost']
        )
    
    @classmethod
    def list_tools(cls) -> List[str]:
        """Return list of available tool names."""
        return list(cls._tools.keys())
    
    @classmethod
    def get_tool_info(cls, name: str) -> Optional[Dict]:
        """Return metadata for a tool."""
        return cls._tools.get(name)
    
    @classmethod
    def clear(cls):
        """Clear registry (useful for testing)."""
        cls._tools.clear()
