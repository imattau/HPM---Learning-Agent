"""
tool_registry.py - Global registry of available tools for HPM agents.
"""

from typing import Dict, Callable, List, Optional, Any
from .base import ToolPattern


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
                 description: str = "",
                 module: Optional[str] = None,
                 function: Optional[str] = None):
        """
        Register a tool function that can be wrapped as a ToolPattern.
        Automatically wraps tool_fn in a validation proxy if module/function provided.
        """
        final_fn = tool_fn
        if module and function:
            # Create a smart proxy that validates signature before calling
            from .innate_substrate import InnateCognitiveSubstrate
            substrate = InnateCognitiveSubstrate()
            
            def smart_proxy(**kwargs):
                # Try to call via safe_call for validation
                # Note: safe_call expects positional args or kwargs
                # We'll pass them as kwargs
                res = substrate.safe_call(module, function, **kwargs)
                return res
            
            final_fn = smart_proxy

        cls._tools[name] = {
            'fn': final_fn,
            'input_keys': input_keys,
            'output_key': output_key,
            'cost': cost,
            'description': description,
            'module': module,
            'function': function
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
    def call(cls, name: str, **kwargs) -> Any:
        """Directly call a registered tool by name."""
        if name not in cls._tools:
            raise ValueError(f"Tool '{name}' not registered.")
        tool_fn = cls._tools[name]['fn']
        return tool_fn(**kwargs)
    
    @classmethod
    def clear(cls):
        """Clear registry (useful for testing)."""
        cls._tools.clear()
