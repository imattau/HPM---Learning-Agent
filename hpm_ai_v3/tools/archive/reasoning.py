"""
reasoning.py - Cognitive reasoning tools backed by learned vector substrates.
Minimal and agnostic version.
"""

from typing import Dict, Any, List, Optional
from .registry import ToolRegistry
from .vector_store import get_mapping_store

try:
    import pint
    ureg = pint.UnitRegistry()
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False


def unit_normalizer(unit_str: str) -> str:
    """Normalize unit strings using Pint. No hardcoded physics concepts."""
    if not PINT_AVAILABLE:
        return unit_str
    try:
        # Basic cleanup for common symbols
        u = unit_str.replace("^2", "**2").replace("mps2", "m/s**2").replace("mps", "m/s")
        q = ureg.parse_expression(u)
        return str(q.units) if hasattr(q, 'units') else str(q)
    except:
        return unit_str


def extract_constraints(text: str) -> Dict[str, Any]:
    """
    Extract numeric constraints from text and attempt to map them to variables
    using the learned vector store. No hardcoded physics keywords.
    """
    from .parsing import extract_numbers
    nums_res = extract_numbers(text)
    nums = nums_res.get("numbers", [])
    
    store = get_mapping_store()
    constraints = {}
    
    for n in nums:
        val = n["value"]
        raw_unit = n["unit"]
        context = n["context"]
        
        norm_unit = unit_normalizer(raw_unit) if raw_unit else None
        
        # Query learned substrate for variable mapping
        res = store.query(context)
        var = res.get("variable")
        
        if var:
            # Map the value to the discovered variable
            constraints[var] = {"value": val, "unit": norm_unit or raw_unit}
            
    return {"constraints": constraints, "status": "success"}


def unit_converter(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
    """Convert between units using Pint."""
    if not PINT_AVAILABLE:
        return {"error": "Pint missing", "status": "failed"}
    try:
        q = value * ureg.parse_expression(from_unit)
        res = q.to(to_unit)
        return {"value": res.magnitude, "unit": str(res.units), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def register_reasoning_tools():
    store = get_mapping_store()
    
    ToolRegistry.register(
        name="map_concept",
        tool_fn=store.query,
        input_keys=["text"],
        output_key="mapping",
        cost=0.01,
        description="Query the learned vector substrate for variable mappings."
    )
    ToolRegistry.register(
        name="add_concept_mapping",
        tool_fn=store.add_mapping,
        input_keys=["description", "variable"],
        output_key="status",
        cost=0.01,
        description="Allow agents to contribute new learned mappings to the substrate."
    )
    ToolRegistry.register(
        name="unit_converter",
        tool_fn=unit_converter,
        input_keys=["value", "from_unit", "to_unit"],
        output_key="converted",
        cost=0.01,
        description="Convert values between different units."
    )
    ToolRegistry.register(
        name="extract_constraints",
        tool_fn=extract_constraints,
        input_keys=["text"],
        output_key="constraints",
        cost=0.01,
        description="Extract numeric constraints and map them to variables."
    )
    print("[ReasoningTools] Registered agnostic reasoning substrate tools.")


register_reasoning_tools()
