"""
perception.py - Diagnostic tools that give HPM AI "eyes" on its input data.
"""
import numpy as np
from typing import Any, Dict, List

def summarize_task(task: Dict[str, Any]) -> str:
    """Return a descriptive string of the task's structure."""
    if not task:
        return "Task is empty."
    
    visible_keys = [k for k in task.keys() if k != "answer"]
    parts = [f"Task has {len(visible_keys)} visible keys: {visible_keys}."]
    for k, v in task.items():
        if k == "answer": continue # Don't reveal the answer
        desc = _describe_any(v)
        parts.append(f"  - '{k}': {desc}")
    
    return "\n".join(parts)

def summarize_pool(pool: List[Any]) -> str:
    """Return a descriptive string of the episodic pool contents."""
    if not pool:
        return "Pool is empty."
    
    parts = [f"Pool has {len(pool)} items."]
    for i, item in enumerate(pool[:10]): # Limit to first 10
        desc = _describe_any(item)
        parts.append(f"  [{i}]: {desc}")
    
    return "\n".join(parts)

def _describe_any(val: Any) -> str:
    """Helper to describe any value's type and shape."""
    t = type(val).__name__
    if isinstance(val, (list, tuple)):
        shape = f"len={len(val)}"
        if len(val) > 0:
            inner_t = type(val[0]).__name__
            shape += f", contains {inner_t}s"
            # Detect 2D grid
            if isinstance(val[0], (list, tuple)):
                shape = f"{len(val)}x{len(val[0])} grid of {type(val[0][0]).__name__}s"
        return f"{t} ({shape})"
    if isinstance(val, np.ndarray):
        return f"numpy array (shape={val.shape}, dtype={val.dtype})"
    if isinstance(val, dict):
        return f"dict (keys={list(val.keys())})"
    if isinstance(val, str):
        return f"string (len={len(val)}, '{val[:30]}...')"
    if isinstance(val, (int, float, bool)):
        return f"{t} (value={val})"
    return f"{t}"
