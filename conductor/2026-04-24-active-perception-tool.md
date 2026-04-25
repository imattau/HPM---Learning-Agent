# Active Perception Tool Implementation Plan

**Goal:** Give HPM AI "eyes" to interpret input structure (keys, shapes, types) by providing an active perception tool it can call before committing to a solver.

**Architecture:** A new `perception.py` module containing diagnostic functions. `base_discovery.py` gains special-case logic in `act()` to pass the full `current_task` dict and `pool` list to these tools.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| CREATE | `hpm_ai_v3/tools/perception.py` | `summarize_task` + `summarize_pool` logic |
| MODIFY | `hpm_ai_v3/tools/innate.py` | Register perception tools in ToolRegistry |
| MODIFY | `hpm_ai_v3/agents/base_discovery.py` | Pass task/pool data to perception tools in `act()` |
| CREATE | `hpm_ai_v3/tests/test_perception_tool.py` | Unit and integration tests |

---

## Task 1: Perception logic

- [ ] **Step 1: Create `hpm_ai_v3/tools/perception.py`**

```python
"""
perception.py - Diagnostic tools that give HPM AI "eyes" on its input data.
"""
import numpy as np
from typing import Any, Dict, List

def summarize_task(task: Dict[str, Any]) -> str:
    """Return a descriptive string of the task's structure."""
    if not task:
        return "Task is empty."
    
    parts = [f"Task has {len(task)} keys: {list(task.keys())}."]
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
```

---

## Task 2: Registration

- [ ] **Step 1: Register tools in `hpm_ai_v3/tools/innate.py`**

Modify `register_innate_tools()`:
```python
    from hpm_ai_v3.tools.perception import summarize_task, summarize_pool
    ToolRegistry.register("summarize_task", summarize_task, ["task"], "result", 0.005,
                          "Describe the keys and data shapes in the current task.",
                          module="hpm_ai_v3.tools.perception", function="summarize_task")
    ToolRegistry.register("summarize_pool", summarize_pool, ["pool"], "result", 0.005,
                          "Describe the shapes and types of values in the episodic pool.",
                          module="hpm_ai_v3.tools.perception", function="summarize_pool")
```

---

## Task 3: Agent Integration

- [ ] **Step 1: Update `base_discovery.py` to handle special-case tool arguments**

In `act()`, when `resolved_args` are being prepared:
Detect if `func` is `summarize_task` or `summarize_pool` and override `resolved_args`.

```python
        # Special case for perception tools that need full objects
        if func == "summarize_task":
            resolved_args = [self.current_task]
        elif func == "summarize_pool":
            resolved_args = [pool]
```

---

## Task 4: Verification

- [ ] **Step 1: Create `hpm_ai_v3/tests/test_perception_tool.py`**

Test that `summarize_task` correctly identifies a grid in a dictionary and that `base_discovery.py` correctly passes the task to the tool.

- [ ] **Step 2: Run tests**
```bash
PYTHONPATH=. pytest hpm_ai_v3/tests/test_perception_tool.py
```
