# Innate Cognitive Substrate Design
Date: 2026-04-22

## Problem
HPM v3's bootstrapping fails because argument wiring is fragile and innate priors
compete with learned patterns in the population. Patterns fail not because the
strategy is wrong but because arguments are mis-typed or mis-wired. The current
`_generate_argument_pool`, `_resolve_binding`, and `trial_args` logic is scattered,
heuristic, and unreliable.

## Solution
A permanent `InnateCognitiveSubstrate` singleton that handles all argument
resolution between population pattern selection and tool execution. The population
selects *what* to call; the substrate handles *how* to call it correctly.

The substrate is never in the population, never subject to replicator dynamics,
and never changes. It is permanent cognitive infrastructure — available at every
phase from -2.0 to 10.0.

## Architecture

```
Population selects pattern (module, function)
    ↓
InnateCognitiveSubstrate.resolve_call(module, function, pool, task_text)
    ↓  [inspects signature → coerces pool values → matches to params]
Validated (fn, resolved_args) pair
    ↓
ToolRegistry.call() executes
    ↓
Result → population reward + absorption
```

## Innate Tool Set

Implemented as plain Python functions, registered permanently at import.
Never in the population. Three groups:

### Group A: Introspection
- `inspect_signature(module, function)` → `{params: [{name, type, default}]}`
- `get_type(value)` → `"int" | "float" | "str" | "list" | "dict" | "bool"`
- `describe_value(value)` → `{type, length, preview, numeric_value_if_coercible}`
- `list_callable_functions(module)` → `[{name, signature}]`

### Group B: Type Operations
- `coerce(value, target_type)` → typed value or None on failure
- `to_int(value)` → int or None
- `to_float(value)` → float or None
- `to_str(value)` → str
- `to_list(value)` → list
- `safe_call(module, function, *args)` → result or error dict (never raises)

### Group C: Pattern Matching / Perception
- `extract_numbers(text)` → `[float, ...]`
- `match_regex(pattern, text)` → `[str, ...]`
- `find_in_list(value, lst)` → index or None
- `split_text(text, delimiter=None)` → `[str, ...]`
- `compare_values(a, b)` → `"equal" | "greater" | "less" | "incomparable"`

## resolve_call Algorithm

```python
def resolve_call(self, module, function, pool, task_text):
    # 1. Inspect target function signature
    sig = self.inspect_signature(module, function)
    if not sig:
        return (module, function, [task_text])  # fallback

    # 2. For each parameter, find best-matching value from pool
    resolved = []
    for param in sig["params"]:
        best = self._match_param(param, pool, task_text)
        resolved.append(best)

    return (module, function, resolved)

def _match_param(self, param, pool, task_text):
    # Try pool values in order, coerce to param type, return first success
    target_type = param.get("type")
    for val in pool:
        coerced = self.coerce(val, target_type)
        if coerced is not None:
            return coerced
    # Fallback: extract from task_text
    if target_type in ("int", "float"):
        nums = self.extract_numbers(task_text)
        if nums:
            return self.coerce(nums[0], target_type)
    return task_text  # last resort
```

## Replacement Scope

Remove from `base_discovery.py`:
- `_generate_argument_pool()` method
- `_resolve_binding()` method
- `trial_args` generation logic in `act()`
- `is_bound` / `arg_bindings` trial sampling loop

Replace with single call:
```python
module, function, resolved_args = self.substrate.resolve_call(
    action_pattern.module,
    action_pattern.function,
    self._get_pool(),  # simple: task inputs + episodic results
    self.current_task.get("text", "")
)
```

`_get_pool()` replaces `_generate_argument_pool()` — same data, no resolution logic.

## Files

- **CREATE**: `hpm_ai_v3/tools/innate_substrate.py` — `InnateCognitiveSubstrate` class + all 15 innate functions
- **MODIFY**: `hpm_ai_v3/agents/base_discovery.py` — remove scattered wiring logic, add substrate call in `act()`
- **MODIFY**: `hpm_ai_v3/agents/discovery_agent.py` — remove redundant argument handling in `act()` override

## What Does Not Change
- Population dynamics (replicator, weights, evaluators)
- Demo absorption (`observe_demo`, `_absorb_discovery`)
- Curriculum loading and advancement
- ToolRegistry and existing tool registrations
- Pattern selection logic in `act()`

## Success Criteria
- Agent can call `math.sqrt(144)` correctly without pre-bound args
- Agent can call `re.findall(r'\d+', text)` correctly from task text
- Agent can call `operator.add(a, b)` with correct int args from pool
- `split` / `arithmetic` innate pattern failures no longer pollute population
- Curriculum advances through Sensory Priming and beyond within 200 episodes
