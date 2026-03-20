# ResourceCostEvaluator Design Spec

**Date:** 2026-03-21

## Goal

Add a per-pattern resource cost signal to the HPM agent evaluator pipeline. Patterns that are computationally expensive (high description length) are penalised more strongly when the system is under real memory/CPU pressure. This grounds the HPM principle that pattern retention is constrained by energy and resource availability.

## Background

HPM explicitly identifies energy, time, risk, and social context as constraints on pattern dynamics. The existing evaluators capture epistemic accuracy, affective curiosity, and social frequency — but none model computational cost. The `ResourceCostEvaluator` fills this gap.

`GaussianPattern.description_length()` already provides a per-pattern complexity measure. The new evaluator scales this by a real-time system pressure scalar derived from `psutil`.

## Formula

```
E_cost_i(t) = -lambda_cost * description_length(pattern_i) * pressure(t)

pressure(t) = w_mem * (mem_percent / 100) + w_cpu * (cpu_percent / 100)
```

- `pressure(t)` ∈ [0, 1]: zero when idle, one when both memory and CPU are maxed
- `E_cost_i` is negative — it lowers `Total_i` for complex patterns under pressure
- `lambda_cost` controls the overall sensitivity of the penalty

The total evaluator sum becomes:

```
Total_i = A_i + beta_aff * E_aff_i + gamma_soc * E_soc_i + delta_cost * E_cost_i
```

`delta_cost = 0.0` by default — the signal is off unless explicitly enabled. All existing agents are completely unaffected.

## Components

### `hpm/evaluators/resource_cost.py`

New class `ResourceCostEvaluator`:

```python
ResourceCostEvaluator(
    lambda_cost: float = 1.0,   # penalty scale
    w_mem: float = 0.5,          # memory weight in pressure
    w_cpu: float = 0.5,          # CPU weight in pressure
)
```

Methods:
- `pressure() -> float` — reads `psutil.virtual_memory().percent` and `psutil.cpu_percent()`, returns weighted scalar in [0, 1]
- `evaluate(pattern) -> float` — returns `-lambda_cost * pattern.description_length() * self.pressure()`

### `AgentConfig` (`hpm/config.py`)

Four new fields, all with defaults that preserve backward compatibility:

```python
delta_cost: float = 0.0     # weight of E_cost in Total_i (0 = off)
lambda_cost: float = 1.0    # penalty scale inside ResourceCostEvaluator
w_mem: float = 0.5          # memory weight in pressure scalar
w_cpu: float = 0.5          # CPU weight in pressure scalar
```

### `hpm/agents/agent.py`

- Instantiate `ResourceCostEvaluator` in `__init__` using config values
- In `step()`: compute `e_cost_i` per pattern, include in `Total_i` as `delta_cost * e_cost_i`
- Add `e_cost_mean` to the returned metrics dict

### `hpm/evaluators/__init__.py`

Re-export `ResourceCostEvaluator`.

## Dependencies

- `psutil` — one new dependency. Add to `pyproject.toml` / `requirements.txt`.

## Backward Compatibility

- `delta_cost = 0.0` default means `E_cost` contributes nothing to `Total_i` — no behaviour change for existing agents or tests.
- All 115 existing tests continue to pass unchanged.

## Testing

### Unit tests (`tests/evaluators/test_resource_cost.py`)

- `test_pressure_zero_when_idle` — mock psutil to return 0% mem/CPU; assert `pressure() == 0.0`
- `test_pressure_one_when_maxed` — mock psutil to return 100% both; assert `pressure() == 1.0`
- `test_evaluate_returns_negative` — any non-zero pressure + non-trivial pattern gives negative output
- `test_evaluate_zero_when_delta_cost_zero` — with `lambda_cost=0`, cost is always 0.0
- `test_complex_pattern_penalised_more_than_simple` — two patterns, same pressure; higher description_length → more negative E_cost
- `test_agent_unaffected_when_delta_cost_zero` — agent step with `delta_cost=0.0` produces same metrics as before

### Integration

- `test_agent_step_includes_e_cost_mean` — agent with `delta_cost > 0` returns `e_cost_mean` in step dict
- `test_high_pressure_prunes_complex_patterns_faster` — simulated high pressure (mocked psutil) with two patterns of different complexity; over N steps, the complex pattern loses weight faster

## Error Handling

- `psutil` unavailable: `ResourceCostEvaluator.__init__` should raise `ImportError` with a clear message directing the user to `pip install psutil`
- `psutil.cpu_percent()` returns `None` on first call on some platforms: treat as `0.0`

## What This Is Not

- Not a hard cap or circuit breaker — it is a soft evaluator signal, not a kill switch
- Not disk monitoring — disk I/O is slower-moving and less relevant to per-step pattern cost; deferred to a future phase
- Not a substrate — it reads system state, not external knowledge
