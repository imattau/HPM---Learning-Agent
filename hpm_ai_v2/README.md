# hpm_ai_v2 — Domain-Agnostic HPM Agent Layer

A reusable, mixin-based agent framework built on top of the `hfn/` core. Replaces the experiment-by-experiment pattern from `hpm_fractal_node/experiments/` with a composable architecture that works across any domain (integers, strings, lists, graphs, etc.) without modification.

## Architecture Overview

```
hpm_ai_v2/
├── agents/
│   ├── base_agent.py       # BaseHFNAgent — foundation for all agents
│   ├── mixins/             # HPM level-specific capabilities
│   │   ├── l2_macro.py     # L2: Schema composition and decomposition
│   │   ├── l3_relational.py # L3: Meta-schema discovery
│   │   ├── l4_forward.py   # L4: Forward model + imaginative BFS
│   │   ├── social.py       # Social field + blackboard scaffolding
│   │   ├── recombination.py # Structural recombination (Appendix E)
│   │   └── sequential_composition.py # AST-based robust composition
│   └── agents.py           # Ready-to-use concrete agent classes
├── domains/
│   ├── base.py             # DomainConfig base class
│   ├── list_domain.py      # List transformation concepts (SP61-67)
│   └── math_physics_domain.py # Extended concepts for multi-domain (SP68+)
├── utils/
│   ├── executor.py         # PythonExecutor — sandboxed code execution
│   ├── oracle.py           # EmpiricalOracle, CountingOracle
│   ├── renderer.py         # ListRenderer — HFN node tree → Python source
│   ├── forward_model.py    # StateTransitionModel — per-node delta learning
│   └── meta_controller.py  # MetaStrategyController, SolveRecord
└── experiments/
    ├── README.md           # Experiments Index — validation suite documentation
    ├── run_sp67.py         # SP67: Multi-agent social recombination example
    └── experiment_sp80_comparative_benchmark.py # SOTA Comparative Benchmark
```

---

## Quick Start

```python
from hpm_ai_v2.agents.agents import InducedSchemaAgent

agent = InducedSchemaAgent()

# Register a strategy (BFS is built-in)
success, code, strategy = agent.solve(
    inputs=[[1, 2, 3]],
    outputs=[[2, 3, 4]],
    goal_type="map",
    task_id="MAP_plus1",
)
print(success, code)
```

---

## Concrete Agents

All live in `agents/agents.py` and are ready to use without subclassing.

| Class | Mixins | Use when |
|---|---|---|
| `InducedSchemaAgent` | Base + L2 + L3 | Schema learning from I/O examples; meta-schema discovery |
| `ImaginativeAgent` | Base + L2 + L4 | Zero-oracle BFS via learned forward model |
| `AnalogicalAgent` | Base + L2 | Cross-domain transfer via structural substitution |
| `SocialAnalogicalAgent` | Base + L2 + L4 + Social + Recombination + Composition | Multi-agent experiments with shared forest, blackboard and sequential composition |

```python
from hpm_ai_v2.agents.agents import (
    InducedSchemaAgent,
    ImaginativeAgent,
    AnalogicalAgent,
    SocialAnalogicalAgent,
)
from hpm_ai_v2.agents.mixins.social import SocialForest

# Social experiment: two agents sharing a forest
shared_forest = SocialForest(D=54, cold_dir="/tmp/shared")
alice = SocialAnalogicalAgent("Alice", shared_forest)
bob   = SocialAnalogicalAgent("Bob",   shared_forest)
```

---

## BaseHFNAgent

`agents/base_agent.py` — the foundation every agent inherits from.

### Constructor parameters

| Parameter | Default | Description |
|---|---|---|
| `cold_dir` | `"data/knowledge_base/hpm_ai_v2"` | Persistent storage directory |
| `forest_class` | `TieredForest` | HFN forest implementation |
| `hot_cap` | `10_000` | Max nodes in hot (in-memory) tier |
| `tau` | `0.5` | Observer learning rate |
| `compression_cooccurrence_threshold` | `2` | Min co-occurrences before node compression |
| `use_density_tracker` | `False` | Enable pattern density tracking (HPM App A.8) |
| `use_affective_evaluator` | `False` | Enable affective state evaluator (HPM §9.3) |
| `n_workers` | `os.cpu_count()` | Workers for parallel BFS evaluation |

### Key attributes

| Attribute | Type | Description |
|---|---|---|
| `forest` | `TieredForest` | Pattern substrate — stores all HFN nodes |
| `observer` | `Observer` | Pattern dynamics — learning, compression, weights |
| `retriever` | `GoalConditionedRetriever` | Goal-conditioned nearest-node lookup |
| `renderer` | `ListRenderer` | Converts HFN node tree → executable Python |
| `executor` | `PythonExecutor` | Runs code strings against input batches |
| `oracle` | `EmpiricalOracle` | Computes empirical state vector from execution results |
| `meta` | `MetaStrategyController` | L5: learns which strategy works in which context |
| `patterns` | `dict[str, HFN]` | Named learned patterns (macros, schemas, etc.) |
| `n_workers` | `int` | CPU workers for parallel BFS |

### Key methods

```python
# Register a custom solve strategy
agent.add_strategy("my_strategy", fn)   # fn(inputs, outputs) -> Optional[List[HFN]]

# Solve a task (tries strategies in meta-controller ranked order)
success, code, strategy_used = agent.solve(inputs, outputs, goal_type="map", task_id="t1")

# Persist and restore
agent.save_state()
agent.load_state()
```

### Built-in strategies

| Strategy | Method | Description |
|---|---|---|
| `exact` | `_try_exact` | Retrieves top-5 nearest nodes, tests directly |
| `bfs` | `_try_bfs` | Beam BFS; evaluates each depth level in parallel |

---

## Mixins

Mixins add HPM level-specific capabilities. Combine them via multiple inheritance.

### L2MacroMixin (`mixins/l2_macro.py`)

Schema composition and decomposition.

```python
# Register a learned macro from a node path
node = agent.register_macro("MAP_plus1", path=[var_inp, add1, list_append], inputs=[[1,2]], outputs=[[2,3]])

# Decomposition strategy: substitute one constituent op
agent.add_strategy("decompose", agent._try_decompose)
```

### L3RelationalMixin (`mixins/l3_relational.py`)

Discovers a common structural prefix across macros — the L3 meta-schema.

```python
meta_node = agent.discover_meta_schema()
# Returns HFN node with id="meta_list_iteration" if ≥2 list macros share a prefix
```

### L4ForwardModelMixin (`mixins/l4_forward.py`)

Learns per-node state deltas and uses them for zero-oracle imaginative BFS.

```python
# Record transitions after solving (call after register_macro)
agent._record_transitions(path, training_inputs)

# Imaginative BFS: navigates predicted state space, oracle called only to verify
agent.add_strategy("imagine", agent._try_imagine)
```

### SocialMixin + SocialForest (`mixins/social.py`)

Shared forest with blackboard for institutional scaffolding (HPM §9.7).

```python
# SocialForest is a TieredForest with a blackboard
shared = SocialForest(D=54, cold_dir="/tmp/shared")

# Post and query failures
shared.post_failure("task_X", "rule_Y", "No solution found")
failures = shared.get_failures("task_X")   # list of {rule, reason, time}

# Exchange patterns between agents
agent.exchange_patterns(partner_agent)
```

### RecombinationMixin (`mixins/recombination.py`)

Structural recombination of two patterns into a novel node (HPM Appendix E).

```python
new_node = agent.recombine_patterns(macro_a, macro_b)
score = agent.insight_score(new_node, test_inputs, test_outputs)  # novelty + effectiveness

agent.add_strategy("recombine", agent._try_recombine)
```

### SequentialCompositionMixin (`mixins/sequential_composition.py`)

Robust sequential composition of two macros via AST transformation. Generates a wrapper function calling the first then the second.

```python
# Create a composite macro (f then g)
composite_node = agent.compose_sequential("macro_add1", "macro_mul2", "compound_add1_mul2")

# Composition strategy: try pairs of existing macros
agent.add_strategy("compose", agent._try_sequential_compose)
```

## Domains

The `hpm_ai_v2` framework is domain-agnostic. Domain semantics are encapsulated in `DomainConfig` objects.

### `domains/base.py`

Defines the `DomainConfig` base class which manages manifold dimensions and concept mappings.

```python
from hpm_ai_v2.domains.list_domain import ListDomainConfig

config = ListDomainConfig()
# config.S_DIM = 20      — empirical state vector length
# config.DIM    = 14     — number of concept dimensions
# config.m_dim  = 54     — total HFN node dimensionality
```

Available domains:
- `ListDomainConfig`: Standard 14-concept set for list transformations.
- `MathPhysicsDomainConfig`: Extended concept set for mathematical and physical reasoning.

---

## Utils

### `utils/executor.py`

```python
from hpm_ai_v2.utils.executor import PythonExecutor

executor = PythonExecutor()
results, errors = executor.run_batch("x = inp\nx += 1", inputs=[1, 2, 3])
# results: [2, 3, 4], errors: [None, None, None]
```

The module also exports `_eval_path_worker` — the picklable worker used by the parallel BFS. You should not need to call it directly.

### `utils/oracle.py`

```python
from hpm_ai_v2.utils.oracle import ListOracle, CountingOracle

oracle = ListOracle(config)
state = oracle.compute_state(results, errors, code_str)  # returns np.ndarray shape (20,)

# CountingOracle wraps any oracle and tracks call count
counting = CountingOracle(oracle)
counting.call_count  # int
```

### `utils/meta_controller.py`

```python
from hpm_ai_v2.utils.meta_controller import MetaStrategyController, SolveRecord

ctrl = MetaStrategyController()
ranked = ctrl.rank_strategies("map", n_macros=3)   # list of strategy names, best first
ctrl.record(SolveRecord(...))
```

### `utils/forward_model.py`

```python
from hpm_ai_v2.utils.forward_model import StateTransitionModel

model = StateTransitionModel()
model.record_path(path, state_sequence)            # learn deltas
predicted = model.predict(current_state, node)     # predict next state
```

---

## Parallel BFS

BFS candidate evaluation uses `ProcessPoolExecutor` by default. Each depth level is evaluated in parallel — the first path that matches exits the pool immediately.

```python
# Use all CPUs (default)
agent = InducedSchemaAgent()

# Limit workers
agent = InducedSchemaAgent(n_workers=2)

# Disable parallelism
agent = InducedSchemaAgent(n_workers=1)
```

Workers re-execute code strings in isolated processes — no shared mutable state, no GIL contention.

---

## Building New Agents with Mixins

The mixin architecture is designed to make creating new agents straightforward and modular. Each mixin adds a specific HPM capability. By combining different mixins you can build agents with any subset of these capabilities without duplicating code.

`BaseHFNAgent` provides the domain-agnostic core — pattern storage, strategy dispatch, BFS, persistence. Mixins then add:

- **New strategies** — `_try_decompose`, `_try_imagine`, `_try_social`, `_try_recombine`
- **New methods** — `register_macro`, `discover_meta_schema`, `exchange_patterns`
- **New state** — `forward_model`, `shared_forest`

### Agent variants

| Agent | Mixins | Capabilities |
|---|---|---|
| `InducedSchemaAgent` | L2, L3 | Macro learning + meta-schema discovery |
| `ImaginativeAgent` | L2, L4 | Macros + mental simulation |
| `AnalogicalAgent` | L2 | Macros + cross-domain analogy (add `_try_analogy` manually) |
| `SocialAnalogicalAgent` | L2, L4, Social, Recombination, Composition | Full L2–L5 + social + recombination + composition |
| Minimal (just `BaseHFNAgent`) | — | Exact match and BFS only |

### Creating a new agent

```python
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.agents.mixins.l2_macro import L2MacroMixin
from hpm_ai_v2.agents.mixins.l4_forward import L4ForwardModelMixin
from hpm_ai_v2.agents.mixins.social import SocialMixin, SocialForest

class MyAgent(BaseHFNAgent, L2MacroMixin, L4ForwardModelMixin, SocialMixin):
    def __init__(self, agent_id: str, shared_forest: SocialForest, **kwargs):
        super().__init__(**kwargs)
        L4ForwardModelMixin.__init__(self)
        SocialMixin.__init__(self, agent_id, shared_forest)
        self.add_strategy("decompose", self._try_decompose)
        self.add_strategy("imagine",   self._try_imagine)
        self.add_strategy("social",    self._try_social)
        self.add_strategy("my_custom", self._my_custom_strategy)

    def _my_custom_strategy(self, inputs, outputs):
        # Return List[HFN] path on success, None on failure
        ...
```

`register_macro`, `discover_meta_schema`, `exchange_patterns`, `_record_transitions` — all inherited from the mixins, no reimplementation needed.

### Why this matters

- **Separation of concerns** — each mixin focuses on exactly one HPM level or mechanism
- **Reusability** — `L2MacroMixin` is shared across `InducedSchemaAgent`, `ImaginativeAgent`, and `SocialAnalogicalAgent`
- **Testability** — each mixin can be tested in isolation
- **Extensibility** — to add a new HPM level, write a new mixin and combine it with existing ones
- **Experimentation** — compare agents with different capability sets by swapping mixins in or out

---

## Relationship to hpm_fractal_node

`hpm_ai_v2` is a **new layer alongside** `hpm_fractal_node/experiments/` — it does not replace or modify it. The experiments in `hpm_fractal_node/` remain the canonical validated implementations of SP61–SP67. `hpm_ai_v2` extracts their reusable patterns into a composable framework for future work.

The `hfn/` package is the shared core used by both.

---

## Running the SOTA Benchmark

```bash
cd /path/to/HPM---Learning-Agent
PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp80_comparative_benchmark.py
```
