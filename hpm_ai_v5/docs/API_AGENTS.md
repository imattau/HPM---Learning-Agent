# HPM v5 Agents & Pipelines Reference

All agents implement the `Agent` protocol: `name`, `step(input: AgentInput) -> AgentOutput`.

---

## Base Types (`hpm_ai_v5.agents.base`)

```python
@dataclass
class AgentInput:
    packet: AdapterPacket
    goal: Any = None
    context: dict = {}

@dataclass
class AgentOutput:
    packet: AdapterPacket
    action: Any = None
    trace: list[dict] = []
```

`BaseAgent` provides default `step()` wiring and trace logging.

---

## Agent Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `SelfAdaptiveAgent` | `agents.adaptive` | Adapts its own strategy based on reward feedback; wraps a base agent |
| `LayeredAgent` | `agents.layered` | Runs multiple sub-agents in layers; aggregates their outputs |
| `HierarchicalPlanningAgent` | `agents.hierarchical_planner` | Decomposes goals into subgoals; uses `PatternManager` for episode structure |
| `MetaPatternDiscoveryAgent` | `agents.meta_pattern` | Discovers reusable meta-patterns across similar tasks; stores in `MetaPattern` objects |
| `ScoringWeightAdaptationAgent` | `agents.scoring` | Learns optimal `α, β, γ, δ` scoring weights from reward (SWA benchmark) |
| `UtilityLearningAgent` | `agents.utility` | Context-conditioned utility learning from reward signals (LUB benchmark) |
| `AutomaticAdapterComposer` | `agents.adapter_composition` | Selects and reuses best adapter pipeline for a task family (AAC benchmark) |
| `OpenAdapterDiscoveryAgent` | `agents.open_adapter_discovery` | Extends AAC with explicit deferral on unsupported task families |

---

## Agent Pipeline (`hpm_ai_v5.agents.pipeline`)

```python
class AgentPipeline:
    agents: list[Agent]

    def run(self, input: AgentInput) -> AgentOutput:
        """Runs agents in sequence, passing output packet as next input."""
```

`AgentStep` is a named wrapper around a single agent for tracing.

---

## Postprocessors (`hpm_ai_v5.postprocessors`)

Postprocessors run after the engine and convert `Action` to domain output.

| Class | Module | Notes |
|-------|--------|-------|
| `CartpoleForecastPostprocessor` | `postprocessors.physics` | Extracts action from forecast state; blends engine + heuristic |
| `AcrobotForecastPostprocessor` | `postprocessors.physics` | Acrobot-specific action extraction |
| `BinaryExplorationPostprocessor` | `postprocessors.physics` | Epsilon-greedy binary action selection |
| `NumericPostprocessor` | `postprocessors.numeric` | Returns scalar from engine forecast |
| `MultiNumericPostprocessor` | `postprocessors.numeric` | Returns vector from ensemble of engines |
| `ExplorationPostprocessor` | `postprocessors.numeric` | Adds exploration noise to numeric output |
| `ValidationOnlyAdapter` | `adapter.validation_only` | Pass-through; used in NLP/code benchmarks |
| `UCodeRenderer` | `postprocessors.code` *(pending SCB)* | Maps `U_*` sequences to Python skeletons |

---

## Pipelines (`hpm_ai_v5.pipelines`)

| Class | Module | Notes |
|-------|--------|-------|
| `AdapterPipeline` | `pipelines.adapter_pipeline` | Chains adapters; respects `requires`/`provides` ordering |
| `AgentPipeline` | `pipelines.agent_pipeline` | Chains agents; mirrors `AdapterPipeline` at the agent level |

Both implement a `.run(packet)` interface and record traces on the packet.
