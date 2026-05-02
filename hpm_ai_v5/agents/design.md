# HPM AI v5 Agents

This note adds an agent pipeline layer above the HPM core.

## Purpose

```text
Adapters transform data.
Agents coordinate behaviour.
```

The goal is to let specialised agents work together while keeping the core small and reusable.

## Updated architecture

```text
Raw input
→ Adapter pipeline
→ Polygraph views
→ HPM core
→ Agent pipeline
→ Postprocessing pipeline
→ Validated output
```

## Core principle

```text
Agents are composable goal-directed behaviours.
Adapters are composable transformations.
```

## Shared packet

Agents pass around a shared packet with raw input, goal, context, views, core actions, candidate outputs, trace, and final output fields.

## Minimal interface

```python
class Agent:
    name: str

    def step(self, packet: dict) -> dict:
        return packet
```

## Agent pipeline

Fixed pipelines should come first.
Dynamic routing can come later.

```text
RouterAgent
→ SpecialistAgent
→ CriticAgent
→ OutputAgent
```

## Relationship to the HPM hierarchy

Agent pipelines are a higher-order pattern layer.
Repeated workflows can become reusable patterns without changing the core.

## Evaluators

Agents should also be evaluated by task success, output quality, confidence, and correction cost.

## Design constraints

- Keep agents specialised.
- Keep the core independent.
- Keep pipelines traceable.
- Start fixed, then route dynamically.

## Minimal implementation path

1. Implement fixed pipelines.
2. Add router and critic agents later.
3. Track agent success and context memory.
4. Create reusable agent sequence patterns only after the fixed path is stable.
