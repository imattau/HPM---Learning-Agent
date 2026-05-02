# HPM AI v5 Agents

## 1. Purpose

This note adds an agent pipeline layer above the HPM core.

```text
Adapters transform data.
Agents coordinate behaviour.
```

The goal is to let specialised agents work together while keeping the HPM core small and reusable.

## 2. Updated Architecture

```text
Raw input
→ Adapter pipeline
→ Polygraph views
→ HPM core
→ Agent pipeline
→ Postprocessing pipeline
→ Validated output
```

Or, for orchestration:

```text
Input
→ RouterAgent
→ SpecialistAgent
→ CriticAgent
→ OutputAgent
→ Final response
```

## 3. Core Principle

```text
Agents are composable goal-directed behaviours.
Adapters are composable transformations.
```

Adapters prepare structure.
Agents decide what to do with it.

## 4. Shared Agent Packet

Agents pass around a shared packet:

```python
packet = {
    "raw_input": None,
    "goal": None,
    "context": {},
    "views": [],
    "core_actions": [],
    "candidate_outputs": [],
    "agent_trace": [],
    "final_output": None,
}
```

Each agent reads the packet, adds or changes fields, then passes it on.
The canonical shared context is `packet.context`; agent and adapter traces are recorded on the same packet so pipelines stay inspectable.
Core decisions should carry an explicit reasoning trace, not free-form prose, so agents can surface why a pattern or sequence won.

## 5. Minimal Agent Interface

```python
class Agent:
    name: str

    def step(self, packet: dict) -> dict:
        return packet
```

This mirrors the adapter style, but at a behavioural level.

## 6. Minimal AgentPipeline

```python
class AgentPipeline:
    def __init__(self, agents):
        self.agents = agents

    def run(self, packet):
        for agent in self.agents:
            step = getattr(agent, "step_packet", None)
            if callable(step):
                packet = step(packet)
            else:
                packet = agent.step(packet)
            packet["agent_trace"].append(agent.name)
        return packet
```

Start with fixed pipelines. Add dynamic routing later.

## 7. Example: Summarisation Pipeline

```text
RouterAgent
→ SummaryAgent
→ CriticAgent
→ OutputAgent
```

RouterAgent determines task type.
SummaryAgent builds a SummaryPlan.
CriticAgent checks coverage, confidence, and invalid assumptions.
OutputAgent renders final summary.

## 8. Example: Chat Pipeline

```text
RouterAgent
→ DialogueAgent
→ ToolAgent or ResponseAgent
→ CriticAgent
→ OutputAgent
```

Example behaviour:

```text
current-info request → ToolAgent
ambiguous request → ClarificationAgent
design request → DesignAgent
simulation request → SimulationAgent
```

## 9. Relationship to HPM Hierarchy

Agent pipelines are a higher-order pattern layer.

```text
State
→ Delta
→ Pattern
→ PatternSequence
→ Agent
→ AgentSequence
```

A repeated agent workflow can itself become a reusable pattern.

## 10. Evaluators at Agent Level

Agents should also be evaluated.

```text
agent_score =
  task_success
+ output_quality
+ confidence
- correction_cost
- validation_failures
```

## 11. Agent Context Memory

Agents should track:

```text
context → agent success
```

This mirrors pattern-level context memory.

## 12. Design Constraints

- Keep agents specialised.
- Keep the core independent.
- Keep pipelines traceable.
- Start fixed, then route dynamically.

## 13. Failure Modes

| Failure | Result |
| --- | --- |
| Too many agents | orchestration overhead |
| Overloaded agents | hard to debug |
| No critic/validator | unreliable output |
| No router | wrong specialist selected |
| No trace | opaque behaviour |
| Agent logic inside core | architecture drift |

## 14. Minimal Implementation Path

1. Implement fixed pipelines.
2. Add RouterAgent.
3. Add CriticAgent.
4. Track agent success and context memory.
5. Create reusable AgentSequence patterns only after the fixed path is stable.

## 15. Key Distinction

```text
Adapters transform representations.
Agents transform task state.
```

Adapters answer:

```text
What structure is present?
```

Agents answer:

```text
What should be done?
```

When an agent consumes a core decision, it should preserve the structured reasoning trace inside its own trace payload.

## 16. Key Takeaway

```text
The HPM core provides pattern intelligence.
Adapters provide domain structure.
Agents provide goal-directed behaviour.
Agent pipelines provide reusable workflows.
```
