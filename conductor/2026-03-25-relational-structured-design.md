# Relational Bundle and Structural Message Design

## Goal

Improve the baseline `hpm` structured stack by adding:

1. A minimal relational bundle layer for cross-level handoff.
2. A constrained, decodable inter-agent structural message protocol.

The design borrows the useful part of `hpm_ai_v1`:

- explicit structural representation
- relation-aware abstraction
- inspectable communication of higher-level patterns

It does **not** import the self-modifying code loop, opaque private dialects, or lossy source-to-source rewriting.

## Scope

In scope:

- new relational dataclasses/helpers in baseline `hpm`
- optional encoding of relational bundles for structured/hierarchical agents
- transport of structural messages through `PatternField`
- optional relay of structural messages in `MultiAgentOrchestrator`
- focused tests for bundle/message behavior

Out of scope:

- code mutation or rewrite systems
- learned opaque private language
- replacement of existing vector encoders
- changes to benchmark logic beyond new tests

## Design Principles

1. Backward compatible by default.
Existing agent and orchestrator behavior must remain unchanged unless structural messaging is explicitly enabled.

2. Decodable and inspectable.
Messages must remain human-readable and serializable into simple primitives.

3. Minimal abstraction.
Add small typed structures near existing seams instead of building a new subsystem.

4. HPM-aligned.
Relations are treated as an additional substrate for structured patterns, not as a separate intelligence stack.

## Proposed Structures

### RelationalEdge

A small typed relation token:

- `source`
- `relation`
- `target`
- `confidence`

Example:

- `agent:l1_0 -> tracks_pattern -> pattern:123`

### RelationalBundle

Extends the current `LevelBundle` idea with:

- `agent_id`
- `mu`
- `weight`
- `epistemic_loss`
- `strategic_confidence`
- `relations`

This allows higher levels to receive both dense belief state and a compact symbolic summary.

### StructuralMessage

A constrained inter-agent communication object containing:

- `source_agent_id`
- `relations`
- `confidence`
- `provenance`

This is a shared protocol, not an unconstrained private language.

## Integration Points

### `hpm/agents/hierarchical.py`

Add helper functions to:

- extract a `RelationalBundle` from an `Agent`
- encode that bundle into a compact numeric vector for backward-compatible higher-level ingestion
- convert a bundle into a `StructuralMessage`

The initial relation set should be simple and reliable:

- agent tracks top pattern
- top pattern has HPM level
- top pattern has current weight signal

### `hpm/field/field.py`

Add a dedicated structural message queue:

- `broadcast_message(source_agent_id, message)`
- `drain_messages()`

This keeps structural messages separate from pattern broadcasts.

### `hpm/agents/multi_agent.py`

Add an optional `structural_messages_enabled` flag.

When enabled:

- after each agent steps, the orchestrator requests one structural message from the agent if available
- messages are queued in the field
- after all agents step, messages are drained and relayed to other agents via a safe optional hook

### `hpm/agents/agent.py`

Add no-op safe hooks:

- `emit_structural_message()`
- `accept_structural_message(message, source_agent_id)`

Default implementation should be conservative and non-disruptive:

- emit a message based on the top-weighted pattern if one exists
- accept by appending to an inbox for later inspection

No learning rule changes are introduced in this patch.

## Encoding Strategy

The baseline still expects vectors between levels, so relational bundles need a small numeric summary.

`encode_relational_bundle()` should preserve current semantics and add only lightweight structural features:

- `mu`
- `weight`
- `epistemic_loss`
- `strategic_confidence`
- relation count
- mean relation confidence

This keeps the encoded representation compact and deterministic.

## Risks and Mitigations

Risk: message spam between agents
Mitigation: emit at most one compact message per step when enabled.

Risk: drift into opaque internal language
Mitigation: fixed typed relation schema and human-readable fields.

Risk: regression in baseline communication path
Mitigation: keep structural messages on a separate queue and behind a flag.

Risk: over-claiming semantics from generic patterns
Mitigation: initial relations only summarize stable agent state, not inferred domain meaning.

## Success Criteria

1. Existing tests continue to pass.
2. New relational helpers are available from baseline `hpm`.
3. Structural messages can be broadcast and drained independently of pattern broadcasts.
4. With structural messaging enabled, orchestrators relay inspectable messages without altering normal pattern communication semantics.
