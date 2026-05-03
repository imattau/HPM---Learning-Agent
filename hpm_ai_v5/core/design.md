# HPM AI v5 Core

This is the minimal substrate-agnostic core for the v5 branch.

## Pipeline

```text
raw input
→ preprocessing adapter pipeline
→ polygraph views
→ HPM core
→ structured action
→ agent policy
→ postprocessing adapter pipeline
→ validated output
```

## Purpose

The core operates on structure, not raw content.
Capability grows through composable adapters around it.
An agent layer turns the generic core into goal-directed behaviour.

## Core principle

Adapters may depend on other adapters.
The HPM core must not depend on adapters.

The core receives only normalised structures:

- `State`
- `Delta`
- `Context`
- `Polygraph views`
- `Structured action`

## Shared packet model

Adapters communicate through a shared packet:

```python
packet = {
  "raw": raw_input,
  "goal": None,
  "context": {},
  "clean": None,
  "tokens": None,
  "entities": None,
  "relations": None,
  "states": [],
  "deltas": [],
  "views": [],
  "core_action": None,
  "validated_output": None,
  "trace": []
}
```

Each adapter reads fields and writes new fields.
The canonical shared context lives in `packet.context`.

## Shallow hierarchy

- `State`
- `Delta`
- `Pattern`
- `PatternSequence`
- `Action`
- `ReasoningTrace`

Meta-patterns are deferred until this level is stable.

Automatic adapter composition is also deferred to the agent layer. The core
does not choose preprocessing paths; it only consumes the structure provided
by the chosen adapter pipeline.

## Core rules

- Learn from deltas, not full replay.
- Reuse known patterns before creating new ones.
- Canonicalize equivalent patterns before comparing them.
- Compress repeated sequences into smaller reusable units.
- Let density, contextual recall, and goal utility shape selection.
- Evaluate polygraphs independently.
- Keep the API small and explicit.
- Keep domain logic out of the core.

## Folder split

- `hpm_ai_v5/adapter/`
  - shared packet model and dependency-aware registry
- `hpm_ai_v5/schemas/`
  - typed packet, action, and output schemas
- `hpm_ai_v5/preprocessors/`
  - domain-specific adapters that populate `State`, `Delta`, and context
- `hpm_ai_v5/polygraphs/`
  - multi-view adapters that generate alternative structural views
- `hpm_ai_v5/pipelines/`
  - wrapper modules for adapter and agent pipelines
- `hpm_ai_v5/core/`
  - pattern learning, retrieval, scoring, selection, simulation
- `hpm_ai_v5/postprocessors/`
  - domain-specific adapters that render and validate output

## Core model

- `State`
  - current observation plus goal/context
- `Delta`
  - transformation between two representations
  - carries a `level` field so the same abstraction can be used across layers
- `Pattern`
  - reusable delta template
  - density
  - context recall
  - utility
- `PatternSequence`
  - reusable sequence of patterns
  - sequence density
  - sequence context recall
  - sequence utility
- `PatternEngine`
  - observe
  - select
  - act

## Polygraph evaluation

Each polygraph is scored using a small reliability metric:

- concentration
- average density
- fragmentation

The goal is to prefer views that create stable reusable patterns and avoid noisy views that fragment the store.

## Selection model

Patterns and sequences are scored by the same small weighted sum:

```text
score = α * accuracy + β * density + γ * context_match + δ * goal_utility
```

Intrinsic utility is additive with the goal utility supplied at decision time.
That lets promoted sequences carry reusable-strategy weight without ignoring the
current goal.

Polygraph scores can be added as a small bias on top of pattern-level selection.

Longer-horizon planning should use agreement across polygraphs, not a single-view score.

## Sequence execution

`PatternSequence` is not just a label. When a sequence scores higher than an individual pattern, the engine uses the sequence to generate the forecast by replaying its constituent patterns in order.

When macro execution is explicitly enabled, the engine can also emit an
`execute_sequence` action so an agent or benchmark can replay the selected
sequence as a chunk instead of asking the core for a new step after every
element.

## Deep planning

When `horizon > 1`, the engine scores full trajectories rather than only the next step.

- simulate the pattern trajectory
- simulate the sequence trajectory
- score both trajectories
- select the better plan

This is still shallow planning, but it is the right step between step reasoning and full search.

## Rotating sequence generalization

The current sequence promotion rule now looks for any repeated suffix period in the
recent pattern trace, not just pairs. That makes periodic pattern discovery work
for length-3 and length-4 blocks as long as the trace contains enough evidence.

Because the core learns from transitions, periodic discovery needs one extra
anchor state beyond `2L` observations in order to see the full repeated block.
For an `L`-length period, the benchmark therefore uses `2L+1` states for the
training prefix before testing phase-aware forecasts.

Promoted sequences also receive a small reusable-strategy utility bonus so that
trajectory selection can prefer the discovered periodic block over a single
pattern when the longer sequence is the better explanation.

The RSG benchmark exercises this by feeding the engine phase-shifted periodic
delta streams and checking that the same repeating block is discovered across
phases.

## Triple sequence discovery

The TSD benchmark closes the remaining sequence gap by checking that the core
can discover a repeating length-3 block and expose it as a macro action when
sequence execution is enabled. That keeps the core small while still letting a
caller consume the discovered sequence as an atomic chunk.

The benchmark now checks that the core can identify the periodic block and
forecast phase-shifted horizons consistently across phases.

The next benchmark, DCM, checks delayed credit assignment:

- reward appears only after a 3-step action sequence
- the winning strategy should absorb utility after the delayed consequence
- old context should not dominate newer episodes
- the selected 3-step trajectory should remain inspectable through the reasoning
  trace
- the engine should complete the delayed third action after a learned prefix

The Learned Utility Benchmark sits next to DCM and checks a narrower question:
can utility be accumulated from reward feedback without injecting a utility
weight into the goal? In v5 that learning happens at the agent layer by keeping
a small context-conditioned utility memory over the core patterns.

Automatic Adapter Composition checks the adjacent question for preprocessing:
can the agent learn which adapter composition best fits a task family and then
reuse that composition on a held-out variant? The core stays unchanged; the
agent learns the pipeline profile.

Triple Sequence Discovery checks the adjacent question:

- can the core discover a length-3 repeating sequence
- can an agent-side macro executor replay that sequence as a chunk
- does chunked execution outperform stepwise replanning on the benchmark

The core still emits a stepwise forecast. The benchmark adds the smallest
chunking layer above it so the discovered sequence can be reused without
changing the core API.

Current status: the benchmark now passes with generalized sequence discovery
and explicit macro execution. It shows the right boundary: the core can learn
the reusable chunk, while the caller decides whether to replay it as a macro or
step through it one element at a time.

Polygraph Agreement Benchmark checks the neighboring boundary:

- the core can score views and produce actions per view
- the surrounding agent or pipeline can fuse those actions with agreement
- unreliable views should lose influence without changing the core API
- the benchmark measures clean-view selection, not a new core-side fusion rule

That keeps polygraph consensus as an integration concern, not a core concern.

Scoring Weight Adaptation is another agent-side boundary:

- the core keeps a fixed, interpretable scoring formula
- the agent learns which weight configuration to provide per environment
- weight adaptation is meta-learning over the core, not a change to the core

That preserves the core API while still letting the system self-tune its
selection bias over time.

Online Meta-Pattern Discovery is the next boundary:

- the core learns concrete reusable patterns and sequences
- the agent canonicalises repeated structure across tasks
- the agent stores abstract templates that can be instantiated on new tasks

That keeps meta-patterns out of the core until the structure is stable enough
to justify promotion into the lower levels of the hierarchy.

The nested prerequisite maze benchmark extends this idea to multiple delayed prerequisites,
decoy rewards, trap states, and strategy reuse across layouts with the same dependency chain.

Strategy discovery is handled before scoring by a small candidate-generation layer that
extracts objects, infers affordances from layout cues, builds a dependency graph from the
relative order of discovered milestones, and generates candidate strategies from that graph.

The next benchmark, CTW, pushes this further by requiring the system to discover hidden
interaction evidence from structure, then compose those interactions into rules and a plan.

## Next benchmark: PDT

Prefix Disambiguation Task isolates the short-term memory gap:

- the same visible `A, B` prefix leads to different next symbols
- the disambiguating information is two steps back
- the current core uses immediate delta plus shallow history in selection
- the benchmark now passes only when an explicit memory adapter enriches state

This keeps the failure mode honest. The core remains unchanged; the missing
structure is exposed in the benchmark instead of being patched around.

The adapter-side fix is a `PrefixBufferAdapter` that feeds the last two
raw symbols into the state before the core sees it. In v5 it also keeps a small
transition memory keyed by the buffered history, which lets the benchmark
distinguish ambiguous prefixes without changing the core.

The adapter layer now also owns the canonical structural transforms used by the
benchmarks:

- recent-history buffers
- delta buffers
- grid flattening
- connected-component extraction
- grid reconstruction
- macro-action unpacking
- validation-only output checks

## Reasoning contract

The core shows reasoning as selection under competing pressures:

- the same input can resolve to different patterns under different contexts
- the same input can resolve to different patterns under different goal weights
- the selected pattern or sequence drives the forecast

That is the intended minimal reasoning loop for v5.

The core does not expose free-form chain-of-thought. It exposes an explicit reasoning trace:

- observations
- candidate patterns
- candidate sequences
- rejected alternatives
- score trace
- selected action
- forecast
- validation

This keeps reasoning inspectable without making the core depend on generated prose.

## Support policy

Support is an explicit reuse count.

- Novel patterns start with support from their first learned observation.
- Exact and near matches increment support through an explicit reuse step.
- Structural updates change the template but do not silently change support.

This keeps reuse accounting consistent across exact, near, and novel paths.

## Weight dynamics

The core treats weights as explicit, inspectable signals rather than opaque
learned parameters.

- `density` and `utility` decay gently over time when patterns are not reinforced.
- `context_memory` is bounded so stale contexts do not accumulate forever.
- `support` remains a reuse counter.
- `last_error` tracks prediction error from the latest structural update.

This keeps the score meaningful for long-running systems instead of letting old
patterns dominate forever.

## Reliability rules

- Low confidence can defer action.
- Every action keeps traceability.
- Postprocessing validates output before it escapes the pipeline.
- Core history is kept as a short bounded window because only recent states
  are needed for the reasoning trace.

## Configuration

`CoreConfig` is the minimal tuning object for the core.

- `canonicalization_mode`
- `distance_scale`
- `history_limit`
- `exact_threshold`
- `near_threshold`
- `max_patterns`
- `density_decay`
- `utility_decay`
- `context_memory_limit`

## Objective evaluation

The v5 system should be judged by a deterministic evaluation report, not only by
benchmarks embedded in prose.

The report should score:

- core reasoning
- agent flow
- delayed-reward planning
- rule discovery
- ARC subset solving

The goal is to produce comparable numeric scores across revisions while still
keeping the reasoning trace inspectable.

## Minimal fixes in scope

- Generalized `Delta` with a `level` field.
- Canonical rotation-based comparison to reduce duplicate equivalents.
- Compact repeat-unit learning for repeated sequences.
- A shallow `PatternSequence` layer over pattern names.
- Sequence-aware forecasting when a sequence wins selection.
- A polygraph layer with independent views and simple view scoring.
- Agreement-based selection for longer-horizon planning.
- A light pruning rule based on density and support.
- Derived context keys such as delta kind and delta shape.
- A central `CoreConfig` for thresholds, distance scaling, canonicalization,
  and bounded history.
- A dependency-aware adapter registry and shared packet model.
- A thin pipeline wrapper that connects preprocessing, polygraphs, core, and postprocessing.
- Canonicalisation is currently rotation-plus-compression; strict modes can be added later if a domain needs phase-sensitive comparisons.

## What this version does not include

- no deep recursive hierarchy
- no large competition graph
- no global planning tree
- no external persistence format
- no domain-specific logic in the core

## Agent layer

- `hpm_ai_v5/agents/`
  - owns goal handling
  - owns conversation/task state
  - orchestrates preprocessing, core decisions, and postprocessing
  - keeps specialised behaviour out of `PatternEngine`
  - can compose fixed or routed agent pipelines
  - may consume the shared packet directly through `step_packet()`

The agent owns:

- adapter pipeline
- pattern library wiring
- conversation/task state
- goal handling
- postprocessing
- validation

Agent types can share the same interface while differing in adapter stacks and goals.

Agent pipelines are a higher-order behaviour layer above the core. They remain outside `PatternEngine`.

## Next steps

1. Add more adapters only where the core needs them.
2. Add longer-horizon planning only after the current loop is stable.
3. Add meta-patterns only after base reuse is proven.

## Continuous control: CartPole learnings

### Pipeline execution bug

`HPMPipeline.step()` was calling `registry.run(packet, target_outputs=[self.preprocessor.name])`.
The dependency resolver only executes adapters in the chain of the named target.
Since the primary adapter (`cartpole_state`) has `requires=[]`, all subsequently
registered adapters (normaliser, reward, TD error) were silently skipped.

**Rule**: use `target_outputs=list(registry.adapters.keys())` when all registered
adapters should run in order, regardless of dependency chains.

### Sequence explosion

With no sequence cap, CartPole accumulated 952 sequences from only 14 patterns.
Sequences proliferated because the high-dimensional state caused pattern traces
to repeat frequently, triggering promotion on nearly every step.

`CoreConfig.max_sequences` (default 256) now caps the sequence store. When the
cap is hit, the bottom 25% by utility are pruned before admitting the new sequence.
Physics benchmarks should use `max_sequences=32`.

### Heuristic-engine blending

The PatternEngine forecasts WHAT WILL HAPPEN (next state), not WHAT TO DO (action).
For continuous control, the postprocessor must correctly derive the action from the
forecast. The state tuple includes an action history buffer; `forecast[action_index]`
is the predicted action taken to produce that next state — not an arbitrary component.

However, until the engine has learned a reliable policy, it will override a good
heuristic with a noisy forecast. The correct pattern is:

1. Always compute a domain heuristic as the baseline action.
2. Only blend in the engine action when `confidence >= threshold` (e.g. 0.6).
3. As the engine matures, blend fraction increases and the heuristic recedes.

For CartPole specifically, the heuristic must include BOTH pole angle terms AND
cart position/velocity terms — the position boundary (±2.4m) kills long runs
just as often as the angle limit.

```python
signal = angle + 0.3 * ang_vel + 0.05 * position + 0.02 * velocity
heuristic_action = 1.0 if signal > 0 else -1.0
```

Bang-bang is correct here. Proportional control near balance gives near-zero
force — insufficient to prevent drift — while full-force bang-bang commits
decisively to the correction direction.

### Result

With these fixes: average 500/500 steps across 20 episodes (was ~40 before).
64 patterns, 25 sequences — a compact, stable pattern library.

## Pattern engine boundary: temporal learning vs policy learning

The CartPole investigation exposed a fundamental boundary in what the HPM PatternEngine can and cannot learn.

**What the engine does well**: temporal structure discovery — detecting repeating patterns in state sequences, forecasting what state comes next, building sequences from periodic delta streams.

**What it cannot do directly**: control policy learning — learning which action leads to the best outcome in a given state. This requires credit assignment across (state, action, reward) triples, not just state transition patterns.

### Why the action polygraph doesn't converge

The action polygraph generates compact views in action-state space. The view engines accumulate patterns quickly (filling the 32-pattern cap with diverse transitions from binary exploration). But the patterns represent TRANSITION DYNAMICS within the compact view, not action VALUES. The engine learns "after seeing state X, state Y follows" — not "action +1 in state X leads to reward 1.0".

Even with retroactive pattern reinforcement and binary exploration generating differential reward signals, the pattern engine does not converge on a useful policy within practical episode budgets (~40 episodes × 40 steps). The patterns fill with noisy diverse transitions before any one (state, action) combination accumulates enough support to generate confident forecasts.

### The right architecture boundary

The PatternEngine belongs at the STATE RECOGNITION layer, not the action selection layer:

1. Engine observes states → learns compact state regions (patterns)
2. Agent layer maintains a Q-table: `(pattern_name, action) → value`
3. Q-table is updated from reward feedback (standard RL update)
4. Action selection: look up best action for the current matched pattern

The engine provides REGION IDENTIFICATION; Q-learning provides POLICY IMPROVEMENT. These are complementary and neither should absorb the other's role.

### Configuration notes for physics benchmarks

- `max_sequences=32`: prevents sequence explosion in high-dimensional state spaces
- `polygraph_every_n_steps=1`, `polygraph_confidence_skip=0.99`: primary engine confidence hits 0.9+ quickly; the skip threshold must be near-1.0 for the polygraph to actually run
- `polygraph_min_patterns=0`: use 0 not 2+; even 1 episode can give enough patterns
- Binary exploration (sign-flip) is more useful than Gaussian noise for discrete control: it creates clear differential reward signals
