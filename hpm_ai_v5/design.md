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

## Architectural mechanisms

### The carry_context feedback loop

The pipeline closes a feedback loop from postprocessing back into preprocessing:

```
preprocess → engine → postprocess → [carry_context]
     ↑                                      ↓
     └──────────────────────────────────────┘
```

Any key written to `packet.context` with the prefix `carry_` is extracted by the
pipeline after postprocessing, stripped of its prefix, and returned in
`PipelineResult.carry_context`. The benchmark (or agent) merges this into the context
dict for the next step.

This lets the postprocessor write state that preprocessing adapters read on the next
step — without coupling the postprocessor to any specific adapter. The carry is
explicit, inspectable, and survives across the engine boundary.

**What goes in carry_context**:
- Q-table learning state: current Q-state key, last action, last Q-confidence
- Trajectory buffers: sliding windows of error, action, or reward history
- Shaped reward inputs: previous-step error values for delta computation
- Any signal the postprocessor computes that a preprocessing adapter needs next step

**Rules**:
- All carry values must be hashable (tuples, floats, ints, strings). Lists corrupt the engine.
- The `carry_` prefix is stripped on delivery — read `ctx.get("error_history")`, not `ctx.get("carry_error_history")`.
- Never use carry to pass large objects. Keep carries small and typed.

### Polygraphs: multi-view observation

The polygraph layer generates alternative structural views of the same raw observation.
Each view is a `PolygraphView(name, state)` passed to a separate PatternEngine instance.

**Why multiple views**: a single state encoding loses information. Different encodings
expose different structural regularities — one view may produce dense reliable patterns
while another fragments on the same data. The polygraph layer lets the system discover
which encoding is useful for a given domain without committing to one at design time.

**Scoring**: each view engine is scored by a reliability metric (concentration, density,
fragmentation). Views with stable reusable patterns score higher. Unreliable views
lose influence without being removed.

**Polygraph vs primary engine**:
- Primary engine: full-dimensional state, used for action selection and forecasting
- Polygraph views: compact encodings, used for multi-hypothesis pattern learning and Q-table voting

**Configuration rules** (from physics benchmarks):
- `polygraph_every_n_steps=1`: run polygraph every step for control tasks
- `polygraph_confidence_skip=0.99`: the primary engine hits 0.9 confidence fast; set skip near 1.0 or polygraphs never fire
- `polygraph_min_patterns=0`: don't wait for a minimum pattern count before scoring

**When to add a new view**: when you have a compact state encoding (≤8 states) that
captures a meaningful structural distinction the primary engine's full state obscures.
Don't add a view for every feature combination — add one when you can name what
structural property it isolates.

**Trajectory vs single-step views**: trajectory views (multi-step window encodings)
take many episodes to accumulate sufficient Q-table data. Add them to the PatternEngine
polygraph for long-term structure learning, but exclude them from fast-converging
single-step Q-table ensembles until they have enough data to vote usefully.

### Postprocessor as policy learner

The postprocessor is the correct location for policy learning (Q-tables, action
selection) because it receives at every step:
- `packet.context["reward"]` — actual env reward from previous step (via carry)
- `action.selected_pattern` — which state region the engine matched
- `packet.context["raw_observation"]` — full current observation
- `packet.context["last_action"]` — what was executed

A Q-table maintained in the postprocessor can update on every step without a separate
agent layer. The carry_context loop threads the postprocessor's learned state back into
preprocessing, closing the RL update cycle.

This keeps RL credit assignment in the postprocessor (where reward and action are both
known), and pattern learning in the engine (where temporal structure is known). These
roles are complementary and should not be merged.

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

## Q-table-as-postprocessor: architecture and findings

### The feedback loop

The postprocessor is the right place for policy learning because it has access at every step to:
- `packet.context["reward"]` — the actual env reward from the previous step (via carry_context)
- `action.selected_pattern` — which state region the engine matched
- `packet.context["raw_observation"]` — full current state
- `packet.context["last_action"]` — what was executed

This means the postprocessor can maintain a Q-table and update it at each step without a separate agent layer. The carry_context mechanism (`PipelineResult.carry_context`) threads postprocessor outputs back into the next preprocessing cycle, closing the loop:

```
preprocess → engine → postprocess → [carry_context]
     ↑                                      ↓
     └──────────────────────────────────────┘
```

### Key engineering lessons

**Terminal reward must be carried forward**: `context["reward"]` is always 1.0 during an episode. When `env.step()` returns `reward=0.0` (done=True), the episode resets and carry is cleared — the terminal penalty never reaches the Q-update. Fix: initialise `carry = {"reward": 0.0}` at episode start, merge actual reward into carry after `env.step()`.

**Absolute Q-diff, not relative confidence**: when Q-values saturate near 10 (common with gamma=0.9 and step rewards of 1.0), the relative difference `|q_pos - q_neg| / (q_pos + q_neg)` ≈ 0.01 even when discrimination is meaningful. Use absolute `|q_pos - q_neg| > threshold` as the gate.

**State resolution determines Q-table utility**: a 2-bit `(sign_angle, sign_ang_vel)` Q-table (4 states) learns correct policy for 3 of 4 quadrants but the `(-angle, +ang_vel)` quadrant is ambiguous — optimal action depends on magnitude ratio, which the coarse state loses. CartPole requires at least 4-bit encoding (magnitude buckets) for a genuinely useful Q-table.

### The pattern-as-Q-table equivalence

The binary_sign polygraph view `(sign_angle, sign_ang_vel, sign_action)` gives the PatternEngine 8 distinct states. With retroactive reward reinforcement, the engine's pattern utility scores play the same role as Q-values — patterns for correct (state, action) pairs accumulate higher utility than patterns for incorrect pairs. This is a direct HPM encoding of a Q-table without any separate data structure.

The limitation is the same: 8 patterns can't resolve the magnitude ambiguity in the `(-angle, +ang_vel)` quadrant.

## What the CartPole benchmark reveals about v5 design

### The learning curve is a phase transition, not gradual improvement

In a 120-episode run, the system performs near-randomly for episodes 1-60 (~45 steps avg),
then snaps into competent policy by episode 61 (150+ steps, sustained). This is not gradual
learning — it is a phase transition driven by Q-table convergence once the shaped reward
provides sufficient gradient signal.

This maps directly to HPM's consolidation dynamic: a period of exploration/noise followed
by rapid stabilisation once the evaluator signal becomes discriminative.

### The HPM engine is not doing the policy learning

The Q-tables in the postprocessor carry the actual control policy. The PatternEngine
provides state representations and confidence signals, but action selection is classical RL.
v5 demonstrates that HPM pattern learning can *coexist* with a policy layer — it has not yet
demonstrated that hierarchical pattern learning *is* the policy.

This is the correct boundary to expose now. Blurring it would mask what is actually working
and why.

### The shaped reward was the critical unlock

Binary survival reward (1.0 per step, 0.0 on failure) gave Q-tables no gradient within an
episode — all non-terminal actions looked identical. The delta-error shaped reward:

```
shaped_reward = -(derived_error_t - derived_error_{t-1}) / max_error
```

provides a dense signal: negative when angle error grew (wrong action), positive when it
shrank. This is a direct implementation of HPM's prediction error dynamics — patterns need
a signal that distinguishes quality of match, not just survival.

**Design rule**: for any continuous control domain, the reward signal fed into the policy
layer must be derivative-of-error, not binary outcome.

### Per-view ensemble voting is structurally sound but shallow

The 5 polygraph views each maintain independent Q-tables and vote on action with weights
proportional to polygraph reliability scores. This is the right structural form — multiple
concurrent hypotheses at different granularities — but currently all views operate at the
same abstraction level (immediate angle/velocity observations).

Genuine hierarchical voting requires views at different timescales:
- Short view: immediate correction (sign of angle)
- Mid view: trajectory trend (error growing or shrinking over 5 steps)
- Long view: strategy-level (oscillation vs. drift vs. boundary proximity)

The ensemble is the scaffold; hierarchy is the next layer.

### Future development directions

**1. Engine utility as policy signal (absorb the Q-table)**

The binary_sign polygraph view already encodes an implicit Q-table through pattern utility.
With retroactive `pattern.reward(reward)` reinforcement, utility scores discriminate correct
from incorrect (state, action) pairs — exactly what a Q-value does. The next step is to
make this convergence fast enough to replace the external Q-table entirely. The Q-table
is a crutch that compensates for the engine's slow utility convergence; fixing that
convergence rate is the deeper solution.

**2. Hierarchical timescale views**

Add polygraph views that operate on sliding windows of 5-20 steps rather than single-step
observations. A "trajectory view" that encodes (error trend, oscillation frequency,
boundary proximity) would allow the ensemble to distinguish recovery strategies from
maintenance strategies — a level of abstraction the current views cannot represent.

**3. Cross-physics transfer via PatternManager**

This is now the next benchmark after single-environment CartPole convergence. The goal is
to test whether the learned state recognisers and policy priors transfer across related
physics variants without retraining from scratch.

Implemented harness scope:

- source training on baseline CartPole
- parameterized target variants: Heavy, Light, Short, Long
- snapshot/export of PatternManager archive, engine state, normaliser state, SWA state,
  and postprocessor Q-tables via agent-side persistence
- zero-shot evaluation mode that freezes learning and restores the pre-eval state after
  scoring
- fine-tune runs from the same seeded snapshot
- catastrophic-forgetting check by evaluating the fine-tuned agent back on source CartPole

The benchmark answers a sharper question than the original single-task run:

- do sign-based and compact state abstractions survive physics scaling?
- does the Q-table/postprocessor transfer as a meaningful policy prior?
- does PatternManager seeding preserve useful patterns without clobbering the source regime?

Current implementation note: the CPT harness is built around the existing parameterized
CartPole dynamics. Acrobot remains a follow-on environment, not part of the current codebase.

**4. Meta-pattern layer for policy consolidation**

The phase transition at episode 60 represents a strategy consolidating. An explicit
meta-pattern layer would represent "I have a reliable policy for this regime" as a
first-class object — triggering reduced exploration, increased confidence thresholds,
and strategy transfer to related tasks. This is the HPM analogue of skill consolidation
in human motor learning.

**5. Engine as state recogniser, Q-table as policy improver**

The cleanest long-term architecture:
1. PatternEngine identifies which state region the system is in (pattern matching)
2. Q-table keyed on `(pattern_name, action)` provides action values per region
3. Utility propagation from reward updates both Q-table and engine pattern utility
4. As pattern utility converges, the Q-table becomes redundant and can be dropped

This preserves the HPM substrate while giving RL the role it is genuinely good at
(credit assignment), without requiring the engine to solve a problem it was not
designed for.

## Engineering principles from CartPole development

These are implementation-level rules derived from building and debugging the CartPole
benchmark. They apply across all v5 domain development.

### Context values must be hashable

Any value written to `packet.context` or carried via `carry_context` will eventually
propagate into `State.context`, which the PatternEngine uses for context-signature hashing.
Lists and dicts will corrupt the engine silently — patterns accumulate but the context
match produces no signal.

**Rule**: all context values must be tuples, floats, ints, strings, or booleans.
Never put lists, dicts, or mutable objects into context. Convert before writing.

```python
# Wrong
ctx["error_history"] = [e1, e2, e3]

# Correct
ctx["error_history"] = tuple([e1, e2, e3])
```

### The carry_context mechanism strips "carry_" prefix

The pipeline extracts keys prefixed with `carry_` from the post-packet context and strips
the prefix before returning them in `PipelineResult.carry_context`. A key written as
`carry_error_history` arrives in the next step as `error_history`.

**Rule**: when reading from carry in an adapter, use the stripped name:
```python
# Written by adapter: ctx["carry_error_history"] = value
# Read next step:     ctx.get("error_history", default)    ← no carry_ prefix
```

### Dependency resolver uses adapter NAMES, not capability names

`AdapterRegistry.resolve()` treats `adapter.requires` as a list of ADAPTER NAMES.
Writing `requires = ["state"]` does nothing unless there is an adapter named `"state"`.
With `target_outputs=list(registry.adapters.keys())`, all adapters run in registration
order regardless of requires/provides metadata.

**Rule**: the requires/provides fields are documentation only in the current resolver.
Register adapters in the order they should run. Don't rely on capability-based resolution.

### Post-/pre-processor design principles from CPT

The CPT benchmark turned the control stack into a useful design test. The main lesson
is that the preprocessor and postprocessor should stay small, explicit, and stable:

- Preprocessors should build state, normalization, error, and short history features.
- Postprocessors should turn engine output into action with one stable policy path.
- Use a simple heuristic baseline first; blend engine output only as a weak hint.
- Keep the primary policy state compact and semantically stable across episodes.
- Add extra action hypotheses only if they are easier to interpret than the baseline,
  not because they are architecturally available.
- Trajectory views and other slower signals belong in the engine or auxiliary scoring
  path unless they can converge within the benchmark budget.
- If a richer control representation lowers performance, revert it quickly.

For other domains, that translates into a KISS rule:

1. Preprocessing should expose the smallest feature set that preserves the domain's
   decision boundary.
2. Postprocessing should prefer one clear action source, not a majority of weak ones.
3. Add extra views or action hypotheses only when they improve convergence on a small
   validation sweep, not when they simply add diversity.
4. Treat transfer benchmarks as a gate on stability, not as a place to debug the source
   control loop.

## Next benchmark: Cross-Physics Transfer (CPT)

Now that CartPole reliably reaches long horizons with the polygraph-weighted Q-table
postprocessor, the next benchmark is transfer rather than raw within-task learning.

### Environment suite

- `cartpole`: source training environment
- `cartpole_heavy`: `mass_pole=0.5`
- `cartpole_light`: `mass_pole=0.02`
- `cartpole_short`: `length=0.25`
- `cartpole_long`: `length=1.0`

These variants preserve the action space and task structure while perturbing the dynamics.
That makes them the right first test for whether the learned structures are physics-invariant
or just tuned to one set of scales.

### Transfer protocol

1. Train on source CartPole and export a transfer snapshot.
2. Run zero-shot evaluation on a target variant with learning disabled.
3. Restore the same source snapshot and fine-tune on the target.
4. Compare target fine-tune against a scratch baseline.
5. Evaluate the fine-tuned agent back on source CartPole to measure forgetting.

### Architectural implication

The CPT harness makes the current v5 boundary explicit:

- `PatternManager` and the engine carry recognisers and reusable structure.
- The CartPole postprocessor carries the fast policy prior through its Q-tables.
- Evaluation uses frozen snapshots so transfer can be measured without contaminating the seed.

If CPT passes on the parameterized CartPole suite, the next meaningful extension is not more
within-task tuning. It is moving the same protocol to a structurally different control domain
such as Acrobot and measuring which abstractions survive that jump.

### Trajectory views belong to the engine, not the Q-table ensemble

Views that operate on multi-step trajectories (error trend, oscillation) take many episodes
to accumulate enough Q-table data to vote usefully. Adding them to the ensemble before
convergence adds noise and degrades performance.

The correct separation: trajectory views generate polygraph observations for the engine
to learn multi-step structure from. Single-step views drive the Q-table ensemble for fast
policy convergence. Both coexist in the pipeline — their roles are distinct.

**Rule**: a new view should be added to the ensemble only when it has a state space
small enough to converge within the expected episode budget (≈ 40 episodes). Larger or
slower views feed the engine's pattern library only.

### The adapter registration order determines execution order

Since the resolver runs adapters in registration order (with `target_outputs=all`), the
registration sequence IS the execution sequence. Always register adapters in the order
they must run:

1. Raw state extraction (e.g., `CartpoleStateAdapter`)
2. Normalisation
3. Reward / goal shaping
4. Error / TD adapters
5. Derived feature adapters (e.g., `TrajectoryBufferAdapter`)

### Reward signals must be derivative-of-error for dense learning

Binary survival reward (1.0 per step, 0.0 on failure) gives the Q-table no gradient
within an episode — every action looks equally good until the terminal step, which is
never observed because the episode resets before carry is processed.

The shaped reward `-(error_t - error_{t-1}) / max_error` provides a dense signal at
every step: negative when the error grew (wrong direction), positive when it shrank.
This is the HPM prediction-error dynamic applied to RL credit assignment.

**Rule**: for any continuous control domain, the reward flowing into the postprocessor
Q-table must be a derivative or relative measure, not a binary outcome. The environment's
sparse reward is correct for evaluation; a shaped signal is required for learning.

### Terminal reward must be carried into the next episode

The environment's terminal reward (0.0 on failure) arrives when `done=True`. At that
point, the episode loop resets and `carry` is cleared — the penalty never reaches the
Q-update that should penalise the action that caused failure.

Fix: initialise `carry = {"reward": 0.0}` at episode start, so the first step of
each episode processes the terminal reward from the previous episode. Then update with
the real reward after `env.step()`.

### Evaluate on the learned policy, not the full training run

RL systems have an exploration phase (high epsilon, random actions) that dominates early
episode averages. Evaluating the overall mean conflates exploration noise with learned
performance.

**Rule**: measure the average of the last half of episodes (or a fixed eval window) as
the benchmark criterion. This reflects the converged policy, not the training curriculum.

---

## NLP Agent Development (from SNLP Benchmark)

### `act().confidence` is not a discriminative signal

The `Action.confidence` value returned by `PatternEngine.act()` decays monotonically
over time as patterns age — it does not reflect match quality. Using it to distinguish
familiar from unfamiliar input produces near-random results.

**Rule**: use `engine.last_match.status` or `view_engine.last_match.status` for
discrimination tasks. Values: `"exact"`, `"near"`, `None`. Assign scores (1.0 / 0.5 /
0.0) from status, not from `confidence`.

### `act().forecast` is a single-token state, not a full skeleton prediction

The forecast returned by `act()` is a one-element tuple representing the first token of
the anticipated next pattern. Comparing it numerically to a multi-element actual state
via L2 distance gives only partial prefix signal.

**Rule**: to measure whether the engine anticipated the next observation, observe the
next state then check `last_match.status`. Do not compare forecast and actual values
numerically unless the state representation is known to be fixed-length and aligned.

### Polygraph view namespace is the agent's working memory index

The polygraph creates named view engines (e.g. `semantic_view_check`,
`skeleton_bigram_view`). For a test sentence to match a trained pattern, the *same
named view engine* must have been populated during training. Isolated seed words and
full sentences generate different view keys from the KB lookup.

**Rule**: training and test inputs must generate overlapping view names. For semantic
slot-filling, train on full sentences whose KB candidates overlap with the test
sentences' KB candidates.

### Task isolation requires PatternStore resets, not just view engine clears

`reset_for_isolation(clear_views=True)` clears the view engine dict but leaves
`engine.store` intact. Patterns from one task silently contaminate the next.

**Rule**: between independent benchmark tasks, reset
`engine.store = PatternStore(config=self.config)`.

### PatternStore has a saturation regime

With `max_patterns=512`, training beyond ~100 episodes on a small corpus causes old
patterns to be evicted and recognition degrades. More training hurts past this point.

**Rule**: tune `max_patterns` per corpus size. Set training episodes to fill but not
overflow the store. Monitor for the inflection point where recognition drops as episodes
increase.

### Sequential ordering requires bigram skeletons; unigrams are insufficient

Single POS-group skeletons collapse ordering information — valid and scrambled sentences
can produce identical unigram skeleton tuples. Discriminating them requires at least
bigram transitions (e.g. `"P_V"`, `"V_D"`).

`SkeletonNgramAdapter` produces bigrams in `context["skeleton_ngrams"]`; the polygraph
exposes them as `skeleton_bigram_view`. The ngram state must be stored in context only
— not appended to `packet.states` — to avoid displacing the primary skeleton state in
the main engine.

### Real spaCy models surface new POS tags requiring explicit mapping

Switching from `spacy.blank("en")` to `en_core_web_sm` introduces tags the
`SkeletonExtractor.POS_GROUP` did not handle: `AUX` → `"V"`, `PART` → `"R"`,
`SYM` falls through as a raw string. Unhandled tags appear verbatim in the skeleton and
prevent near-match generalisation.

**Rule**: audit any new domain input for unhandled POS tags before deploying a skeleton
adapter.

### Small synthetic corpora produce measurement variance, not learning failure

The SNLP benchmark (9 sentence templates, 3 intent classes) is sufficient for
structural pattern learning. Run-to-run variance of ~10 percentage points on T1
(skeleton recognition) reflects stochastic test sampling, not architectural
instability. Report a range across multiple runs, not a single score.

### Multi-view ensemble voting converges faster than single Q-table

With 5 views voting with polygraph-score weights, the CartPole Q-table ensemble reaches
>150 avg by episode 41-60. A single Q-table with the same 4-state encoding took >80
episodes. The ensemble is more robust because:
- Views with reliable polygraph scores dominate
- Views that have not converged contribute low weight (0.1 floor)
- Different state encodings catch different failure modes

This is a direct implementation of HPM's multi-level concurrent hypothesis evaluation.


---

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


---

# HPM AI v5 Planning

## Purpose

Planning remains a shallow, inspectable layer above the core. The planner
constructs symbolic candidate trajectories, scores them, and returns the
best prerequisite-respecting strategy.

## Nested prerequisite maze benchmark

The nested maze test checks whether the planner can:

- preserve delayed reward
- satisfy multiple prerequisites in order
- avoid traps and decoy rewards
- reuse a successful strategy on a later maze with the same dependency chain
- explain why weaker strategies were rejected

## Strategy discovery layer

The maze benchmark now includes a small candidate-generation agent before scoring:

- object extraction
- affordance inference from layout cues
- dependency graph building from inferred milestone order
- strategy generation from the graph

This keeps candidate discovery separate from trajectory scoring. The planner still
scores trajectories, but it no longer needs hand-shaped strategy candidates.

## Design rule

The planner should reason over trajectories, not just next-step scores.

For the maze benchmark, that means:

- candidate strategies are explicit
- candidate strategies are discovered from maze structure
- dependency edges are inferred from the relative order of discovered milestones
- each strategy is simulated end-to-end
- the full trajectory is scored
- strategy reuse is rewarded across mazes with the same dependency structure

## Scope

This is still not a general search planner. It is a minimal benchmark harness
for delayed-reward reasoning, subgoal ordering, and strategy reuse.

## Next benchmark: RSG

Rotating Sequence Generalization checks the other side of the shallow hierarchy:

- discover periodic blocks of any length
- reuse the same canonical pattern across phase shifts
- score trajectories over a long enough horizon to expose the full block
- prefer reuse over splitting the same periodic structure into duplicate patterns

The benchmark uses phase-shifted periodic delta streams because the core learns
from transitions. That keeps the test aligned with the actual substrate while
still stressing canonicalization and sequence promotion.

Current status: the core can now promote repeating blocks of arbitrary length,
and phase-shifted horizon forecasts are now tested directly instead of being
treated as an expected failure.

Because the core consumes transitions, the training prefix needs one extra
anchor state beyond `2L` to expose a complete repeating block. The benchmark
therefore uses `2L+1` states before scoring held-out forecasts.

## Next benchmark: DCM

Delayed Consequence Maze checks the other side of the planning loop:

- sparse delayed reward after a 3-step action sequence
- utility propagation onto the reusable strategy
- decay and bounded history under repeated episodes
- phase-aware selection over the learned 3-step trajectory
- a teacher-forced prefix followed by engine-controlled completion of the delayed third action

The harness is deliberately small. It uses a deterministic curriculum with
winning, decoy, and trap episodes so delayed credit assignment can be tested
without adding a full exploration policy.

## Next benchmark: LUB

Learned Utility Benchmark checks reward-driven utility learning:

- utility is learned from observed reward rather than injected through the goal
- context-specific preference is learned at the agent layer
- the same candidate patterns are preferred differently by context
- the benchmark scores preference accuracy, reward quality, and utility separation

The HPM core already exposes explicit utility and decay hooks. The benchmark
uses an agent-side utility memory to update context-conditioned preferences
without changing the core scoring loop.

## Next benchmark: TSD

Triple Sequence Discovery checks whether the system can discover and then reuse
a length-3 sequence as a usable chunk:

- discover a repeated three-step pattern in the pattern trace
- prefer the chunked sequence over stepwise replanning
- use a macro execution mode to replay the discovered sequence
- compare macro execution reward against a stepwise baseline

The core still reasons step-by-step by default, but it can emit an
`execute_sequence` action when a discovered sequence is sufficiently better than
the best single-step pattern. The benchmark turns that on explicitly so the
triple can be replayed as an atomic chunk.

Current status: TSD now passes with generalized sequence discovery plus macro
execution. It is the benchmark that closes the remaining gap between
discovering a repeated chunk and actually using it.
The benchmark accepts any reusable chunk of length at least three, because the
selected trace can include extra bookkeeping states even when the useful macro
is the triple itself.

## Next benchmark: PAB

Polygraph Agreement Benchmark checks whether the system can use multiple views
of the same numeric stream and downweight an unreliable one:

- generate exact, noisy, and trend views from the same input
- score each view with `PolygraphEvaluator`
- combine actions with polygraph agreement
- prefer the clean views over the noisy one
- outperform the noisy view alone

This benchmark is intentionally agent-oriented. The core already exposes
polygraph scoring and agreement helpers; the benchmark verifies that the
surrounding stack actually uses them.

## Next benchmark: SWA

Scoring Weight Adaptation checks whether the agent can learn which scoring
weights work best in different environments:

- noisy environments should downweight density
- stable repeating environments should favor density
- context-switching environments should favor context and utility
- weights are learned from reward feedback, not hand-tuned

This is agent-side meta-learning over the core's fixed scoring formula. The core
still computes scores; the agent learns which `α, β, γ, δ` to provide.

## Next benchmark: OMPD

Online Meta-Pattern Discovery checks whether the agent can abstract reusable
structure from several similar tasks and transfer it to a novel task:

- learn a canonical dependency-chain signature from training tasks
- store the abstract meta-pattern, not just the concrete task names
- match a new task to the stored structure zero-shot
- instantiate the template into concrete actions without exploration

This is model-extending behavior at the agent layer. The core still handles
patterns and sequences; the agent now learns higher-level reusable templates
from them.

## Next benchmark: AAC

Automatic Adapter Composition checks whether the agent can discover which
preprocessing pipeline a new task needs and reuse that pipeline on a held-out
variant:

- compare a small set of candidate adapter pipelines on a calibration prefix
- store the best pipeline as a reusable structural profile
- reuse the learned profile on a new task with the same structure but different
  surface values
- prefer the pipeline that produces the most stable downstream predictions

This stays agent-side. The core still receives normalized structures, but the
agent now learns which adapter composition turns raw input into the right
structure for that task family.

## Next benchmark: Open Adapter Discovery

Open Adapter Discovery pushes AAC one step further:

- compare candidate adapter compositions on a support set
- reuse a learned adapter profile on a held-out query task
- defer cleanly when the adapter library cannot represent the task
- separate solvable numeric/grid families from an unsupported graph family

This is the first benchmark that tests the boundary of the adapter catalog
itself. It checks whether the system can choose among known adapter
compositions and still refuse unsupported structure instead of forcing a bad
representation.

## Next benchmark: CTW

The next benchmark is Compositional Transformation World.

- discover hidden rules from structure
- discover interaction evidence before naming rules
- compose rules into a strategy
- apply the same rule set in a new layout
- explain why the strategy was generated

This is the benchmark that should expose reasoning flaws in candidate discovery,
not just scoring.

## Next benchmark: PDT

Prefix Disambiguation Task checks the remaining short-term memory gap:

- identical current symbols can require different next-symbol predictions
- the disambiguating signal lives in the symbol two steps back
- the current core only uses the immediate delta and shallow history in selection
- this should therefore fail until an explicit short-term memory adapter or
  a richer selection context is introduced

The benchmark is intentionally simple:

- same visible `A, B` prefix
- different hidden two-step context
- different next symbol
- score only on the ambiguous `B` positions

The fix is a `PrefixBufferAdapter`:

- maintain the last two raw symbols
- expose them as structured state before the core
- keep a small transition memory keyed by the buffered history
- let the benchmark distinguish `(C, A, B)` from `(A, A, B)` without changing
  the core

The adapter is in place and the buffered benchmark now passes by using that
history window as short-term memory. The core remains unchanged.

More generally, the newer numeric transforms belong in the adapter layer:

- `PrefixBufferAdapter`
- `StateFusionAdapter`
- `NormalisationAdapter`
- `DifferencingAdapter`
- `RollingStatsAdapter`
- `AutocorrelationAdapter`
- `EntropyAdapter`
- `SymbolicAdapter`

The high-value canonical adapter set also includes:

- `RecentBufferAdapter`
- `DeltaBufferAdapter`
- `FlattenGridAdapter`
- `ConnectedComponentsAdapter`
- `GridPostprocessor`
- `ActionSequenceUnpacker`
- `ValidationOnlyAdapter`

## Continuous control: CartPole benchmark

CartPole is the first physics control benchmark. Key design learnings:

### Postprocessing is the action interface

The engine forecasts the next state. The state tuple includes an action history
buffer as its first N elements: `(a0, a1, ..., aN, obs...)`. In the forecast,
`forecast[N-1]` is the action predicted to have been taken — this is the action
to execute now. The postprocessor must extract and de-normalise this component,
not an arbitrary observable (e.g. sin θ).

Selecting the wrong index (e.g. sin θ at index 5 instead of the action at index 2)
is a silent bug — the output stays in [-1,1] and looks plausible but is meaningless.

### Heuristic-as-baseline, engine-as-refinement

The engine should not replace the heuristic until it has earned confidence.
The right pattern for continuous control:

1. Compute the domain heuristic unconditionally.
2. If `engine.confidence >= 0.6`, blend engine action in with weight `alpha ≤ 0.4`.
3. Below threshold, use pure heuristic.

This prevents the engine from producing catastrophic overrides during the early
episodes when its pattern store is sparse.

### Position matters as much as angle

Short CartPole runs (~40 steps) are usually caused by the pole angle exceeding
the threshold. Long runs (200+) are limited by the cart drifting to the position
boundary (±2.4m). The heuristic must correct both simultaneously:

```
signal = angle + 0.3 * ang_vel + 0.05 * position + 0.02 * velocity
```

A pure angle-based heuristic plateaus around 100-200 steps. Adding cart terms
allows sustained 500-step runs.

### Bang-bang not proportional

Proportional control applies near-zero force when state variables are small.
For CartPole, this allows small perturbations to accumulate uncorrected.
Bang-bang (±1) commits to a direction and applies maximum corrective force
regardless of magnitude — which is correct given the threshold-based termination
condition.

### Result

Avg 500/500 steps across 20 episodes. 64 patterns, 25 sequences.

### CPT carry-forward principles

Cross-Physics Transfer reinforced a small set of rules for future domains:

- Keep preprocessing minimal and explicit.
- Keep the postprocessor policy path stable and easy to inspect.
- Use the engine as a recogniser, not as the final control policy.
- Prefer one strong heuristic plus one stable learned policy over multiple weak
  action voters.
- Add extra hypotheses or views only if they improve convergence within the
  benchmark budget.
- If a richer representation reduces stability, revert it quickly.
- Run transfer only after the source policy is already solved and reproducible.


---

# ARC HPM v5

## Purpose

ARC is handled as a small, substrate-specific extension around the shared HPM v5 stack.

```text
ARC task JSON
→ ARC adapter pipeline
→ ARC polygraphs
→ ARC agent pipeline
→ validated output grid
```

## Phase 1 scope

Only a minimal subset is implemented initially:

- connected components
- colour mapping
- translation
- simple exact-match validation

The current implementation also covers a useful extension of that subset:

- rotation
- horizontal reflection
- vertical reflection
- crop-to-object
- edge lists
- object and shape signatures
- pairwise spatial relations
- relation graphs over object candidates
- grid, object, colour, and structural deltas
- image/object/transformation polygraphs

This is enough to test the architecture without overbuilding the solver, while
still giving the agent a richer structured representation bank to compose.

The low-level region labelling and distance views are now backed by standard
libraries (`scipy.ndimage`) rather than custom flood-fill code, which keeps the
adapter layer smaller and makes the ARC image views more reliable.

Shape feature extraction now uses `skimage.measure.regionprops` for region
metrics such as perimeter, solidity, extent, and Euler number. That keeps the
shape adapter aligned with the library-first direction instead of maintaining a
custom geometry classifier.

The regression suite now also uses real ARC-AGI-2 training JSON tasks from
`data/ARC-AGI-2/data/training` for the routes the current solver can actually
close end-to-end. That keeps the adapter and pattern-management path honest
without pretending the solver is broader than it is.

The objective report now includes both:

- a small ARC smoke subset for quick regression
- an opt-in full labeled ARC-AGI-2 training split for dataset-wide evaluation

The full split is behind `HPM_RUN_FULL_ARC=1` so normal development runs stay
fast, while a dedicated full benchmark still exists for boundary testing.

## Folder split

- `hpm_ai_v5/arc/adapters/`
  - task parsing
  - grid normalization
  - object extraction
  - colour mapping
  - geometry hints
  - example-pair deltas
  - structural analysis views
- `hpm_ai_v5/arc/polygraphs/`
  - image view
  - object view
  - transformation view
  - colour view
  - geometry view
- `hpm_ai_v5/arc/agents/`
  - router
  - hypothesis generation
  - simulation
  - critic
  - output
- `hpm_ai_v5/arc/`
  - shared grid/object/transformation helpers
  - pipeline and solver entry point

## Design rule

Adapters create structural candidates.
Polygraphs create competing interpretations.
Agents test and select transformations.

The shared HPM core remains reusable and domain-agnostic.

ARC-specific structure now lives in the adapter layer:

- `EdgeList` exposes adjacency structure.
- `EdgeList` is now graph-backed through `networkx.grid_2d_graph`.
- `ObjectDetector` and `ShapeDefiner` expose discrete object and shape views.
- `SpatialRelation` exposes between-object geometry.
- `ArcHypothesisAdapter` turns the structural views into explicit candidate transformations.
- `SymmetryCompletionAdapter` surfaces rotational and reflection completions.
- `LineExtensionAdapter` surfaces border and stroke extension candidates.
- `relation_graph(...)` uses `networkx` to package object relations as graph views.
- `ColorMapper` exposes palette statistics.
- `PatternMiner` exposes repeating sub-blocks.
- `GridDelta`, `ObjectDelta`, `ColorDelta`, and `StructuralDelta` expose training-time transformation evidence.
- `ImagePolygraph`, `ObjectPolygraph`, and `TransformationPolygraph` provide competing views for agreement.

## Next phases

1. Add composition of multiple simple patterns.
2. Add strategy memory keyed by task signature.
3. Expand hypothesis ranking across ARC families.


---

# HPM AI v5 Objective Evaluation

## Purpose

The v5 stack needs an objective evaluation layer that turns the existing
benchmarks into comparable numeric scores.

## What it measures

- core reasoning
- agent pipeline flow
- delayed-reward planning
- rule discovery and reuse
- learned utility from reward feedback
- triple sequence discovery and macro reuse
- polygraph agreement over multiple views
- scoring weight adaptation across environments
- online meta-pattern discovery across structurally similar tasks
- automatic adapter composition across preprocessing pipelines
- open adapter discovery with explicit defer on unsupported structure
- ARC transformation solving

Polygraph agreement is scored as a separate benchmark that measures whether the
stack prefers the clean views over the noisy one when multiple structural
representations are available.

Scoring weight adaptation is scored separately as agent-side meta-learning over
the core's fixed `α, β, γ, δ` formula.

Online meta-pattern discovery is scored separately as agent-side structural
abstraction and zero-shot transfer across similar tasks.

Automatic adapter composition is scored separately as agent-side pipeline
selection and reuse over held-out task variants.

Open adapter discovery is scored separately as agent-side adapter selection
and calibration over known numeric and grid families, plus a held-out graph
family that must be deferred.

Those pipelines are now treated as adapter compositions rather than a separate
preprocessing tier.

## Design rule

Evaluation should be metrics-first.

- benchmark result
- numeric score
- compact trace

The report is meant to be comparable across revisions without depending on
natural-language explanations.

## Output contract

An evaluation report should include:

- per-benchmark score
- per-benchmark metrics
- pass/fail flag
- overall score
- summary of passed and failed benchmarks

## Scope

This is not a training loop and not a general judge. It is a deterministic
regression-style evaluation harness for the v5 system surface.
