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
