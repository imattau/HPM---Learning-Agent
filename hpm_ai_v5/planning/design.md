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

The fix is a `PrefixBufferPreprocessor`:

- maintain the last two raw symbols
- expose them as structured state before the core
- keep a small transition memory keyed by the buffered history
- let the benchmark distinguish `(C, A, B)` from `(A, A, B)` without changing
  the core

The adapter is in place and the buffered benchmark now passes by using that
history window as short-term memory. The core remains unchanged.
