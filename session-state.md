# Session State Checkpoint
Generated: 2026-04-24
Reason: Context threshold exceeded (95%)

## Execution Mode

**Mode**: unattended
**Auto-Continue**: true

> **CRITICAL**: Do NOT pause for user confirmation. Complete ALL remaining work.

## Current Task
Write spec and plan for HPM v4 Wikipedia character-stream simulation.

## Prior Work Completed This Session
- Fixed 3 bugs in hpm_ai_v4/simulations/hpm_ai_simulation_1.py (committed 5d4772a0 on hpm-ai-v4-dev)
- Reasoning layer spec/plan written (docs/superpowers/specs/2026-04-24-reasoning-layer-design.md, docs/superpowers/plans/2026-04-24-reasoning-layer.md)

## Current Branch
hpm-ai-v4-dev

## The Design Request (full text from user)

The user provided a detailed simulation design for HPM AI learning from Wikipedia character streams. Key points:

### Environment
- WikipediaStream class: reads chars from file, converts to char IDs (ASCII 32-126 + newline = ~95 symbols)
- Emits one char at a time, loops when exhausted

### Pattern Architecture
- Hierarchical HMM with 3 latent levels: z3 (K=8 topic), z2 (K=16 phrase), z1 (K=16 char class)
- Transition matrices A3 (8×8), A32 (8×16), A21 (16×16), emission B (16×95)
- Flat Bernoulli baseline patterns

### Training
- Single agent (or 3 for social)
- Online EM, sliding window 100 chars
- Evaluators: epistemic, affective (curiosity + compression), social
- Replicator dynamics with decay, recombination every 500 steps
- 100k characters, log every 1k

### Reasoning Queries (PROGRAMMATIC - user flagged this)
The Reasoner interface is programmatic (char IDs in/out), NOT natural language:
a) Next-char prediction: compose_predictions(obs_seq: List[int]) -> np.ndarray (prob dist over 95 chars)
b) Word completion: encode prefix as char IDs, call compose_predictions repeatedly until space
c) Planning: plan(goal_state=space_id, horizon=5, num_rollouts=20) -> List[int]
d) Counterfactual: counterfactual(obs_seq, intervention_char_id) -> np.ndarray
e) Explanation: explain() -> str (human-readable description of learned pattern)

### Metrics
- Prediction accuracy > 50% (baseline 1/95 ≈ 1%)
- Word completion top-1 > 30%
- Planning success rate > 60%
- Counterfactual KL > 0.1
- Compression mutual info: increases from ~0 to >0.2

### Code snippet from user shows TotalHPMSystem integration (system.py)

## Key Architectural Question to Resolve
The existing HierarchicalPattern in hpm_ai_v4/pattern.py is 2-level HMM.
The design calls for a 3-level HMM (z3, z2, z1).
Options:
A) Use existing 2-level as-is (simpler, fits existing codebase)  ← RECOMMENDED
B) New 3-level pattern class
C) Composition of two 2-level patterns

The user said "spec and plan" without wanting to answer many questions. Go with Option A (use existing 2-level HierarchicalPattern with latent_dim=16, obs_dim=95) as the simplest path that validates the simulation concept. Note this in the spec.

## v4 Codebase Files
- hpm_ai_v4/pattern.py — HierarchicalPattern (2-level HMM), FlatPattern
- hpm_ai_v4/agents/agent.py — HPMAgent with self.reasoner = Reasoner(self), act()
- hpm_ai_v4/agents/reasoning.py — Reasoner: compose_predictions, plan, counterfactual, explain, simulate
- hpm_ai_v4/system.py — TotalHPMSystem
- hpm_ai_v4/io/adapters.py — DiscreteInputAdapter, etc.
- hpm_ai_v4/simulations/hpm_ai_simulation_1.py — existing simulation (reference)

## What Needs to Be Done
1. Write spec: docs/superpowers/specs/2026-04-24-wikipedia-simulation-design.md
2. Write plan: docs/superpowers/plans/2026-04-24-wikipedia-simulation.md
3. Commit both

## Spec Content to Cover
1. Overview / goal
2. WikipediaStream environment (char vocabulary, file loading, streaming)
3. Pattern architecture decision: use existing 2-level HierarchicalPattern, latent_dim=16, obs_dim=95
4. Training loop (online EM, evaluators, replicator dynamics, recombination params)
5. Programmatic reasoning interface — ALL queries are char-ID in/out:
   - next_char_predict(prefix_chars: str) -> Dict[str, float] (top-5 predictions with probs)
   - word_complete(prefix: str, max_len=10) -> str
   - plan_to_word_boundary(horizon: int, num_rollouts: int) -> str
   - counterfactual_shift(context: str, forced_char: str) -> Dict[str, float]
   - explain_best_pattern() -> str
6. Metrics and success criteria
7. Files to create/modify
8. NOT a natural language interface — all reasoning is programmatic

## Files to Create (Plan Tasks)
- hpm_ai_v4/simulations/wikipedia_sim.py — main simulation script
- hpm_ai_v4/simulations/data/download_wikipedia.py — script to get corpus
- hpm_ai_v4/simulations/text_reasoning.py — TextReasoningInterface (wraps Reasoner with char encoding/decoding)
- hpm_ai_v4/tests/test_wikipedia_sim.py — unit tests for WikipediaStream + TextReasoningInterface

## Continuation Instructions
1. Read hpm_ai_v4/agents/reasoning.py to understand current Reasoner interface signatures
2. Read hpm_ai_v4/pattern.py to confirm HierarchicalPattern constructor params
3. Write the spec document (comprehensive, no TBDs)
4. Self-review the spec
5. Write the plan document (TDD, bite-sized tasks with actual code)
6. Commit both files
7. Tell user "Spec written to <path> and plan written to <path>. Please review."
