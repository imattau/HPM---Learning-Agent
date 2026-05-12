# v6 Reasoning Agent Enhancements

## Background & Motivation
The `ReasoningAgent` in the `hpm_ai_v6` codebase provides the top-down cognitive loop, navigating the hierarchical pattern graph built by the `MultiAgentReader`. Currently, the reasoning logic is functional but relies on exact structural matches, isolated path evaluation, and fixed beam search parameters. To make the reasoning engine more robust, especially on sparse or noisy graphs, we will implement three core enhancements in sequence.

## Scope & Impact
This plan outlines changes strictly confined to `hpm_ai_v6/agents/reasoning_agent.py` and its corresponding test suite `hpm_ai_v6/tests/test_reasoning_agent.py`. The interface of `ReasoningAgent` (`reason` and `reason_with_trace`) will remain structurally identical, but the internal pathfinding and evaluation logic will be upgraded.

## Proposed Solution & Implementation Steps

### Phase 1: Enhanced Subgraph Pattern Matching (Analogical Binding)
**Objective:** Relax strict node unifications to allow analogical substitutions when matching subgraph antecedents.
1. **Modify `_edge_matches_template`:** Introduce a similarity threshold parameter. If a variable binding fails the exact key match, calculate the similarity (using the analogy cache) between the required cell and the candidate cell.
2. **Update Template Unification:** Allow a template binding to succeed if the candidate cell is a known top analog to the previously bound cell (e.g., similarity > 0.85).
3. **Score Penalty:** Apply a minor decay penalty to the resulting edge score based on the degree of analogical substitution, reflecting lower confidence compared to exact matches.
4. **Testing:** Add a test verifying that the agent can complete a backward chain using an analogically equivalent intermediate node when the exact node is missing.

### Phase 2: Cross-Agent Evidence Synthesis
**Objective:** Aggregate scores from multiple independent paths that reach the same target to boost overall confidence.
1. **Modify `_beam_search_path` and `_backward_chain_path` returns:** Instead of only tracking the absolute `best_path`, maintain a registry of all valid paths found that successfully reach the goal.
2. **Path Synthesis Logic:** In `reason_with_trace`, iterate over the `candidate_paths`. If multiple paths exist, compute a synthesized score (e.g., using a noisy-OR calculation `1 - product(1 - path_score)`).
3. **Trace Update:** Update the `chosen_path` to reflect the synthesized evidence, potentially promoting an edge that had multiple weak supporting paths over a single brittle path.
4. **Testing:** Add a test proving that two parallel weak paths (e.g., one from `word` agent, one from `phrase` agent) outscore a single slightly stronger path.

### Phase 3: Adaptive Beam Width & Pruning
**Objective:** Replace fixed search bounds with dynamic sizing based on path saliency.
1. **Update `ReasoningAgent.__init__` parameters:** Refactor `beam_width` to `max_beam_width` and add `min_beam_width` and `pruning_threshold` (e.g., drop paths with score < 0.05).
2. **Refactor Search Loops:** In `_beam_search_path` and `_prove_edge_backward`, calculate the score drop-off at each expansion step. If the delta between the best and worst beam candidate is very large, dynamically shrink the beam. If candidates are tightly clustered, maintain `max_beam_width`.
3. **Aggressive Pruning:** Immediately discard any expanded path step whose cumulative score falls below the `pruning_threshold`.
4. **Testing:** Add a test verifying search termination behavior and ensuring high-confidence deep paths are not pushed out by noisy broad transitions.

## Verification
- Run `pytest hpm_ai_v6/tests/test_reasoning_agent.py` after each phase.
- Ensure all 31 existing tests continue to pass without regression.
- Execute `hpm_ai_v6/experiments/reasoning_capability_eval.py` to confirm that benchmark accuracy (synthetic and corpus) is maintained or improved.