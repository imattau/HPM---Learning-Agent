# Implementation Plan: HPM AI v4 - Completing the Social & Deliberative Stack

## Background & Motivation
The HPM AI v4 architecture successfully integrates Core, I/O, Reasoning, and Meta layers. However, several critical components remain as stubs or minimal implementations. Specifically, the "Social" aspect of the framework (cultural convergence and gossip) is one-way, the reasoning layer lacks robust planning over hierarchical structures, and the system lacks baseline adapters for simple discrete interaction. This plan aims to bridge these gaps to fulfill the HPM framework's predictions regarding social knowledge propagation and multi-level meta-cognition.

## Scope & Impact
- **Substrate Layer:** Enabling bidirectional pattern exchange (gossip).
- **Reasoning Layer:** Implementing full hierarchical planning and predictive composition.
- **Social Layer:** Refining structural signatures for cultural field convergence.
- **Reflection Layer:** Implementing window-based stagnation detection and meta-interventions.
- **I/O Layer:** Providing concrete baseline adapters.

## Proposed Solution
We will implement a bidirectional "Collective Memory" loop. Agents will not only broadcast their best patterns to the `ExternalSubstrate` but also periodically "gossip" by pulling random high-density patterns from the substrate. We will complete the `Reasoner` by implementing a Monte Carlo-style planning algorithm that uses the `HierarchicalPattern` generative model to simulate future trajectories. Finally, we will refine the `ReflectionEngine` to monitor population-level epistemic metrics and trigger "Curiosity Interventions" when learning plateaus.

## Alternatives Considered
- **Direct Peer-to-Peer Gossip:** Rejected in favor of a Substrate-mediated approach to better align with the HPM "Pattern Field" theory and support asynchronous collective memory.
- **Deterministic Planning:** Rejected in favor of stochastic rollouts to better handle the probabilistic nature of HFN/HMM patterns.

## Phased Implementation Plan

### Phase 1: Bidirectional Collective Memory (Gossip)
- **Task 1.1:** Update `ExternalSubstrate` (`hpm_ai_v4/tools/substrate.py`) to store full pattern objects and support random retrieval.
- **Task 1.2:** Implement `gossip_with_substrate` in `HPMAgent` (`hpm_ai_v4/agents/agent.py`).
- **Task 1.3:** Integrate gossip into the `perceive_and_learn` loop with appropriate frequency (e.g., every 20 steps).

### Phase 2: Deliberative Reasoning & Planning
- **Task 2.1:** Refine `predict_next_distribution` in `HierarchicalPattern` (`hpm_ai_v4/pattern.py`) to correctly chain $A_3 \to A_{32} \to A_{21} \to B$.
- **Task 2.2:** Implement `Reasoner.plan()` (`hpm_ai_v4/agents/reasoning.py`) using stochastic rollouts over the latent state space.
- **Task 2.3:** Refine `compose_predictions()` to use replicator weights for population-level blending.

### Phase 3: Meta-Cognitive Social & Reflection
- **Task 3.1:** Enhance `SocialNetwork` (`hpm_ai_v4/social.py`) with more granular structural signatures to drive faster cultural convergence.
- **Task 3.2:** Update `ReflectionEngine` (`hpm_ai_v4/reflection.py`) to detect learning plateaus over a moving window and trigger global curiosity (beta_aff) boosts.

### Phase 4: Baseline I/O & Integration
- **Task 4.1:** Implement `DiscreteInputAdapter` and `ConsoleOutputAdapter` in `hpm_ai_v4/io/adapters.py`.

## Verification
- **Unit Tests:** New tests in `hpm_ai_v4/tests/test_v4_stubs.py` for each component.
- **Integration Test:** Verify that two agents in a shared environment converge on identical patterns more quickly when gossip is enabled.
- **System Test:** Run `TotalHPMSystem` with `DiscreteInputAdapter` to verify the full loop.

## Migration & Rollback
- All changes are additive or refactorings of existing stubs.
- Rollback involves reverting to the current git commit (HPM v4 integrated).
