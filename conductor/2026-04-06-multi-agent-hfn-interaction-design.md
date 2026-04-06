# SP40: Experiment 16 — Multi-Agent HFN Interaction (Social Learning)

## 1. Overview and Rationale
The **Multi-Agent HFN Interaction** experiment evaluates whether the Hierarchical Fractal Node (HFN) architecture can support social learning and knowledge transfer between independent agents. While multi-agent dynamics were tested in the legacy HPM codebase, they have not been fully validated using the new, purely geometric HFN structure. 

In this experiment, two or more autonomous `Observer` instances (each maintaining its own private `Forest` and `Meta-Forest`) will interact. By exchanging nodes or observing each other's generated outputs, the agents should be able to bootstrap their world models faster than a solitary agent. This tests if HFN representations are universally decipherable and transmissible between distinct sovereign learners.

## 2. Setup & Execution
- **Environment & Curriculum:** A complex, multi-part curriculum (e.g., a set of distinct structural rules or patterns). The total curriculum is divided.
- **Agent Configurations:**
  - **Agent A (Solo Baseline):** Explores the entire curriculum alone.
  - **Agent B & Agent C (Social Learners):** Each explores a *subset* of the curriculum initially (developing specialized, partial world models).
- **The Interaction Channel (Social Phase):**
  - After initial independent learning, Agent B and Agent C interact.
  - **Mechanism:** Agent B generates ("dreams") observations based on its learned structure and broadcasts them to Agent C, and vice versa. Alternatively, they directly broadcast their highest-weight, most compressed HFN nodes.
  - The receiving agent uses its own `Observer` to ingest the broadcasted nodes/observations, attempting to integrate the foreign structure into its own forest.
- **The Evaluation Phase:** Both social agents are then tested on the *full* curriculum to see if they successfully absorbed each other's specialized knowledge.

## 3. Evaluation Metrics
1. **Learning Acceleration (Speed):** Do the interacting agents reach high accuracy on the full curriculum in fewer total observations than the Solo agent?
2. **Convergence vs. Divergence:** Do Agent B and Agent C's forests converge toward a shared structural representation of the world, or do they develop incompatible, divergent hierarchies?
3. **Knowledge Transfer Efficiency:** How many "social interactions" does it take for Agent B to master Agent C's domain without ever seeing the raw training data?

## 4. Why This Matters
*Can HFN support knowledge transfer between agents?*
If an agent can only learn from raw environmental data, its learning speed is bottlenecked by its individual experience. True intelligence scales through culture and communication. If HFN nodes can be reliably communicated, absorbed, and utilized by foreign agents, it proves the framework can support massive, decentralized "Sovereign Clusters" that parallelize learning and share high-level abstractions instantly.

## 5. Implementation Roadmap
1. **Curriculum Partitioning:** Create a synthetic dataset with two distinct concepts (Concept 1 and Concept 2).
2. **Social Interaction Protocol:** Implement a mechanism for `Observer` instances to exchange HFN nodes or decode their top nodes into synthetic observations for the other to ingest.
3. **Comparative Loop:** Run the Solo agent on Concept 1+2. Run Agent B on Concept 1, Agent C on Concept 2, then run the social exchange phase.
4. **Experiment Script:** Create `hpm_fractal_node/experiments/experiment_multi_agent_social.py` to orchestrate the agents and output comparative metrics on convergence and transfer efficiency.
