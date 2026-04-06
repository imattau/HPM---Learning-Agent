# SP36: Experiment 12 — Self-Curiosity (Autonomous Demand-Driven Learning)

## 1. Overview and Rationale
The **Self-Curiosity** experiment evaluates the HFN system's capacity for true autonomy by removing the passive external data stream. Instead of waiting for a curriculum to be spoon-fed, the agent must drive its own learning trajectory through a continuous loop of generative exploration and self-evaluation. 

This experiment builds upon the earlier "gap query" mechanisms but inverts the dynamic: the system uses its own `Decoder` to generate observations from its existing structural knowledge, then feeds these generated observations back into its `Observer`. By doing so, it probes the boundaries of its own understanding, detects internal inconsistencies or "surprising" structural gaps, and refines its world model without external prompting.

## 2. Setup & Execution
- **Curriculum (Removed):** The system is initialized with a small seed of fundamental priors (e.g., basic spatial or mathematical transformations), but there is no external stream of training data.
- **The Self-Curiosity Loop:**
  1. **Generate (Dream):** The system randomly samples an abstract node from its `Forest` (or a combination of nodes) and uses the `Decoder` to synthesize a concrete observation.
  2. **Observe (Perceive):** The generated observation is fed back into the `Observer`.
  3. **Evaluate (Detect Surprise):** The `Observer` attempts to explain the generated observation. If the generation process exposed a structural gap or conflict (e.g., the combination of two rules produced an unexpected interference pattern), the `residual_surprise` will be high.
  4. **Expand (Refine):** High surprise triggers the creation of new nodes (or gap queries to an external oracle/environment if configured). The system creates new structural representations to resolve the inconsistency it just hallucinated.

## 3. Evaluation Metrics
1. **Generative Diversity:** Does the system continue to explore new regions of the manifold, or does it collapse into generating the same exact observation repeatedly?
2. **Structural Growth (Complexity-Accuracy):** Does the `Forest` size grow in a principled way? We track the number of new, valid structural concepts discovered without external supervision.
3. **Surprise Trajectory:** We expect to see cyclic spikes in surprise as the system discovers new combinations, followed by compression phases as it integrates them.

## 4. Why This Matters
*Can the system drive its own learning trajectory?*
Passive learners require massive, carefully curated datasets to cover all edge cases. An autonomous agent with **Self-Curiosity** can self-play. By combining its known rules in novel ways and evaluating the results, it can autonomously map out the logical consequences of its own knowledge base, discovering new patterns and abstractions entirely on its own. This is the foundation of unsupervised "Play."

## 5. Implementation Roadmap
1. **Generative Sampler:** Implement a method to select abstract nodes (or pairs of nodes) and decode them into concrete vectors. Add slight noise to encourage exploration.
2. **Autonomous Loop:** Create the `generate → observe → evaluate → expand` cycle. Ensure the `Observer` processes the self-generated data.
3. **Experiment Script:** Create `hpm_fractal_node/experiments/experiment_self_curiosity.py` to run the autonomous loop for a set number of "dreams," tracking forest size, generation diversity, and surprise spikes.
