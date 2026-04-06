# SP41: Experiment 17 — Unified Cognitive Loop Test (The Core Agent)

## 1. Overview and Rationale
The **Unified Cognitive Loop Test** is the culmination of all previous experiments. It integrates goal-conditioned retrieval, multi-step reasoning, self-curiosity, belief revision, and world model simulation into a single, continuous autonomous agent loop. 

Until now, we have tested these capabilities in isolation. To prove the HFN architecture can function as a fully autonomous general agent, it must seamlessly orchestrate these mechanisms in response to a dynamic, evolving environment. The agent will be given a high-level goal, seeded with only partial knowledge, and must navigate a world that changes the rules mid-way through.

## 2. Setup & Execution
- **The Environment (Evolving World):** 
  - A continuous state space where reaching a target requires a sequence of specific transformations (e.g., `A -> B -> C -> Target`).
  - **Phase 1 (Partial Knowledge):** The agent knows some basic rules but not all. It must use **Self-Curiosity** to discover the missing transitions through generative play before attempting the goal.
  - **Phase 2 (Action):** The agent is given a goal state. It must use **Goal-Conditioned Multi-Step Reasoning** to formulate a plan and execute it.
  - **Phase 3 (Environmental Shift):** The environment suddenly changes (e.g., the transition rule for `B -> C` is altered). 
  - **Phase 4 (Adaptation):** The agent's previously successful plan will now fail. It must detect this failure (**Belief Revision**), use **World Model Simulation** to evaluate alternative paths in its head, and formulate a new plan to reach the goal.

- **The Cognitive Loop:**
  `Plan → Act → Fail → Revise → Simulate → Retry`

## 3. Evaluation Metrics
1. **Improvement Over Episodes:** Does the agent require fewer physical steps to reach the goal in subsequent episodes after discovering the optimal path?
2. **Structure Reuse:** Are the invariant rules discovered in Phase 1 reused effectively during Phase 4's adaptation, or does the agent build completely redundant structures?
3. **Adaptation Speed (Stability):** How quickly does the agent recover from the Phase 3 environmental shift? Does it revise its beliefs smoothly or collapse into oscillation?
4. **Transfer Across Tasks:** Can the agent apply the structural rules it learned in one sequence to reach a completely different target state in the same environment?

## 4. Why This Matters
*This is the one experiment that matters now.*
It tests whether the HFN framework can sustain the complete lifecycle of agency. An agent must explore when it doesn't know what to do (curiosity), plan when it has a goal (multi-step reasoning), act on that plan, recognize when the world has changed (falsification), update its internal model (revision), and mentally test new solutions before acting again (simulation). Success here means HFN is a viable substrate for Artificial General Intelligence (AGI).

## 5. Implementation Roadmap
1. **Dynamic Environment Simulator:** Create a mock environment class that accepts actions, updates state, and can undergo a "rule shift" at a specific epoch.
2. **The Unified Agent Class:** Wrap the `Observer`, `Decoder`, and `Retriever` into a unified `Agent` class that manages the `Plan -> Act -> Revise` loop.
3. **Experiment Script:** Create `hpm_fractal_node/experiments/experiment_unified_cognitive_loop.py` to run the multi-phase curriculum, tracking the agent's success rate, step efficiency, and forest complexity over time.
