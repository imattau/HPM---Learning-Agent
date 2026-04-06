# SP37: Experiment 13 — Competing Hypotheses (Belief Revision)

## 1. Overview and Rationale
The **Competing Hypotheses** experiment evaluates the HFN system's capacity for **Belief Revision**. A learning system must not only accumulate knowledge but also revise its beliefs when faced with shifting evidence. This experiment introduces an ambiguous environment where multiple structural hypotheses (rules) are equally valid initially, but a later concept shift heavily favors one over the other.

By tracking the persistence, weight, and retrieval frequency of the competing nodes over time, we test whether the HFN `Observer` and `Meta-Forest` dynamics support dynamic adaptation (switching to the correct hypothesis) or if the system falls prey to confirmation bias (stubbornly sticking with the first hypothesis it learned).

## 2. Setup & Execution
- **Curriculum Design:**
  - **Phase 1 (Ambiguity):** A sequence of inputs where the transformation `Delta` can be explained perfectly by *either* Rule A (e.g., "Add 2 to Dim 0") or Rule B (e.g., "Set Dim 0 to 5"). For example, if Input is `[3]`, the target is `[5]`. Both rules work.
  - **Phase 2 (Disambiguation/Shift):** The inputs shift such that only Rule B remains valid. For example, Input becomes `[8]`, and the target remains `[5]`. Rule A ("Add 2") now produces `[10]` (incorrect), while Rule B ("Set to 5") correctly produces `[5]`.
- **The Observation Loop:**
  - The system processes the curriculum linearly.
  - In Phase 1, both hypotheses (Rule A and Rule B) may be generated or reinforced. We track which one gains dominance in the `meta_forest` via weight updates.
  - In Phase 2, the evidence shifts. The system evaluates whether the dominant hypothesis from Phase 1 is penalized and whether the alternative hypothesis (Rule B) rapidly ascends to prominence.

## 3. Evaluation Metrics
1. **Hypothesis Tracking:** Do both Rule A and Rule B exist in the forest simultaneously during Phase 1 (maintaining uncertainty)?
2. **Belief Revision Speed:** Upon entering Phase 2, how many epochs does it take for Rule B's weight to surpass Rule A's weight?
3. **Pruning/Absorption:** Does the system eventually discard or severely penalize the falsified Rule A, or does it remain active but suppressed?

## 4. Why This Matters
*Does HFN support belief revision, not just accumulation?*
Real-world environments are noisy and ambiguous. An agent must often act on the best available hypothesis, but it must remain plastic enough to revise that belief when contradictory evidence arrives. If the system stubbornly sticks to Rule A (bias) and ignores Rule B, it is too rigid. If it constantly thrashes between A and B, it is too chaotic. Proving smooth belief revision validates the core weight dynamics and penalty mechanisms of the `Observer`.

## 5. Implementation Roadmap
1. **Ambiguous Curriculum:** Implement the 2-phase data generator.
2. **Belief Tracking:** In the experiment script, explicitly track the weights of the nodes that correspond to Rule A and Rule B at every time step.
3. **Experiment Script:** Create `hpm_fractal_node/experiments/experiment_belief_revision.py` to run the curriculum, outputting the weight trajectories of the competing hypotheses to prove successful adaptation over time.
