# SP30: The Structured Curriculum (Bootstrapping Mastery) — Design Specification

## 1. Overview and Rationale

The **Structured Curriculum** experiment (SP30) builds on SP29 by addressing the "Mastery Gap." In SP29, the system failed to hit the 6/10 study threshold because the first 10 tasks in the ARC-AGI-2 dataset were too complex for the initial prior library.

SP30 implements a **Bootstrapped Curriculum**: the Governor filters the task pool to find "Toddler-appropriate" tasks for the Study Phase. By solving simpler tasks first (e.g., identity transforms, single color shifts), the system stabilizes high-utility HFN nodes that act as the necessary scaffolding for more complex Test Tasks.

## 2. The Curriculum Filter

The Governor will implement a "Complexity Heuristic" to select the Study Set:
1.  **Pixel Delta Entropy**: Prefers tasks where the output grid has minimal changes from the input grid.
2.  **Color Stability**: Prefers tasks where the number of unique colors is preserved.
3.  **Dimension Invariance**: Prefers tasks where input and output grid shapes are identical.

## 3. The "Educational" Lifecycle

The experiment follows a 3-phase progression:
1.  **Phase 1: Bootstrapped Study (Mastery Target: 6/10)**: 
    *   Governor selects 10 "Simple" tasks.
    *   Iterates until 6/10 are solved (as per SP29 logic).
    *   Stabilizes foundational "Identity" and "Simple Shift" nodes.
2.  **Phase 2: Graduated Study (Stretching)**:
    *   Governor adds 10 "Medium" tasks.
    *   System continues to learn, reusing nodes from Phase 1.
3.  **Phase 3: The Challenge Test**:
    *   System is given 10 "Hard" tasks (randomly sampled).
    *   Retains all learned knowledge.

## 4. Evaluation Metrics

1.  **Time-to-Mastery**: Number of iterations required to hit the 6/10 threshold compared to SP29.
2.  **Scaffolding Efficiency**: Percentage of "Hard" tasks solved that used L2 nodes born during the "Simple" Study Phase.
3.  **Topological Depth**: Does the curriculum approach lead to deeper HFN trees (L3+) compared to random exposure?

## 5. Implementation Roadmap

1.  **Heuristic Implementation**: Add `calculate_task_complexity(task)` to the loader or Governor.
2.  **Curriculum Sorter**: Sort the entire 1000-task pool by complexity.
3.  **Tiered Experiment Script**: Implement `experiment_structured_curriculum.py`.
    *   Tier A (Study): First 10 simplest tasks.
    *   Tier B (Test): 10 tasks of significantly higher complexity.
