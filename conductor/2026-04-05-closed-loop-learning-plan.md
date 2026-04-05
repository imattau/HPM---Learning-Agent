# Objective
Design and implement "Experiment 7: Closed-Loop Learning" to validate that the HFN system improves its own world model over time through the continuous `observe -> explain -> fail -> create -> re-observe` cycle.

# Background & Motivation
The core premise of the Hierarchical Pattern Modelling (HPM) framework is that an agent continuously minimizes surprise by abstracting and integrating new structural patterns into its world model. While we have tested top-down reasoning and induction, we need a definitive test to show the continuous *bottom-up learning* loop in action. This experiment will serve as the primary validation that the system's `residual_surprise` decreases while its `structure` (nodes and edges) increases as it processes a curriculum of observations.

# Scope & Impact
- **New File:** `hpm_fractal_node/experiments/experiment_closed_loop.py`
- **Metrics Tracked:** 
  - `residual_surprise` per observation over time
  - `forest_size` (node count, edge count) over time
- **Impact:** Demonstrates the fundamental, domain-agnostic learning capability of the HFN framework without task-specific reasoning scaffolding.

# Proposed Solution
1. **Curriculum Selection:** Use a sequence of structured observations (e.g., a subset of ARC training grids converted to manifold vectors, or a stream of simple repeating geometric transformations).
2. **Closed-Loop Engine:** 
   - Initialize a baseline `Forest` and `Observer`.
   - Loop over the curriculum for multiple epochs.
   - For each item, call `res = observer.observe(x)`.
   - Record `res.residual_surprise` and `len(forest)`.
   - The observer's native dynamics will attempt to explain the input. If it fails (high residual surprise), it naturally creates new nodes to capture the anomaly. On subsequent epochs, these new nodes will be used to explain the input, resulting in lower surprise.
3. **Metric Evaluation:** After processing, output a chronological report showing the expected trends: surprise goes down, structure goes up and then stabilizes (compression).

# Alternatives Considered
- **Using Full ARC Tasks:** We could run the full solver loop, but that introduces top-down reasoning noise. Focusing solely on the `Observer`'s `observe` method isolates the world-model building process.
- **Plotting vs Terminal Output:** We will start with a clear tabular terminal output (Epoch | Avg Surprise | Forest Size) to maintain CLI compatibility, with an option to dump a CSV for plotting.

# Implementation Steps
- [ ] **Step 1:** Create `hpm_fractal_node/experiments/experiment_closed_loop.py`.
- [ ] **Step 2:** Implement an observation generator that cycles through a set of distinct, structured patterns.
- [ ] **Step 3:** Implement the closed loop: tracking `residual_surprise` and forest size across multiple epochs.
- [ ] **Step 4:** Format a clear, readable report that proves `surprise ↓` and `structure ↑` over iterations.

# Verification & Testing
- Run `python hpm_fractal_node/experiments/experiment_closed_loop.py`.
- Verify that `residual_surprise` is high in Epoch 1 and significantly lower in Epoch 3+.
- Verify that the forest grows as it encounters novel patterns, and the growth rate slows as the world model becomes comprehensive.
