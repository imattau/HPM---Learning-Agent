# HPM Hierarchical Abstraction Experiments (SP54-SP56)

This directory contains experiments for **HPM-Native Synthesis, Library Discovery, and Compositional Abstraction**.

## Experiment 45: Library Discovery & Probing (SP55)
Demonstrates an HFN agent autonomously discovering an external Python library, probing its functions to determine their behavioral "portraits," and using those portraits to solve data-transformation tasks.

### Key Components
- `mock_tool_lib.py`: An opaque external library with functions like `find_pairs`, `uniquify`, and `invert_nested`.
- `library_query.py`: Implements `LibraryScannerQuery` (discovery) and `LibraryProbingConverter` (active sensing).
- `experiment_library_discovery.py`: The main experiment driver.

### HPM Principles Demonstrated
1.  **Gap-Driven Exploration**: Discovery is triggered by coverage gaps ($Accuracy - Complexity$) when existing priors fail.
2.  **Behavioral Probing**: Functions are "recognized" by their input-output deltas, not their names or source code.
3.  **No Oracle Leakage**: Uses a generic, hash-based dense projection oracle (`BehavioralOracle`) instead of hand-crafted features.
4.  **Probabilistic Retrieval**: Tools are selected based on HFN log-likelihood (`node.log_prob(goal_vec)`).

## Experiment 46: Compositional Abstraction (SP56)
Demonstrates the **Hierarchical Pattern Stack** by abstracting second-order patterns (Meta-Relations) from first-order relations.

### Key Components
- `sp56_oracle.py`: A stateful oracle that populates a 90D manifold (30D Content | 30D Relation | 30D Meta-Relation).
- `experiment_compositional_abstraction.py`: The main experiment driver.

### HPM Principles Demonstrated
1.  **Compositional Abstraction**: Building Level 3 (Meta-Relation) nodes from sequences of Level 2 (Relation) nodes.
2.  **Zero-Shot Cross-Domain Transfer**: Using abstract structural principles (e.g., "Oscillation") discovered in Math/Spatial domains to predict patterns in a strictly novel Boolean domain.
3.  **Manifold Factorization**: Demonstrates how hierarchical constraints emerge naturally from the geometric factorization of a shared latent space.

### Running the Experiments
```bash
# Run SP55: Library Discovery
PYTHONPATH=. .venv/bin/python hpm_fractal_node/experiments/experiment_library_discovery.py

# Run SP56: Compositional Abstraction
PYTHONPATH=. .venv/bin/python hpm_fractal_node/experiments/experiment_compositional_abstraction.py
```
