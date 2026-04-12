# HPM Library Discovery Experiments (SP55)

This directory contains experiments for **HPM-Native Library Discovery and Behavioral Recognition**.

## Experiment 45: Library Discovery & Probing
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
5.  **Compositional Reasoning**: Successfully chains discovered tools to solve multi-step tasks (e.g., "Uniquify then Invert").

### Running the Experiment
```bash
PYTHONPATH=. .venv/bin/python hpm_fractal_node/experiments/experiment_library_discovery.py
```
