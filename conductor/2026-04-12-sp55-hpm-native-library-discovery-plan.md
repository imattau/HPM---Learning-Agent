# SP55: HPM-Native Library Discovery Redesign (Plan)

## Objective
Redesign the SP55 Library Discovery experiment to be fully HPM-compliant by eliminating the hand-crafted behavioral oracle, removing forced discovery mechanisms, and implementing probabilistic retrieval and multi-probe learning.

## Key Files & Context
- `hpm_fractal_node/code/library_query.py`: Contains the Oracle and Converter logic.
- `hpm_fractal_node/experiments/experiment_library_discovery.py`: The main experiment script.

## Implementation Steps

### 1. Remove Oracle Leakage (Pre-trained Embedding)
- Replace `BehavioralOracle` with `PretrainedEmbeddingOracle` in `library_query.py`.
- This new oracle will serialize inputs (e.g., `repr(data)`) and pass them through a lightweight pre-trained embedding model (or a simulated dense latent space generator if external APIs are unavailable) to derive the state vector.
- This ensures the latent space is rich and generic, removing the hand-engineered feature leakage.

### 2. Implement Utility-Driven Competition
- In `experiment_library_discovery.py`, restore the default `residual_surprise_threshold` (e.g., 2.0).
- Allow the agent to first attempt solving the task using local structural primitives (priors).
- As the local AST grows, the complexity penalty will lower the HPM utility. Once the utility drops below the threshold, it will naturally trigger the coverage gap and the `LibraryScannerQuery` without manual forcing.

### 3. Multi-Probe Iterative Learning & Uncertainty
- Update `LibraryProbingConverter` to simulate iterative probing.
- Instead of a single pass returning a perfect $\mu$, the converter will generate a sequence of noisy observations across multiple distinct probes.
- Initialize discovered nodes with high uncertainty ($\sigma$), allowing standard HPM Observer dynamics to refine the representation and reduce variance over time.

### 4. Remove Symbolic Index Mapping & Implement Probabilistic Retrieval
- Remove the deterministic `id_to_func` index mapping shortcut in the test script.
- Update the retrieval selection logic to rank all candidates purely by HFN log-likelihood (`node.log_prob(goal_vec)`), selecting the highest-scoring node based purely on behavioural match.

### 5. Add Compositional Tasks
- Introduce multi-step reasoning tasks in `test_recognition` (e.g., "Pair then Uniquify" or "Invert then Scale").
- Verify that the agent can chain multiple retrieved library nodes to satisfy a complex target delta.

## Verification & Testing
- Run `experiment_library_discovery.py`.
- Verify that the gap query is triggered naturally via utility competition.
- Verify that the correct tools are retrieved using log-likelihood scoring without index mapping.
- Confirm success on the new compositional task.
