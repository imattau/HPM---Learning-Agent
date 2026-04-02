# Implementation Plan: Iterative "Thinking" Solver (SP28)

## 1. Goal
Upgrade the ARC Solver to support 30x30 grids and implement an iterative "Thinking" loop that validates rule hypotheses against training data before solving the test case.

## 2. Manifold Expansion (30x30)
- Update `hpm_fractal_node/arc/arc_sovereign_loader.py`:
    - `S_DIM = 900` (30x30).
    - `COMMON_D = 900 + 30 + 20 = 950`.
    - Update `normalize` to pad/clip to 30x30.

## 3. The "Thinking" Loop Algorithm
1.  **Induce Candidates**: Each specialist returns top 3 winners.
2.  **Form Hypotheses**: Generate permutations of (Spatial, Symbolic) pairs.
3.  **Validate**: For each hypothesis $H_i$:
    - For each train example $(I_j, O_j)$:
        - Decode $H_i$ using $I_j$ as context.
        - If decoded $\Delta$ results in $O_j'$, and $O_j' \neq O_j$, then $H_i$ is **INVALID**.
    - If $H_i$ is INVALID:
        - **Negative Anchoring**: Register $H_i$ in the `Failure_Manifold`.
        - Calculate the "Error Vector" ($O_j - O_j'$) and log the Euclidean distance.
    - If $H_i$ is valid for ALL train examples, mark as **SUCCESS**.
4.  **Execute**: Use the first valid $H_i$ to generate the test output.

## 4. Implementation Code Changes

### 4.1 Update `hpm_fractal_node/arc/arc_sovereign_loader.py`
```python
# Constants
S_DIM = 900  # 30x30 Spatial
M_DIM = 30   
C_DIM = 20   
COMMON_D = S_DIM + M_DIM + C_DIM

def extract_spatial(input_grid: np.ndarray, output_grid: np.ndarray) -> np.ndarray:
    def normalize(g):
        res = np.zeros((30, 30))
        r, c = min(30, g.shape[0]), min(30, g.shape[1])
        res[:r, :c] = g[:r, :c]
        return res / 9.0
    delta = normalize(output_grid) - normalize(input_grid)
    return delta.flatten()
```

### 4.2 Experiment Script: `hpm_fractal_node/experiments/experiment_thinking_arc_solver.py`
The experiment will extend `experiment_sovereign_arc_solver.py` with the following changes:
- `WorkerConfig.common_d` set to 950 (or sliced sizes).
- Governor loop uses `for hypothesis in hypothesis_list:` logic.
- Adds `Failure_Manifold` (Forest) to store rejected hypotheses.
- Logs the "Geometric Distance to Solution" ($L_2$ norm of the output error).

## 5. Review against Specification
- **30x30 Support**: Yes, implemented via `S_DIM = 900`.
- **Iterative Thinking**: Yes, implemented via the internal validation loop against all training examples.
- **Top-K Induction**: Yes, Governor collects multiple candidates per specialist to form the hypothesis list.
- **Smoke Tests**: Will include the heartbeat and retrieval tests from SP27.
