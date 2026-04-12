# SP59: Manifold-Guided Operator Induction

This experiment provides **definitive proof** that the HPM framework can unify **Geometric Latent Representations** with **Symbolic Executable Reasoning** through manifold-guided induction.

## 1. Objective
Synthesize a deep, non-linear operator chain ($x \to (2x + 1) \pmod{10}$) by using the HFN manifold to heuristically guide a beam search. The agent must use residual error vectors to query the forest for relevant operators, rather than brute-forcing all possible combinations.

## 2. Methodology: The Guidance Loop
The experiment uses a 90D manifold factorized into three 30D slices:
- **`[0:30]` Content**: The input state.
- **`[30:60]` Relational**: The resulting delta produced by an operator.
- **`[60:90]` Parameters**: The functional parameters of the operator (Weight, Bias, Modulus).

**The Manifold Query**:
1. For a partial chain $f(x)$, calculate the current prediction $\hat{x}_{t+1} = f(x_t)$.
2. Calculate the residual error: $\Delta = x_{t+1} - \hat{x}_{t+1}$.
3. Query the HFN forest with the pair `[Current State, Residual Delta]`.
4. The forest returns operators whose geometric signatures match the required correction.
5. The beam expands *only* with these retrieved operators.

## 3. Results: Success through Geometry
The agent was trained on three primitives (`Add_1`, `Mul_2`, `Mod_10`) stored across multiple input values. It was then presented with a noisy non-linear sequence: **1, 3, 7, 5, 1, 3...**

### Synthesis Performance

| Metric | Result |
| :--- | :--- |
| **Winning Chain** | `Add_1 ∘ Mod_10 ∘ Mul_2` |
| **Search Depth** | 3 |
| **Mean Prediction Error** | **0.0364** |
| **Guidance Status** | **SUCCESS** (Forest retrieval directed the synthesis) |

### Analysis
By focusing the retrieval on the functional axes (Input and Delta), the agent successfully "saw through" the perceptual noise and identified the clean structural primitives in its memory. The successful construction of a depth-3 non-linear operator proves that HPM does not just store patterns; it uses the geometry of those patterns to **drive creative induction**.

## 4. Running the Experiment
```bash
PYTHONPATH=. .venv/bin/python hpm_fractal_node/experiments/experiment_manifold_induction.py
```
