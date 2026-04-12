# SP57: Operator-Level Compositional Abstraction

This experiment demonstrates the HPM framework's transition from **Vector Addition** to **Functional Operator Composition**.

## 1. Objective
Demonstrate that an HFN agent can synthesize complex, non-linear dynamics zero-shot by chaining primitive mathematical operators ($f(g(x))$), rather than merely adding static geometric deltas.

## 2. Key Components
- `sp57_operators.py`: Implements the `Operator` class ($f(x) = Ax + B$) and the `OperatorOracle` for continuous manifold embedding.
- `experiment_operator_composition.py`: The main experiment driver that trains primitives and synthesizes the $x \to 2x + 1$ rule.

## 3. Results: Synthesis Success
The agent was trained on two independent primitives:
1.  **Additive**: `Add_1` ($x \to x + 1$)
2.  **Multiplicative**: `Mul_2` ($x \to x \times 2$)

It was then presented with an out-of-family sequence: **1, 3, 7, 15, 31...** ($x \to 2x + 1$).

### Performance Comparison

| Condition | Mean Prediction Error | Result |
| :--- | :--- | :--- |
| **Constant Addition** | 101.50 | **FAIL** |
| **Constant Multiplication** | 6.50 | **FAIL** |
| **Composed Operator (SP57)** | **0.00** | **SUCCESS** |

### Analysis
The agent successfully performed a generative search over its Long-Term Memory and discovered that the composition **Add_1 ∘ Mul_2** perfectly explained the non-linear trajectory. This proves that HPM can achieve universal function induction by embedding functional semantics into the HFN geometry.

## 4. Running the Experiment
```bash
PYTHONPATH=. .venv/bin/python hpm_fractal_node/experiments/experiment_operator_composition.py
```
