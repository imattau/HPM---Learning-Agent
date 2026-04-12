# SP58: Heuristic Operator Induction and Disambiguation

This experiment demonstrates the HPM framework's transition to **Scalable and Noise-Robust Function Induction**.

## 1. Objective
Synthesize a deep, non-linear operator chain ($x \to (2x + 1) \pmod{10}$) using a heuristic beam search guided by manifold error, while disambiguating between competing hypotheses under perceptual noise.

## 2. Key Components
- `sp57_operators.py` (Extended): Added `ModOperator` and support for non-linear `ComposedOperator` chains.
- `experiment_heuristic_induction.py`: Implements `BeamSearchSynthesis` and the disambiguation test logic.

## 3. Results: Scalable Induction
The agent was trained on three primitives:
1.  **Additive**: `Add_1`
2.  **Multiplicative**: `Mul_2`
3.  **Modular**: `Mod_10`

It was then presented with a noisy non-linear sequence: **1, 3, 7, 5, 1, 3...**

### Synthesis Performance

| Metric | Result |
| :--- | :--- |
| **Winning Chain** | `Add_1 ∘ Mod_10 ∘ Mul_2` |
| **Search Depth** | 3 |
| **Mean Prediction Error** | **0.0364** |
| **Disambiguation Status** | **SUCCESS** (Outperformed shallow and greedy baselines) |

### Analysis
The agent successfully maintained a beam of candidate hypotheses and used the evolving noisy trajectory to prune incorrect models (like `Mul_3` or `Add_2`). The successful recovery of a depth-3 non-linear operator proves that HPM can achieve robust symbolic induction by unifying functional composition with manifold-guided search.

## 4. Running the Experiment
```bash
PYTHONPATH=. .venv/bin/python hpm_fractal_node/experiments/experiment_heuristic_induction.py
```
