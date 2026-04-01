# Agnostic Decoder Experiment (SP22)

This experiment evaluates the **Agnostic Decoder**, the generative half of the HPM architecture. It tests the system's ability to "collapse" abstract, high-variance HFN goals into concrete, low-variance actions using purely geometric and topological signals.

## Core Theory: Variance Collapse

The decoder operates on the principle that **abstraction is variance**. A high-level goal has high $\Sigma$ (it is a "cloud" of potentiality), while a concrete action has low $\Sigma$ (it is a "point" of execution). 

The `hfn.Decoder` algorithm uses three agnostic mechanisms:
1.  **Concrete Check**: If a node's variance is below a threshold, it is returned as an output.
2.  **Explicit Expansion**: If a node has children, the decoder recursively unfolds them (the "Scripted" path).
3.  **Implicit Resolution**: If an abstract node has no children, the decoder queries a target manifold and uses **Topological Scoring** to find the best-fitting concrete node (the "Search" path).

## Architecture

- **`hfn/decoder.py`**: A domain-agnostic engine residing in the core `hfn/` folder. It contains no language or physics logic.
- **Topological Scoring**: Constraints are resolved by checking DAG edges. If a goal requires a specific relationship (e.g., `HAS_COLOR -> RED`), the decoder rejects any candidate that lacks that edge, even if it is geometrically closer.

## Experiment: 1D Block Stacking

To prove the decoder is agnostic, we use a synthetic 1D coordinate task:
- **Target Manifold**: Specific positions ($X=1.0, 2.0, 5.0$).
- **Topological Prior**: Position $2.0$ is bound to the concept `RED`.
- **Test Cases**:
    - **Test 1**: Expand a sequence goal into two specific positions.
    - **Test 2**: Find a "Red thing" near $X=2.1$ (successfully resolves to $pos\_2.0$).
    - **Test 3**: Find a "Red thing" near $X=4.9$ (rejects the closer $pos\_5.0$ and correctly selects $pos\_2.0$).

## Key Insights

- **Zero-Logic Generativity**: Proves that synthesis can be achieved without if-then rules or domain-specific templates.
- **Topological Integrity**: Verified that the HFN graph structure is a functional constraint system for top-down reasoning.
- **Uniformity**: The decoder uses the same HFN primitives as the Observer, fulfilling the scale-free architectural goal.

## Usage

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_agnostic_decoder.py
```
