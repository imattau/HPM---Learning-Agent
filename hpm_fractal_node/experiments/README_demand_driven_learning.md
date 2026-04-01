# Demand-Driven Learning Experiment (SP24)

This experiment evaluates the interaction between the generative (top-down) and perceptual (bottom-up) halves of the HPM architecture. It demonstrates the "Fail-Learn-Retry" loop, where the Decoder acts as a "Curiosity Engine" that triggers targeted learning by the Observer to fulfill missing generative constraints.

## Core Concept: Active Curiosity

In HPM, learning is not just passive observation. It is driven by the necessity of achieving a goal.

1.  **The Generative Gap**: The `Agnostic Decoder` attempts to collapse a high-variance goal into concrete actions. If it lacks the necessary leaves in its target manifold (e.g., trying to point to a "Green Block" without a prior for green), it stalls and emits a `ResolutionRequest`.
2.  **Targeted Observation**: The Governor receives this request and asks the `Observer` to scan its "Historical Buffer" (raw environmental data) for any evidence matching the missing geometry ($\mu$) and topology.
3.  **Grounded Creation**: If the Observer finds matching evidence, it creates a new node and binds it to the requested topological constraints.
4.  **Resumption**: The Decoder retries the path, finds the newly created node, and successfully executes the goal.

## Anti-Hallucination Guard

A critical feature of this architecture is that the Decoder **cannot hallucinate**. 

If the Decoder requests a node to fulfill a goal (e.g., "Point to the Yellow Block"), but the Observer finds **no historical evidence** of a Yellow Block in the environment, the Observer refuses to create the node. The goal definitively fails. The system only learns what is both *needed* and *real*.

## The Experiment

The script runs a 1D block-stacking simulation:
*   **Priors**: The system knows about Red ($X=2.0$) and Blue ($X=5.0$).
*   **Buffer**: The system has passively seen $X=8.0$ (Green), but ignored it because it had no goal associated with it.

### Test 1: The Green Block (Valid Gap)
*   **Goal**: `Goal_Point(Target=Green_Block)`.
*   **Result**: The Decoder stalls, requests a node near $X=8.0$ with a `HAS_COLOR->GREEN` edge. The Observer finds $8.0$ in the buffer, creates the node, and the Decoder successfully retries and outputs the coordinate.

### Test 2: The Yellow Block (Hallucination Guard)
*   **Goal**: `Goal_Point(Target=Yellow_Block)` at $X=10.0$.
*   **Result**: The Decoder stalls and requests a node near $X=10.0$. The Observer scans the buffer, finds nothing, and blocks the hallucination. The goal correctly fails.

## Usage

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_demand_driven_learning.py
```
