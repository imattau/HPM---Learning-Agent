# [SP95] Inverse Sprinkler Rotation — Non-Intuitive Physical Invariants

## Objective
Demonstrate HPM's ability to discover non-intuitive physical laws (fluid dynamics) that defy common statistical correlations. Specifically, show that it can learn the rotation sign of a sprinkler system regardless of whether fluid is flowing in (inverse) or out (normal).

## Physical Domain: The Inverse Sprinkler
A classic fluid dynamics problem: In certain geometries, an "inverse sprinkler" (sucking fluid in) rotates in the **same direction** as a normal sprinkler (spraying fluid out). This is highly non-intuitive as the "obvious" (but wrong) statistical shortcut would be to flip the rotation sign when flipping the flow sign.

## Experimental Phases

### 1. One-Shot Discovery (Inflow)
- **Goal**: Discover the rotation sign for a single "Inflow" configuration.
- **Process**: The agent explores different torque and geometry primitives.
- **Outcome**: Discovered a macro that correctly predicts rotation for inflow.

### 2. Generalization (Outflow)
- **Goal**: Apply the learned macro to an "Outflow" configuration.
- **Process**: The agent tests its discovered macro on data where the flow direction is reversed.
- **Outcome**: The agent successfully predicts the **same rotation sign**, correctly identifying the physical invariant that torque is independent of flow direction in this specific geometry.

### 3. Zero-Shot Novel Geometry
- **Goal**: Apply the discovered physics to a completely new sprinkler geometry (different number of arms and arm length).
- **Process**: Zero-shot application of the learned macro.
- **Outcome**: Successfully predicted the rotation sign for the novel case at Depth 1.

## Key HPM Mechanisms

### Invariant Identification
HPM succeeds where simple "correlation-fitters" fail because its search is guided by **structural consistency** across different examples. It prefers a single, robust macro that works for both In and Out flow over two separate, parameter-sensitive models.

### Causal Scaffolding
By using a **FluidOracle**, HPM can verify its discovered programs against physical conservation laws (e.g., angular momentum), ensuring that the final output is not just a statistical fit but a physically plausible "theory."

## Running the Experiment
```bash
PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp95_inverse_sprinkler.py
```
