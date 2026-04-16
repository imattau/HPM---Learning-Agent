# [SP97] Spurious Correlation Resistance — Structure over Statistics

## Objective
Prove that HPM can detect the failure of statistical shortcuts (spurious correlations) under distribution shift and recover by discovering robust, underlying physical structure.

## Core Idea: The "Illusion" Test
Most ML systems fail when an input feature ($Z_1$) is highly correlated with the target ($Y$) during training but disconnected at test time. This experiment tests if HPM can "escape" this statistical trap.

## Experimental Setup
- **True Structure**: $y = 0.5 g t^2$.
- **Spurious Feature ($Z_1$)**: 
  - **Training**: $z_1 = y + \text{noise}$ (Perfectly predictive).
  - **Test**: $z_1 = \text{random}$ (Broken correlation).
- **Distractors**: Multiple other noisy and random variables are included to ensure non-trivial selection.

## Experimental Phases

### 1. Initial Exploitation (Confounded Data)
- **Outcome**: The agent initially finds a solution using $Z_1$. 
- **Rationale**: $Z_1$ is simpler and genuinely predictive during training. HPM correctly prioritizes efficiency when utility is high.

### 2. Distribution Shift (Failure Detection)
- **Goal**: Evaluate the $Z_1$-based model on data where the $Z_1 \approx Y$ correlation is broken.
- **Outcome**: The model **fails immediately**.
- **HPM Dynamics**: The **Oracle** detects a total collapse in Accuracy, causing the pattern's Utility to plummet and triggering the search for alternatives.

### 3. Structural Recovery (Invariant Discovery)
- **Goal**: Re-solve the task on the shifted data.
- **Process**: The agent triggers a deep BFS search. Guided by amplified **Scientific Insight** (correlations with physical variables like $t^2$), it ignores the now-useless $Z_1$.
- **Outcome**: Successfully discovered the invariant $0.5 g t^2$ structure, completely abandoning the spurious shortcut.

## Key Insights

### Structural Integrity
SP97 demonstrates that HPM is not "locked in" to surface-level statistics. Its ability to **evaluate and replace** patterns based on empirical performance allows it to recover physical "common sense" even after being initially misled.

### Recovery-Based Resistance
Unlike "adversarial robustness" in Deep Learning (which often requires special training), HPM's resistance to spurious correlations is a natural byproduct of its **hierarchical search and evaluation dynamics**. It resists statistics by preferring structure when statistics fail.

## Running the Experiment
```bash
PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp97_spurious_resistance.py
```
