# Task 1 Implementation Plan

## Objective
Implement Task 1: Hierarchical Function Learning with Surface Distractors to validate the HPM AI v3 core.

## Key Files
- `hpm_ai_v3/task1/task1_data.py`: Data generation logic.
- `hpm_ai_v3/task1/regression_pattern.py`: Regression-specific pattern.
- `hpm_ai_v3/task1/run_task1.py`: Training and evaluation script.

## Implementation Steps

### 1. Data Generation
Create `hpm_ai_v3/task1/task1_data.py` to generate structural (a, b) and surface (color, shape) features.

### 2. RegressionPattern Implementation
Create `hpm_ai_v3/task1/regression_pattern.py` by extending `CausalPattern` for regression tasks (predicting y from inputs).

### 3. Training & Evaluation
Create `hpm_ai_v3/task1/run_task1.py` with:
- `train_hpm_agent()`: Population-based learning.
- `evaluate_sensitivity()`: Sensitivity analysis comparing structural vs. surface perturbations.
- `train_baseline()`: MLP baseline for comparison.
- `main()`: Full orchestration.

## Verification
1. Compare MSE of HPM vs MLP.
2. Verify structural vs surface sensitivity asymmetry.
3. Observe symbolic pattern emergence.
