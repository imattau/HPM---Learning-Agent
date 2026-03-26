# HPM AI v2: Implementation Plan (Refined)

## Overview
This plan outlines the steps to build a fresh iteration of the HPM framework into a new directory `hpm_ai_v2`. It incorporates developmental levels 1-5, dynamic conflict discovery, and institutional field logic as defined in the "Human Learning as Hierarchical Pattern Modelling v1.25" paper.

## Steps

- [ ] **Step 1: Setup Workspace**
  - Create the directory `hpm_ai_v2/`.
  - Create the corresponding test directory `tests/hpm_ai_v2/`.
  - Add initialization files and basic typing definitions.

- [ ] **Step 2: Composite Pattern Representations (`patterns.py`)**
  - Implement `CompositePattern` class with constituent features $S$ and developmental level $L$.
  - Implement latent hierarchy $z^{(2)} \to z^{(1)} \to x_t$ with level-appropriate priors.
  - Implement Entropy ($H_q$) and Mutual Information for the $Comp_i(t)$ metric.

- [ ] **Step 3: Developmental & Institutional Evaluators (`evaluators.py`)**
  - Implement Epistemic, Affective, and Social evaluators.
  - Implement `InstitutionalField` with a "Replication Filter" logic (Step 9.7 in paper).
  - Implement `MaturationPenalty` logic for high-level patterns with low-level epistemic gaps.

- [ ] **Step 4: Meta Pattern Rule & Conflict Dynamics (`dynamics.py`)**
  - Implement `PatternPopulation` container for weights $w_i(t)$.
  - Implement `ConflictDetector`: Dynamic update of $\kappa_{ij}$ based on predictive divergence.
  - Implement the discrete-time replicator update equation with conflict inhibition.

- [ ] **Step 5: Structural Recombination & Insight (`recombination.py`)**
  - Implement merging of composite pattern constituents in $\mathcal{R}$.
  - Implement `ConstraintSatisfaction` $\mathcal{C}(h^*)$ based on structural coherence.
  - Add `InsightEvaluator` to boost affective reward and initial weight for novel, effective recombinations.

- [ ] **Step 6: Pattern Density & Substrate Shifting (`density.py`)**
  - Implement connectivity metric $C(h)$ based on feature co-occurrence.
  - Implement `DensityFunction` $D(h)$ and its contribution to the `StabilityBias`.
  - Implement `SubstrateTransition` logic (shifting density into efficient but rigid substrates).

- [ ] **Step 7: Verification & Milestone Testing**
  - **Milestone A**: Level 1 vs. Level 2 emergence (verify that Level 2 patterns require stabilized Level 1 base).
  - **Milestone B**: Superstition Persistence (verify that high-density, low-accuracy patterns are "sticky").
  - **Milestone C**: Scientific Refinement (verify that Institutional Fields accelerate convergence on accurate structure).
  - **Milestone D**: Recombinative Insight (verify that successful merges gain a temporary competitive advantage).
  - **Milestone E**: Abstract & Generative Levels (verify Level 4 immunity to surface noise and Level 5 generative utility).