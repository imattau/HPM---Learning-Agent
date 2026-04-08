# Experiment 29: Robust Intent Inference and AST Synthesis

## Objective
To redesign the Intent-Driven Reasoning pipeline to address critical architectural flaws: Oracle Leakage, lossy state representation, and brittle string-based rendering.

## Background: "Verifiable Structural Synthesis"
A review of Experiment 28 revealed that the system was "cheating" by using an oracle that provided the ground truth output. This experiment replaces that with a **Constraint Oracle**: the agent is given only partial semantic goals (e.g., "Expected Loop Count = 1", "Type = List"). It must also utilize an **ASTRenderer** to ensure that synthesized code is syntactically valid by construction.

## Setup
- **Domain**: Constraint-to-AST Space.
- **State Representation**: 14D extended semantic vector including structural markers (LoopCount, BranchCount, ASTDepth) and execution features (MeanVal, TypeSignature).
- **ASTRenderer**: Utilizes Python's `ast` module to map HFN nodes directly to abstract syntax tree nodes (e.g., `ast.For`, `ast.If`, `ast.AugAssign`).
- **Constraint Oracle**: Maps natural language prompts to 14D constraint vectors + bitmasks.
- **Planner**: Multi-objective DFS planner minimizing a composite loss of semantic distance, complexity, and prior confidence.

## Results
- **True Synthesis from Partial Intent**: The agent successfully synthesized a loop-based program for *"Double all numbers"* and a conditional program for *"Filter even numbers"* based only on structural and statistical constraints.
- **AST Robustness**: 100% of synthesized programs were syntactically valid and passed `ast.parse()` validation, a massive improvement over the previous string-concatenation model.
- **Knowledge Base Integration**: The agent successfully loaded its persistent knowledge base from the `curiosity` phase, utilizing learned priors to navigate the expanded 14D space.
- **Semantic Resolution**: The 14D state space proved sufficient to distinguish between different algorithmic structures (Map vs Filter) without collisions.

## Metrics Summary
| Metric | Baseline (Exp 28) | Redesign (Exp 29) | Status |
|---|---|---|---|
| Oracle Dependence | Full Output | Partial Constraints | **REFINED** |
| Rendering Safety | String (Brittle) | AST (Robust) | **STABLE** |
| State Dimensionality | 9D (Lossy) | 14D (Expressive) | **IMPROVED** |
| Syntactic Validity | 80% | 100% | **PASSED** |

## Verification Command
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_robust_intent.py
```
