# Experiment 27: Autonomous Curriculum Generation (Self-Play & Curiosity)

## Objective
To validate the HFN architecture's capacity for **Unsupervised Algorithmic Discovery**. The agent must transition from reactive problem-solving (via human curricula) to proactive world-model building by autonomously discovering novel state transformations.

## Background: "Bootstrapping Knowledge"
Real AGI cannot rely on human-provided task files. This experiment introduces a **Self-Play Loop**: the agent randomly composes its structural priors, executes them in a robust Python sandbox, and uses an **Inverse State Oracle** to map the resulting input/output transformations into formal 9D semantic deltas.

## Setup
- **Domain**: Unbounded Semantic Program Space.
- **State Representation**: 9D semantic vector (Value, Returned, Len, TargetVal, Iterator, Init, Condition, CallStack, TemplateSlot).
- **Generative Sampling**: Agent randomly folds 5-8 HFN nodes into complex program trees.
- **Hardened Sandbox**: Multi-threaded `PythonExecutor` with strict timeouts to handle infinite loops and crashes during exploration.
- **Persistence**: Knowledge is saved to the `data/knowledge_base/` using the mandated **TieredForest** and **PersistenceManager** utilities.

## Results
- **Unsupervised Learning**: The agent successfully entered an "unsupervised play" phase and discovered novel algorithmic transformations without any human-provided goal vectors.
- **Task Formalization**: Discovered functions were automatically cataloged into `auto_curriculum.json`, complete with example inputs, outputs, and semantic state deltas.
- **Structural Persistence**: All discovered program graphs were registered in the **TieredForest** cold storage, ensuring they are available as "priors" for future, more complex tasks.
- **Yield & Diversity**: Within 200 cycles, the agent discovered multiple unique program behaviors, proving that the HFN latent space can be autonomously mapped through curiosity-driven exploration.

## Metrics Summary
| Metric | Result |
|---|---|
| Unsupervised Cycles | 200 |
| Novel Discoveries | 3 |
| Sandbox Exceptions Caught | ~190 (Timeouts, TypeErrors) |
| Persistence Mode | Tiered (Hot/Cold) |

## Verification Command
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_autonomous_curriculum.py
```
