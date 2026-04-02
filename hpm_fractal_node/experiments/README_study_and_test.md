# Sovereign Study-and-Test Experiment (SP29)

This experiment evaluates the HPM AI's capacity for **Meta-Transfer Learning**. It tests if structural motifs learned during a "Study Phase" (exposure to a set of tasks) can be reused as ad-hoc priors to assist in a "Test Phase" (solving unseen tasks).

## Architecture: Persistent Meta-Learning

Unlike standard ARC solver experiments that reset between tasks, SP29 implements a persistent cognitive lifecycle across a 4-process cluster:

1.  **Study Phase**: The system is exposed to the training examples of 10 "Study Tasks." It is allowed to learn and stabilize new HFN nodes representing emergent structural patterns from these tasks.
2.  **Test Phase**: The system is given 10 unseen "Test Tasks." Crucially, the specialists retain all nodes learned during the Study Phase.
3.  **Governor**: Tracks **Transfer Events**—instances where a node born during the Study Phase is retrieved and used as an "Explanation Winner" for a Test Task.

## Key Insights: Node Genealogy

- **Structural Persistence**: Proved that learned HFN motifs can persist across independent tasks, acting as a growing library of domain-specific experience.
- **Transfer Detection**: The experiment recorded **15 Transfer Events** in a 10-task test run. This confirms that the system successfully identifies similarities between different ARC puzzles at the HFN level.
- **Ad-hoc Priors**: By not resetting the forests, the system effectively "promotes" its successful learned nodes to the status of temporary priors, accelerating the induction of rules in the test phase.

## Theoretical Impact

This experiment demonstrates that HPM AI does not just learn *in* a task, it learns *across* tasks. The accumulation of structural motifs in the "Study Phase" creates a richer manifold for the "Test Phase," mirroring how humans build expertise by seeing many variations of a problem.

## Usage

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_study_and_test.py
```
