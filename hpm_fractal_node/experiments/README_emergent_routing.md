# Emergent Routing Experiment (SP26)

This experiment demonstrates **Decentralized Sovereignty** in the HPM AI architecture. It refactors the multi-process cluster to remove the central Governor routing table, proving that specialized processes can autonomously filter and claim relevant observations using their internal world models.

## Architecture: The Competence Gate

Unlike previous experiments where the Governor decided which core processed which data, SP26 implements a **Broadcast & Claim** model:

1.  **Broadcaster (Governor)**: The central process acts as a simple transmitter, broadcasting all raw observations ($x$) to every active worker process simultaneously.
2.  **Sovereign Worker**: Each specialist (e.g., `Spec_Signal`) implements a high-speed **Competence Gate** in its process loop:
    *   It uses `Evaluator.accuracy(x, self.forest)` to calculate how well the incoming data fits its *global domain identity*.
    *   If the accuracy is below a `competence_threshold`, the worker remains dormant and returns a negative response.
    *   If the data fits its domain, it triggers the expensive `Observer.observe(x)` expansion.

## The Experiment: Autonomous Signal Filtering

The system starts with a single **Explorer** (Generalist). As it matures, it discovers a dense cluster and promotes it to a new process (**Spec_Signal**). 

In the execution phase, the Governor broadcasts a mix of structured signal and uniform noise. 
*   **Result**: The `Spec_Signal` specialist autonomously claimed ~60% of the broadcast (matching the signal frequency) and ignored the noise, while the `Explorer` continued to monitor the entire space.

## Key Insights

- **Elimination of the Bottleneck**: Removing the central routing table allows the system to scale to a large number of domains (e.g., the "Sovereign Octet") without increasing the Governor's complexity.
- **Self-Organizing Sovereignty**: Specialists define their own boundaries. A "Math Specialist" only wakes up when it "hears" math, based entirely on its own internal HFN topography.
- **Natural Stereo Vision**: If multiple specialists claim the same observation, the Governor receives multiple explanations. This overlap is the functional basis for analogy and multi-perspective reasoning.

## Usage

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_emergent_routing.py
```
