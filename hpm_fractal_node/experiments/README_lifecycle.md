# Multi-Observer Lifecycle Experiment (Multi-Process)

This experiment evaluates the lifecycle and orchestration of multiple HFN observers in a parallelized, multi-process environment. It compares three different architectural strategies for managing cross-domain knowledge (Math and Text) across a staged workload.

## Overview

The experiment uses a **Common Latent Space** (216D) where specific slices are allocated to different domains:
- **Math Slice**: Dimensions 0–108 (109D)
- **Text Slice**: Dimensions 109–215 (107D)

By embedding domain-specific observations into this shared space, we can test how a system handles "pure" observations (only one slice active) and "mixed" observations (both slices active, representing a cross-domain task).

## Strategies Compared

1.  **Monolithic (Baseline)**: A single, large HFN observer that attempts to model all domains in one forest. It stays "hot" throughout the entire experiment.
2.  **Specialists**: Separate dedicated processes for Math and Text. Each models only its slice of the world. In mixed stages, both processes observe the same input, each explaining the part it understands.
3.  **Specialists + Mixed**: Same as Specialists, but with an additional **on-demand Mixed Worker**. When a mixed stage is detected, a new process is spawned, and its state is initialized by "cloning" the active nodes from the specialists.

## Multi-Process Architecture

To achieve true multi-core utilization for the CPU-bound HFN observation logic, the experiment uses a **Controller-Worker** model:

- **HFNWorker**: Each observer (Math, Text, Mixed) runs in its own dedicated Python process. It maintains its own `TieredForest` (RAM/Disk tiered storage) and `Observer` instance.
- **MultiProcessController**: Orchestrates the experiment. It routes batches of observations to the relevant workers via `multiprocessing.Queue`.
- **Batching**: Observations are sent in batches (default size 20) to minimize Inter-Process Communication (IPC) overhead.

## Experiment Stages

The experiment runs through a 6-stage lifecycle:
1.  `math_seed`: Initial math training.
2.  `text_seed`: Initial text training.
3.  `math_text_mix`: First exposure to combined math+text observations.
4.  `math_revisit`: Return to pure math to check for retention/interference.
5.  `text_revisit`: Return to pure text.
6.  `math_text_mix_revisit`: Final test of cross-domain explanation.

## Key Insights

- **True Parallelism**: By using separate processes, the system bypasses the Python Global Interpreter Lock (GIL), allowing each observer to utilize a full CPU core simultaneously during mixed stages.
- **100% Coverage**: The parallel "Specialists" approach maintains 100% explanation coverage. The system effectively "tiles" the common latent space, with each specialist claiming the dimensions it understands.
- **Memory Isolation**: Offloading the `TieredForest` to worker processes keeps the main controller process lean (low RSS), allowing for very large total world models (1000+ nodes) distributed across the system.
- **State Transfer**: The experiment proves that HFN state (nodes and edges) can be successfully transferred between processes to spawn "Mixed" specialists on-demand, enabling dynamic architectural scaling.

## Issues & Bottlenecks

- **IPC Overhead**: The primary performance bottleneck is the cost of serializing (Pickling) 216D vectors and HFN node objects across process boundaries. For small batches, this overhead can make the multi-process version slower than a serial monolithic version.
- **Mixed Worker Initialization**: Cloning nodes from multiple specialists into a new mixed worker requires a full forest sync, which is a high-latency event.
- **Coordination Complexity**: Managing the lifecycle (start, stats, stop) of multiple processes adds architectural complexity compared to a single-threaded loop.

## Usage

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_multi_observer_lifecycle.py
```
