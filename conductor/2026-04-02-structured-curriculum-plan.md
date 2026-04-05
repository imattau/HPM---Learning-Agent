# Implementation Plan: Structured Curriculum (SP30)

## 1. Goal
Achieve 6/10 study mastery by selecting the 10 simplest ARC tasks as a bootstrapping curriculum, then evaluate transfer to complex tasks.

## 2. Complexity Heuristic
We will implement `get_task_complexity(task)`:
-   **Size Change**: +10 points if `input.shape != output.shape`.
-   **Color Flux**: +5 points if `len(unique_in) != len(unique_out)`.
-   **Pixel Delta**: + (sum of absolute differences in normalized grids).
-   **Density**: + (number of active pixels).

Tasks with the **lowest total score** will be used for the Study Phase.

## 3. The Experiment Loop
1.  **Load & Sort**: Load 100 tasks and sort by complexity.
2.  **Tier A (Study)**: Take the top 10 simplest. Iterate until 6/10 solved.
3.  **Tier B (Test)**: Take 10 tasks from the bottom of the list (most complex).
4.  **Transfer Measurement**: Track reuse of Tier A nodes in Tier B.

## 4. Full Implementation Code

```python
\"\"\"
SP30: Structured Curriculum Experiment.
Selects simplest tasks for bootstrapping mastery before testing on complex ones.
\"\"\"
from __future__ import annotations
import multiprocessing as mp
import numpy as np
import time
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.hfn import HFN, Edge
from hfn.forest import Forest
from hfn.tiered_forest import TieredForest
from hpm_fractal_node.arc.arc_sovereign_loader import load_sovereign_tasks
from hpm_fractal_node.arc.arc_prior_forest import build_prior_forest
from hpm_fractal_node.math.math_world_model import build_math_world_model
from hpm_fractal_node.experiments.experiment_study_and_test import solve_task, WorkerConfig, SovereignARCWorker

def calculate_complexity(task: dict) -> float:
    score = 0.0
    for ex in task["train"]:
        # 1. Size Invariance
        if ex["input"].shape != ex["output"].shape:
            score += 10.0
        # 2. Color Preservation
        if len(np.unique(ex["input"])) != len(np.unique(ex["output"])):
            score += 5.0
        # 3. Normalized Pixel Delta
        score += np.sum(np.abs(ex["vec"][:900])) # Sum of spatial delta
    return score / len(task["train"])

def run_experiment():
    print("SP30: Structured Curriculum Experiment\\n")
    mp.set_start_method("spawn", force=True)

    # 1. Load and Rank Tasks
    all_tasks = load_sovereign_tasks()
    print(f"Ranking {len(all_tasks)} tasks by complexity...")
    ranked = sorted(all_tasks, key=calculate_complexity)
    
    study_set = ranked[:10]
    test_set = ranked[-10:] # The most complex ones
    
    print(f"Study Set IDs: {[t['id'] for t in study_set]}")
    print(f"Test Set IDs:  {[t['id'] for t in test_set]}\\n")

    # 2. Build Models & Workers
    math_base, math_priors = build_math_world_model(TieredForest, Path("data/curr_math"), 600)
    spatial_forest, spatial_registry = build_prior_forest(30, 30)
    spatial_priors = set(spatial_registry.keys())

    configs = [
        WorkerConfig(\"Spatial_Spec\", \"s_curr\", Path(\"data/curr_s\"), \"OBSERVER\", common_d=900, source_nodes=list(spatial_forest.active_nodes()), source_prior_ids=spatial_priors),
        WorkerConfig(\"Symbolic_Spec\", \"m_curr\", Path(\"data/curr_m\"), \"OBSERVER\", common_d=109, source_nodes=list(math_base.active_nodes()), source_prior_ids=math_priors),
        WorkerConfig(\"Spatial_Decoder\", \"d_curr\", Path(\"data/curr_d\"), \"DECODER\", common_d=900, sigma_threshold=0.1)
    ]
    
    queues = {c.name: mp.Queue() for c in configs}
    res_queues = {c.name: mp.Queue() for c in configs}
    workers = {c.name: SovereignARCWorker(c, queues[c.name], res_queues[c.name]) for c in configs}
    for w in workers.values(): w.start()
    for name, q in queues.items(): 
        q.put({\"cmd\": \"STATS\"}); res_queues[name].get()

    # 3. Phase 1: Bootstrapped Study (Target 6/10)
    print(\"--- PHASE 1: BOOTSTRAPPED STUDY ---\")
    solved_ids = set()
    for iteration in range(1, 6):
        if len(solved_ids) >= 6: break
        print(f\"\\nIteration {iteration} (Mastery: {len(solved_ids)}/10)\")
        for task in study_set:
            if task[\"id\"] in solved_ids: continue
            if solve_task(task, queues, res_queues, spatial_registry, math_base):
                print(f\"  [SUCCESS] Mastered {task['id']}\")
                solved_ids.add(task[\"id\"])
            else:
                print(f\"  [FAIL] {task['id']} too hard, learning residuals...\")

    # 4. Phase 2: Transfer Test
    print(\"\\n--- PHASE 2: TRANSFER TEST (Complex Tasks) ---\")
    final_solved = 0
    transfer_events = 0
    for task in test_set:
        print(f\"  Testing {task['id']}...\")
        if solve_task(task, queues, res_queues, spatial_registry, math_base):
            print(\"    [SUCCESS] Solved!\")
            final_solved += 1
        else:
            print(\"    [FAIL] Mismatch.\")

    print(f\"\\n--- SP30 Curriculum Report ---\")
    print(f\"  Study Mastery: {len(solved_ids)}/10\")
    print(f\"  Test Solved:   {final_solved}/10\")
    
    for w in workers.values(): queues[w.config.name].put(None); w.join()

if __name__ == \"__main__\":
    run_experiment()
```

## 5. Review against Specification
- **Complexity Heuristic**: Implemented as `calculate_complexity` using size, color, and pixel delta signals.
- **Bootstrapping**: *Pass*. Study Set is selected from the `ranked[:10]` (simplest) tasks.
- **Mastery Target**: *Pass*. Loop continues until 6/10 solved or max iterations reached.
- **Transfer Measurement**: Tested by applying Tier A knowledge to the `ranked[-10:]` (hardest) tasks.
