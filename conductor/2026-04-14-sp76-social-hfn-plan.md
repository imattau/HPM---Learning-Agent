# SP76: Social Sharing of L4/L5 HFN Nodes

## Objective
Implement and verify the SP76 experiment, demonstrating zero-shot transfer of L4 (Forward Model) and L5 (Meta-Controller) knowledge between agents using a shared `TieredForest`.

## Scope
- Create `hpm_ai_v2/experiments/experiment_sp76_social_hfn.py`.
- Configure two agents (Agent A and Agent B) to use the same cold storage directories for their main forest, forward model (`l4`), and meta-controller (`l5`).
- Train Agent A on a set of tasks to populate the shared forests.
- Evaluate Agent B on L4 predictions, L5 rankings, and zero-shot problem solving using Agent A's learned nodes.

## Implementation Steps
1. **Create Shared Forests:** Define common `cold_dir` paths for the main forest, forward model, and meta-controller.
2. **Initialize Agent A (Expert):** Instantiate `BaseHFNAgent` pointing to the shared directories. Enable `use_hfn_forward_model` and `use_hfn_meta_controller`.
3. **Train Agent A:** Have Agent A solve standard tasks (e.g., `MAP_add1`, `MAP_mul2`) to generate L2 macros, L4 state transition deltas, and L5 strategy performance stats. Ensure `save_state()` is called to flush nodes to cold storage.
4. **Initialize Agent B (Novice):** Instantiate a second `BaseHFNAgent` using the exact same shared directories. It will load the `.npz` files populated by Agent A via the `TieredForest` indexing mechanism.
5. **Evaluate Agent B (L4):** Perform mental simulation (`predict`) on a primitive node and verify the predicted delta matches the true state transition error (< 0.1).
6. **Evaluate Agent B (L5):** Call `rank_strategies` and assert that the top strategy reflects Agent A's successful experience (e.g., `["decompose", "exact"]`).
7. **Zero-Shot Solve:** Have Agent B solve a new, related task (e.g., `MAP_add2`). Verify it succeeds and selects the optimal strategy (not `bfs`) using the shared knowledge base.

## Verification
- Run `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp76_social_hfn.py`.
- The script must complete successfully, passing all internal assertions for L4 prediction, L5 ranking, and zero-shot solving.
