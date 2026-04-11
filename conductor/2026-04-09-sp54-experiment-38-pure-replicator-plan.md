# SP54: Experiment 38 — Pure Replicator Compression and Schema Transfer

## Objective
Address the final impurities in the HPM planning layer to achieve a perfectly un-biased self-organizing search process over structured programs, and validate schema learning and transfer across distinct algorithmic tasks.

## Background & Rationale
1. **Hard Threshold in Compression:** `if advantage > 5.0` is a fixed heuristic gate. It should be replaced with relative tracking—allowing the `Observer` to natively manage compression thresholds based on co-occurrence counts and relative utility.
2. **True Replicator Scaling:** The `sigmoid(adv / 3)` function compresses dynamic range. We will switch to `weight = np.exp(advantage / 5.0)` to create sharper competition and better separation of strong vs weak patterns, closer to true replicator equations.
3. **Schema Transfer Validation:** We will test if the system can cleanly reuse a compressed MAP schema in "Map double" without rediscovering the loop, and whether it can compose the MAP schema with a novel CONDITION in "Filter positive".

## Implementation Steps
- [ ] Update `experiment_schema_transfer.py` weight scaling to use `np.exp(advantage / tau)`.
- [ ] Relax the hard `advantage > 5.0` compression gate to rely on relative advantage and the global Observer's native `compression_cooccurrence_threshold`.
- [ ] Verify Task C (Map double) converges in < 1/10th the iterations of Task B.
- [ ] Verify Task D (Filter positive) correctly inserts the condition inside the reused loop schema.