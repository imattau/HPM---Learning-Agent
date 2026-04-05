# Session State — Complete 3 HFN Fixes (2 Remaining)

execution_mode: unattended
auto_continue: true

## Status
✅ Fix 1: observer.py _expand() (lines 324-382) — DONE
⏳ Fix 2: evaluator.py density_ratio() (lines 83-125) — IN PROGRESS
⏳ Fix 3: decoder.py _score_topological_fit() (lines 109-130) — PENDING

## Current Progress
- Read evaluator.py lines 1-130 to identify current density_ratio() method
- Current method at lines 84-126 uses mean nearest-neighbour distance approach
- Need to replace with k-neighbor shell-based approach from spec below

## Fix 2: evaluator.py density_ratio() (lines 83-125)
REPLACE entire method (lines 84-126) with:

```python
def density_ratio(
    self,
    x: np.ndarray,
    nodes: Sequence[HFN],
    radius: float,
    k: int = 5,
) -> float:
    """Local vs shell density ratio (k→2k neighbors). Scale-invariant, region-aware."""
    node_list = list(nodes)
    if len(node_list) < 3:
        return 0.0
    x = np.asarray(x, dtype=float)
    mus = np.array([n.mu for n in node_list], dtype=float)
    dists_to_x = np.sort(np.linalg.norm(mus - x, axis=1))
    
    k_actual = min(k, len(dists_to_x) - 1)
    k2_actual = min(k * 2, len(dists_to_x) - 1)
    d_k = dists_to_x[k_actual] + 1e-9
    d_2k = dists_to_x[k2_actual] + 1e-9
    
    local_count = float(np.sum(dists_to_x < radius))
    local_density = local_count / (radius ** 2 + 1e-9)
    
    shell_count = float(np.sum((dists_to_x >= d_k) & (dists_to_x < d_2k)))
    shell_density = (shell_count + 1e-9) / ((d_2k ** 2 - d_k ** 2) + 1e-9)
    
    return local_density / (shell_density + 1e-9)
```

## Fix 3: decoder.py _score_topological_fit() (lines 109-130)
First add to Decoder class (above __init__):
```python
_RELATION_WEIGHTS: dict[str, float] = {
    "MUST_SATISFY": 3.0,
    "PART_OF": 2.0,
    "spatial": 1.5,
    "temporal": 1.5,
    "recombined": 0.5,
}
_DEFAULT_RELATION_WEIGHT = 1.0
```

Then REPLACE method with:
```python
def _score_topological_fit(self, abstract_node: HFN, concrete_node: HFN) -> float:
    """Score topological fit with relation-weighted edges."""
    abstract_edges = abstract_node.edges()
    if not abstract_edges:
        return 0.0
    
    concrete_edge_map: dict[str, set[str]] = {}
    for e in concrete_node.edges():
        concrete_edge_map.setdefault(e.target.id, set()).add(e.relation)
    
    score = 0.0
    for e in abstract_edges:
        w = self._RELATION_WEIGHTS.get(e.relation, self._DEFAULT_RELATION_WEIGHT)
        if e.target.id in concrete_edge_map:
            if e.relation in concrete_edge_map[e.target.id]:
                score += w
            else:
                score += w * 0.5
        else:
            score -= w * 0.5
    return score
```

## Next Steps (UNATTENDED - DO NOT PAUSE)
1. Replace evaluator.py density_ratio() method
2. Test: python3 -m pytest hfn/tests/ -v (expect 35/35 pass)
3. Replace decoder.py _score_topological_fit() method
4. Test again: 35/35 pass
5. Commit: "Fix HFN scaling & correctness: expansion memoization, local density, relation-aware topology"
6. Push to origin master
7. Report: all tests pass, commit hash, push successful

## Files
- /home/mattthomson/workspace/HPM---Learning-Agent/hfn/observer.py (✅ DONE)
- /home/mattthomson/workspace/HPM---Learning-Agent/hfn/evaluator.py (⏳ NEXT)
- /home/mattthomson/workspace/HPM---Learning-Agent/hfn/decoder.py (⏳ AFTER)
