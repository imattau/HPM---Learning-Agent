# Corrected Refactor Plan (Generic, Macro‑Agnostic)

## Core Principles

- **L4 delta node**: references the pattern node (any level) it predicts via `inputs`.
- **Solve record node**: references the pattern node used to solve the task (any level).  
  `relation_type = "solve_record"` (not L5).
- **L5 aggregate meta‑node**: references a collection of solve record nodes (or directly the pattern nodes) and stores aggregated stats.  
  `relation_type = "meta_pattern"`.
- **Core HFN**: gains a generic `aggregate` factory (no interpretation).

---

## Phase 1: Core HFN – Generic Aggregate Node Factory

**File:** `hfn/recombination.py`

```python
def aggregate(self, nodes: List[HFN], agg_func: Callable[[List[np.ndarray]], np.ndarray],
              new_id: str, relation_type: str = "aggregate") -> HFN:
    """Create a node whose inputs are `nodes` and mu = agg_func([n.mu for n in nodes])."""
    mus = [n.mu for n in nodes]
    new_mu = agg_func(mus)
    new_sigma = np.ones_like(new_mu)
    node = HFN(mu=new_mu, sigma=new_sigma, id=new_id, use_diag=True)
    node.inputs = nodes
    node.relation_type = relation_type
    return node
```

**No other core changes.**

---

## Phase 2: L4 – Delta Nodes (Fractal)

**File:** `hpm_ai_v2/agents/mixins/l4_forward.py` (or `hfn/transition.py` if moved)

- When recording a transition for any pattern node `pnode` (L1 primitive, L2 macro, L3 schema, etc.):
  - Create (or update) a delta node with:
    - `id = f"delta:{pnode.id}"`
    - `inputs = [pnode]`
    - `mu = delta_vector` (running average)
    - `relation_type = "transition"`
- Use `Recombination.aggregate` only if multiple deltas are averaged; otherwise simple node creation.

**Pseudo‑code:**

```python
def record_delta(self, pnode: HFN, delta: np.ndarray):
    delta_id = f"delta:{pnode.id}"
    existing = self.delta_forest.get(delta_id)
    if existing is None:
        agg_func = lambda mus: delta   # initial
        new_node = self.recombination.aggregate([pnode], agg_func, delta_id, "transition")
        self.delta_forest.register(new_node)
    else:
        # EMA update of existing.mu, keep inputs unchanged
        existing.mu = 0.9 * existing.mu + 0.1 * delta
```

---

## Phase 3: L5 – Solve Records and Aggregate Meta‑Nodes

**File:** `hpm_ai_v2/agents/mixins/l5_meta.py` (or new file)

### 3.1 Solve Record Node

When a solve attempt finishes (success or failure), create a node:

```python
def create_solve_record(self, context, strategy, pattern_used: HFN, success: bool, oracle_calls: int, depth: int):
    node_id = f"solve:{context}:{strategy}:{int(time.time())}"
    mu = np.array([1.0 if success else 0.0, float(oracle_calls), float(depth), time.time()])
    node = HFN(mu=mu, sigma=np.ones(4), id=node_id, use_diag=True)
    node.inputs = [pattern_used]   # reference the pattern that solved it
    node.relation_type = "solve_record"
    self.solve_forest.register(node)
    return node
```

### 3.2 Aggregate Meta‑Node (L5)

For each context (goal_type, n_macros_bucket, strategy), maintain an aggregate node:

```python
def update_meta_node(self, context, strategy, new_solve_record_node):
    agg_id = f"meta:{context}:{strategy}"
    existing = self.meta_forest.get(agg_id)
    # Collect all solve record nodes for this context (from forest query or stored list)
    solve_nodes = self._get_solve_records(context, strategy) + [new_solve_record_node]
    # Keep only last N (e.g., 100)
    solve_nodes = solve_nodes[-self.max_history:]
    # Compute aggregated mu: success rate, avg oracle calls, count, last timestamp
    successes = sum(1 for n in solve_nodes if n.mu[0] > 0.5)
    total_calls = sum(n.mu[1] for n in solve_nodes)
    new_mu = np.array([successes/len(solve_nodes), total_calls/len(solve_nodes), len(solve_nodes), time.time()])
    # Replace existing aggregate node
    if existing:
        self.meta_forest.deregister(agg_id)
    new_agg = self.recombination.aggregate(solve_nodes, lambda mus: new_mu, agg_id, "meta_pattern")
    self.meta_forest.register(new_agg)
```

### 3.3 Ranking Strategies

`rank_strategies` reads the `mu` of the aggregate meta‑node (cached) – no traversal needed for speed.

---

## Phase 4: Backward Compatibility

- Keep old dict‑based `MetaStrategyController` as fallback.
- New fractal L5 is opt‑in via `use_fractal_meta=True` in agent constructor.

---

## Summary of Changes

| File | Change |
|------|--------|
| `hfn/recombination.py` | Add `aggregate` method. |
| `hpm_ai_v2/agents/mixins/l4_forward.py` | Delta nodes now store `inputs = [pnode]`. |
| `hpm_ai_v2/agents/mixins/l5_meta.py` | New file with solve record nodes + aggregate meta‑nodes. |
| `hpm_ai_v2/agents/base_agent.py` | In `solve`, call `meta.record(rec, pattern_used)`. |
| `hpm_ai_v2/agents/agents.py` | Add `use_fractal_meta` flag. |