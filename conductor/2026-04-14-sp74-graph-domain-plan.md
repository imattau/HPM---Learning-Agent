# SP74: Graph Domain Few-Shot Learning Plan

## Objective
Demonstrate that HPM can learn a graph transformation (adding a new node and connecting it to all existing nodes - a "star" extension) from a single example and generalize to a novel graph. This introduces non-Euclidean structured data capabilities.

## Scope & Impact
- **New Domain:** Adds graph structure processing capabilities to `hpm_ai_v2`.
- **Dependency:** Introduces `networkx` for graph manipulation.
- **Components:** `GraphDomainConfig`, `GraphRenderer`, and `GraphOracle`.
- **Constraint:** Zero modifications to `BaseHFNAgent` or other generic utilities. The agent is entirely domain-agnostic.

## Proposed Solution
Following the domain-agnostic architecture:

### 1. `hpm_ai_v2/domains/graph_domain.py`
- Create `GraphDomainConfig(DomainConfig)` with graph-specific concepts (`ADD_NODE`, `ADD_EDGE`, `REMOVE_NODE`, `REMOVE_EDGE`, `CLEAR_GRAPH`, `COPY_GRAPH`).
- Create `get_graph_primitive_nodes(config)`.

### 2. `hpm_ai_v2/domains/graph_renderer.py`
- Create `GraphRenderer(Renderer)` to translate concepts into executable `networkx` code.
- Implement AST/string generation for operations like adding a node and connecting to all existing nodes.

### 3. `hpm_ai_v2/utils/oracle/graph_oracle.py`
- Add `GraphOracle(BaseOracle)` to extract empirical state (node count, edge count, average degree) from `networkx` graphs.
- Add it to `hpm_ai_v2/utils/oracle/__init__.py` for easy importing.

### 4. `hpm_ai_v2/experiments/experiment_sp74_graph_fewshot.py`
- Implement the experiment script validating the one-shot learning of the "add-star" macro.
- Phase 1: Train on a chain graph (0-1-2).
- Phase 2: Test generalization on a triangle graph.
- Note: To ensure `BaseHFNAgent._check_outputs` (which relies on `==`) works correctly without modification, we may need to monkey-patch `nx.Graph.__eq__` within the experiment script or rely on exact object state if feasible.

## Implementation Steps
- [ ] Add `networkx>=3.0` to `requirements.txt` and run `uv pip install`.
- [ ] Implement `graph_domain.py`.
- [ ] Implement `graph_renderer.py`.
- [ ] Implement `graph_oracle.py` and update oracle `__init__.py`.
- [ ] Create and run `experiment_sp74_graph_fewshot.py`.
- [ ] Verify success.

## Verification
- Run `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp74_graph_fewshot.py`.
- Assert successful one-shot learning and generalization with macro reuse.
