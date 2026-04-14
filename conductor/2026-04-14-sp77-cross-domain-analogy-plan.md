# SP77: Cross-Domain Analogy Using Shared L3 Schemas

## Objective
Demonstrate that a relational pattern (L3 meta-schema) learned in one domain (List) can be transferred to a different domain (Graph) via structural analogy. This enables zero-shot construction of new macros in the target domain without any prior training examples.

## Key Files & Context
- **Domain extensions**: `hpm_ai_v2/domains/graph_domain.py`, `hpm_ai_v2/domains/graph_renderer.py`
- **Analogy engine**: `hpm_ai_v2/agents/mixins/l3_analogy.py` (New file)
- **Agent updates**: `hpm_ai_v2/agents/social_agent.py` (Or similar base agent to integrate analogy mixin)
- **Experiment script**: `hpm_ai_v2/experiments/experiment_sp77_cross_domain_analogy.py` (New file)

## Proposed Strategy: Lightweight L3AnalogyMixin
Instead of porting the complex AST-based `DomainTransferBridge` from the v1 codebase, we will implement a direct structural mapping engine native to the HFN architecture (`TieredForest`).

The `L3AnalogyMixin` will introduce a `_try_l3_analogy` strategy that explicitly retrieves nodes with `relation_type="meta_schema"` from the shared forest. It will identify the substitutable "slots" in the schema (the generic or domain-specific inner loop operations) and structurally replace them with the corresponding operations in the target domain (e.g., mapping List's inner operation to Graph's `RELABEL_NODE`). 

## Implementation Plan

### 1. Extend the Graph Domain & Renderer
- **Domain Config**: Add new concepts `FOR_EACH_NODE` and `RELABEL_NODE` to `GraphDomainConfig` in `graph_domain.py`.
- **Renderer**: Update `GraphRenderer.render` in `graph_renderer.py` to generate python/NetworkX code for these concepts.
  - `FOR_EACH_NODE` maps to `for node in list(G.nodes()):`.
  - `RELABEL_NODE` modifies the specific node (e.g., using `nx.relabel_nodes`). Note that we need to handle iteration context (i.e. replacing the loop body). The renderer will be modified to support a basic loop structure or substitution mapping identical to the List domain's `FOR_LOOP`.

### 2. Implement `L3AnalogyMixin`
- **Create**: `hpm_ai_v2/agents/mixins/l3_analogy.py`.
- **Method `_try_l3_analogy(inputs, outputs)`**:
  - Retrieve all shared schemas (`relation_type="meta_schema"`) from the `TieredForest`.
  - For each schema, identify the structure (e.g. prefix inputs: `VAR_INP`, `LIST_INIT`, `FOR_LOOP`, `ITEM_ACCESS`).
  - Attempt to map the List domain scaffold components to Graph domain equivalents (e.g. `VAR_INP` -> `VAR_INP`, `FOR_LOOP` -> `FOR_EACH_NODE`, `ITEM_ACCESS` -> `RELABEL_NODE`).
  - Compose a new macro node natively in the `TieredForest` from the mapped primitives.
  - Render and execute the code to verify if it solves the target task.

### 3. Update Social Agent
- Import and inject `L3AnalogyMixin` into `SocialHFNAgent`.
- Add `"analogy"` to the agent's strategy registry and ensure the meta-controller can rank it.

### 4. Create Experiment Script `experiment_sp77_cross_domain_analogy.py`
- **Phase 1 (Expert - List Domain)**: Agent A is initialized with `ListDomainConfig`. It is trained on list map tasks (e.g. `MAP_add1`, `MAP_mul2`, `FILTER_pos`). It calls `discover_meta_schema()` to synthesize `meta_list_iteration`. The state is saved to the shared root.
- **Phase 2 (Novice - Graph Domain)**: Agent B is initialized with `GraphDomainConfig` and the shared root directory (L4/L5 HFN nodes).
- **Phase 3 (Zero-Shot Target)**: Agent B is given a novel Graph task: "increment the label of every node in a path graph". 
- **Verification**: Ensure Agent B solves the graph task zero-shot using the `bfs` or `analogy` strategy (which relies on `meta_list_iteration`) and successfully registers the structural transfer macro.

## Verification & Testing
- The experiment script will pass successfully with zero training examples for Agent B in the target Graph domain.
- `assert` that the final solution macro utilizes the L3 schema structure transferred from Agent A.
