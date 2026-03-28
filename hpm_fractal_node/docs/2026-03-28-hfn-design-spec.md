# HPM Fractal Node (HFN) — Design Specification

**Date:** 2026-03-28
**Status:** Draft for review

---

## 1. Purpose and Context

The HPM Fractal Node (HFN) is a data structure that represents a single pattern in the HPM framework. It is the core unit from which all hierarchical pattern representations are built.

This specification defines the interface, invariants, and dynamics of HFN nodes, together with two supporting components: the **Observer** (which uses nodes to explain an incoming signal) and the **Forest** (which maintains the population of nodes and drives learning through competitive dynamics).

This specification does not define encoder architecture, proximity search data structures, how the attention budget is set, or evaluator signal sources. Those are out of scope.

---

## 2. Core Concept

Every HFN node has two faces:

**Compressed face** — a multivariate Gaussian N(μ, Σ) in a shared D-dimensional latent space Z. This is the node's identity: it represents what the pattern looks like as a probability distribution over signals.

**Structural face** — a DAG of child HFN nodes with typed edges. This is the node's internal composition: how the pattern is built from sub-patterns.

Both faces are always present. The Gaussian is the node's compressed summary. The DAG is its polygraph body. Together they satisfy the three HPM pattern requirements:

- **Internal coherence**: the DAG (polygraph) defines structural coherence among children.
- **Functional utility**: the Gaussian supports prediction — it can assign probability to signals.
- **Evaluator reinforcement**: tracked externally by the Observer and Forest, not by the node itself.

---

## 3. Levels Are Observer-Relative

HFN does not have a fixed level number. What counts as "level N" depends on the Observer's current position in the expansion tree. A node is a root from one Observer's perspective and a leaf from another's simultaneously.

This is a design invariant: **no node stores its own level**. The same node can be a child of multiple parent nodes. Hierarchy is a property of a particular traversal, not a property of nodes.

---

## 4. HFN Node Interface

### 4.1 Identity operations

```python
def log_prob(x: ndarray) -> float
```
Returns the log-probability of signal `x` under the node's Gaussian N(μ, Σ). Scalar surprise: higher value means the node finds `x` more probable (less surprising). Used by the Observer to measure how well this node explains the current signal.

```python
def overlap(other: HFN) -> float
```
Returns the Gaussian overlap integral ∫ N(μ_self, Σ_self) · N(μ_other, Σ_other) dz. This is the competition coefficient κ between two nodes. Used by the Forest to determine competitive pressure.

```python
def description_length() -> float
```
Returns the complexity of this node. Used in the scoring function S = accuracy − λ · complexity, where λ is a compression pressure constant set by the Forest. Simpler nodes are preferred when accuracy is equal.

### 4.2 Structure operations (read-only)

```python
def children() -> list[HFN]
```
Returns the immediate child nodes in this node's DAG. Returns an empty list for leaf nodes.

```python
def edges() -> list[Edge]
```
Returns the typed edges between this node's children. Each edge is a triple (source: HFN, target: HFN, relation: str). Edge types are domain-defined strings (e.g. "next", "part-of", "causes").

```python
def expand(depth: int) -> HFN
```
Returns the sub-tree rooted at this node to the given depth. At depth 0, or if the node is a leaf, returns the node itself. This is the fractal property: `expand` has the same return type at every depth, and the returned sub-tree exposes the same interface.

### 4.3 Recombination

```python
def recombine(other: HFN) -> HFN
```
Returns a new HFN node with `self` and `other` as children. The new node's Gaussian is derived from the combined structure (e.g. parameters fit to the union of the children's distributions). No existing node is mutated. The new node starts as a candidate — the Forest decides whether to register it.

---

## 5. HFN Node Invariants

1. **No parent references.** A node has no knowledge of which nodes contain it as a child. Parent structure is maintained solely by the containing nodes.

2. **No mutation from queries.** Calling `log_prob`, `overlap`, `description_length`, `children`, `edges`, or `expand` never alters any node's state.

3. **Shared child references.** The same node object can be a child of multiple parent nodes simultaneously. This is not an error; it is how reuse is expressed.

4. **Fractal uniformity.** Every node exposes the same interface regardless of depth. There is no "leaf interface" or "root interface".

5. **Gaussian required.** Every node, including leaves, has a Gaussian N(μ, Σ). Leaves have no children; their Gaussian is their sole identity.

---

## 6. Observer Interface

The Observer is responsible for explaining a single incoming signal `x`. It is stateless between signals.

### 6.1 State during one explanation pass

- `x`: the current signal in Z (D-dimensional vector)
- `B`: the attention budget (integer, decremented each time a node is expanded)
- `frontier`: the set of nodes whose children have not yet been examined

### 6.2 Expansion decision

The Observer expands a frontier node `h` only when **both** conditions hold:

1. **Surprise condition:** D_KL(P(x) ‖ N(μ_h, Σ_h)) ≥ τ

   The node's Gaussian does not adequately explain `x`. τ is the surprise threshold (a scalar constant set externally).

2. **Utility condition:** node `h` has the highest expected surprise reduction Δ(h) among all current frontier nodes.

   Δ(h) estimates how much expanding `h` is likely to reduce residual surprise, based on the distribution of its children's Gaussians.

When a node passes both conditions, the Observer calls `expand(1)` on it, adds the returned children to the frontier, and decrements B by 1.

### 6.3 Expansion loop

```
initialise frontier with candidate root nodes (provided by Forest)
while True:
    compute surprise for all frontier nodes against x
    if no frontier node exceeds τ: stop (signal is explained)
    if B == 0: stop (budget exhausted)
    select h = argmax Δ(h) over frontier nodes exceeding τ
    new_children = h.expand(1).children()
    frontier = (frontier \ {h}) ∪ new_children
    B -= 1
```

### 6.4 Observer outputs

- **Explanation tree**: the set of nodes and edges visited during expansion. Read-only; the Observer does not modify nodes.
- **Residual surprise**: the aggregate unexplained surprise remaining after the loop terminates. Passed to the Forest as a signal that new nodes may be needed.

---

## 7. Forest Interface

The Forest maintains the population of all active HFN nodes. It is the component responsible for learning dynamics.

### 7.1 State

- Registry of all active nodes, indexed by Gaussian identity (μ, Σ)
- Weight w_h ∈ [0, 1] for each node h, representing accumulated predictive success
- Score S_h for each node h, updated after each query: S_h = accuracy_h − λ · h.description_length()

### 7.2 Responsibilities

**Retrieval.** Given the current signal `x`, the Forest performs a proximity search in Z and returns the k nearest nodes (by Gaussian mean) as candidate roots for the Observer. The Forest reads only Gaussian identities for this purpose; it does not push data into nodes.

**Competitive weight update.** After each Observer query, nodes that appeared in the explanation tree gain weight proportional to their contribution to reducing surprise. Nodes that did not participate lose weight proportional to their overlap with the nodes that did:

```
for each active node h:
    if h was in explanation tree:
        w_h += α · (1 - w_h) · accuracy_h
    else:
        for each explaining node j:
            w_h -= β · κ_hj · w_h
        where κ_hj = h.overlap(j)
```

This is a discrete per-query update, not a continuous ODE. α and β are learning rate constants.

**Structural absorption.** When node `j` has persistently lost weight to node `h` over the last N queries, **and** κ_hj exceeds threshold α_struct, the Forest absorbs `j` as a child of `h`:

- A new parent node is created via `h.recombine(j)` or by directly adding `j` to `h`'s child list (depending on implementation).
- `j` is not destroyed; it becomes a structural component of `h`.
- This is how competition resolves into hierarchy: weaker, highly overlapping nodes become sub-patterns of stronger nodes.

**Node creation — residual surprise.** When residual surprise (returned by Observer) exceeds a threshold, the Forest spawns a new leaf node. The new node's Gaussian is initialised from the unexplained component of `x`. It starts with low weight w = w_init and is registered in the Forest.

**Node creation — query-induced compression.** The Forest tracks which sub-structures recur across explanation trees. When a set of nodes [h₁, h₂, ... hₙ] consistently co-appear in the same expansion path across M consecutive queries, the Forest creates a new node whose children are [h₁ ... hₙ] and whose Gaussian is derived from their combined structure. This node represents the compressed form of the recurring relationship. It enters the Forest as a candidate with low initial weight w = w_init and must earn stability through continued predictive accuracy. This is the primary mechanism by which new abstractions form: not from unexplained signals alone, but from structural patterns discovered through repeated querying.

### 7.3 Forest invariant

The Forest never writes to or mutates any HFN node. It reads Gaussian identities via `log_prob`, `overlap`, and `description_length` to compute weights and scores. Structural changes (creating new parent nodes, adding children) produce new node objects; they do not alter existing ones.

---

## 8. Cross-Cutting Invariants

| Invariant | Statement |
|---|---|
| Structural ignorance | A node knows only its children. No node knows its parents, its Forest weight, or its level. |
| Immutability under query | Any call to any node method returns a value; it never changes any node's fields. |
| Fractal uniformity | The HFN interface is identical at every depth. Depth is a property of traversal. |
| Compression as default | The scoring function S = accuracy − λ · complexity means simpler nodes are preferred when predictive power is equal. |
| Pattern density as emergent stability | Nodes persist by earning accurate predictions repeatedly. Weight is not assigned; it is accumulated. |

---

## 9. HPM Alignment

| HPM Concept | HFN Implementation |
|---|---|
| Pattern substrate | The HFN node — both Gaussian and DAG structure |
| Pattern dynamics | Forest competitive updates, structural absorption, node creation |
| Pattern evaluator / gatekeeper | Observer surprise threshold τ; Forest score S_h with compression penalty |
| Pattern field | The Forest's active population and the proximity structure of Z |
| Hierarchy of abstraction levels | Depth-first expansion by Observer; absorption resolves competition into hierarchy |
| Innate priors | Initial Gaussian parameters; w_init for new nodes vs. established nodes |

---

## 10. Out of Scope

The following are **explicitly excluded** from this specification. They are implementation decisions to be made separately:

- How Gaussians are initialised from raw (non-latent) signals (encoder architecture)
- Data structure for proximity search in Z (e.g. k-d tree, HNSW, ball tree)
- How the attention budget B is set (may be fixed, signal-dependent, or externally controlled)
- Sources of evaluator signal (affect, social reward, curiosity) that modulate λ or τ
- Training procedure for Gaussian parameters within a node over time

---

## 11. Glossary

| Term | Definition |
|---|---|
| HFN | HPM Fractal Node — the core pattern unit |
| Z | The shared D-dimensional latent space in which all Gaussians live |
| N(μ, Σ) | Multivariate Gaussian with mean μ and covariance Σ |
| D_KL | Kullback-Leibler divergence — measures how surprised a Gaussian is by a signal |
| τ | Surprise threshold — below this the Observer considers a node to explain the signal adequately |
| κ_hj | Overlap coefficient between nodes h and j: ∫ N_h · N_j dz |
| B | Attention budget — limits the number of expansions per signal |
| Δ(h) | Expected surprise reduction from expanding node h |
| S_h | Score of node h: accuracy minus compression penalty |
| λ | Compression pressure constant — weight of complexity in the scoring function |
| α_struct | Structural absorption threshold for overlap coefficient κ |
| w_h | Weight of node h in the Forest — accumulated predictive success |
| Polygraph body | The DAG of children and typed edges that forms the structural face of a node |
