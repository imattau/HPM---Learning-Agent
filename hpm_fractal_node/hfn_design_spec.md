# HPM Fractal Node (HFN) Design Spec

## 1. Core Principles

### 1.1 Node Definition
HFN =
- subnodes (HFN[])
- relationships (edges between subnodes)
- L5 (derived meta-pattern)

### 1.2 Constraints
- Nodes are self-contained
- No parent references
- No global graph
- No mutation from queries
- All meaning is relative to the node

### 1.3 Allowed Operations
- Inspect node (read-only)
- Query subnodes (internal)
- Recombine nodes (external → new node)
- Derive L5 (pure function)

---

## 2. Data Structures

### 2.1 HFN Node

```python
class HFN:
    def __init__(self, id, subnodes=None, edges=None):
        self.id = id
        self.subnodes = subnodes or []
        self.edges = edges or []
```

### 2.2 Edge Structure

```python
Edge = (source_index, target_index, relation_type)
```

---

## 3. Core Functions

### 3.1 Inspect Node

```python
def inspect(node):
    return {
        "subnodes": node.subnodes,
        "edges": node.edges
    }
```

### 3.2 Flatten

```python
def flatten(node, depth=1):
    if depth == 0:
        return [node.id]
    result = []
    for sub in node.subnodes:
        result.extend(flatten(sub, depth-1))
    return result
```

### 3.3 Derive L5

```python
def derive_L5(node):
    labels = [sub.id for sub in node.subnodes]
    relations = [
        f"{node.subnodes[src].id}->{node.subnodes[dst].id}"
        for (src, dst, _) in node.edges
    ]
    return {
        "sequence": labels,
        "relations": relations
    }
```

### 3.4 Query Subnodes

```python
def query_subnodes(node, condition_fn):
    return [s for s in node.subnodes if condition_fn(s)]
```

### 3.5 Query Edges

```python
def query_edges(node, relation_type):
    return [
        (node.subnodes[src], node.subnodes[dst])
        for (src, dst, rel) in node.edges
        if rel == relation_type
    ]
```

---

## 4. Recombination

### 4.1 Basic Recombine

```python
def recombine(node_a, node_b):
    new_subnodes = [node_a, node_b]
    new_edges = []
    if node_a.subnodes and node_b.subnodes:
        new_edges.append((0, 1, "next"))
    return HFN(
        id=f"{node_a.id}+{node_b.id}",
        subnodes=new_subnodes,
        edges=new_edges
    )
```

### 4.2 Structural Merge

```python
def merge_structures(node_a, node_b):
    subnodes = node_a.subnodes + node_b.subnodes
    edges = []
    edges.extend(node_a.edges)
    offset = len(node_a.subnodes)
    for (src, dst, rel) in node_b.edges:
        edges.append((src + offset, dst + offset, rel))
    return HFN(
        id=f"merge({node_a.id},{node_b.id})",
        subnodes=subnodes,
        edges=edges
    )
```

---

## 5. Example

```python
A = HFN("A")
B = HFN("B")
C = HFN("C")
E = HFN("E")

AB = HFN("A→B", subnodes=[A, B], edges=[(0,1,"next")])

ABC = HFN("A→B→C", subnodes=[AB, C], edges=[(0,1,"next")])
ABE = HFN("A→B→E", subnodes=[AB, E], edges=[(0,1,"next")])
```

---

## 6. Properties

- Fractality
- Reuse
- Multi-context support
- Observer relativity

---

## 7. Excluded

- Learning
- Optimisation
- Probabilities
- Global graph
- Parent references

---

## 8. Summary

HFN is a minimal fractal structure for representing patterns.

**Key Rule:**
Nodes store structure. Functions interpret structure. New nodes are constructed from structure.
