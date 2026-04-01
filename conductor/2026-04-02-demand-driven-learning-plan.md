# Implementation Plan: Demand-Driven Learning (SP24)

## 1. Goal
Implement a "Fail-Learn-Retry" loop where the Agnostic Decoder triggers targeted learning by the Observer to fulfill missing generative constraints, demonstrating active "Curiosity."

## 2. Core Modification: `hfn/decoder.py`
The `Decoder` must return structured requests when a gap is found, rather than just returning an empty list.

```python
from dataclasses import dataclass
from typing import List, Union

@dataclass
class ResolutionRequest:
    missing_mu: np.ndarray
    missing_sigma: np.ndarray
    required_edges: list[Edge]

# Decoder.decode return type becomes:
def decode(self, node: HFN) -> Union[List[HFN], ResolutionRequest]:
```
*If topological scoring fails to find a candidate > 0.0, the Decoder raises the Request.*

## 3. The Experiment: `experiment_demand_driven_learning.py`

### 3.1 Setup
- **Manifold**: 1D Hand coordinates.
- **Priors**: Red=2.0, Blue=5.0. No prior for Green.
- **Buffer**: A list of raw observations `[2.1, 5.0, 8.0, 8.1]`.
- **Goal**: Find Green (`mu=8.0`, edge to `Concept_Green`).

### 3.2 The Governor Loop
```python
def execute_goal(decoder, observer, goal, buffer):
    max_retries = 3
    for attempt in range(max_retries):
        result = decoder.decode(goal)
        
        if isinstance(result, list):
            return result # Success!
            
        elif isinstance(result, ResolutionRequest):
            print(f"  [Governor] Decoder stalled. Requesting target near {result.missing_mu}")
            # Trigger Observer to scan buffer
            found = False
            for obs in buffer:
                # If observation matches request mu closely enough
                if np.linalg.norm(obs - result.missing_mu) < 0.5:
                    print(f"  [Observer] Found evidence in buffer: {obs}. Stabilising new node.")
                    # 1. Create concrete leaf
                    leaf_id = f"leaf_discovered_{np.random.randint(100)}"
                    new_leaf = HFN(mu=obs, sigma=np.array([0.001]), id=leaf_id)
                    # 2. Apply requested constraints (Binding!)
                    for edge in result.required_edges:
                        new_leaf.add_edge(new_leaf, edge.target, edge.relation)
                    # 3. Register in Forest
                    observer.register(new_leaf)
                    found = True
                    break
                    
            if not found:
                print(f"  [Observer] No evidence found for {result.missing_mu}. Hallucination blocked.")
                return "FAILURE: Ungrounded Request"
                
    return "FAILURE: Max Retries Exceeded"
```

## 4. Test Cases
1. **The Green Block (Valid Gap)**: 
    * Decoder fails $\rightarrow$ Requests `mu=8.0` with `HAS_COLOR->GREEN` edge.
    * Observer finds `8.0` in buffer, creates node, adds edge.
    * Decoder retries $\rightarrow$ Success!
2. **The Yellow Block (Hallucination Guard)**:
    * Decoder fails $\rightarrow$ Requests `mu=10.0` with `HAS_COLOR->YELLOW` edge.
    * Observer scans buffer $\rightarrow$ Finds nothing near `10.0`.
    * Governor aborts $\rightarrow$ Success (Blocked hallucination).

## 5. Review Against Specification
- **Decoder Update**: Yes, returns `ResolutionRequest`.
- **Historical Buffer**: Yes, implemented as a simple list array for the Observer to scan.
- **Fail-Learn-Retry Loop**: Yes, Governor manages retries and binding.
- **Anti-Hallucination Guard**: Yes, explicitly tested with the Yellow Block scenario.
