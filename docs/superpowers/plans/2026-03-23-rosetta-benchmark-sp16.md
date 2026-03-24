# Rosetta Benchmark — Sub-project 16 Implementation Plan

**Goal:** Bridge concept understanding across divergent mathematical substrates.

---

### Task 1: Rosetta Simulator
- [ ] **Create `benchmarks/rosetta_sim.py`**
  - Implement `EuclideanEncoder`: Side counts and lengths.
  - Implement `CoordinateEncoder`: Vertex coordinates.
  - Implement `generate_shared_shapes(n, seed)`: 
    - Generates a list of shapes (Square, Rectangle, Triangle).
    - Returns each shape in *both* Euclidean and Coordinate formats (The shared set).

### Task 2: Implementation of Rosetta Benchmark
- [ ] **Create `benchmarks/rosetta_alignment.py`**
  - Use `benchmarks/multi_domain_alignment.py` logic for Procrustes rotation.
  - **Step 1 (Learning)**: Both agents learn the "Identity" law (Transformation: same shape) using the shared Litmus set.
  - **Step 2 (Alignment)**: Find $R$ such that $R @ M_{Euclidean} \approx M_{Coordinate}$.
  - **Step 3 (The Litmus Turn)**:
    - Agent A (Euclidean) picks a "Target Shape" (Square).
    - Agent A encodes the Square Law ($L3_A$).
    - Agent B (Coordinate) receives $L3_{rotated} = R @ L3_A$.
    - Agent B must use $L3_{rotated}$ to pick the Square from a set of Coordinate-based distractors.

### Task 3: Verification
- [ ] **Create `tests/benchmarks/test_rosetta.py`**
  - Smoke test for the Rosetta stone alignment.
- [ ] **Run benchmark** and report communication accuracy.
