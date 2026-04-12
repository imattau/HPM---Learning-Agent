# GEMINI.md

## Project Overview
HPM Learning Agent is a Python-based implementation of the **Hierarchical Pattern Modelling (HPM)** framework, a theory of human learning developed by Matt Thomson. The project aims to build AI agents that learn by progressively discovering, stabilizing, and refining hierarchical patterns across multiple levels of abstraction.

### Core Architecture
The system is built around four structural roles defined by the HPM framework:
1.  **Pattern Substrates (`hpm/patterns/`, `hfn/hfn.py`)**: Where patterns are encoded. Supports Gaussian, Laplace, Categorical, Beta, and Poisson types.
2.  **Pattern Dynamics (`hpm/dynamics/`, `hfn/observer.py`)**: How patterns change via weight updates, recombination, and structural absorption.
3.  **Pattern Evaluators (`hpm/evaluators/`, `hfn/evaluator.py`)**: Determine pattern utility via Epistemic (accuracy), Affective (complexity), Social, and Coherence metrics.
4.  **Pattern Fields (`hpm/field/`, `hfn/forest.py`)**: The shared environment where patterns from multiple agents/processes compete and interact.

### Key Components
*   **`Agent` (`hpm/agents/agent.py`)**: The primary learning unit that integrates substrates, evaluators, and dynamics.
*   **`HierarchicalOrchestrator` (`hpm/agents/hierarchical.py`)**: Manages multi-level agent stacks.
*   **`HFN` (Hierarchical Fractal Node)**: A new probabilistic substrate implementing HPM as a forest of nodes with fractal geometry and native Observer dynamics.
*   **`Observer`**: Drives learning dynamics, including online weight updates and surprise-driven node creation.
*   **`TieredForest`**: Pluggable storage backend supporting hot/cold tiering for large knowledge bases.

---

## Building and Running

### Prerequisites
*   Python 3.11 or higher.
*   Recommended: `uv` for fast dependency management.

### Setup
```bash
# Using uv (recommended)
uv sync

# Using pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Latest Experiments (SP54-SP56)
The latest breakthrough enables 100% HPM-native planning, schema discovery, tool integration, and hierarchical abstraction.

```bash
# Run Execution-Guided Synthesis (SP54)
PYTHONPATH=. .venv/bin/python hpm_fractal_node/experiments/experiment_execution_guided_synthesis.py

# Run Library Discovery & Recognition (SP55)
PYTHONPATH=. .venv/bin/python hpm_fractal_node/experiments/experiment_library_discovery.py

# Run Compositional Abstraction (SP56)
PYTHONPATH=. .venv/bin/python hpm_fractal_node/experiments/experiment_compositional_abstraction.py

# Run Schema Transfer Discovery
PYTHONPATH=. .venv/bin/python hpm_fractal_node/experiments/experiment_schema_transfer.py
```

### Running Traditional Benchmarks
Most benchmarks support a `--smoke` flag for rapid end-to-end verification.

| Superpower | Domain | Key Metric | Benchmark Script |
|---|---|---|---|
| **SP4** | ARC | Accuracy | `benchmarks/structured_arc.py` |
| **SP5-6** | Math | Accuracy | `benchmarks/structured_math_l4l5.py` |
| **SP7-8** | Physics | Accuracy | `benchmarks/structured_phyre.py` |
| **SP10** | Transfer | Alignment | `benchmarks/phyre_delta_alignment.py` |
| **SP16** | Rosetta | Discovery | `benchmarks/rosetta_geometric_benchmark.py` |
| **SP54** | Planning | Utility | `hpm_fractal_node/experiments/experiment_execution_guided_synthesis.py` |
| **SP55** | Tool Integration | Discovery | `hpm_fractal_node/experiments/experiment_library_discovery.py` |
| **SP56** | Compositional | Transfer | `hpm_fractal_node/experiments/experiment_compositional_abstraction.py` |

---

### Development Conventions

*   **HPM Consistency**: All new planning and learning dynamics MUST use the HPM utility equation (`Accuracy - Complexity + Coherence`) and replicator-style dynamics (Observer-managed weights).
*   **Tiered Knowledge Base**: All experiments MUST use `TieredForests` for hot/cold storage.
*   **Cold Storage Persistence**: Use `PersistenceManager` to save structural nodes to `data/knowledge_base/` at the end of each run.
*   **Type Hinting**: Required for all new function and class definitions.

### Testing Practices
*   **Unit Tests**: New features MUST include unit tests in the `tests/` directory.
*   **Simulator Validation**: Any changes to the `PhyRE` physics engine MUST be verified using `tests/benchmarks/test_phyre_sim.py`.
*   **Regression Testing**: Major architectural changes should be verified against existing "Structural Immunity" and "Elegance Recovery" metrics.

---

## Documentation & Planning

### Specs and Plans
- **Location**: All documentation must be placed in `docs/superpowers/`.
  - **Plans**: Implementation roadmaps go in `docs/superpowers/plans/`.
  - **Specs**: Technical design specifications go in `docs/superpowers/specs/`.
- **Naming Convention**:
  - **Plans**: `YYYY-MM-DD-<feature-slug>[-sp<X>].md`
  - **Specs**: `YYYY-MM-DD-<feature-slug>[-sp<X>]-design.md`

### Design Principles
- **Agentic Protocol**: Plans should be structured for autonomous execution using checkbox (`- [ ]`) syntax.
- **Traceability**: Link results back to the theoretical grounding in the HPM working paper.
- **Modularity**: Design for domain-agnostic reuse within the `hpm/` core package.
