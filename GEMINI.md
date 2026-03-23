# GEMINI.md

## Project Overview
HPM Learning Agent is a Python-based implementation of the **Hierarchical Pattern Modelling (HPM)** framework, a theory of human learning developed by Matt Thomson. The project aims to build AI agents that learn by progressively discovering, stabilizing, and refining hierarchical patterns across multiple levels of abstraction.

### Core Architecture
The system is built around four structural roles defined by the HPM framework:
1.  **Pattern Substrates (`hpm/patterns/`)**: Where patterns are encoded. Supports various probabilistic types:
    *   `GaussianPattern`: Multivariate Gaussian for continuous data.
    *   `LaplacePattern`: Heavier-tailed continuous data.
    *   `CategoricalPattern`: Discrete sequences.
    *   `BetaPattern`: Bounded [0,1] data.
    *   `PoissonPattern`: Count data.
2.  **Pattern Dynamics (`hpm/dynamics/`)**: How patterns change.
    *   `MetaPatternRule`: Governs weight updates.
    *   `RecombinationOperator`: Creates new patterns by combining existing ones.
3.  **Pattern Evaluators (`hpm/evaluators/`)**: Determine pattern utility.
    *   `Epistemic`: Predictive accuracy/loss.
    *   `Affective`: Balance of complexity and utility.
    *   `Social`: Frequency and prevalence in a shared `PatternField`.
    *   `ResourceCost`: Memory and CPU overhead.
4.  **Pattern Fields (`hpm/field/`)**: The shared environment where patterns from multiple agents compete and interact.

### Key Components
*   **`Agent` (`hpm/agents/agent.py`)**: The primary learning unit that integrates substrates, evaluators, and dynamics.
*   **`HierarchicalOrchestrator` (`hpm/agents/hierarchical.py`)**: Manages multi-level agent stacks where higher levels (L2+) process abstracted "bundles" (mu, weight, epistemic loss) from lower levels.
*   **`MultiAgentOrchestrator` (`hpm/agents/multi_agent.py`)**: Coordinates ensembles of agents sharing a pattern field.
*   **`Store` (`hpm/store/`)**: Pluggable storage backends including `InMemoryStore` and `SQLiteStore`.
*   **`PhyRE Simulator` (`benchmarks/phyre_sim.py`)**: Custom 2D physics engine for evaluating physical reasoning in SP7–SP10.

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

### Running Benchmarks
The project includes a suite of benchmarks testing various "Superpowers" (SP) of the HPM framework. Most benchmarks support a `--smoke` flag for rapid end-to-end verification.

| Superpower | Domain | Key Metric | Benchmark Script |
|---|---|---|---|
| **SP4** | ARC | Accuracy | `benchmarks/structured_arc.py` |
| **SP5-6** | Math | Accuracy | `benchmarks/structured_math_l4l5.py` |
| **SP7-8** | Physics | Accuracy | `benchmarks/structured_phyre.py` |
| **SP10** | Transfer | Alignment | `benchmarks/phyre_delta_alignment.py` |
| **SP11** | DS-1000 | Accuracy | `benchmarks/structured_ds1000_l4l5.py` |
| **SP12** | Chemistry | Accuracy | `benchmarks/structured_chem_logic_l4l5.py` |

**Example Command:**
```bash
# Run a fast smoke test of the SP6 benchmark
python benchmarks/structured_math_l4l5.py --smoke
```

### Running Tests
The repository uses `pytest` for both unit tests and automated benchmark smoke tests.
```bash
# Run all unit tests
pytest tests/

# Run benchmark smoke tests specifically
pytest tests/benchmarks/
```

---

## Development Conventions

### Coding Style
*   **Type Hinting**: Required for all new function and class definitions.
*   **Documentation**: Use descriptive docstrings for classes and complex methods.
*   **Configuration**: All agent hyperparameters should be managed via `AgentConfig` in `hpm/config.py`. Use `BENCH_CONFIG` in `benchmarks/common.py` for benchmark-specific defaults.
*   **Immutability**: Patterns should generally be treated as immutable; `update()` and `recombine()` return new instances.

### Testing Practices
*   **Unit Tests**: New features or pattern types MUST include unit tests in the `tests/` directory.
*   **Benchmark Smoke Tests**: All new benchmark scripts should support a `--smoke` or `--n_tasks 2` flag and be added to the automated test suite in `tests/benchmarks/`.
*   **Simulator Validation**: Any changes to the `PhyRE` physics engine should be verified using `tests/benchmarks/test_phyre_sim.py`.
*   **Regression Testing**: Major architectural changes should be verified against existing "Structural Immunity" (`benchmarks/structural_immunity.py`) and "Elegance Recovery" (`benchmarks/elegance_recovery.py`) metrics.

### Contribution Guidelines
*   **Research Alignment**: Ensure implementations align with the structural principles described in the `Human Learning as Hierarchical Pattern Modelling v1.25.pdf` working paper.
*   **Surgical Updates**: Prefer targeted changes to specific modules (`agents`, `patterns`, `evaluators`) rather than monolithic refactors.

---

## Documentation & Planning

The project uses a rigorous "Superpowers" (SP) methodology to design, implement, and track major capabilities. All significant features or architectural changes must follow this process.

### Specs and Plans
- **Location**: All documentation must be placed in `docs/superpowers/`.
  - **Plans**: Implementation roadmaps go in `docs/superpowers/plans/`.
  - **Specs**: Technical design specifications go in `docs/superpowers/specs/`.
- **Naming Convention**: Use the following mandatory format for consistency:
  - **Plans**: `YYYY-MM-DD-<feature-slug>[-sp<X>].md`
  - **Specs**: `YYYY-MM-DD-<feature-slug>[-sp<X>]-design.md`
  - *Example*: `2026-03-23-structured-arc-sp4-design.md`

### Design Principles
- **Agentic Protocol**: Plans should be structured for autonomous execution, using checkbox (`- [ ]`) syntax for task tracking and clear verification steps.
- **Traceability**: Link implementation results back to the theoretical grounding in the working paper whenever possible.
- **Modularity**: Design for domain-agnostic reuse within the `hpm/` core package.
