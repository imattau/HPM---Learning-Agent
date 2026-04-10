# HPM Learning Agent

AI learning agents built on the **Hierarchical Pattern Modelling (HPM)** framework — a theory of how humans learn, developed by Matt Thomson. The full working paper is included in this repo.

---

## Repository Overview

- `hpm/` — baseline HPM framework (core agents, field dynamics, hierarchy, benchmarks)
- `hpm_ai_v1/` — structured multi-agent self-repair and project-manifold reasoning
- `hpm_fractal_node/` — HFN (Hierarchical Fractal Node) substrate: a new pattern substrate implementing HPM principles as a forest of probabilistic nodes with fractal geometry, Observer dynamics, and an extensive experiment suite
- `hfn/` — the core HFN library (nodes, forest, observer, async controller, fractal metrics, query/converter pipeline)

## What is HPM?

HPM is a framework for understanding learning as the progressive discovery and refinement of patterns across multiple levels of abstraction. Rather than treating learning as a single process (like gradient descent or reinforcement), HPM proposes that all learning systems — including humans — operate through four structural roles working together:

1. **Pattern substrates** — where patterns are stored (weights, memory, symbolic representations)
2. **Pattern dynamics** — how patterns are created, updated, stabilised, and lost
3. **Pattern evaluators** — what determines which patterns are worth keeping (prediction error, reward, coherence)
4. **Pattern fields** — the shared environment that shapes which patterns compete and survive

Patterns at higher levels of abstraction are built from regularities detected at lower levels. Learning is not just accumulating data — it is the emergence of structure that generalises, predicts, and supports action.

---

## Benchmarks & Superpowers (SP)

The project tracks major architectural breakthroughs as "Superpowers" (SP), validated through rigorous benchmarks.

### Core Framework Benchmarks
- **Reber Grammar**: Discrete sequence learning (AUROC 0.934).
- **Structural Immunity**: Noise resilience and rapid recovery (T_rec = 5 steps).
- **Substrate Efficiency**: Pareto-optimal representation (beats GMM at comparable complexity).
- **Elegance Recovery**: Identification of specific generative mathematical laws.

### Hierarchical Encoders (SP4–SP16)
- **SP4–SP6**: Structured ARC and Math (98%+ accuracy via hierarchicalRelational Laws).
- **SP7–SP8**: Physics Reasoning (PhyRE) hierarchical grounding.
- **SP10**: Delta Alignment (Procrustes-based cross-domain transfer).
- **SP11–SP14**: Grounding in DS-1000, Molecular Discovery (Chem-Logic), and Linguistic Register Shifts.
- **SP16**: Geometric Rosetta (Relational translation between fundamantally different substrates).

### Sovereign Agency & Unified Loops (SP17–SP54)
The transition from passive pattern matching to autonomous, intent-driven synthesis.

| breakthrough | Evidence |
|---|---|
| **Belief Revision** | SP37: Successfully penalizes falsified rules in ambiguous environments |
| **Unified Cognitive Loop** | SP41: Plan -> Act -> Fail -> Explore -> Re-Plan loop closed |
| **Long-Horizon Reasoning** | SP42: 100% success on 20-step reasoning chains |
| **Recursive Scaling** | SP45: Discovered 7-step nested graph for map/loop execution |
| **Non-Linear Synthesis** | SP48: Rendered if/else blocks for conditional mapping |
| **Curiosity / Self-Play** | SP51: Autonomously cataloged novel programs via state oracles |
| **AST Synthesis** | SP53: Robust code generation via partial constraints and AST module |
| **HPM-Native Planning** | **SP54 SUCCESS**: 100% HPM-native planning via population dynamics, temporal credit assignment, and replicator contrast dynamics. |

---

## HPM Fractal Node (HFN) Experiment Suite

The **HFN** substrate implements HPM principles directly as a forest of probabilistic Gaussian nodes.

| Experiment | Domain | Key Finding |
|---|---|---|
| `experiment_math.py` | Arithmetic | Discovers perfect-power clusters and modular prime regions (18× above random). |
| `experiment_nlp.py` | Language | Semantic category purity 0.780 mean using QueryLLM. |
| `experiment_unified_cognitive_loop.py` | Core Agent | Capstone: closed-loop belief revision and re-planning. |
| `experiment_recursive_scaling.py` | Scaling | Discovered deep nested graphs for complex execution. |
| `experiment_non_linear_synthesis.py` | Logic Forks | Turing-complete if/else branching. |
| `experiment_modular_abstraction.py` | Functions | Encapsulated procedures and O(1) planning calls. |
| `experiment_execution_guided_synthesis.py` | SP54 Planning | Replaced beam search with native HPM population dynamics and backpropagated utility. |
| `experiment_schema_transfer.py` | Transfer | **Discovery**: Autonomous emergence of MAP and FILTER schemas via Replicator Contrast Dynamics. |

---

## HPM AI — Sovereign Recursive Intelligence (v3.0)

The **HPM AI** (implemented in `hpm_ai_v1/`) treats the entire HPM codebase as its primary algebraic substrate.

### Core Capabilities
- **Cascading Dependency Repair**: Automatically identifies and repairs project-wide dependency breaks.
- **Global Project Manifold**: Understands the relational ecology of the 220+ module repository.
- **Agentic Negotiation**: L4 Architects and L5 Monitors design and gate ChangeSets through a shared Pattern Field.

---

## Setup

**Requirements:** Python 3.11+

```bash
# Using uv (fastest)
uv sync

# Create a virtual environment and install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Benchmarks & Experiments

```bash
# Run latest SP54 Execution-Guided Synthesis
PYTHONPATH=. .venv/bin/python hpm_fractal_node/experiments/experiment_execution_guided_synthesis.py

# Run Schema Transfer Discovery
PYTHONPATH=. .venv/bin/python hpm_fractal_node/experiments/experiment_schema_transfer.py

# Run traditional benchmarks
python benchmarks/reber_grammar.py
python benchmarks/structured_math_l4l5.py
```

### Run tests
```bash
python -m pytest tests/ -v
```

---

## Background reading

The working paper *Human Learning as Hierarchical Pattern Modelling* (included as a PDF) gives the full theoretical grounding for the framework.
