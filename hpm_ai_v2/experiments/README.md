# hpm_ai_v2 Experiments — Benchmark and Stretch Tests

This directory contains the experimental validation suite for the `hpm_ai_v2` framework. These experiments demonstrate HPM's superior sample efficiency, compositional abstraction, and cross-domain transfer capabilities.

## Featured Benchmarks (SP80–SP81)

### [SP80] Comparative Benchmark: HPM vs. Published Few-Shot Results
**Script**: `experiment_sp80_comparative_benchmark.py`

This benchmark compares HPM's performance on symbolic list transformations against state-of-the-art results for GPT-4, fine-tuned Transformers, and MAML (Model-Agnostic Meta-Learning).

| Task | k-shot | HPM Acc | GPT-4 (Lit) | Transformer (Lit) | MAML (Lit) |
|------|--------|---------|-------------|-------------------|------------|
| `add_one` | 1 | **100.0%** | ~95% | ~10% | ~70% |
| `double` | 1 | **100.0%** | ~95% | ~10% | ~70% |
| `filter_positive` | 2 | **100.0%** | ~70% | ~5% | ~50% |
| `compose_add1_double`| 1 | **100.0%** | ~40% | ~0% | ~30% |

**Key Breakthroughs**:
- **Zero-Shot Composition**: HPM solves the composite task `(x+1)*2` at k=1 by reusing previously learned scalar macros within a discovered iteration motif.
- **Motif-Guided Search**: High-level `MAP_START` and `MAP_END` motifs reduce the BFS planning depth, enabling reliable solutions within a shallow search horizon.
- **Macro Prioritization**: Uses `MacroPrioritizingRetriever` to ensure learned patterns are preferred over primitive operations during search.

---

### [SP81] Stretch Test: Filter Positive Then Double
**Script**: `experiment_sp81_stretch_test.py`

Evaluates **zero-shot sequential composition** of two high-level macros: `filter_positive` and `double`.

- **Goal**: `[-1, 2, 3, -4] → [4, 6]` (filter positives, then double them).
- **Result**: **100.0% accuracy at k=1**.
- **Mechanism**: The agent uses `SequentialCompositionMixin` to chain the learned `filter_pos` and `map_double` macros via AST transformation. It achieves perfect accuracy without any training examples for the composite task itself.

---

## Physical and Chemical Intelligence (SP94–SP97)

These experiments demonstrate HPM's ability to discover and transfer robust physical and chemical laws from noisy, continuous, or confounded data.

| Superpower | Domain | Script | Docs |
|---|---|---|---|
| **SP94** | Chemistry | `experiment_sp94_reaction_prediction.py` | [README_SP94.md](./README_SP94.md) |
| **SP95** | Fluid Dynamics | `experiment_sp95_inverse_sprinkler.py` | [README_SP95.md](./README_SP95.md) |
| **SP96** | Physics | `experiment_sp96_projectile_motion.py` | [README_SP96.md](./README_SP96.md) |
| **SP97** | Causal | `experiment_sp97_spurious_resistance.py` | [README_SP97.md](./README_SP97.md) |

**Key Breakthroughs**:
- **One-Shot Reaction Prediction (SP94)**: Learns complex bit-level molecular transformations from a single example.
- **Inverse Sprinkler Problem (SP95)**: Discovers non-intuitive physical invariants that defy simple statistical correlations.
- **Noise-Robust Discovery (SP96)**: Extracts quadratic laws ($0.5 g t^2$) from noisy continuous trajectories.
- **Spurious Resistance (SP97)**: Detects the failure of statistical shortcuts under distribution shift and recovers true structural invariants.

---

## Social and Lifelong Learning

### [MS-SL] Multi-Specialist Social Learning
**Script**: `experiment_ms_sl.py` | **Docs**: [README_MS_SL.md](./README_MS_SL.md)

Demonstrates pattern field convergence and institutional scaffolding. Specialist agents (Alice, Bob, Charlie) share learned macros via a social forest and coordinate via a shared blackboard to solve cross-domain tasks they were not individually trained on.

---

## Domain-Specific Few-Shot Experiments

| Superpower | Domain | Script | Description |
|---|---|---|---|
| **SP71** | Image | `experiment_sp71_image_fewshot.py` | PIL-based image transformations (blur, rotate, enhance). |
| **SP73** | Audio | `experiment_sp73_audio_fewshot.py` | Librosa-based audio processing (pitch shift, time stretch). |
| **SP74** | Graph | `experiment_sp74_graph_fewshot.py` | NetworkX-based graph manipulations (relabeling, edge addition). |
| **SP77** | Analogy | `experiment_sp77_cross_domain_analogy.py`| Zero-shot transfer of iteration schemas from List to Graph domain. |

---

## Core System Validation

- **SP68 (Full Stack)**: `experiment_sp68_full_stack.py` — Verifies integration of L2–L4 capabilities.
- **SP69 (Trust)**: `experiment_sp69_trust.py` — Evaluates pattern reliability and "trust" weights in the observer.
- **SP70 (Lifecycle)**: `experiment_sp70_lifecycle.py` — Tests node aging, pruning, and long-term persistence.
- **SP72 (Lifelong)**: `experiment_sp72_lifelong.py` — Continuous learning across a changing task distribution.

---

## Running Experiments

All experiments should be run from the project root using `PYTHONPATH=.`:

```bash
# Run the comparative benchmark
PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp80_comparative_benchmark.py

# Run the composition stretch test
PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp81_stretch_test.py
```
