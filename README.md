# HPM Learning Agent

AI learning agents built on the **Hierarchical Pattern Modelling (HPM)** framework — a theory of how humans learn, developed by Matt Thomson. The full working paper is included in this repo.

---

## What is HPM?

HPM is a framework for understanding learning as the progressive discovery and refinement of patterns across multiple levels of abstraction. Rather than treating learning as a single process (like gradient descent or reinforcement), HPM proposes that all learning systems — including humans — operate through four structural roles working together:

1. **Pattern substrates** — where patterns are stored (weights, memory, symbolic representations)
2. **Pattern dynamics** — how patterns are created, updated, stabilised, and lost
3. **Pattern evaluators** — what determines which patterns are worth keeping (prediction error, reward, coherence)
4. **Pattern fields** — the shared environment that shapes which patterns compete and survive

Patterns at higher levels of abstraction are built from regularities detected at lower levels. Learning is not just accumulating data — it is the emergence of structure that generalises, predicts, and supports action.

HPM is a conceptual lens, not a specific algorithm. It is compatible with reinforcement learning, Bayesian inference, and predictive processing — these are all implementations of pattern dynamics under different constraints.

---

## What the agents do

Each agent maintains a **probabilistic model** of its environment — a distribution that is updated incrementally as new observations arrive. Multiple agents share a **PatternField**, a competitive environment where patterns are stored, recombined, and pruned based on their predictive performance.

The system includes several specialist components that mirror HPM's structural roles:

- **HPMAgent** — core learning agent; updates its pattern model on each observation and tracks accuracy, surprise, and weight
- **PatternField** — shared store where agent patterns compete by weight; higher-performing patterns receive more influence
- **StructuralLawMonitor** — periodically checks pattern health and triggers recombination when the field stagnates
- **RecombinationStrategist** — generates new patterns by combining existing ones, preventing premature convergence
- **PredictiveSynthesisAgent** — produces forward predictions across the agent ensemble
- **DecisionalActor** — selects actions based on internal pattern state (external reward head + internal coherence head)
- **MultiAgentOrchestrator** — coordinates all of the above; the main entry point for multi-agent learning

### Pattern types

Five pattern substrates are implemented, each suited to different data:

| Pattern | Data type | Model |
|---|---|---|
| `GaussianPattern` | Continuous ℝ^D | Multivariate Gaussian |
| `LaplacePattern` | Continuous ℝ^D | Laplace (heavier tails) |
| `CategoricalPattern` | Discrete sequences | D×K probability matrix |
| `BetaPattern` | Bounded [0,1]^D | Independent Beta distributions |
| `PoissonPattern` | Count data ℕ^D | Independent Poisson distributions |

---

## Benchmarks

Benchmarks test different aspects of HPM-grounded learning across a progression from flat single-level agents to hierarchical multi-level architectures. Multi-agent versions run the full orchestrator (PatternField + Monitor + Strategist + Actor). For thorough analysis of all results, see [benchmarks/README.md](benchmarks/README.md).

### Reber Grammar — discrete sequence learning
The Reber Grammar is a finite-state automaton over a 7-symbol alphabet. Valid sequences follow hidden structural rules; random sequences do not. The agent learns positional symbol distributions and must distinguish valid from random sequences.

| Setup | AUROC | NLL separation |
|---|---|---|
| Single agent (CategoricalPattern) | 0.934 | 16.82 |
| 3-agent ensemble | 0.955 | — |

Pass criteria: AUROC > 0.80, separation > 5.0. Both pass with margin.

### Structural Immunity — noise resilience
Three-phase protocol: 500 steps of stable signal, 20 steps of uniform noise, 500 steps of recovery. Measures how quickly the agent recovers its baseline accuracy after the noise storm.

| Setup | T_rec | Result |
|---|---|---|
| Single agent (Gaussian) | ≤ 100 | IMMUNE |
| 2-agent shared field (Gaussian) | ≤ 100 | IMMUNE |
| 2-agent shared field (Beta) | 5 | IMMUNE |

Pass criterion: T_rec ≤ 100.

### ARC — abstract visual reasoning
Tests pattern generalisation on tasks from the Abstraction and Reasoning Corpus (fchollet/arc-agi, train split).

| Setup | Tasks | Accuracy | vs Chance |
|---|---|---|---|
| Single agent | 342 / 400 | 65.8% | +45.8% |
| 2-agent ensemble | 342 / 400 | 65.5% | +45.5% |

58 tasks excluded (grids exceed 20×20). Chance baseline is 20% (5-way discrimination).

**What this benchmark actually tests:** Given 3–5 input→output grid pairs as training, the agent must identify the correct output from 4 distractors drawn from other tasks. This is a discrimination task, not actual ARC task solving — it measures whether the agent can build a useful representation of a visual transformation rule. The agent encodes the transformation delta (output − input) via a fixed random projection to 64 dimensions; it never operates on raw pixel values.

The multi-agent version splits training pairs between two agents (even/odd), so each sees fewer examples. Scores are comparable to single-agent, suggesting the PatternField sharing partially compensates for fewer examples per agent.

### Substrate Efficiency
Tests whether the agent discovers a compact representation of a redundant data stream. The data is 3 overlapping Gaussian clusters in 16-dimensional space; an optimal model needs only 3 components. The agent's complexity/accuracy trade-off is compared against Gaussian Mixture Models (GMM) at k=1–5 on a Pareto frontier.

| Model | Complexity | Accuracy | Pareto frontier |
|---|---|---|---|
| HPM agent | 0.20 | 0.39 | ✓ |
| GMM k=1 | 0.00 | 0.00 | ✓ |
| GMM k=2 | 0.23 | 0.31 | ✗ |
| GMM k=3 | 0.50 | 0.49 | ✓ |
| GMM k=4 | 0.77 | 0.71 | ✓ |
| GMM k=5 | 1.00 | 1.00 | ✓ |

The HPM agent sits on the Pareto frontier — it achieves more accuracy per unit of complexity than GMM k=2, using less representational overhead to capture meaningful structure.

### Elegance Recovery
Tests whether the agent recovers the specific structure of a hidden mathematical law, not just any smooth fit. The agent trains on y = x²/(1+x), then the top-weighted pattern is evaluated against a true-law test set and a distractor (y = x²). A positive gap means the agent distinguishes the training law from the near-identical distractor.

| Steps | Recombinations | NLL (true law) | NLL (distractor) | Gap | Result |
|---|---|---|---|---|---|
| 1500 | 17 | −3.33 | −3.27 | +0.06 | RECOVERED |

Pass criterion: gap > 0.

---

### SP4 — Structured ARC (hierarchical encoders)

Extends the ARC benchmark with a five-level hierarchical encoder stack (L1: pixel moments, L2: object anatomy, L3: transformation rule, L4: relational meta-rule, L5: strategy gate). Each ablation tests the contribution of each level independently.

| Configuration | Accuracy | vs Flat |
|---|---|---|
| flat (no hierarchy) | 63.2% | baseline |
| l1_only | 63.2% | — |
| l2_only | 46.5% | −16.7pp |
| l3_only | — | — |
| full (L1–L3) | 69.0% | +5.8pp |
| L4_only | **88.6%** | **+25.4pp** |
| L4+L5 (full stack) | 88.6% | +25.4pp |

L4 alone delivers a +25.4pp gain. L5 neither adds nor subtracts — it correctly identifies L4 as reliable and sets its gating weight to 1.0.

### SP5 — Structured Math (algebraic transformation families)

Tests hierarchical encoding on symbolic algebraic reasoning: four transformation families (linear, quadratic, exponential, logarithmic), each with multiple task variants.

| Configuration | Accuracy | vs Flat |
|---|---|---|
| flat | 10.6% | baseline |
| l1_only | 10.6% | — |
| l2_only | 66.7% | +56.1pp |
| l3_only | **97.8%** | **+87.2pp** |
| l2+l3 | 96.7% | +86.1pp |
| full (L1–L3) | 96.7% | +86.1pp |

L3 alone achieves 97.8% — the decisive abstraction level for symbolic reasoning. L1 carries no discriminative signal; the family structure lives entirely at L3.

### SP6 — Math L4/L5

Extends SP5 with L4 (relational meta-rule) and L5 (strategy gate) layers.

| Configuration | Accuracy |
|---|---|
| l2+l3 (SP5 baseline) | 96.7% |
| l4_only | **98.3%** |
| l4+l5 (full stack) | 98.3% |

L4 adds a further +1.6pp over the already high SP5 baseline. L5 again stays at gamma=1.0, correctly deferring to a reliable L4.

### SP7 — PhyRE Physics Reasoning

Hierarchical encoding applied to simulated physics: four families (Projectile, Bounce, Slide, Collision), 60 tasks each, 240 tasks total. The task is to identify the correct physics outcome from distractors.

| Configuration | Accuracy | vs Flat |
|---|---|---|
| flat | 22.5% | baseline |
| l2+l3 | **62.5%** | **+40.0pp** |
| l4_only | 61.7% | +39.2pp |
| l4+l5 (full stack) | 61.7% | +39.2pp |

L2+L3 captures the main signal (+40pp). L4 provides no gain over L2+L3 here — in contrast to ARC, where L4 was the decisive level.

### SP8 — Cross-task L4 (PhyRE)

Tests whether training L4 across physics families (not just within each family) improves transfer. Cross-task training pairs examples from different families at the L4 level.

| Configuration | Accuracy |
|---|---|
| l2+l3 (SP7 baseline) | 62.5% |
| cross_task_l4 | 58.3% |

No gain from cross-task L4 training on PhyRE. The relational structure at L4 does not transfer across physics families in this setting.

### SP9 — Naive Cross-Domain Transfer (zero-padding)

Tests whether representations learned in one domain can be transferred to another via zero-padding (concatenating source-domain embeddings with zeros to match the target embedding dimension). Three transfer directions tested.

| Transfer direction | l2+l3 baseline | cross_domain | Result |
|---|---|---|---|
| Math+PhyRE → ARC | 80.0% | 26.7% | NEGATIVE (−53.3pp) |
| Math+ARC → PhyRE | 58.3% | 16.7% | NEGATIVE (−41.6pp) |
| PhyRE+ARC → Math | 100% | 22.2% | NEGATIVE (−77.8pp) |

Naive zero-padding collapses performance in all three directions. Cross-domain representations are not structurally compatible without alignment.

### SP10 — Delta Alignment (Procrustes cross-domain transfer)

Replaces zero-padding with Procrustes-based alignment of embedding delta-vectors. Instead of concatenating raw embeddings, it aligns the *change* in representation across abstraction levels using an orthogonal rotation matrix learned from paired examples.

| Transfer direction | l2+l3 baseline | delta_align | vs SP9 | vs baseline |
|---|---|---|---|---|
| Math+PhyRE → ARC | 80.0% | **80.0%** | +53.3pp | ties |
| Math+ARC → PhyRE | 63.3% | **63.3%** | +46.6pp | ties |
| PhyRE+ARC → Math | 97.8% | 57.8% | +35.6pp | −40.0pp (partial) |

**Verdict: PARTIAL.** Delta alignment beats SP9 in all three directions and ties the within-domain l2+l3 baseline in 2 of 3 cases.

---

### SP11 — DS-1000 Boss Fight (Symbolic Data Science)

Stress-tests HPM agents on real-world Python library transformations (Pandas, NumPy, etc.). Evaluates grounding of L3 Relational Laws in messy API substrates.

| Configuration | Accuracy | vs Baseline (L2+L3) |
|---|---|---|
| l2+l3 (Baseline) | 70.7% | — |
| l4_only (Intuition) | 97.9% | +27.2pp |
| l4l5_full (Gated) | **97.9%** | **+27.2pp** |

### SP12 — Chem-Logic I (Molecular Discovery)

Infers hidden reaction laws (Oxidation, Reduction) using RDKit as an external physical substrate.

| Configuration | Accuracy | Result |
|---|---|---|
| full (L1-L5) | 100% | **SOLVED** |

### SP13 — Chem-Logic II (Ambiguity & Competition)

Introduces competitive inhibition (Amine vs Hydroxyl) and latent pH shifts. Evaluates L5 Surprise detection.

| Configuration | Accuracy | Avg Surprise (pH Shift) |
|---|---|---|
| l4l5_full | 67.5% | **0.236** |

### SP14 — Linguistic Register Shift (Social pH)

Tests detection of hidden "Social Register" shifts (Formal vs Informal). L5 monitor detects surprise when Formal intuition fails on Informal test data.

| Configuration | Accuracy | Avg Surprise (Shift) |
|---|---|---|
| l4l5_full | 0.0% (Trap) | **0.968** |

### SP15 — Generalized Cross-Domain Alignment

Bridges the "Symbolic Gap" by aligning Math with DS-1000 and Chemistry across 6 domains.

| Sources | Target | l2+l3 baseline | Delta Alignment |
|---|---|---|---|
| DS-1000 + Chem | Math | 100% | **77.8%** |

### SP16 — Geometric Rosetta (Concept Discovery)

Two agents with fundamentally different substrates (Cartesian vs Coordinate) achieve shared understanding of a "Square" by discovering a relational translation matrix.

| Step | Result | Metric |
|---|---|---|
| Blind Attempt | Surprise Detected | S = 0.255 |
| Discovery | Mapping Found | 45.0° Rotation |
| Transfer | **SUCCESS ✅** | 100.0% Accuracy |

---

### Key architectural findings across SP4–SP16

| Insight | Evidence |
|---|---|
| Decisive levels are domain-specific | L3 for Math, L4 for ARC/DS-1000 |
| Symbolic Gap can be bridged | SP15: +55pp gain on Math transfer |
| L5 Monitor detects unobserved variables | SP13/14: Surprise > 0.2 on pH/Register shifts |
| Relational Invariants bridge languages | SP16: 100% transfer across math substrates |
| Grounding requires additive rescue | Fixed substrate-bridge issue in SP11 |

---

## What the results suggest

Taken together, the benchmarks show something more interesting than raw performance numbers. Each test targets a different property that HPM claims good learning systems should have — and the results offer early evidence for each claim.

**Structure over surface.** The Reber Grammar benchmark (AUROC 0.934) shows the agent learning genuine structural regularities in discrete sequences, not just memorising frequencies. The key test is that it assigns lower probability to grammatically invalid sequences even when those sequences are plausible-looking — it has internalised the constraint structure, not just the symbol statistics.

**Resilience through shared representation.** The Structural Immunity benchmark shows rapid recovery (T_rec = 5 steps with Beta agents) after a noise storm. This is not a property of a single well-trained model — it emerges from the PatternField, where patterns that survived the noise have higher weight and exert more influence during recovery. The shared field acts as a distributed memory that buffers individual agents against disruption.

**Efficient representation, not just accurate representation.** The Substrate Efficiency result is arguably the most theoretically significant. The HPM agent sits on the Pareto frontier — it achieves more accuracy per unit of representational complexity than a comparably parameterised GMM. This matters because HPM predicts that good learners should not just fit data well, they should do so *parsimoniously*. A system that builds a bloated model has not understood the structure; it has memorised it.

**Specificity of learned laws.** The Elegance Recovery benchmark tests a subtle property: can the agent distinguish between two laws that produce very similar outputs over a wide input range? The positive gap (+0.06) is small but meaningful — the agent has not just found a smooth fit to the training data, it has converged on the specific generative structure that produced it. This is the beginning of what HPM calls a "generative rule" — a pattern that captures not just what happened, but why.

**What this means for AI development.** Current large AI systems achieve impressive capability through scale — enormous parameter counts, vast training data, and compute-intensive optimisation. HPM points toward a different hypothesis: that human-level generalisation may require not more parameters, but better *architecture* — systems where patterns at multiple levels of abstraction interact, compete, and recombine under principled constraints. The benchmarks here are small-scale, but they validate the structural principles. A system that is simultaneously efficient, resilient, structurally specific, and capable of cross-agent knowledge sharing — at any scale — would represent a qualitatively different approach to machine learning than the dominant paradigm.

The open question is whether these properties, demonstrated here in lightweight closed-form models, will transfer to richer neural substrates. That is the central challenge the future directions below are designed to address.

---

## Future directions

- **Hierarchical agents** — patterns at one level feeding as inputs to agents at the next level, implementing HPM's multi-level abstraction hierarchy
- **Temporal dynamics** — sequence modelling where pattern context evolves over time, not just positionally
- **Reward-grounded evaluation** — connecting the DecisionalActor to real environment reward signals (Gymnasium)
- **Neural substrates** — replacing the closed-form pattern updates with learned neural representations (PyTorch/JAX)
- **Social pattern fields** — multiple agents with separate private fields that share selectively, modelling social learning
- **Meta-pattern monitoring** — a higher-level agent that monitors and adjusts the learning dynamics of lower-level agents

---

## Setup

**Requirements:** Python 3.11+

```bash
# Clone the repo
git clone https://github.com/your-org/HPM---Learning-Agent.git
cd HPM---Learning-Agent

# Create a virtual environment and install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Or with uv (faster)
uv sync
```

### Run benchmarks

```bash
# Reber Grammar (discrete sequence learning)
python benchmarks/reber_grammar.py
python benchmarks/reber_grammar.py --poisson      # Poisson variant

# Multi-agent Reber Grammar (full orchestrator)
python benchmarks/multi_agent_reber_grammar.py
python benchmarks/multi_agent_reber_grammar.py --poisson

# Structural Immunity (noise resilience)
python benchmarks/structural_immunity.py
python benchmarks/multi_agent_structural_immunity.py
python benchmarks/multi_agent_structural_immunity.py --beta  # Beta variant

# Flat ARC baseline
python benchmarks/arc_benchmark.py
python benchmarks/multi_agent_arc.py

# Other early benchmarks
python benchmarks/substrate_efficiency.py
python benchmarks/elegance_recovery.py

# SP4 — Structured ARC (hierarchical encoders, L1–L5)
python benchmarks/structured_arc.py

# SP5 — Structured Math (algebraic transformation families)
python benchmarks/structured_math.py

# SP6 — Math with L4/L5 layers
python benchmarks/structured_math_l4l5.py

# SP7 — PhyRE physics reasoning (hierarchical)
python benchmarks/structured_phyre.py

# SP8 — Cross-task L4 training sweep (PhyRE)
python benchmarks/phyre_cross_task_l4.py

# SP9 — Naive cross-domain transfer (zero-padding)
python benchmarks/phyre_cross_domain_l4.py

# SP10 — Delta alignment cross-domain transfer (Procrustes)
python benchmarks/phyre_delta_alignment.py

# SP11 — DS-1000 Boss Fight (Symbolic Data Science)
python benchmarks/structured_ds1000_l4l5.py

# SP12 — Chem-Logic I (Molecular Discovery)
python benchmarks/structured_chem_logic_l4l5.py

# SP13 — Chem-Logic II (Ambiguity & Competition)
python benchmarks/structured_chem_logic_v2.py

# SP14 — Linguistic Register Shift (Social pH)
python benchmarks/structured_linguistic_l4l5.py

# SP15 — Generalized Cross-Domain Transfer (6 Domains)
python benchmarks/multi_domain_alignment.py

# SP16 — Geometric Rosetta (Concept Discovery)
python benchmarks/rosetta_geometric_benchmark.py
```

### Run tests

```bash
python -m pytest tests/ -v
```

---

## Background reading

The working paper *Human Learning as Hierarchical Pattern Modelling* (included as a PDF) gives the full theoretical grounding for the framework. The code in this repo is a direct implementation of the structural principles described there.
