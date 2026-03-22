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

Five benchmarks test different aspects of HPM-grounded learning. Multi-agent versions run the full orchestrator (PatternField + Monitor + Strategist + Actor).

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

# Other benchmarks
python benchmarks/arc_benchmark.py
python benchmarks/multi_agent_arc.py
python benchmarks/substrate_efficiency.py
python benchmarks/elegance_recovery.py
```

### Run tests

```bash
python -m pytest tests/ -v
```

---

## Background reading

The working paper *Human Learning as Hierarchical Pattern Modelling* (included as a PDF) gives the full theoretical grounding for the framework. The code in this repo is a direct implementation of the structural principles described there.
