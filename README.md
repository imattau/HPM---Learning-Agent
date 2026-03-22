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
Compares how efficiently different pattern substrates represent learned structure — measured by description length relative to predictive accuracy.

### Elegance Recovery
Tests whether the system can recover a more parsimonious (lower description-length) solution after perturbation, analogous to human insight after confusion.

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
