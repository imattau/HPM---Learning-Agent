# Human Learning as Hierarchical Pattern Modelling: A Computational Validation

**Matt Thomson**  
imatt.au@protonmail.com

## Abstract

The Hierarchical Pattern Modelling (HPM) framework proposes that human learning is best understood as the progressive construction and refinement of hierarchical patterns across multiple substrates, driven by a committee of evaluators and shaped by social pattern fields. This paper presents a computational implementation of HPM and validates its core predictions through five targeted experiments. We demonstrate that HPM agents: (1) develop hierarchical abstractions that are robust to surface distractors but sensitive to structural changes; (2) leverage institutional replication pressure to unlearn spurious correlations faster than isolated agents; (3) acquire meta-cognitive policies that modulate exploration in non-stationary environments; (4) exhibit creative compositional generalization through structural recombination of learned primitives; and (5) learn to selectively offload computation to external tools based on cost-accuracy trade-offs. These results provide strong empirical support for HPM as a unifying account of learning, abstraction, social convergence, meta-learning, creativity, and extended cognition.

## 1. Introduction

Human learning remains one of the most remarkable yet poorly unified phenomena in cognitive science. Researchers have produced detailed accounts of reinforcement learning, memory consolidation, concept formation, and language acquisition, but a coherent framework linking these processes has been elusive. The Hierarchical Pattern Modelling (HPM) hypothesis [Thomson, 2026] proposes that diverse cognitive phenomena can be understood as variations of a single underlying activity: the discovery, stabilization, manipulation, and recombination of hierarchical patterns across multiple levels of abstraction and substrates.

This paper provides the first comprehensive computational validation of the HPM framework. We implement the core architectural commitments—patterns as generative models, evaluator-driven selection, population dynamics with recombination, and multi-agent pattern fields—and test the framework's predictions across five experiments targeting different aspects of cognition. Our results confirm that HPM agents exhibit hallmarks of intelligent learning: abstraction of causal structure from surface noise, institutional convergence on truth, meta-cognitive adaptation, creative recombination, and flexible tool use.

## 2. The HPM Framework

We briefly summarize the key components of HPM as implemented in our experiments. For a full theoretical treatment, see Thomson [2026].

### 2.1 Patterns and Substrates

A **pattern** is a generative model that assigns probabilities to streams of experience and supports prediction, action, and simulation. Patterns are not restricted to neural representations; they can be encoded in symbolic rules, motor routines, or external tools (substrates). Each pattern maintains internal parameters and a set of evaluator scores.

### 2.2 Evaluators

Patterns are selected and weighted by a committee of evaluators, including:
- **Epistemic evaluator:** Prediction accuracy (negative log-likelihood).
- **Curiosity evaluator:** Learning progress or compression improvement.
- **Coherence evaluator:** Internal consistency and independence from surface features.
- **Affective evaluator:** Reward signals from the environment.
- **Social evaluator:** Feedback from other agents in a pattern field.

The total score of a pattern is a weighted sum of these evaluator contributions. Patterns with higher scores gain population weight through replicator dynamics.

### 2.3 Population Dynamics and Recombination

Patterns compete within a population. Weights evolve via a replicator equation with inhibition (to prevent redundancy) and decay (forgetting). Occasionally, high-weight patterns are recombined—splicing structural components—to create novel patterns. Successful recombinations receive an "insight" boost, accelerating their integration.

### 2.4 Pattern Fields and Institutions

Multiple HPM agents can form a **pattern field**, where agents publish their top patterns for cross-validation. Patterns that replicate successfully on other agents' private data receive positive social signals; those that fail are penalized. This institutional pressure simulates scientific peer review and accelerates convergence on veridical structure.

### 2.5 Meta-Patterns

Higher-order patterns can observe the performance of lower-level patterns and adjust evaluator weights or exploration policies. This recursive architecture enables meta-learning and adaptive strategy selection.

## 3. Computational Implementation

We implemented HPM in Python using PyTorch and Pyro for probabilistic modeling, with Ray for distributed multi-agent simulation. The core classes are:

- : Abstract base class defining the pattern interface (, , , ).
-  / : Neural patterns with hierarchical latent variables (, ) for regression tasks.
- : Manages a population of patterns, applying replicator dynamics, recombination, and pruning.
- : Computes and updates evaluator scores.
- : Coordinates multiple agents, replication testing, and social signal propagation.

All code is available at https://github.com/imattau/HPM---Learning-Agent.

## 4. Experiments

We designed five experiments to test specific predictions of the HPM framework. Each experiment compares HPM agents against appropriate baselines or control conditions.

### 4.1 Experiment 1: Hierarchical Abstraction and Sensitivity Asymmetry

**Prediction:** HPM agents will develop patterns that are robust to surface feature variations but sensitive to changes in underlying causal structure.

**Task:** Regression with two structural features and two surface features.

**Results:**
- Surface ΔMSE: 0.021 (near zero)
- Structural ΔMSE: –0.197 (significant change)
- Asymmetry ratio: ≈ 9.4

**Interpretation:** The agent successfully learned to ignore irrelevant surface distractors while remaining sensitive to the true causal variables.

### 4.2 Experiment 2: Institutional Pressure and Superstition Unlearning

**Prediction:** Agents in a pattern field with robust replication will unlearn spurious correlations faster than isolated agents.

**Task:** Binary prediction with one causal feature and one spurious feature ("Monday effect").

**Results:**
- Isolated agents: Slow, linear decline in spurious importance.
- Pattern Field agents: Sharp drop within 10–15 steps.

**Interpretation:** Institutional cross-validation applies selective pressure that individual epistemic updating cannot match.

### 4.3 Experiment 3: Meta-Cognitive Control in Non-Stationary Environments

**Prediction:** HPM agents can acquire meta-patterns that modulate exploration/exploitation based on environmental volatility.

**Task:** Regression in an environment alternating between stable (fixed weights) and volatile (weights change) regimes.

**Results:**
- Meta-Pattern Agent learned to reduce curiosity during stable periods and increase it during volatile periods, achieving the lowest cumulative regret.

**Interpretation:** The meta-pattern successfully extracted regularities about the environment's volatility and adjusted exploratory policy accordingly.

### 4.4 Experiment 4: Creativity via Structural Recombination

**Prediction:** HPM agents can recombine structural components of previously learned patterns to solve novel compositional tasks.

**Task:** Learn primitives  and  separately, then tested on composition .

**Results:**
- HPM + Recombination: Rapid MSE drop, reaching low error within 50–100 steps.
- HPM (no recombination): Slower learning.
- Baseline MLP: High error.

**Interpretation:** Recombination enables compositional generalization by reusing functional modules.

### 4.5 Experiment 5: Tool Use and External Substrate Shifting

**Prediction:** HPM agents can learn to selectively offload computation to external tools when evaluator pressures favor cost-accuracy trade-offs.

**Task:** Regression on a computationally expensive function with a costly external oracle tool available.

**Results:**
- Meta-Tool learned to use the tool in high-frequency regions and internal neural patterns in smooth regions, achieving the lowest cost-weighted error.

**Interpretation:** HPM agents flexibly shift between substrates based on evaluator pressures.

## 5. General Discussion

The five experiments collectively validate the core trajectory proposed by the HPM framework. From surface-invariant abstraction to institutional truth-seeking, meta-cognitive adaptation, creative recombination, and extended tool use, HPM agents exhibit the hallmarks of intelligent learning without task-specific engineering.

## 6. Conclusion

We have presented a computational implementation of the Hierarchical Pattern Modelling framework and empirically validated its core predictions. The results demonstrate that HPM is a viable and unifying account of learning, abstraction, social convergence, meta-learning, creativity, and extended cognition.

## References

[Thomson, 2026] Thomson, M. (2026). Human Learning as Hierarchical Pattern Modelling. Working Paper v1.25.
