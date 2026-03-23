# Benchmarks — HPM Learning Agent

This document provides a comprehensive analysis of all benchmarks in this repository, covering what each test measures, how it works, the results obtained, and what those results reveal about hierarchical pattern learning. The benchmarks form a progression: early tests validate the core multi-agent framework, while the SP series (SP4–SP10) systematically explore hierarchical encoding across three distinct reasoning domains.

For a summary table of all results, see the [root README](../README.md). This document provides the analytical depth behind those numbers.

---

## Table of Contents

1. [Overview — what the benchmarks test and why](#overview)
2. [Early benchmarks — core framework validation](#early-benchmarks)
   - [Reber Grammar](#reber-grammar--discrete-sequence-learning)
   - [Structural Immunity](#structural-immunity--noise-resilience)
   - [ARC flat baseline](#arc--abstract-visual-reasoning-flat-baseline)
   - [Substrate Efficiency](#substrate-efficiency)
   - [Elegance Recovery](#elegance-recovery)
3. [SP series — hierarchical encoder ablations](#sp-series--hierarchical-encoder-ablations)
   - [SP4 — Structured ARC](#sp4--structured-arc)
   - [SP5 — Structured Math](#sp5--structured-math)
   - [SP6 — Math L4/L5](#sp6--math-l4l5)
   - [SP7 — PhyRE Physics](#sp7--phyre-physics-reasoning)
   - [SP8 — Cross-task L4](#sp8--cross-task-l4-phyre)
   - [SP9 — Naive cross-domain transfer](#sp9--naive-cross-domain-transfer)
   - [SP10 — Delta alignment](#sp10--delta-alignment-procrustes)
4. [Cross-benchmark analysis — patterns across domains](#cross-benchmark-analysis)
5. [Architectural insights — what the results reveal about HPM](#architectural-insights)
6. [How to run all benchmarks](#how-to-run)

---

## Overview

The benchmarks test three separable claims made by the HPM framework:

**Claim 1: Hierarchical structure improves learning.** A system that builds explicit representations at multiple levels of abstraction should outperform a flat single-level system, because higher levels capture structure that is invisible at lower levels.

**Claim 2: The decisive abstraction level is domain-specific.** Different domains place their discriminative information at different levels of the hierarchy. A well-designed hierarchical system should route information to whichever level carries signal, without requiring manual specification.

**Claim 3: Cross-domain transfer is possible if representations are aligned at the right structural level.** Naive transfer fails because embedding spaces are not aligned across domains. But if the *relational geometry* of embeddings is preserved via alignment, transfer should partially succeed.

The benchmarks are designed to test each claim with controlled ablations, so that the contribution of each architectural component can be isolated.

---

## Early Benchmarks — Core Framework Validation

These benchmarks were developed before the SP series and validate the foundational multi-agent framework: the PatternField, the StructuralLawMonitor, the RecombinationStrategist, and the DecisionalActor.

### Reber Grammar — discrete sequence learning

**File:** `reber_grammar.py`, `multi_agent_reber_grammar.py`

**What it tests:** Whether the agent can learn genuine structural regularities in discrete sequences — as opposed to surface-level symbol frequencies. The Reber Grammar is a finite-state automaton over a 7-symbol alphabet. Valid sequences must follow transition rules that are not directly observable; random sequences do not. The test is whether the agent assigns systematically lower probability to invalid sequences.

**How it works:** The agent trains on 500 valid sequences, then discriminates 100 valid/invalid pairs. The AUROC measures how well the agent's negative log-likelihood scores separate the two classes. The multi-agent version uses a 3-agent ensemble sharing a PatternField.

| Setup | AUROC | NLL separation |
|---|---|---|
| Single agent (CategoricalPattern) | 0.934 | 16.82 |
| 3-agent ensemble | 0.955 | — |

Pass criteria: AUROC > 0.80, separation > 5.0. Both configurations pass with margin.

**Interpretation:** The CategoricalPattern model learns positional symbol distributions that encode the grammar's transition constraints. The key evidence for genuine structure learning (rather than frequency matching) is the large NLL separation (16.82): invalid sequences are not just slightly less probable, they are strongly penalised. The 3-agent ensemble adds approximately +2pp AUROC — a modest but consistent gain from the PatternField's pattern competition, where agents that have learned tighter transition constraints receive more influence.

The Reber Grammar is a clean test because the structural rules are exact: there are no ambiguous sequences. This makes it a good calibration point for whether the substrate and dynamics are working correctly before moving to noisier domains.

---

### Structural Immunity — noise resilience

**File:** `structural_immunity.py`, `multi_agent_structural_immunity.py`

**What it tests:** Whether the system can recover its learned representations after a disruption event — specifically, a burst of uniform noise that temporarily overwhelms the signal. This maps to HPM's prediction that patterns with high evaluator weight should be resilient to temporary interference, because the shared PatternField acts as a distributed buffer.

**How it works:** Three-phase protocol over a single Gaussian-distributed stream: 500 steps of stable signal (mean=[1,2,3], cov=0.1I), 20 steps of uniform noise (covering all signal values), then 500 steps of recovery signal. T_rec is the number of recovery steps before accuracy returns to within 5% of the pre-noise baseline.

| Setup | T_rec | Result |
|---|---|---|
| Single agent (Gaussian) | ≤ 100 | IMMUNE |
| 2-agent shared PatternField (Gaussian) | ≤ 100 | IMMUNE |
| 2-agent shared PatternField (Beta) | 5 | IMMUNE |

Pass criterion: T_rec ≤ 100.

**Interpretation:** All configurations recover well within the threshold. The Beta variant's T_rec of 5 is notably fast: the Beta distribution's bounded support [0,1] means it has less to update when recovering from a noise burst — its sufficient statistics return to the pre-noise regime quickly because the noise cannot permanently shift the Beta parameters far from the signal's true distribution.

The single-agent and shared-field Gaussian results are comparable, which suggests that the PatternField's shared memory is not the dominant recovery mechanism here — the individual agent's own pattern dynamics are sufficient. The value of the shared field is more likely to emerge in settings where the noise is prolonged or where different agents experience different noise types.

---

### ARC — abstract visual reasoning (flat baseline)

**File:** `arc_benchmark.py`, `multi_agent_arc.py`

**What it tests:** Pattern generalisation on visual transformation tasks from the Abstraction and Reasoning Corpus (ARC-AGI, train split). Given 3–5 input→output grid pairs as training examples, the agent must identify the correct output from 4 distractors drawn from other tasks.

**Important:** This is a discrimination task, not the ARC solving task. The agent does not produce an output grid — it selects the correct one from a candidate set. This tests whether the agent can build a useful internal representation of the transformation rule, not whether it can execute the rule. Distractors are drawn from other tasks, so they tend to follow different transformation rules — making this a test of transformation-rule discrimination.

**How it works:** Each task's training pairs are encoded via a fixed random projection of the pixel-delta (output − input) to 64 dimensions. The agent builds a pattern from the training deltas, then scores each candidate by its probability under that pattern. The highest-scoring candidate is selected. 58 of 400 ARC tasks are excluded because their grids exceed 20×20.

| Setup | Tasks | Accuracy | vs Chance |
|---|---|---|---|
| Single agent | 342 / 400 | 65.8% | +45.8% |
| 2-agent ensemble | 342 / 400 | 65.5% | +45.5% |

Chance baseline: 20% (5-way discrimination including the correct output and 4 distractors).

**Interpretation:** +45.5pp over chance is a substantial margin for a system that never operates on raw pixels and uses a fixed random projection with no learned visual features. It indicates that the transformation delta — even as a random projection — carries enough structural signal to distinguish transformation types. The multi-agent ensemble is marginally lower than single-agent, not because the PatternField hurts, but because each agent sees fewer training pairs (even/odd split), and the field's sharing partially but not fully compensates for the data reduction.

This benchmark establishes the flat baseline that the SP4 hierarchical encoder will later improve upon.

---

### Substrate Efficiency

**File:** `substrate_efficiency.py`

**What it tests:** Whether the agent discovers a compact representation of a structured but redundant data stream, without being told how many components to use. The data is 3 overlapping Gaussian clusters in 16-dimensional space; an optimal model needs 3 components.

**How it works:** The agent's complexity/accuracy trade-off is compared against Gaussian Mixture Models (GMM) at k=1–5 on a Pareto frontier. Points on the frontier achieve more accuracy per unit of complexity than points off the frontier.

| Model | Complexity | Accuracy | Pareto frontier |
|---|---|---|---|
| HPM agent | 0.20 | 0.39 | Yes |
| GMM k=1 | 0.00 | 0.00 | Yes |
| GMM k=2 | 0.23 | 0.31 | No |
| GMM k=3 | 0.50 | 0.49 | Yes |
| GMM k=4 | 0.77 | 0.71 | Yes |
| GMM k=5 | 1.00 | 1.00 | Yes |

**Interpretation:** The HPM agent reaches the Pareto frontier between GMM k=1 (trivially compact, zero accuracy) and GMM k=3 (correct structure, high accuracy). It achieves more accuracy per unit of complexity than GMM k=2, which uses a comparable number of components but is a fixed model rather than an adaptive one. The result is not that the HPM agent outperforms GMM — it does not match GMM k=3's accuracy — but that it is efficiently positioned on the frontier without being told the correct k.

This reflects HPM's prediction that good learning systems should not just fit data, they should do so parsimoniously. A system that bloats its representational complexity to match data perfectly has not understood the structure; it has overfit it.

---

### Elegance Recovery

**File:** `elegance_recovery.py`

**What it tests:** A subtle property — whether the agent can distinguish between a true generative law and a near-identical distractor law after training. This tests structural specificity: not just accuracy on training data, but whether the system has converged on the specific generative mechanism that produced the data.

**How it works:** The agent trains on y = x²/(1+x) for 1500 steps. The top-weighted pattern is then evaluated on a true-law test set versus a distractor set generated by y = x². A positive gap means the agent's learned pattern assigns higher probability to the true law than to the distractor.

| Steps | Recombinations | NLL (true law) | NLL (distractor) | Gap | Result |
|---|---|---|---|---|---|
| 1500 | 17 | −3.33 | −3.27 | +0.06 | RECOVERED |

**Interpretation:** The gap (+0.06) is small but directionally correct and consistently positive across runs. The agent has not just fit a smooth curve — it has converged on the specific functional form that governs the training data, to a degree sufficient to penalise the near-identical distractor. The 17 recombinations during training indicate that the RecombinationStrategist is actively exploring the pattern space, not just refining a single initial pattern. This is the beginning of what HPM calls a "generative rule" — a pattern that captures not just what happened, but the underlying structure that produced it.

---

## SP Series — Hierarchical Encoder Ablations

The SP (structured pattern) series tests a five-level hierarchical encoder stack applied to three distinct domains: visual reasoning (ARC), symbolic algebra (Math), and simulated physics (PhyRE). Each benchmark ablates which levels of the hierarchy are active, allowing direct measurement of each level's contribution to task performance.

### The five-level encoder stack

| Level | Role | ARC encoding | Math encoding | Physics encoding |
|---|---|---|---|---|
| L1 | Pixel/surface statistics | Mean pixel delta, variance | Raw coefficient values | Raw state variables |
| L2 | Object/structural anatomy | Object count, bounding boxes, colour distributions | Term structure, operator counts | Physical quantities (velocity, mass, angle) |
| L3 | Transformation rule | Delta structure: what changed and how | Transformation family signature | Force-outcome relationship |
| L4 | Relational meta-rule | Cross-example rule consistency | Family-level algebraic constraint | Cross-scenario physical regularity |
| L5 | Strategy gate | Reliability-weighted combination (gamma) | Reliability-weighted combination (gamma) | Reliability-weighted combination (gamma) |

L5 is not a feature extractor — it is a gating mechanism that learns to weight the contributions of L1–L4 based on their predictive reliability. When a lower level is consistently accurate, L5 sets gamma → 1.0 and passes it through without modification.

---

### SP4 — Structured ARC

**File:** `structured_arc.py`

**What it tests:** Whether hierarchical encoding improves ARC discrimination, and which level of the hierarchy carries the most signal. The flat baseline (63.2%) uses a single fixed random projection; the hierarchical version builds a structured embedding at each level.

| Configuration | Accuracy | Delta vs flat |
|---|---|---|
| flat | 63.2% | baseline |
| l1_only | 63.2% | 0pp |
| l2_only | 46.5% | −16.7pp |
| full (L1–L3) | 69.0% | +5.8pp |
| l4_only | **88.6%** | **+25.4pp** |
| l4+l5 (full stack) | 88.6% | +25.4pp |

**Findings:**

L1 alone matches the flat baseline exactly — the raw delta statistics carry the same signal as the random projection baseline. L2 alone actually *hurts* performance (−16.7pp): object anatomy without transformation context is misleading, because tasks with different transformation rules can have similar object layouts.

The full L1–L3 stack recovers the loss and adds +5.8pp over flat. But the large gain comes from L4 (+25.4pp solo). This is a striking result: the meta-relational level — which encodes *consistency of the transformation rule across training examples* — is the decisive discriminative signal for ARC.

**Why L4 is decisive for ARC:** ARC tasks are defined by a rule that holds consistently across all training pairs. L4 measures the coherence of the transformation signature across examples. If the same transformation rule applies to all training pairs (as it must in a valid ARC task), L4 will produce a tight, consistent representation. Distractors from other tasks will have different rules and therefore different L4 signatures. This coherence signal is not available at L3 (which encodes each example independently) or at L1/L2 (which are too low-level to capture rule consistency).

L5 adds nothing over L4 alone (88.6% ties) because L4 is already highly reliable — L5 correctly recognises this and sets gamma=1.0, passing L4 through unchanged.

---

### SP5 — Structured Math

**File:** `structured_math.py`

**What it tests:** Whether hierarchical encoding improves classification of algebraic transformation families (linear, quadratic, exponential, logarithmic), and which level carries signal for symbolic reasoning.

| Configuration | Accuracy | Delta vs flat |
|---|---|---|
| flat | 10.6% | baseline |
| l1_only | 10.6% | 0pp |
| l2_only | 66.7% | +56.1pp |
| l3_only | **97.8%** | **+87.2pp** |
| l2+l3 | 96.7% | +86.1pp |
| full (L1–L3) | 96.7% | +86.1pp |

**Findings:**

The flat baseline of 10.6% is near chance (10% for a 10-way discrimination) — raw coefficient values without structural encoding carry almost no discriminative signal. L2 jumps to 66.7% by encoding term structure, but L3 alone reaches 97.8%: the transformation family signature is the decisive level for symbolic reasoning.

**Why L3 is decisive for Math (contrast with ARC):** In symbolic algebra, the transformation family is directly encoded in the functional form — the relationship between input and output variables is determined by the algebraic structure, not by consistency across examples. L3's transformation rule encoding captures this directly. L4, which measures consistency *across* examples, adds relatively little because L3 already has near-perfect discrimination: there is no additional signal in cross-example consistency when each example already reveals the family.

The slight drop from L3_only (97.8%) to l2+l3 (96.7%) and full (96.7%) suggests a mild interference effect: adding L2 introduces some noise into the combined representation that marginally reduces accuracy from the L3-only peak. This is consistent with HPM's prediction that lower-level features can sometimes compete with higher-level representations when the higher level already provides complete discrimination.

---

### SP6 — Math L4/L5

**File:** `structured_math_l4l5.py`

**What it tests:** Whether adding L4 and L5 to the already-high SP5 baseline produces further gains in symbolic reasoning.

| Configuration | Accuracy |
|---|---|
| l2+l3 (SP5 baseline) | 96.7% |
| l4_only | **98.3%** |
| l4+l5 (full stack) | 98.3% |

**Findings:**

L4 adds +1.6pp over the strong SP5 baseline — a small but real gain. This is because L4's cross-example consistency signal can resolve the rare ambiguous cases where L3 alone is uncertain (near the boundaries between families). L5 again stays at gamma=1.0, deferring to a reliable L4.

The combined picture for Math: L3 does the heavy lifting (87.2pp gain), L4 refines it (+1.6pp), L5 is correctly passive. This is a well-calibrated hierarchy where each level contributes proportionally to the signal available at that level of abstraction.

---

### SP7 — PhyRE Physics Reasoning

**File:** `structured_phyre.py`

**What it tests:** Hierarchical encoding applied to simulated physics tasks — four families (Projectile, Bounce, Slide, Collision), 60 tasks each, 240 total. The task is to identify the correct physical outcome from distractors generated by other scenarios in the same family.

| Configuration | Accuracy | Delta vs flat |
|---|---|---|
| flat | 22.5% | baseline |
| l2+l3 | **62.5%** | **+40.0pp** |
| l4_only | 61.7% | +39.2pp |
| l4+l5 (full stack) | 61.7% | +39.2pp |

**Findings:**

The flat baseline (22.5%) is low but above chance (20% for a 5-way discrimination) — raw physical state variables carry minimal discriminative signal without structural encoding. L2+L3 delivers a large +40pp gain by encoding physical quantities and force-outcome relationships. But L4 provides no additional gain over L2+L3 — in sharp contrast to ARC, where L4 was the decisive level.

**Why L4 does not help for PhyRE:** In ARC, L4 captures cross-example consistency of a transformation rule — and this consistency is highly discriminative because ARC tasks are defined by a consistent rule. In physics, the "rule" (Newton's laws) is the same across all scenarios. What varies is the *configuration*: the initial conditions, masses, angles, and velocities. Two Projectile scenarios can have identical physical laws but very different outcomes depending on initial conditions. L4's cross-example consistency signal measures rule coherence, not configurational identity — and since the rule is always "the same" (classical mechanics), L4 cannot discriminate between scenarios from different families that happen to use similar physical quantities.

This points to a fundamental difference between the domains: ARC is a rule-discrimination task, while PhyRE is a configuration-discrimination task. L4 is well-suited to the former and adds little to the latter.

L5 stays at gamma=1.0 (deferring to L4), but since L4 itself provides no discriminative gain, L5 is effectively gating a non-informative level — a principled but ultimately unhelpful response to a level that has been asked to do more than its structural design allows.

---

### SP8 — Cross-task L4 (PhyRE)

**File:** `phyre_cross_task_l4.py`

**What it tests:** Whether training the L4 encoder across all physics families jointly (cross-task L4) — rather than within each family separately — improves generalisation. The hypothesis is that cross-family training might expose family-invariant physical regularities at L4.

| Configuration | Accuracy |
|---|---|
| l2+l3 (SP7 baseline) | 62.5% |
| cross_task_l4 | 58.3% |

**Findings:**

Cross-task L4 training does not improve over the within-domain L2+L3 baseline — it slightly *hurts* performance (−4.2pp). This rules out the hypothesis that family-invariant physical regularities exist at L4 in this representation. The families (Projectile, Bounce, Slide, Collision) are physically distinct enough that cross-family L4 training introduces interference rather than transfer.

This result is consistent with SP7's finding that L4 adds nothing over L2+L3 in physics. The configuration-based discrimination in PhyRE is simply not addressable at the meta-relational level, regardless of whether L4 is trained within or across families.

---

### SP9 — Naive Cross-Domain Transfer

**File:** `phyre_cross_domain_l4.py`

**What it tests:** Whether representations learned in one domain can be transferred to another via zero-padding — concatenating a source-domain embedding with zeros to match the target domain's embedding dimension, then using the combined embedding for target-domain discrimination.

Three transfer directions were tested:

| Transfer direction | l2+l3 baseline | cross_domain | Delta |
|---|---|---|---|
| Math + PhyRE → ARC | 80.0% | 26.7% | −53.3pp |
| Math + ARC → PhyRE | 58.3% | 16.7% | −41.6pp |
| PhyRE + ARC → Math | 100% | 22.2% | −77.8pp |

**Findings:**

Naive zero-padding collapses performance in all three directions, in every case reducing accuracy below what the within-domain baseline achieves. The cross-domain embeddings are not just unhelpful — they actively interfere with the target-domain representations.

**Why zero-padding fails:** Zero-padding assumes that source-domain embeddings occupy a subspace compatible with the target-domain embedding space. This assumption does not hold: the embedding spaces of Math, PhyRE, and ARC are independently constructed with no shared structure. Concatenating a Math embedding with a PhyRE embedding and then appending zeros does not create a richer representation — it creates a high-dimensional vector where most of the dimensions are uninformative zeros, diluting the signal from the target-domain dimensions and breaking whatever structure the target-domain L4 encoder has learned to exploit.

The PhyRE+ARC → Math direction is particularly severe (−77.8pp from a 100% baseline) because Math's L3 is already near-perfect — any perturbation of the embedding space degrades the tight discriminative boundary that L3 has established.

SP9 establishes the floor: naive cross-domain transfer, without alignment, is uniformly harmful. This motivates SP10.

---

### SP10 — Delta Alignment (Procrustes)

**File:** `phyre_delta_alignment.py`

**What it tests:** Whether aligning the *relational geometry* of embedding delta-vectors via Procrustes rotation can enable cross-domain transfer. Instead of concatenating raw embeddings, SP10 computes the change in representation across abstraction levels (the "delta") and aligns these deltas from source to target domains using an orthogonal rotation matrix learned from paired examples.

The Procrustes alignment finds the rotation R that minimises the Frobenius distance between source deltas (rotated) and target deltas. R is learned from paired examples where the source and target domains represent the same underlying relational structure.

| Transfer direction | l2+l3 baseline | delta_align | vs SP9 | vs baseline |
|---|---|---|---|---|
| Math + PhyRE → ARC | 80.0% | **80.0%** | +53.3pp | ties |
| Math + ARC → PhyRE | 63.3% | **63.3%** | +46.6pp | ties |
| PhyRE + ARC → Math | 97.8% | 57.8% | +35.6pp | −40.0pp |

**Findings:**

Delta alignment beats SP9 in all three directions by large margins (35–53pp). On two of three directions it *ties* the within-domain l2+l3 baseline — meaning that aligned cross-domain representations are as informative as within-domain representations for those transfer directions.

The third direction (PhyRE+ARC → Math) achieves partial recovery: it recovers 35.6pp over SP9 but falls 40pp below the Math baseline of 97.8%. This is not a failure of the alignment method per se — it reflects the structural asymmetry of the transfer: Math L3 has near-perfect within-domain performance precisely because the algebraic family structure is tightly encoded at that level. Aligning physics or visual deltas to match algebraic deltas requires a rotation that must bridge a large geometric gap, and the Procrustes solution cannot fully recover the algebraic structure from configurations that are not algebraically related.

**Why delta alignment succeeds (where zero-padding failed):** The key insight is that relational geometry — the *structure* of how embeddings change across abstraction levels — can be more transferable than absolute embedding positions. Two domains may have different absolute representations, but if they share a similar relational structure (e.g., "the transformation from L2 to L3 involves identifying a consistent rule"), then the delta between L2 and L3 embeddings will have a similar geometric structure across domains. Procrustes rotation finds the best rigid alignment of this geometry, without scaling or distorting the relationships.

The success on 2/3 directions suggests that ARC and Math share enough relational structure that their delta geometries are alignable, and similarly for ARC and PhyRE. The partial failure on Math→PhyRE/ARC→Math suggests the algebraic domain's relational geometry is more distinctive — possibly because symbolic structure has less overlap with visual or physical structure than visual and physical structure have with each other.

---

## Cross-Benchmark Analysis

### The decisive abstraction level varies by domain

The most important cross-domain finding is that the abstraction level that carries discriminative signal differs systematically across domains:

| Domain | Decisive level | Accuracy at decisive level | Interpretation |
|---|---|---|---|
| Math | L3 (transformation rule) | 97.8% | Algebraic family is directly encoded in functional form |
| ARC | L4 (relational meta-rule) | 88.6% | Cross-example rule consistency is the discriminative signal |
| PhyRE | L2+L3 (structural encoding) | 62.5% | Configuration variety limits the ceiling; no higher-level gain |

This is not a failure of any level — it reflects the structural character of each domain. Math is a rule-classification problem where the rule is visible at L3. ARC is a rule-consistency problem where the rule must be inferred from cross-example coherence at L4. PhyRE is a configuration-discrimination problem where the relevant signal is in the physical quantities and their relationships, not in rule consistency.

### L5 is a reliable null result

Across all three domains and all configurations tested, L5 (the strategy gate) consistently sets gamma=1.0 when lower levels are reliable and has no information to add when lower levels are uninformative. This is the correct behaviour for a reliability-weighted gate: it does not add noise when the underlying signal is already clean, and it cannot synthesise information that is absent at lower levels.

The absence of L5 gain is not evidence that L5 is useless — it is evidence that the benchmark tasks are not in the regime where L5 would add value. L5 should be most useful in tasks where different lower levels are reliable in different conditions, requiring adaptive weighting. None of the current benchmarks create this condition. Future benchmarks with noisy or inconsistent training examples should test this.

### Cross-domain transfer is geometrically possible but domain-dependent

SP9 shows that naive concatenation always hurts. SP10 shows that Procrustes alignment of delta-vectors recovers the baseline on 2/3 directions. The third direction (Math as target) fails to match the baseline because Math's L3 representation is highly discriminative on its own — adding aligned cross-domain deltas introduces perturbations that outweigh the potential benefit.

This suggests a generalisation: cross-domain transfer via delta alignment works best when the target-domain baseline is not already near ceiling. If the target domain has high within-domain performance (Math: 97.8%), there is less room for cross-domain transfer to add value, and the alignment noise is more likely to hurt than help.

---

## Architectural Insights

### 1. Epistemic threading: information must flow through the hierarchy, not bypass it

A recurring pattern in the ablations is that lower levels alone often perform worse than expected, while higher levels alone often perform better. L2 alone on ARC actually hurts (−16.7pp) because object anatomy without transformation context is misleading. L3 alone on Math achieves 97.8% because it captures the complete discriminative structure.

This suggests that the hierarchy should not be treated as a set of independent feature extractors — it is a processing pipeline where each level refines the output of the one below. Skipping levels (L2 without L1, or L4 without L2+L3 as context) can produce representations that are partial in ways that are worse than no representation at all.

In HPM terms: patterns at each level are defined by the regularities they detect in the output of lower-level patterns. A level-2 pattern that does not receive level-1 input is not a "level-2 pattern" — it is a differently-constructed level-1 pattern with different features, and it may not generalise as expected.

### 2. Rule-discrimination vs. configuration-discrimination tasks require different architectures

ARC and Math are fundamentally rule-discrimination tasks: the right answer is determined by identifying the correct rule (transformation type, algebraic family). PhyRE is a configuration-discrimination task: the right answer is determined by the specific initial conditions and dynamics of each scenario.

Rule-discrimination tasks benefit from higher abstraction levels (L4 for ARC, L3 for Math) because rules are precisely the kind of structure that higher levels are designed to capture. Configuration-discrimination tasks saturate at L2+L3 because the relevant information (specific physical quantities) is already fully captured at those levels, and higher levels abstract away from the configuration details that distinguish scenarios.

This has a direct design implication: when building an HPM agent for a new domain, the first question should be "what is the discriminative information, and at what level of abstraction does it live?" This determines which levels of the hierarchy to invest in.

### 3. The geometry of relational transfer

SP10's partial success with Procrustes delta alignment points toward a deeper principle: transfer is possible when the *relational geometry* of representations is compatible across domains, even when the absolute representations are not. This is consistent with HPM's prediction that pattern fields at higher levels of abstraction should be more transferable than lower-level substrates, because higher levels encode structural relationships rather than domain-specific features.

The Procrustes alignment is a rigid rotation — it can align similar geometries but cannot reshape them. Future work should explore whether a learned (non-rigid) alignment could recover the Math→target direction, where the geometric gap is large enough that a rigid rotation is insufficient.

### 4. Hierarchy is not universally beneficial — it must match domain structure

The L2_only result on ARC (−16.7pp) and the cross-task L4 result on PhyRE (−4.2pp) are important negative results. They show that adding hierarchy indiscriminately can hurt performance, not just fail to help. This is consistent with HPM's view that hierarchy is a structural hypothesis that must be validated against the domain's actual pattern structure — it is not a free improvement.

The practical implication: hierarchical agents should include ablation protocols to verify that each level is contributing positively before deploying the full stack.

---

## How to Run

All benchmarks assume Python 3.11+ and a virtual environment with dependencies installed.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Early benchmarks

```bash
# Reber Grammar — discrete sequence learning
python benchmarks/reber_grammar.py
python benchmarks/reber_grammar.py --poisson      # Poisson variant
python benchmarks/multi_agent_reber_grammar.py
python benchmarks/multi_agent_reber_grammar.py --poisson

# Structural Immunity — noise resilience
python benchmarks/structural_immunity.py
python benchmarks/multi_agent_structural_immunity.py
python benchmarks/multi_agent_structural_immunity.py --beta

# ARC flat baseline
python benchmarks/arc_benchmark.py
python benchmarks/multi_agent_arc.py

# Substrate Efficiency
python benchmarks/substrate_efficiency.py

# Elegance Recovery
python benchmarks/elegance_recovery.py
```

### SP series — hierarchical encoder ablations

```bash
# SP4 — Structured ARC (L1–L5 ablations)
python benchmarks/structured_arc.py

# SP5 — Structured Math (algebraic transformation families)
python benchmarks/structured_math.py

# SP6 — Math with L4/L5
python benchmarks/structured_math_l4l5.py

# SP7 — PhyRE physics reasoning
python benchmarks/structured_phyre.py

# SP8 — Cross-task L4 training (PhyRE)
python benchmarks/phyre_cross_task_l4.py

# SP9 — Naive cross-domain transfer (zero-padding)
python benchmarks/phyre_cross_domain_l4.py

# SP10 — Delta alignment cross-domain transfer (Procrustes)
python benchmarks/phyre_delta_alignment.py
```

### Run all tests

```bash
python -m pytest tests/ -v
```

---

## Summary of all results

| Benchmark | Best configuration | Result | Key finding |
|---|---|---|---|
| Reber Grammar | 3-agent ensemble | AUROC 0.955 | Structural regularities learned with margin |
| Structural Immunity | 2-agent Beta | T_rec = 5 | Fast recovery via shared PatternField |
| ARC flat | Single agent | 65.8% (+45.8pp) | Transformation delta carries strong signal |
| Substrate Efficiency | HPM agent | Pareto frontier | Efficient representation without known k |
| Elegance Recovery | Single agent | Gap +0.06, RECOVERED | Structural specificity beyond smooth fit |
| SP4 Structured ARC | L4_only | 88.6% (+25.4pp) | L4 meta-relational level is decisive for ARC |
| SP5 Structured Math | L3_only | 97.8% (+87.2pp) | L3 transformation rule is decisive for Math |
| SP6 Math L4/L5 | L4_only | 98.3% (+1.6pp over SP5) | L4 refines near-ceiling baseline |
| SP7 PhyRE | L2+L3 | 62.5% (+40pp) | L3 captures physics structure; L4 adds nothing |
| SP8 Cross-task L4 | — | 58.3% (below baseline) | Cross-family training hurts in PhyRE |
| SP9 Naive transfer | — | 16.7–26.7% (all NEGATIVE) | Zero-padding destroys cross-domain signal |
| SP10 Delta alignment | — | Ties baseline 2/3 directions | Procrustes alignment enables partial transfer |
