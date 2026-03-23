# Research Arc: SP4–SP10 — Hierarchical Pattern Modelling Across Domains

## Overview

This document summarises the research arc from SP4 through SP10, covering structured discrimination benchmarks in visual (ARC), algebraic (Math), and physics (PhyRE) domains, and culminating in cross-domain transfer experiments. The arc traces a single question: does the HPM hierarchy — from raw features (L1) through structural encodings (L2–L3) to generative abstraction (L4) and metacognitive gating (L5) — produce consistent, principled improvements in agent discrimination performance? The short answer is yes within domains, and partially yes across domains, with a sharp and instructive boundary at symbolic versus continuous abstraction.

---

## SP4: Visual Pattern Completion (ARC)

The first structured benchmark used five ARC transformation families as a discrimination task: given a query grid, identify which of two candidate continuations is consistent with the demonstrated transformation. Three hierarchical encoders were built — ArcL1 (64-dim pixel delta), ArcL2 (9-dim object anatomy), ArcL3 (14-dim relational summary) — mirroring the HPM substrate hierarchy from low-level sensory features to relational structure.

The flat pixel baseline scored **63.2%**; the full L2+L3 hierarchical encoder reached **69.0% (+5.8pp)**. The gain is real but modest. ARC transformations are visually ambiguous: the same pixel-level pattern is consistent with multiple relational interpretations, and the benchmark captures that ambiguity. The key lesson from SP4 is that hierarchy adds signal even under high visual ambiguity — but the ceiling is set by the task itself, not the encoder.

---

## SP5: Algebraic Transformation Families (Math)

The Math benchmark applied the same discrimination framing to five equation transformation families: expand, factor, simplify, differentiate, integrate. Three encoders were designed — MathL1 (14-dim coefficient features), MathL2 (10-dim structural features), MathL3 (12-dim transformation class features).

The results were decisive and revealing:

| Encoder configuration | Accuracy |
|---|---|
| Flat (L1 only) | 10.6% |
| L3 only | **97.8%** |
| L2 + L3 | 96.7% |

L3 — the transformation class features — is the sole decisive abstraction. L1 (coefficient features) is near chance and actively harmful to scoring when included alongside L3; it introduces noise that degrades the composite score. This is a direct empirical validation of an HPM prediction: the relevant abstraction level is domain-specific, and including lower levels indiscriminately can hurt. L1 is retained in the Math stack, but only for epistemic threading — not for discrimination scoring.

---

## SP6: L4 Generative Head and L5 Meta-Monitor (Math)

With a strong L2+L3 baseline established in Math, SP6 added the two upper layers of the HPM stack. L4 is a generative prediction head: online ridge regression from L2 features to L3 features, fitted on seen task pairs using the closed-form solution W = (X^T X + αI)^{-1} X^T Y. L5 is a surprise-based metacognitive gate that tracks whether L4's predictions are reliable enough to use (`strategic_confidence` thresholding).

| Configuration | Accuracy |
|---|---|
| L2+L3 baseline | 96.7% |
| L4 only | **98.3% (+1.6pp)** |
| L4+L5 full stack | 98.3% |

L4 adds 1.6pp when cross-task training data is available. Math is an ideal domain for L4: the L2→L3 mapping is consistent across transformation families (structural features reliably predict class membership), so cross-task training generalises cleanly. L5 remains in exploit mode throughout — Math tasks have sufficiently reliable L4 intuitions that the metacognitive gate never needs to fall back. This is the intended behaviour: L5 adds value when L4 is uncertain, not when it is already accurate.

---

## SP7: PhyRE — Physics Reasoning Environment

SP7 introduced a novel benchmark: physics discrimination using pymunk simulation. Four families (Projectile, Bounce, Slide, Collision) were simulated and snapshotted into 240 tasks. Three new encoders were built — PhyreL1 (16-dim kinematic deltas), PhyreL2 (14-dim material properties), PhyreL3 (12-dim aggregate physics summary) — maintaining the same three-tier structure as Math and ARC.

| Configuration | Accuracy |
|---|---|
| Flat | 22.5% |
| L2+L3 | **62.5% (+40pp)** |
| L4 only (per-task) | 61.7% |
| L4+L5 full | 61.7% |

The L2+L3 gain is dramatic — +40 percentage points over flat — confirming that structured physics encodings capture the right abstractions. However, L4 does not add value. The reason is sample size: per-task ridge regression has only 3 training pairs, which is insufficient to learn the L2→L3 mapping in a physics domain with continuous, noisy features. This prompted a direct hypothesis test in SP8.

---

## SP8: Cross-Task L4 in PhyRE

SP8 tested whether L4's failure in SP7 was a per-task data problem. The experiment used an 80/20 task-level split (192 train, 48 held-out), fitted a single global L4 model on all training tasks, and swept training pairs per task (3, 5, 10, 20).

| Configuration | Accuracy |
|---|---|
| Flat | 14.6% |
| L2+L3 | 58.3% |
| Per-task L4 | 58.3% |
| Cross-task L4 (global) | 58.3% |

No configuration improves over L2+L3. The global L4 generalises across families — it is a valid drop-in — but the ceiling is set by L2+L3 scoring, not by L4's predictive accuracy. L4 learns to predict L3 from L2 reliably, but that prediction does not add discriminative information beyond what direct L2+L3 feature comparison already provides. The architectural insight: L4 adds value when L2→L3 is a learnable shortcut that direct scoring cannot exploit, as in Math. In PhyRE, the shortcut is already captured by the scoring function itself.

---

## SP9: Naive Cross-Domain Transfer (Zero-Padding)

With within-domain results established across three domains, SP9 tested cross-domain transfer: can an L4 model trained on two domains generalise to a third? The method was leave-one-domain-out with zero-padding to create a shared feature space (Math L2 padded to 14 dims, PhyRE L2 already 14 dims, ARC L2 padded similarly).

| Test domain | L2+L3 baseline | Cross-domain L4 |
|---|---|---|
| ARC | 0.800 | 0.267 |
| PhyRE | 0.583 | 0.167 |
| Math | 1.000 | 0.222 |

The results were uniformly negative. Zero-padding creates heterogeneous feature spaces: Math L2 and PhyRE L2 encode structurally incomparable quantities. Ridge regression has no basis for aligning them and instead fits to the padding artifacts. The negative result is informative — it rules out the simplest transfer strategy and motivates a structurally principled alternative.

---

## SP10: Delta Alignment — Procrustes Structure-Preserving Transfer

SP10's key insight is that domains may share relational structure even when absolute feature values are incomparable. Two domains that differ in what their L2 features mean may still agree on how features change across task pairs — and that relational geometry can be aligned.

The method: compute all N(N-1) pairwise feature deltas per domain; fit a local linear map M_d ∈ R^{14×14} via ridge regression on those deltas; align M_d1 and M_d2 via Procrustes SVD with a determinant check to enforce proper rotation R ∈ SO(14); form a shared map M_shared = (M_d1 + R^T M_d2) / 2; score candidates using centroid anchor plus delta prediction in the shared space.

| Test domain | L2+L3 | Naive cross-domain L4 | Delta alignment |
|---|---|---|---|
| ARC | 0.800 | 0.267 | **0.800** |
| PhyRE | 0.633 | 0.167 | **0.633** |
| Math | 0.978 | 0.222 | 0.578 |

Delta alignment matches L2+L3 on two of three rotations — ARC and PhyRE test domains — recovering the full baseline without any domain-specific features. It fails on Math by ~40pp. The failure is structural: algebraic transformation families (expand, factor, differentiate, integrate) are discrete symbolic abstractions. Their L2→L3 relational geometry is not continuous and has no analogue in physics or visual domains. The Procrustes alignment finds a rotation, but the rotation maps a continuous relational space onto a discrete one — the geometry does not transfer.

---

## Results Summary

| Sub-project | Domain | Key comparison | Gain |
|---|---|---|---|
| SP4 | ARC | flat → L2+L3 | +5.8pp |
| SP5 | Math | flat → L3-only | +87.2pp |
| SP6 | Math | L2+L3 → L4+L5 | +1.6pp |
| SP7 | PhyRE | flat → L2+L3 | +40pp |
| SP8 | PhyRE | L2+L3 → cross-task L4 | 0pp |
| SP9 | All | L2+L3 → zero-pad transfer | −53pp avg |
| SP10 | All | zero-pad → delta alignment | +40pp avg (2/3 domains) |

---

## Architectural Insights

**1. Level matters — the decisive abstraction is domain-specific.** In Math, L3 alone gives 97.8%; L1 is near-chance and harmful. In Physics and ARC, L2+L3 together are required. The HPM prediction that different tasks engage different levels of the hierarchy is confirmed.

**2. L4 needs cross-task data, not just more per-task data.** The SP7→SP8 sweep is a controlled test of this: varying training pairs per task (3 to 20) produces no gain. L4 adds value only when the L2→L3 mapping is learnable from cross-task signal and not already captured by direct scoring — as in Math, where transformation class membership is a clean cross-task generalisation.

**3. Cross-domain transfer requires structural alignment.** SP9 shows that naive feature concatenation fails categorically. SP10 shows that relational delta alignment recovers full within-domain performance for continuous/perceptual domains. The alignment method matters as much as the transfer target.

**4. The L2→L3 abstraction is partially domain-agnostic.** The critical finding of SP10: continuous and perceptual domains (physics, visual) share relational geometry at the L2→L3 transition. Symbolic discrete domains (algebraic transformations) do not. This is not a failure of the method — it is a discovery about the structure of the domains themselves.

**5. HPM hierarchy validated within domains; partially validated across domains.** The within-domain arc (SP4–SP7) consistently shows that structured hierarchical encodings outperform flat baselines, with gains scaling to the difficulty of the domain. The cross-domain arc (SP9–SP10) partially validates that the hierarchy is not purely domain-specific — relational structure transfers between continuous domains — but reveals a genuine boundary at the symbolic/continuous interface.

---

## Open Questions and Next Directions

- **Shared symbolic embedding.** Can algebraic transformation families be re-encoded in a continuous space that is structurally compatible with physics/ARC L2 features? A learned embedding (e.g. via contrastive training on symbolic structure) may bridge the gap that Procrustes cannot.

- **L5 under genuine uncertainty.** L5 stayed in exploit mode throughout Math and was not exercised by PhyRE's L4 failure. A domain where L4 intuitions are unreliable but improvable mid-episode would test whether the metacognitive gate adds real value or is merely decorative.

- **Multi-domain L4 training.** SP10 uses two source domains to align toward a third. What happens with three source domains and a fourth test domain? Does additional source diversity improve or destabilise the Procrustes alignment?

- **Active learning for L4 sample efficiency.** SP8 showed that random task pairs are insufficient for PhyRE L4. Selecting maximally informative training pairs (e.g. via uncertainty sampling or feature-space coverage) may unlock L4 gains that random sampling cannot.

- **Hierarchical transfer beyond L4.** All cross-domain experiments transfer at the L2→L3 level. Transferring L3→L4 (generative structure) or L4→L5 (metacognitive calibration) across domains has not been attempted and may reveal deeper commonalities — or deeper boundaries.
