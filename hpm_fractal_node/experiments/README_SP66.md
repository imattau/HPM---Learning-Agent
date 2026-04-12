# SP66: Enhanced Cross-Domain Structural Analogy

This experiment integrates pattern density tracking, affective evaluation, and robust AST-level substitution into the cross-domain analogy framework, adhering to HPM's fractal uniformity principle.

## 1. Objective
Demonstrate an HPM agent that not only transfers structural scaffolds across domains but also manages its own internal state (density, affect) using the same HFN primitives used for domain data.

## 2. Methodology: HPM-Aligned Enhancements

### A. Pattern Density Tracking (HPM Appendix A.8)
- **Metrics**: Tracks Connectivity (C), Reinforcement (E), and Field Amplification (F).
- **Density Equation**: $D(h) = \alpha C(h) + \beta E(h) + \gamma F(h)$.
- **Fractal Storage**: All density and usage state is stored as HFN nodes in dedicated tiered forests.
- **Persistence**: Dense patterns persist even under moderate epistemic loss, while low-density patterns are pruned.

### B. Affective Evaluator (HPM Section 9.3 & 9.4)
- **Arousal/Valence**: Global affective state is managed as an HFN node `affective:global`.
- **Anxiety/Frustration**: High arousal modulated by surprise leads to the persistence of familiar patterns even when they fit poorly (HPM §9.3).
- **Curiosity**: Exploration probability peaks at intermediate learnability levels, driving discovery of new structure (HPM §9.4).

### C. AST-Level Substitution
- **Robustness**: Replaces fragile string-based code manipulation with Python's `ast` module.
- **Scaffold Transformation**: Automatically identifies the "payload" operation in MAP and FILTER macros and substitutes it with novel domain primitives.

## 3. Results: Validation against HPM Core Principles
The experiment achieved a 100% success rate across all 10 phases:

| Phase | Test | Status | Result |
| :--- | :--- | :--- | :--- |
| **5** | Learnability Probe | **SUCCESS** | Identified 'analogous' transfer opportunity |
| **6** | MAP Transfer | **SUCCESS** | Zero-shot transfer to string domain (Depth 2) |
| **7** | FILTER Transfer | **SUCCESS** | Zero-shot transfer to string domain (Depth 2) |
| **8** | Affective Persistence| **SUCCESS** | Spurious pattern retained under anxiety (Loss 0.67) |
| **9** | AST Substitution | **SUCCESS** | Valid, executable code produced via AST |
| **10**| Robust Classification| **SUCCESS** | Correctly distinguished random/trivial/analogous |

### Analysis
By treating meta-cognitive state (density, affect) as HFN nodes, the system achieves **fractal uniformity**. This allows the same learning dynamics (surprise-driven updates, structural absorption) to apply to both domain knowledge and the agent's own cognitive strategies.

## 4. Running the Experiment
```bash
PYTHONPATH=. .venv/bin/python hpm_fractal_node/experiments/experiment_cross_domain_analogy_enhanced.py
```
