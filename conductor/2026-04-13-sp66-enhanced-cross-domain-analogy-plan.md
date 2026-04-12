# SP66: Enhanced Cross-Domain Structural Analogy (Design & Plan)

## 1. Objective
Implement Experiment 66 (`experiment_cross_domain_analogy_enhanced.py`), which extends the SP64 cross-domain analogy transfer with three HPM‑aligned enhancements:
- **Pattern Density Tracking** (HPM Appendix A.8): Track D(h) = α·C(h) + β·E(h) + γ·F(h), storing state in HFN nodes.
- **Affective Evaluator** (HPM Section 9.3 & 9.4): Manage affective state (arousal, valence, curious/anxious) as HFN nodes to modulate pattern persistence and exploration.
- **AST-Level Substitution**: Safely transform code via `ast` rather than string replacements.

## 2. Implementation Steps
- [ ] Create `hpm_fractal_node/experiments/experiment_cross_domain_analogy_enhanced.py` and populate it with the provided complete script from the prompt.
- [ ] Ensure all imports (`TieredForest`, `MetaAwareAgent`, `ASTMacroSubstitutor`, etc.) are working properly by running the script.
- [ ] Execute the experiment to verify all 10 phases run correctly and the 6 success conditions are met.

## 3. Verification & Testing
- Run `PYTHONPATH=. .venv/bin/python hpm_fractal_node/experiments/experiment_cross_domain_analogy_enhanced.py`.
- Check that the affective persistence test (Phase 8) successfully retains a spurious pattern under anxiety.
- Check that AST substitution (Phase 9) creates valid Python code.
- Confirm full success output for the experiment.
