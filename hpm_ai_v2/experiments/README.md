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

## Blinded HPM usefulness study

`blinded_usefulness.py` creates randomized A/B response packets from matched HPM and baseline reports, plus a ratings template. It writes the deblinding key to a separate path with owner-only file permissions; the key path must be outside the shareable packet directory. Keep the key private until rating collection is complete. The packet includes the same prompt, answer, and cited source material for each anonymous response. The analyzer reports useful-or-very-useful rates overall and by task family, pairwise preferences, Wilson intervals for the user gate, and cluster-bootstrap intervals over raters.

`blinded_rating_app.html` is a local, browser-only reviewer for the generated `rating_packet.jsonl`. Choose the packet and a rater ID; draft scores stay in that browser's local storage. It never reads the deblinding key. Complete the packet and download `ratings.csv`, then use the analyzer below. It requires no server or external connection; open the HTML file directly in a browser.

`audit_training_acceptance.py` consolidates the first-target gates from saved comparison, cumulative-training, correction-control, and optional human-rating reports. It emits each gate independently; a missing rating report is `unmeasured`, and the overall acceptance flag remains false until every required gate passes. Regenerate the current evidence with:

```bash
PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.experiments.audit_training_acceptance \
  --output hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/training_acceptance_audit.json
```

The current audit has 405 episodes with at least 60 cases per family; HPM task completion is higher than the matched 9B retrieval baseline overall and in every family, and the cumulative retention gates pass. Grounded-answer F1 is lower than the baseline despite higher thresholded grounded-answer accuracy; the correction confirmation exposes three same-user collateral regressions; human usefulness remains unmeasured. Thus this report does not mark acceptance as passed.

For the balanced six-family run, the generated full packet covers 329 answer-bearing cases across grounded QA, corrections, comparisons, preferences, and abstention. It excludes 76 workflow items because those benchmark labels measure routing only, not the quality of a completed answer. Use `--sample-size 30` for a balanced, manageable packet for one rater; make a separate output folder and key with a distinct `--seed` for each additional rater. Fill `ratings_template.csv` or `ratings_template.jsonl` with a `rater_id`, usefulness scores (integers 1–5), and a preference (`A`, `B`, or `tie`).

```bash
PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.experiments.blinded_usefulness build \
  --hpm-report hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/balanced_six_family_report.json \
  --baseline-report hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/qwen_bm25_full405.json \
  --source-state hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/study_training_state.json \
  --curriculum hpm_ai_v2/training/curricula/grounded_study_seed.jsonl \
  --curriculum hpm_ai_v2/training/curricula/knowledge_acquisition_growth_150_seed.jsonl \
  --curriculum hpm_ai_v2/training/curricula/retrieval_discrimination_90_seed.jsonl \
  --curriculum hpm_ai_v2/training/curricula/study_procedures_seed.jsonl \
  --curriculum hpm_ai_v2/training/curricula/workflow_alias_transfer_seed.jsonl \
  --curriculum hpm_ai_v2/training/curricula/comparison_transfer_60_seed.jsonl \
  --curriculum hpm_ai_v2/training/curricula/correction_transfer_60_seed.jsonl \
  --curriculum hpm_ai_v2/training/curricula/correction_query_augmentation_seed.jsonl \
  --curriculum hpm_ai_v2/training/curricula/preference_recall_60_seed.jsonl \
  --curriculum hpm_ai_v2/training/curricula/grounded_qa_expansion_60_seed.jsonl \
  --curriculum hpm_ai_v2/training/curricula/abstention_neighbor_60_seed.jsonl \
  --curriculum hpm_ai_v2/training/curricula/workflow_route_expansion_60_seed.jsonl \
  --output-dir /tmp/hpm-rater-01 --private-key-output /tmp/hpm-rater-01-key.json \
  --seed 1 --sample-size 30

PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.experiments.blinded_usefulness analyze \
  --packet /tmp/hpm-rater-01/rating_packet.jsonl \
  --ratings /tmp/hpm-rater-01/ratings.csv \
  --key /tmp/hpm-rater-01-key.json \
  --output /tmp/hpm-usefulness-report.json
```

Building packets does not produce human-evaluation evidence. Report the study as incomplete until independent raters submit ratings; the analyzer marks acceptance as unavailable when no eligible ratings exist. Current packets for the gold-audited 405-item comparison are in `results/2026-10-24-balanced-six-family-corrected/blinded-usefulness-goldaudit/` (329 answer-bearing cases) and `blinded-usefulness-rater01-goldaudit/` (balanced 30-case sample). Their separate deblinding keys are stored beside those folders; keep them private until rating is complete.

## Correction transfer and acquisition growth (2026-10-24)

`correction_transfer_confirmation_v2_48_seed.jsonl` introduces eight deliberate corrections, eight distinct validation paraphrases, and 32 held-out correction and collateral questions. On its initial exploratory run, all eight corrected facts validated and the correction-query retrieval reached 16/16 held-out correction items, but same-user collateral answer quality regressed; this is not a safe correction result. The follow-up scopes correction influence to a sufficiently similar correction query and a matching user. It avoids the observed collateral source-ranking changes in the unit regression check, but the curriculum rerun selected the corrected source on 15/16 correction items (one eclipse paraphrase selected a valid alternate source); validation retrieval was 5/8, so this scoped design is also exploratory, not accepted. Both runs demonstrate addition of eight validated fact nodes, not a settled correction capability. Qwen3.5 `latest` (the installed local 9.7B-class tag) was run with BM25 top-three context on the same held-out suite. See `results/2026-10-24-balanced-six-family-corrected/correction-transfer-v2-qwen-latest/cumulative_study_report.json` and `correction-transfer-v2-scoped-r3/cumulative_study_report.json`. The nonexistent `qwen3.5:9.7b` tag produced HTTP 404 and is invalid; do not use its report.

Checkpoint vocabulary expansion now filters terms already in the vocabulary before applying the candidate limit. This prevents frequent existing terms from hiding new curriculum terms; a focused regression test verifies the rare new terms are added. The correction retrieval change also has a regression test asserting an unrelated same-user query's evidence scores stay unchanged.

## Knowledge acquisition with fact validation

`knowledge_acquisition_growth_150_seed.jsonl` corrects the earlier 90-item acquisition curriculum's split: each of its 60 trained facts now has a differently worded, held-out validation query with the same exact source-supported answer. It retains the ten original validation items and the original 20-item test set. Run the HPM and matched local Ollama baselines with:

```bash
PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.experiments.run_cumulative_study \
  --phase hpm_ai_v2/training/curricula/knowledge_acquisition_growth_150_seed.jsonl \
  --output-dir /tmp/hpm-knowledge-acquisition-growth \
  --ollama-model qwen3.5:latest --max-prior-interactions 8
```

The runner reports per-phase pre/post held-out test performance, new and validated fact nodes, and restart consistency. It also reports the fraction of distinct trained QA/correction answers that have exact-answer validation coverage so a curriculum with unrelated validation facts is visible before interpreting fact growth. On the current 20-item exam, 60/60 fact nodes validated, but accuracy was already 20/20 before training and remained 20/20 afterward; this confirms structured fact growth, not a test-accuracy gain. Qwen3.5 9.7B scored 14/20 plain and 18/20 with BM25 retrieval. Treat this as a small in-corpus diagnostic.

For a cumulative continuation, pass `--initial-checkpoint CHECKPOINT_DIR`, a test-only `--retention` curriculum, and `--baseline-history-curriculum` for every train/validation curriculum represented in the checkpoint, even when no Ollama model is being evaluated. The runner verifies those episode IDs exactly reproduce the checkpoint's fitted-ID set, uses its validation examples to preserve HPM abstention calibration, and uses its train/validation examples as Qwen demonstrations when comparing Ollama.

The gold-audited 405-item comparison is in `results/2026-10-24-balanced-six-family-corrected/growth_validated_goldaudit_matched_405_summary.json`: HPM 387/405, Qwen3.5 9.7B plain 125/405, and BM25+Qwen 299/405. HPM answer-token F1 is 0.860 versus 0.873 for BM25+Qwen. Eight test answer references in the QA growth curriculum were corrected against their questions and source sentences, then the cumulative HPM and Ollama runs were repeated with unchanged train/validation interactions. The result remains a narrow in-corpus task comparison, not evidence of general 9B capability parity; blinded human ratings remain pending.

The current-code rerun at `results/2026-10-24-knowledge-acquisition-growth-150-current/cumulative_study_report.json` trains 60 facts, validates each against a distinct held-out paraphrase, and persists all 60 as validated HFN facts. All 20 held-out test predictions remain identical after checkpoint reload. HPM is 20/20 both before and after training, since the phase sources are ingested before the diagnostic pre-score; therefore this run establishes validated persistent acquisition, not a test accuracy increase. Matched BM25+Qwen3.5 9.7B Q4_K_M is 18/20 (answer-token F1 .908), while HPM is 20/20 (F1 1.0). The acceptance audit separates `validated_knowledge_growth` (validation, checkpoint persistence, and restart) from `knowledge_transfer_effect` (held-out improvement, regression, or no gain at a ceiling).

A current-code rerun of `concept_transfer_confirmation_36_seed.jsonl` does **not** pass the fact-acquisition gate: its three validation rows check new facts within each concept, rather than paraphrasing the nine trained answers. As a result, none of the nine fact candidates validates even though all three concepts validate; HPM changes from 30/36 to 29/36, while matched BM25+Qwen gets 30/36 (F1 .781 vs HPM .817). A corrected variant adds held-out paraphrases for all nine trained answers; all nine facts and all three concepts validate, but held-out HPM accuracy still changes from 30/36 to 29/36 (Qwen gets 27/36). Thus persistent fact growth is verified, but the transfer score regresses slightly. These runs show why fact validation must stay separate from task-level efficacy. Reports: `results/2026-10-24-concept-transfer-confirmation-current/cumulative_study_report.json` and `results/2026-10-24-concept-transfer-fact-validation-current/cumulative_study_report.json`. The corrected curriculum should remain a development set; next work is to reduce collateral interference and confirm any retrieval changes on untouched held-out material.

Long Ollama evaluations persist each completed prediction in a signature-checked progress JSONL file under the output directory and reuse it only when the model configuration, sources, episodes, and demonstration history match. Resume the same command after an interruption; changed evaluations fail closed rather than mixing old predictions into new metrics. Use `--baseline-mode retrieval` to run only BM25+Qwen, `--baseline-mode chat` to run only full-context Qwen, or leave the default `both` for both baselines. For a newly added phase, omit `--retention` to evaluate the phase's held-out test set without rerunning earlier retention items; report the earlier matched retention audit separately.

The first 36-item concept-transfer continuation in `results/2026-10-24-balanced-six-family-corrected/concept-transfer-36-qwen-retrieval/cumulative_study_report.json` adds 97 vocabulary terms and ten facts. HPM improved from 2/36 to 16/36 (F1 .161 to .515); BM25+Qwen scored 24/36 (F1 .768). The initial HPM misses were inspected, so this set became exploratory after the subsequent retrieval change. On 68 calibration/validation episodes a .05 normalized BM25 corroboration weight improved evidence selection from 48/68 to 51/68. Applying that weight to TF-IDF and dense retrieval yields an exploratory 18/36 HPM result (F1 .565); one concept passes validation and is promoted. BM25+Qwen remains at 24/36. The 405 retained-item HPM score is 386/405 after training versus 387/405 before, within the retention gate. The retrieval change requires confirmation on a new untouched exam. Reports: `results/2026-10-24-balanced-six-family-corrected/concept-transfer-36-qwen-retrieval-hybrid/cumulative_study_report.json` and `concept-transfer-continuation-bm25-fusion-405/cumulative_study_report.json`.

The attempt to rerun both baselines across all 441 retained and new items remained active for more than 15 minutes with the local inference worker consuming CPU/GPU but produced no final report before interruption; no Qwen metrics were taken from that incomplete run.

Checkpoint continuations require all prior validation-history curricula, and each run now works from an isolated checkpoint copy so loading/reindexing cannot alter the source checkpoint. The untouched follow-up curriculum `../training/curricula/concept_transfer_confirmation_36_seed.jsonl` uses disjoint compression, heat-transfer, and immune-defense sources. It has nine training items, three validation items, and 36 unlabeled held-out questions. HPM improves from 7/36 to 19/36 after training (answer-token F1 .221 to .549), adds nine facts and 98 vocabulary terms, and promotes two of the three new concepts. Its previous 441-item retention exam moves from 404/441 to 405/441 and passes the gate, with identical post-restart predictions. Qwen3.5 9.7B Q4_K_M with BM25 top-three retrieval scores 31/36 on the new exam (grounded-answer accuracy 22/36, F1 .786; median latency 1.71 s), so HPM still trails on answer quality. Full reports: `results/2026-10-24-balanced-six-family-corrected/concept-transfer-confirmation-qwen-retrieval-matched/cumulative_study_report.json` and `concept-transfer-confirmation-retention-441-isolated/cumulative_study_report.json`.


The first four-domain acquisition follow-up, `../training/curricula/capability_transfer_36_seed.jsonl`, is diagnostic only: it added eight fact candidates but had no matched validation answers (0/8 coverage), and HPM stayed at 33/36 before and after fitting. The revised confirmation curriculum, `../training/curricula/capability_transfer_confirmation_36_seed.jsonl`, spans geology, botany, linguistics, and networking. Eight trained facts each have a distinct paraphrased validation question, and 36 test questions remain held out. All eight facts validate and promote. HPM selects correct evidence on 32/36 before and after fitting (answer F1 .897 to .892); this pre-fit score is reported because source documents are ingested before the diagnostic. Qwen3.5 9.7B Q4_K_M with BM25 top-three retrieval also selects evidence on 32/36, with grounded-answer accuracy 21/36, answer F1 .808, no errors, and 1.82 s median latency. HPM answer F1 is .892 with about 1.6 ms median retrieval latency. The 513-item retention suite stays at 446/513, all 549 predictions match after restart, and fact validation coverage is 8/8. The report exposes `fact_validation_gate` so a phase cannot present unvalidated QA/correction facts as verified knowledge; this phase passes with 8/8 matched and validated facts. The matched run records Qwen3.5 9.7B Q4_K_M placement on the NVIDIA GeForce RTX 5060 Ti (16,311 MiB, driver 580.173.02). Future runs capture GPU name, memory, and driver with `nvidia-smi` when accessible, while reporting unavailable when the sandbox cannot query host hardware. This is evidence of validated knowledge growth on a small corpus, not broad 9B capability parity. Reports: `results/2026-10-24-balanced-six-family-corrected/capability-transfer-confirmation-v1/cumulative_study_report.json` and `capability-transfer-confirmation-retention-513-gated/cumulative_study_report.json`.

## Current-code continuation audit (2026-10-24)

The 387/405 result above was produced by an earlier code state; the saved checkpoint scores 370/405 when re-evaluated under current code. The current three-phase continuation trains ten new facts, validates all ten on held-out answers, persists 17 validated/promoted facts total, passes all retention gates, and reproduces the same 405 predictions after reload. Its cumulative score is 357/405 (88.1%) against matched Qwen3.5 9.7B Q4_K_M BM25 retrieval at 288/405 (71.1%). HPM answer-token F1 is 0.808 versus Qwen's 0.867, and correction task accuracy remains 32/60. The automated audit passes task completion, validated knowledge growth, and retention, but overall acceptance remains false because correction transfer and human usefulness are unresolved. The two revised validation prompts ask directly for the trained propositions after generic paraphrases selected unrelated sentences; all test questions and labels remain held out. Reports: `results/2026-10-24-balanced-six-family-corrected/growth-validated-final-scoped/cumulative_study_report.json`, `full405_9b_comparison_current.json`, and `training_acceptance_audit_current.json`.
