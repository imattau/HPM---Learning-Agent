# HPM Training Regime: First Target at 9B-Model Usability

**Status:** active implementation; acceptance criteria not met  
**Scope:** first user-facing target for `hpm_ai_v2`; no claim that HPM has matched a 9B model yet.

## 1. Objective

Build a personal study assistant that is useful enough, on a clearly bounded set of everyday learning tasks, to compare favorably with a fixed approximately 9B-parameter language model. The target is usability, not parameter-count parity or a claim of general intelligence.

The assistant should let a user provide documents, ask questions in ordinary language, request study aids and small transformations, correct misunderstandings, and see where factual answers came from. It should retain useful concepts, preferences, and corrections across sessions, apply them to new material, and make uncertainty visible.

### First target user and task boundary

The first user is one person studying or working with a changing collection of non-fiction material. Supported tasks:

1. Ingest user-provided notes, articles, and short documents.
2. Answer questions from those sources, with passage-level evidence and an explicit unknown when evidence is insufficient.
3. Explain, compare, define, outline, and summarize concepts at a requested level.
4. Generate flashcards, quizzes, and a study plan from the material.
5. Apply simple learned transformations and preferences, such as a preferred summary format or a recurring extraction workflow.
6. Incorporate corrections and demonstrate the effect of a correction in a later session.

The first release is not expected to replace broad open-domain conversation, coding, or web research. These are out of the first training and evaluation boundary.

## 2. What “comparable to a 9B LLM” means

Choose and pin one locally runnable, general-purpose model in the 8–10B parameter range before the benchmark is assembled. Record its exact model/revision, quantization, prompt, decoding settings, hardware, and tool access. The comparison is against the complete assistant experience on the task boundary above, not a parameter count.

Use three systems on the same held-out tasks:

- **HPM assistant:** the proposed learning system.
- **9B baseline:** the pinned model with a fixed system prompt and the same user-visible tools and documents.
- **Retrieval baseline:** the same 9B model with a conventional document retrieval store, where feasible. This separates gains from ordinary retrieval from gains due to HPM learning and reuse.

An HPM result only counts if the tested configuration, including any language encoder or generator, is recorded. The language interface may initially be a frozen pretrained component; it must not silently provide answers or hidden labels to the HPM learner. During evaluation, all systems see the same source corpus, conversation history, and tool results.

### Initial acceptance target

On a preregistered set of at least 300 user-style episodes, with at least 60 episodes in each major task family:

- HPM task completion is within 5 percentage points of the 9B baseline overall and within 10 points on every task family.
- Source-grounded factual answers are at least as accurate as the retrieval baseline; unsupported factual claims are no more frequent than the 9B baseline.
- At least 80% of users rate answers as useful or very useful in a blinded pairwise preference study, and HPM is no more than 10 points below the 9B system on that measure.
- After a user correction, the corrected behavior is applied on a held-out paraphrase or later-session task at least 80% of the time, without changing unrelated answers.
- Retention on previously mastered tasks stays within 5 percentage points after each new curriculum phase.
- Median response time and user effort are reported. A response that is technically correct but requires substantially more prompting or waiting does not meet the usability target.

Report confidence intervals, per-task results, abstention quality, evidence precision, correction uptake, time-to-answer, and user preference. Do not collapse them into one aggregate score. These thresholds are gates for the first target, not results already achieved.

## 3. HPM learning model

Every learning episode should update four explicit parts of the system:

| HPM role | First-target implementation |
|---|---|
| Pattern substrate | Persistent, typed nodes for source passages, concepts, claims, relations, procedures, user preferences, and task outcomes. Edges record composition and provenance. Keep source text and learned abstractions separately addressable. |
| Pattern dynamics | Propose candidate patterns from observation, retrieval, prediction error, and user correction. Reinforce supported patterns; revise or mark contradicted patterns with scope and evidence; consolidate repeated structure; decay or archive patterns that are unused and low utility. |
| Evaluators | Score task success, evidence support, calibration, correction uptake, transfer, structural reuse, latency, and resource cost. No single fluency or novelty score can promote a pattern on its own. |
| Pattern fields | Condition retrieval and evaluation on the user, current task, source collection, time, and trust/provenance. Keep private user knowledge separate from generally reusable patterns and imported content. |

Use HPM levels operationally:

- **L1 – observations:** passages, entities, claims, questions, feedback, and task outcomes.
- **L2 – reusable chunks:** definitions, source-grounded facts, summaries, extraction templates, and stable user preferences.
- **L3 – relations:** concept relations, analogies, and task schemas that connect chunks across documents.
- **L4 – generative procedures:** question answering, comparison, quiz generation, and learned transformations executed against retrieved evidence.
- **L5 – meta-control:** choose retrieval, answer, ask a clarifying question, run a tool, defer, or abstain; monitor confidence and learning value.

Promotion to a more general level requires evidence of utility on held-out examples. A repeated phrase is not by itself a learned concept; one successful answer is not by itself a trusted procedure.

## 4. Training sequence

Training is cumulative and session-based. Each episode is a tuple of context, user request, available evidence, action/answer, observed outcome, and provenance. Keep training, validation, and final evaluation users and source documents disjoint where possible.

### Phase 0 — reliable substrate and measurement

Before learning, make ingestion, retrieval, citation, persistence, versioning, and evaluation deterministic enough to audit. Store each source with stable IDs and offsets. Every generated claim should link to source spans or be labeled as an inference. Add a fixed benchmark runner and baseline wrappers. Establish a clean-user and restart test so persistence is part of the measured system.

**Exit gate:** repeatable answers and scores across process restarts; citations resolve to the original source; all model and tool calls are logged with configuration.

### Phase 1 — grounded observation and retrieval (L1)

Expose short, coherent documents in small batches. Train/teach the agent to segment passages, identify candidate concepts and claims, index provenance, retrieve relevant evidence, and say “not in the material” when evidence is absent. Include paraphrased questions, answerable and unanswerable questions, conflicting sources, and near-duplicate passages.

Use direct labels for a compact seed set, then weak supervision and self-generated questions only as proposals. A human or verified source check gates accepted answers. Prefer learning a stable representation and evidence link over memorizing a question-answer pair.

**Exit gate:** high evidence precision and calibrated abstention on unseen passages and paraphrases.

### Phase 2 — reusable concepts and schemas (L2–L3)

Interleave documents from related topics. Ask for definitions, comparisons, causal/part-whole relations, summaries at different levels, and cross-document questions. Encourage the system to build concept nodes that point back to multiple supporting passages. Discover reusable answer and study-task schemas from successful episodes.

Test with new documents that use different wording and with distractor sources. A proposed abstraction must predict or answer at least one held-out task before it is stabilized. Keep incompatible claims as scoped alternatives rather than averaging them into one fact.

**Exit gate:** demonstrate fewer steps or less evidence-search cost on later tasks while preserving citations and accuracy.

### Phase 3 — procedures and composition (L4)

Teach multi-step workflows: retrieve evidence → synthesize → cite; compare two concepts; turn a section into flashcards; generate a quiz, score answers, and select a review item. Begin with demonstrations and worked examples, then remove scaffolding and vary order, source, and phrasing. Compose previously validated procedures for new requests.

Reward a correct, evidence-backed result and successful reuse. Penalize unsupported claims, irrelevant retrieval, excessive steps, and failure to follow the requested format. Validate each procedure on held-out inputs before registering it as reusable.

**Exit gate:** solve unseen compositions with fewer new patterns than a from-scratch run, while meeting the grounded-answer gate.

### Phase 4 — user interaction, correction, and active learning (L5)

Collect realistic sessions with ambiguous, incomplete, and changing requests. Teach the controller when to answer, ask a concise clarification, retrieve more evidence, use a tool, or abstain. Present a correction as evidence with scope, not an unconditional rewrite of all similar knowledge. Ask for feedback only when its expected value exceeds the interruption cost.

Use near-misses, high uncertainty, and expected transfer value to select practice episodes. Do not use curiosity or novelty alone as a success reward. Measure how many corrections are needed and whether the fix transfers without collateral changes.

**Exit gate:** reliable correction uptake, appropriate clarification/abstention, and no measurable regression on unrelated retained tasks.

### Phase 5 — consolidation and lifelong retention

After each batch/session, replay a small, stratified sample of prior tasks. Merge duplicate concepts only when provenance and meaning agree; retain links to their original observations. Recompute pattern utility from long-term success, reuse, contradiction, and cost. Archive weak patterns instead of deleting evidence. Evaluate after restart and after topic switches.

**Exit gate:** retained performance, bounded storage growth, and increasing transfer or decreasing task cost across successive curriculum blocks.

## 5. Episode sampling and data mix

Start with a hand-authored, auditable seed set and then mix in user-like sessions. A proposed early mix is:

| Episode type | Share | Purpose |
|---|---:|---|
| Grounded question answering and abstention | 30% | Factual usefulness and evidence discipline |
| Explain, summarize, define, compare | 20% | Abstraction and level adaptation |
| Study aid creation and retrieval practice | 15% | Multi-step task utility |
| Cross-document relations and transfer | 15% | L2/L3 structure and recombination |
| User preference and workflow learning | 10% | Persistent personalization |
| Corrections, contradictions, and adversarial distractors | 10% | Belief revision and robustness |

Within each batch, balance common cases, edge cases, unanswerable queries, conflicting sources, and paraphrases. Increase difficulty only after the current phase passes its validation gate. Mix earlier material into later practice to detect forgetting. Keep final evaluation tasks and their source documents out of curriculum generation and meta-controller tuning.

## 6. Feedback and pattern lifecycle

For each episode, record:

1. What was observed and from which source or user correction.
2. Which candidate patterns were retrieved or proposed.
3. What action was selected and why, including uncertainty.
4. Whether the outcome was correct, useful, and properly supported.
5. Which patterns were reinforced, revised, linked, quarantined, or archived.
6. The resource cost and whether the episode yielded useful new learning.

Promotion rules:

- A source fact needs one or more traceable source spans and conflict status.
- A generalized concept/schema needs support across distinct examples or explicit user confirmation, plus a held-out utility check.
- A procedure needs repeated success on varied inputs and an execution or evidence check.
- A user preference is scoped to that user and can be superseded by a newer explicit preference.
- A contradiction lowers confidence or splits scope; it does not silently erase prior evidence.

## 7. Evaluation design

Use session-shaped episodes, not isolated trivia alone. Reserve separate splits for:

- **Within-session task quality:** direct task completion, evidence, format, and usefulness.
- **Few-shot acquisition:** performance after 1, 3, and 10 demonstrations of a new concept or workflow.
- **Cross-session learning:** correction and preference retention after restart and time-separated use.
- **Transfer:** new documents, paraphrases, and combinations of known workflows.
- **Retention/interference:** old tasks after each new topic or curriculum phase.
- **Calibration:** confidence versus correctness; appropriate abstention on absent or conflicting evidence.
- **Efficiency:** user turns, retrieval calls, latency, token/tool cost, new-pattern count, pattern reuse, and storage growth.

Blind human raters should judge usefulness, clarity, and trustworthiness without knowing which system generated each answer. Ground truth checks should be separate from preference ratings. Publish per-family results and failure examples, including cases where HPM is less usable than the baseline.

## 8. Current v2 fit and work required

Existing code provides useful starting components: `ReaderAgent` handles passage ingestion and retrieval; the HFN forest and observer provide pattern storage and dynamics; reader experiments cover curiosity, themes, analogy, spelling, and QA; SP72 explores persistence and lifelong learning. These components do not yet constitute this regime.

The first implementation tranche should:

1. Define a stable episode/evidence schema and provenance links for every answerable claim.
2. Add a session trainer that can ingest a curriculum, run tasks, collect outcomes/corrections, and persist a replayable event log.
3. Separate candidate pattern creation from promotion, with explicit evaluator gates and contradiction handling.
4. Add held-out curriculum exams for acquisition, transfer, retention, calibration, and usability.
5. Make the reader path honest about its text representation limits. Current `TextDomainConfig` uses a TF-IDF-style vocabulary vector, so it should not be treated as robust open-ended semantic understanding; compare a fixed semantic encoder as a controlled substrate option.
6. Fix and document user-facing v2 API examples before treating them as the training interface; current README constructor and `solve()` return examples are out of sync with the implementation.

### Implementation evidence (2026-10-23)

The current implementation in `hpm_ai_v2/training/study.py` reads JSONL sources and episodes, builds passage, sentence, source-linked fact, user-scoped concept, negative-pattern, preference, and procedure nodes, applies feedback to Observer weights, calibrates abstention on validation data, returns extractive answers with exact offsets, checkpoints its state, and evaluates test episodes read-only. Facts are candidates until held-out answer/provenance checks validate them; repeated support across sources can promote them. Explicit `concept_key` labels group facts under typed HFN concepts; they receive no retrieval influence until multiple source supports and a successful held-out answer check validate the concept, after which a small concept-level bonus is applied. Learned fact nodes also contribute a small, trust-weighted retrieval bonus to their source passages, making supervised L2 knowledge operational during evidence selection. A supervised abstain episode creates a persistent HFN negative pattern, which can reject a similar question for that user without reducing the global utility of its source. Explicit preference episodes persist scoped values and revision history; the chat planner applies supported summary-format preferences. Supervised procedure episodes learn user-scoped trigger aliases for existing define, summarize, outline, and flashcard executors; they require two distinct training triggers and a successful held-out validation match before chat can use them. These nodes select existing cited executors, not arbitrary learned programs. For the frozen semantic encoder, a configurable BM25 corroboration bonus supports retrieval; near-threshold lexical rescue is opt-in and remains subject to hard-negative evaluation. Correction episodes create `corrects` graph edges and use stronger reinforcement and rejection penalties. The runner reloads each saved checkpoint and reports whether held-out predictions remain identical. `fit_phase` measures a fixed retention suite before and after a phase; `fit_curriculum` runs ordered phases against one cumulative HFN state and reports knowledge growth and retention gates after every phase. The cumulative command can compare plain Ollama and BM25 plus the same Ollama model against HPM, with matching user history. An optional fixed Ollama embedding substrate provides the semantic encoder control; its parameters are frozen while HPM trains its retrieval and fact weights. Reports capture per-task and per-episode metrics, validation resubstitution diagnostics, knowledge growth, retention, latency, model/configuration metadata, and BM25 recall/context size.

The hand-authored smoke fixtures were compared with pinned Qwen3.5 9.7B Q4_K_M on Ollama, both plain and with BM25 retrieval. Results by fixture (HPM default / plain Qwen / BM25+Qwen task accuracy) are: grounded QA 4/4 / 3/4 / 3/4; procedures 4/4 / 4/4 / 4/4; semantic paraphrase transfer 4/5 / 5/5 / 5/5; correction transfer 3/3 / 3/3 / 3/3. On default semantic paraphrase transfer HPM evidence accuracy is 0.75 and answer-token F1 is 0.75; plain and BM25+Qwen have answer F1 0.483 and 0.441. The curriculum now trains one user-scoped unsupported pattern (“food warmth”) and validates transfer to a near-negative temperature question. The pattern rejects that held-out validation query while preserving the adjacent supported distance query. With opt-in lexical rescue, HPM gets 5/5 test accuracy and 0.833 answer F1; its calibration resubstitution is 3/3. The default remains at 4/5; broader unseen negative queries are needed before enabling rescue by default. On correction transfer, HPM's answer F1 is 1.0 versus 0.779 and 0.602; all three abstain correctly on its unsupported item. HPM's extractive references are unusually close to the authored gold answers, so its perfect F1 on several fixtures is not a comparable free-form generation result. Median HPM retrieval latency is 0.02–0.12 ms versus 0.81–1.13 seconds for Ollama in these runs (semantic-encoder setup/precomputation is not included). Every run recorded the same model digest and GPU placement, and HPM predictions matched after checkpoint reload. The correction fixture creates one validated correction link. The sequential retention test passes across three cumulative curriculum phases, with one new fact node per phase and no loss on its single retained task. A new `run_cumulative_study.py` command runs ordered session curricula through one HFN state, validates source/episode boundaries, reports per-phase fact growth and retention gates, and reload-checks the accumulated checkpoint; an integration test confirms two-session cumulative growth and restart consistency. Its optional baseline receives the same accumulated train/validation examples on the retained exam. A `preference` supervision type now stores explicit `(user_id, preference_key)` settings in persistent HFN nodes, supports replacement with a provenance history, exposes a retrieval method, and scores held-out preference recall without folding it into evidence calibration. The Ollama evaluator scopes demonstrations to the matching user. The chat planner now retrieves `summary_format` per user and applies `bullets` or `numbered` formatting to summaries/outlines; unsupported values are ignored. Tests cover two-user separation, preference updates, test scoring, checkpoint restoration, and application on actual chat replies. A larger retrieval-discrimination run uses 10 topic documents compiled from 90 passages and 29 query episodes (11/9/9 by split). On the 9-item test, HPM, plain Qwen3.5 9.7B, and BM25 plus Qwen all score 9/9 task accuracy with 20 shared pre-test interactions; HPM and plain Qwen have answer-token F1 1.0, BM25 plus Qwen 0.912, and HPM's unsupported rejection is 1.0. Neutral-weight HPM already retrieves all eight answerable test topics correctly, so supervised weights do not improve retrieval accuracy on this split. Checkpoint predictions are identical after reload. The 95% Wilson interval is still wide ([0.701, 1.0]); test wording uses a fixed template and topics overlap across splits. Full report and compact episode summary are in `hpm_ai_v2/experiments/results/2026-10-23-retrieval-discrimination-90-report.json` and `...supervised-seed-benchmarks.json`. This is an expanded diagnostic, still not usability or 9B-capability evidence.

The answer baseline is Qwen3.5 9.7B Q4_K_M, digest `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`; Ollama reported GPU placement. The frozen encoder is Qwen3 embedding, 595.78M Q8_0, 1,024 dimensions, digest `ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d`. The host CPU is Intel Core i5-14600K on Linux x86_64; the GPU model could not be identified with the available driver tools.

A separate workflow-routing transfer set tests four supported operations with 8 training examples, 4 held-out validation examples, and 16 test phrases on four topic sources. Procedure scores now use the explicit `workflow_key` label, rather than treating any no-evidence abstention as correct. With the frozen 595.78M Qwen3 embedding encoder, HPM routed 16/16 test phrases (95% Wilson interval [0.806, 1.0]); plain Qwen3.5 9.7B Q4_K_M also scored 16/16 ([0.806, 1.0]), and BM25 plus the same Qwen scored 15/16 ([0.717, 0.989]). All systems saw the same 12 train/validation interactions; HPM predictions matched after checkpoint reload. The phrases vary sentence framing and topic context but retain known task words or explicit aliases such as “recap” and “definition.” The report is `hpm_ai_v2/experiments/results/2026-10-23-workflow-alias-semantic-qwen35/study_report.json`. This is parity on a small, fixed workflow-routing exam only; it does not show parity on open-ended study tasks, fact acquisition, retention, or user-rated usefulness.

A new 90-fact knowledge-acquisition curriculum splits 60 train, 10 validation, and 20 test episodes across 10 source topics. Training creates 60 source-linked fact nodes and 60 provenance edges; HPM gets 20/20 exact cited test answers and remains identical after checkpoint reload. The neutral HPM retrieval baseline also gets 20/20, so the experiment verifies persistent knowledge growth but no test-set improvement from the learned fact weights. Qwen3.5 9.7B gets 19/20 without demonstrations; with all 70 train/validation demonstrations it drops to 2/20, as does BM25+Qwen. The evaluator's bounded BM25 history retrieval (mean two selected examples, with 70 available) restores both Qwen baselines to 19/20. HPM's and Qwen's confidence intervals overlap substantially. The sources are supplied at test time to every system, so this does not demonstrate a memory advantage. Full results and the HPM checkpoint are in `hpm_ai_v2/experiments/results/2026-10-23-knowledge-acquisition-90/`. The default baseline evaluator now selects a bounded relevant history; `all` remains available as a diagnostic.

Thirty-seven focused supervised-study and chat tests pass after the cumulative-calibration and negative-anchor regressions were added; `compileall` and `git diff --check` pass. The full suite has known verification limits: collection reports 12 import errors because `tests/hpm_ai_v2/` imports the absent `hpm_ai_v2.patterns` package; a run excluding that directory showed unrelated failures and was interrupted at a slow controller test before completion.

A paired three-phase cumulative run over the knowledge-acquisition, retrieval-discrimination, and workflow-alias curricula used 79 training episodes, 45 phase-test episodes, and 14 source documents. HPM added 70 fact nodes, one negative pattern, and four workflow nodes; all retention gates passed and checkpoint predictions matched after reload. The current-code paired comparison scored HPM 43/45 (95.6%, Wilson 95% CI [0.852, 0.988]), plain Qwen3.5 9.7B Q4_K_M 44/45 (97.8%, [0.884, 0.996]), and BM25+Qwen 43/45 (95.6%, [0.852, 0.988]). Per family, HPM achieved 26/28 grounded QA, 16/16 workflow routing, and 1/1 unsupported rejection; plain Qwen achieved 27/28, 16/16, and 1/1. HPM answer-token F1 was 0.944, and median measured latency was 0.32 ms (excluding startup/precompute). A real failure exposed destructive phase-local abstention calibration: the retrieval phase had caused 15/20 false abstentions on the earlier knowledge exam. Cumulative fitting now retains prior validation examples without using test labels; learned negative patterns use validated content-word anchors, and the current paired run confirms paraphrase rejection. This 45-item diagnostic is well below the 300-item, six-family target, has only one unsupported item, uses shared source documents, and has no blinded usefulness ratings. Current report: `hpm_ai_v2/experiments/results/2026-10-23-cumulative-anchor-transfer-qwen35/cumulative_study_report.json`. A matched run with frozen `qwen3-embedding:0.6b` scored the same 43/45 task accuracy and all same per-family accuracies, but lower answer-token F1 (0.847 vs 0.944 with TF-IDF); it did not improve this corpus. See `hpm_ai_v2/experiments/results/2026-10-23-cumulative-anchor-qwen35-semantic/cumulative_study_report.json`.

Still outstanding from this design: independent and substantially larger task-family exams, semantic-encoder controls on a larger controlled set, concept/relation discovery beyond explicit `concept_key` supervision, learning arbitrary executable procedures beyond trigger routing to fixed executors, comparison synthesis and study-aid quality checks, broad clarification evaluation, preferences beyond summary formatting, broader correction transfer and collateral-regression measurement, realistic replay and retention checks across many curriculum phases, exact GPU-model capture, and blinded user preference evaluation. Treat the current work as infrastructure and smoke experiments only.

Do not begin with autonomous web crawling or unconstrained self-play. First establish trustworthy grounded learning from a user-selected corpus. Expand sources and autonomy only after evidence, correction, retention, and cost gates pass. The target checkpoint must also start from a **large pretrained HPM knowledge base**: a broad, curated, deduplicated, source-attributed corpus ingested into HFN before supervised task learning. The current 38-document checkpoint is only a benchmark fixture and does not meet this target. Measure corpus size, domain coverage, provenance/license completeness, held-out retrieval, factual answer quality, and retention as the corpus grows; do not equate document count alone with 9B capability.

## 9. Delivery plan

1. **Benchmark contract:** select/pin the 9B baseline and freeze task families, data splits, scoring rubric, and system configurations.
2. **Pretrained HPM knowledge base:** ingest a broad, user-selected, provenance-checked corpus into HFN before supervised task learning; measure scale, domain coverage, retrieval quality, and retention.
3. **Episode and provenance layer:** standardize observations, answers, citations, corrections, outcomes, and persistence.
4. **Grounded study loop:** implement Phases 1–2 with source-linked retrieval, abstention, candidate concepts, and promotion checks.
5. **Composed study workflows:** implement Phase 3 and evaluate new-workflow acquisition.
6. **Personalization loop:** implement Phase 4 corrections and preferences with scope and rollback.
7. **Lifelong evaluation:** implement Phase 5 exams, replay, archive, and cost/retention reporting.
8. **Usability decision:** run blinded comparison against the pinned baseline; report whether each gate passed and revise the target only with explicit evidence.

The first milestone is the benchmark contract plus the grounded study loop. A 9B-usability claim is premature until the full acceptance target is measured.

### Source-only corpus pretraining input (2026-10-24)

`run_cumulative_study.py` now accepts repeatable `--pretraining-corpus` JSONL files containing source rows only. The `prepare_pretraining_corpus` utility converts local books (text/EPUB/PDF), wiki exports, conversation exports, saved websites (HTML/XML), notebooks, and common programming-language files into source rows with source type, path, content hash, and license metadata. It can also import pinned Hugging Face dataset revisions with an explicit declared license and row limit; dataset/config/revision/split, row and conversation parent IDs, language, and source URLs travel with each item. OpenAssistant OASST1 is a plausible conversation source under its Apache-2.0 dataset card. The Stack v2 is not a drop-in corpus: its files have varied original licenses and its card states additional access/training conditions. The runner ingests sources into HFN before the first phase's held-out exam and records source counts, type/license counts, metadata coverage, IDs, and text volume. This supports grounding a checkpoint in varied curated material before task supervision. It is plumbing, not completion of the corpus target: no large broad-domain corpus has been selected or ingested yet, and the existing 38-document benchmark checkpoint remains insufficient evidence of 9B-equivalent knowledge capability. Web crawling and bulk dataset download remain deliberately separate from corpus preparation; source terms and provenance must travel with the data.

### Comparison-source routing follow-up (2026-10-23)

Reviewing the four-family exam exposed a selector defect: comparison questions that explicitly named two source titles could still take a topically related distractor for one side. The HPM comparison path now prioritizes explicit title/ID mentions, resolves overlapping aliases by preferring the longest title, and uses a cited sentence from the explicitly selected source if its title is not repeated in the source body. Focused supervised-study tests pass. Rescoring the same trained checkpoint on all 105 held-out items yields 103/105 overall (98.1%, Wilson 95% CI [0.933, 0.995]), with 60/60 comparisons and comparison evidence precision/recall of 1.00; grounded QA remains 26/28, workflow 16/16, and unsupported rejection 1/1. The original plain-Qwen and BM25+Qwen scores were not rerun after this HPM-only fix, so treat the change as a corrected HPM task score, not a newly matched baseline comparison. Summary: `hpm_ai_v2/experiments/results/2026-10-23-cumulative-four-family-hpm-explicit-source-v2/rescore_summary.json`.

### Correction transfer phase follow-up (2026-10-24)

The first 60-item correction exam produced 43/60 transfer accuracy. Inspection showed that each corrected fact stored only its answer sentence, so alternative user phrasings often failed to retrieve it. Correction episodes now retain their user-scoped query vectors and text patterns; matching corrected facts reinforce the corrected source and penalize the misconception only for that user, without changing shared source weights. Two additional supervised query forms per corrected fact raised the held-out correction score to 55/60 (91.7%, Wilson 95% CI [0.817, 0.969], answer-token F1 0.698). The cumulative 165-item HPM exam is 155/165 (93.9%), with grounded answer accuracy 63/88 at F1 >= 0.8. The correction family improves by 15 points while every other retained family stays unchanged; the per-family retention gate passes. Eight new correction links were added during query augmentation, all after a wrong pre-training prediction. On the same 60 items, with the same 38 documents and up to eight relevant prior interactions, Qwen3.5 9.7B Q4_K_M scores 1/60 plain and 12/60 with BM25; plain Qwen has six malformed outputs, BM25+Qwen none. BM25+Qwen answer-token F1 is 0.757 despite low exact source-ID accuracy. Results are a narrow hand-authored correction-memory diagnostic, not the full usability target. Reports: `hpm_ai_v2/experiments/results/2026-10-24-correction-query-augmentation/correction_augmentation_report.json` and `hpm_ai_v2/experiments/results/2026-10-24-correction-query-augmentation/correction_9b_baseline_comparison.json`.

### Preference recall continuation (2026-10-24)

Continued from the corrected checkpoint with a sixth supervised family: 12 simulated users, five explicit keys per user, and 60 training, 60 validation, and 60 held-out episodes. HPM recalled all 60 held-out `(user_id, preference_key)` values exactly (Wilson 95% CI [0.940, 1.000]); these pairs were taught explicitly, while evaluation changed the request wording. All five prior-family retention gates passed on the same 165 test episodes, with no score changes. The resulting six-family exam contains 225 episodes, and the checkpoint persists 60 preferences. These synthetic examples demonstrate scoped storage and lookup, not generalization to unseen preferences or preference usefulness. The benchmark remains below 300 episodes, with only one abstention test item and no blinded user-rating study. Report/checkpoint: `hpm_ai_v2/experiments/results/2026-10-24-six-family-preference/preference_phase_report.json` and `agent_state.pkl`.

### Balanced six-family continuation (2026-10-24)

Added 60 held-out grounded-QA episodes, 60 unsupported requests with supported-neighbor validation examples, and 60 varied routing episodes, with separate supervised train/validation/test splits. A first attempt exposed that several QA gold sentences were mispaired with the questions; those labels were corrected before the scored continuation, and no test labels were used for training. Training added six source-linked fact nodes, validated nine fact patterns, stored 12 user-scoped negative patterns, and added workflow triggers to the four validated procedures. All three phase retention gates passed. The 405-item cumulative HPM exam scores 385/405 (95.1%): grounded QA 73/88, correction 55/60, workflow 76/76, comparison 60/60, preference recall 60/60, abstention 61/61. On the matched new 180-item slice, HPM scores 169/180 (49/60 QA sources, 46/60 grounded QA answers at F1 >= 0.8, 60/60 abstentions, 60/60 workflow routes). Qwen3.5 9.7B Q4_K_M with BM25 scores 165/180 (46/60 QA sources, 43/60 grounded answers, 60/60 abstentions, 59/60 workflow routes); its mean QA answer-token F1 is higher (0.921 vs HPM's 0.790). Plain Qwen scores 52/180 under the exact source-ID/workflow-label rubric. This matched slice shows comparable task results for BM25+Qwen and HPM, not generalized 9B parity: all systems use a shared set of 38 source documents, tests are hand-authored, and there are no blinded usefulness ratings. The Qwen comparison covers only the new 180 episodes; do not compare it directly with HPM's full 405-item total. HPM report/checkpoint and paired results are in `hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/`; the Qwen raw report with answer scores regraded against corrected gold is `hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family/ollama_expansion_baselines.json`.

The matched full-exam Qwen3.5 9.7B Q4_K_M plus BM25 run scores 288/405 (71.1%, Wilson 95% CI [0.667, 0.753]) against HPM's 385/405 (95.1%, [0.925, 0.968]). Both systems have the same 38 source documents and 357 train/validation interactions; Qwen receives relevant history capped at eight demonstrations per request. By family Qwen scores 67/88 grounded QA, 12/60 correction, 14/60 comparison, 74/76 workflow, 60/60 preferences, and 61/61 abstention. Aggregate answer-token F1 favors Qwen (0.885 vs HPM 0.832), while HPM's task-accuracy advantage is concentrated in comparison and correction. Since questions are hand-authored from this shared corpus, the result supports learned HPM capability on this benchmark, not broad or open-world 9B parity. Full matched summary: `hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/full405_9b_comparison.json`. A cold reload reproduced all 405 predictions exactly after JSON tuple/list normalization; accuracy remained 385/405. The audit result is `hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/restart_consistency.json`.

To verify that a curriculum actually increases stored knowledge and improves held-out use, cumulative phases now score their test exam read-only both before and after fitting. Test examples remain excluded from fitting and calibration. A diagnostic on the old `grounded_qa_expansion_60_seed.jsonl` exposed a split flaw: its validation answers did not match the exact facts created from training answers, so no candidate facts could be validated. In a fresh neutral state that run added ten fact candidates but validated none, with no held-out evidence-accuracy gain (45/60 before and after) and a small answer-F1 decline (.773 to .759). The replacement `grounded_qa_acquisition_growth_60_seed.jsonl` pairs ten training facts with held-out paraphrase validation, then tests 20 paraphrases of those facts plus 40 novel facts from the same sources. On a fresh state, HPM improved from 40/60 to 42/60 evidence/task accuracy and answer-token F1 .625 to .662; it corrected two test selections, regressed none, and validated all ten trained facts. This is a modest result on a small, hand-authored in-corpus exam, not evidence of broad capability by itself.

On the same 60-item acquisition exam, Qwen3.5 9.7B Q4_K_M with the same 20 train/validation interactions and an eight-example relevant-history cap scored 6/60 (.100) without retrieval and 47/60 (.783) with BM25 retrieval; answer-token F1 was .172 and .727 respectively. BM25 answerable-source recall@k was 59/60 (.983). HPM's post-training 42/60 and F1 .662 sit below that retrieved Qwen baseline on this task. The plain-Qwen result is particularly sensitive to exact evidence IDs and abstention behavior and should not be read as general model capability. Full system report: `hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/qwen_fresh_qa_growth_60.json`; HPM before/after report: `hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/fresh_qa_growth_60.json`. The blinded usefulness packet is ready, but ratings have not been collected, so human usefulness remains unevaluated.

### Blinded usefulness study harness (2026-10-24)

Added `hpm_ai_v2/experiments/blinded_usefulness.py` to randomize HPM and Qwen answer order, produce anonymous response packets and rating templates, keep the deblinding key outside the shareable packet directory, and analyze usefulness scores after collection. The generated full packet has 329 answer-bearing cases over five families; the 76 workflow cases are excluded because the benchmark only tests their route label, not the quality of a finished answer. It supports balanced per-rater samples and reports per-family usefulness, per-rater acceptance outcomes, Wilson intervals for the user gate, and rater-cluster bootstrap intervals. The current six-family packet and private key are under `hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/`. This prepares the user study but supplies no human ratings; the usability acceptance gate remains unmeasured.

### Training growth and restore audit (2026-10-24)

The curriculum audit found `knowledge_acquisition_90_seed.jsonl` had 60 distinct train facts and zero exact fact matches in validation, so none of those facts could be promoted. Added `knowledge_acquisition_growth_150_seed.jsonl`: all 60 training answers have exact, source-supported validation counterparts with changed query wording; the original ten other validation items and 20 disjoint test items remain. The new coverage metric in `SupervisedStudyTrainer.fit` reports distinct training facts, exact validation matches, and coverage fraction. This keeps a zero-promotion curriculum split flaw visible. Its matched 20-item experiment validated 60/60 facts and added 60 source-support edges; HPM was already 20/20 before training and remained 20/20 after, so no test accuracy gain is claimed. Qwen3.5 9.7B scored 14/20 without retrieval and 18/20 with BM25, using the same 130 train/validation interactions with an eight-example relevant-history cap. HPM answer-token F1 was 1.0; the Qwen variants scored .933 and .908. This is a small shared-corpus QA diagnostic, not evidence of broad 9B capability. Full report: `hpm_ai_v2/experiments/results/2026-10-24-knowledge-acquisition-growth-150-fixed/cumulative_study_report.json`.

The larger run also exposed a TieredForest scaling bottleneck: every node registration searched all active nodes and loaded cold entries to find parents. Hierarchy registration now tracks parent IDs incrementally, and bulk source ingestion and checkpoint restoration defer cache rebuilding until the batch is complete. The corrected acquisition run ingested ten sources in 0.19 seconds, restored 948 patterns and 60 facts in 0.94 seconds, and reproduced the same test predictions after reload. Tests cover incremental parent tracking and bulk checkpoint restore. The larger prior attempt was interrupted before baselines because it entered the slow restore path; its partial checkpoint was superseded by the completed `-fixed` run.

### Retention-calibrated 405-item comparison (2026-10-24)

The balanced six-family continuation initially recalibrated abstention without carrying forward the initial checkpoint's 100 validation examples, which produced a false retention drop. `fit_curriculum` now carries those examples into later phases; the corrected run retains all 225 prior exam items and all three retention gates pass. A later curriculum audit found eight held-out QA answers paired with neighboring source sentences; these labels were corrected against their questions and source passages. The rerun scores HPM 387/405 (95.6%), plain Qwen3.5 9.7B Q4_K_M 125/405 (30.9%), and BM25+Qwen 299/405 (73.8%). HPM answer-token F1 is 0.860 and BM25+Qwen's is 0.873. The new QA phase moves held-out source selection from 49/60 to 50/60; answer-token F1 rises from 0.808 to 0.839. It adds ten fact nodes, with eight validation promotions. The rerun uses identical train/validation interactions, test episode IDs, and sources for all systems; no test labels enter fitting or calibration. All 405 HPM predictions match after checkpoint reload. This hand-authored, shared-source benchmark does not establish general 9B capability parity. Summary: `hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/growth_validated_goldaudit_matched_405_summary.json`; detailed report: `hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/growth-validated-goldaudit/cumulative_study_report.json`.

The refreshed blinded usefulness packets cover 329 answer-bearing cases; a 30-case sample balances five families with six examples each. Procedure routing is excluded because it does not produce comparable answers. No independent ratings have been collected, so the user-rated usefulness gate remains open. Current packets are in `hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/blinded-usefulness-goldaudit/` and `blinded-usefulness-rater01-goldaudit/`.

### Concept-transfer continuation and matched 9.7B retrieval check (2026-10-24)

Added `concept_transfer_36_seed.jsonl` with 12 new sources in energy storage, error correction, and feedback control. Nine labeled training questions create source-backed facts and three scoped concepts; three separate validation questions and 36 test questions remain held out. Continuing from the corrected gold-audit checkpoint expands the reader vocabulary with 97 terms and adds ten facts. On the new exam, HPM improves from 2/36 to 16/36 correct (answer-token F1 .161 to .515); its 405-item retained exam changes from 388/405 to 387/405, passes the configured retention gate, and reproduces predictions after restart. All three concepts remain candidates because their held-out evidence checks fail. This is an actual knowledge and transfer gain with limited magnitude, not a claim of 9B parity. Detailed HPM report: `hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/concept-transfer-continuation-user-scope-fixed/cumulative_study_report.json`.

For a matched check on the 36 new items, Qwen3.5 9.7B Q4_K_M with BM25 top-three source retrieval and up to eight relevant training demonstrations scores 24/36 (66.7% task/evidence accuracy), grounded-answer accuracy 21/36 (58.3%), and answer-token F1 .768. Its median latency is 1.76 seconds. HPM post-training scores 16/36 and F1 .515 on this same exam, so the current HPM learner remains behind the retrieved 9.7B baseline on cross-topic transfer. The earlier 405-item comparison remains the broad matched benchmark. This focused report is `hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/concept-transfer-36-qwen-retrieval/cumulative_study_report.json`.

The initial attempt to run both 9.7B baselines over all 441 retained and new questions remained active for more than 15 minutes with the local inference worker consuming CPU/GPU, but produced no final report before interruption. Since that runner version did not checkpoint individual predictions, no Qwen scores from the partial run are used. The cumulative runner now writes each baseline prediction to a signature-checked JSONL progress file and resumes only when model settings, sources, episodes, and training history match. `--baseline-mode retrieval` permits an explicit BM25+LLM comparison without also launching the much longer full-context chat baseline. The completed 36-item output has all predictions saved with zero evaluation errors. This makes future long evaluations resumable; it does not turn the interrupted run into a completed result.

### Validation-calibrated lexical corroboration (2026-10-24)

The new-transfer failures showed the TF-IDF/HFN score sometimes ranked a broadly related old source above a source with strong query-term evidence. On the 68 calibration/validation episodes saved with the checkpoint, a .05 normalized BM25 corroboration weight improved evidence selection from 48/68 to 51/68; the weight was selected on validation, not by optimizing the held-out labels. The existing lexical corroboration hook now applies to TF-IDF as well as frozen dense embeddings. A fresh continuation adds the same 97 terms and ten facts and passes one of three concept validations (promoting `feedback_control`). On the 36-item phase set HPM moves from 5/36 to 18/36 (F1 .244 to .565), while Qwen3.5 9.7B with BM25 top-three scores 24/36 (F1 .768). Because the initial HPM misses on this set were inspected before the retrieval change, these post-change 36-item scores are exploratory and need confirmation on a new untouched exam. On the separate 405 retained items HPM scores 386/405 after training versus 387/405 before the phase; the retention gate passes, and all 441 HPM predictions reproduce after restart. Reports: `concept-transfer-36-qwen-retrieval-hybrid/cumulative_study_report.json` and `concept-transfer-continuation-bm25-fusion-405/cumulative_study_report.json` under `hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/`.

Checkpoint continuations now require all prior validation-history curricula, and the runner isolates the initial checkpoint before loading so cold-store reindexing cannot change the source. An untouched confirmation curriculum, `concept_transfer_confirmation_36_seed.jsonl`, uses 12 disjoint sources from information compression, heat transfer, and immune defense. It has nine training episodes, three separate validation episodes, and 36 test questions with no concept labels. Continuing cumulatively, HPM rises from 7/36 to 19/36 on the new test (answer F1 .221 to .549), adds nine facts and 98 vocabulary terms, and promotes two of the three new concepts. Its 441-item prior exam changes from 404/441 to 405/441 and passes the retention gate; restart predictions match. On the same new exam, Qwen3.5 9.7B Q4_K_M with BM25 top-three scores 31/36 (86.1% evidence/task accuracy), grounded-answer accuracy 22/36 (61.1%), and answer F1 .786, with 1.71 s median latency. HPM remains behind on evidence and answer quality, with about 1.5 ms median retrieval latency. This clean confirmation supports growth from training and retention on this curriculum, but not capability parity. Reports: `concept-transfer-confirmation-qwen-retrieval-matched/cumulative_study_report.json` and `concept-transfer-confirmation-retention-441-isolated/cumulative_study_report.json` under `hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/`.


The first four-domain acquisition follow-up created eight fact candidates but matched none to held-out validation answers (0/8 coverage), and test accuracy stayed at 33/36 before and after training. Treat that run as diagnostic, not evidence of improvement. The revised `capability_transfer_confirmation_36_seed.jsonl` uses geology, botany, linguistics, and networking sources, with eight train facts, eight paraphrased validation questions covering all eight facts, and 36 held-out questions. Validation promotes all eight new fact nodes. HPM and Qwen3.5 9.7B Q4_K_M with BM25 top-three retrieval each select correct evidence on 32/36; HPM answer F1 is .892 versus .808 for Qwen, while grounded-answer accuracy is 32/36 for HPM and 21/36 for Qwen. HPM's pre-training test was also 32/36 because the sources are ingested before supervision, so this phase proves validated knowledge growth and matched retrieval performance, not an accuracy increase caused by fitting. The 513-item retention score remains 446/513, and the full 549-item post-restart predictions match. Qwen had no errors on the 36-item comparison; median latency was 1.82 s versus about 1.6 ms for HPM. This remains a small, hand-authored corpus and does not satisfy the 300-episode first-target acceptance gate or establish broad 9B parity. Reports: `capability-transfer-confirmation-v1/cumulative_study_report.json` and `capability-transfer-confirmation-retention-513-gated/cumulative_study_report.json` in `hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/`.


The supervised fact report now includes a `fact_validation_gate`: a QA/correction phase passes only when every distinct trained answer is represented in held-out validation and its fact node reaches validated/promoted status after evidence checking. The four-domain confirmation passes 8/8; the preceding diagnostic phase failed at 0/8. Experiment host metadata now captures NVIDIA GPU name, memory, and driver version when accessible; the matched run used an RTX 5060 Ti with Qwen3.5 9.7B Q4_K_M at 100% GPU placement. This closes the missing hardware-recording field for new runs, but does not close the human usefulness, broader task, or general capability gates.

### Correction transfer check (2026-10-24)

The new eight-correction curriculum has eight matched held-out validation paraphrases and 32 held-out correction/collateral cases. An exploratory fit added and validated 8/8 facts and selected the corrected source on 16/16 correction tests, but inspection exposed same-user collateral answer regressions, so that result is not accepted. Correction evidence influence is now scoped to matching user and correction-query similarity, with a unit regression check for unchanged unrelated evidence scores. On a rerun, HPM scored 15/16 correction cases and 5/8 validation retrieval cases; the retained 32-item exam changed from 26/32 to 25/32 and restart predictions matched. The matched Qwen3.5 `latest` BM25 top-three run scored 4/16 on correction and 12/16 on the collateral QA, with overall answer F1 .780. These small hand-authored results are exploratory: the correction matcher trades one held-out correction for reduced collateral effects and needs a new independent curriculum before any capability claim. The scoped report is `correction-transfer-v2-scoped-r3/cumulative_study_report.json`; the baseline report is `correction-transfer-v2-qwen-latest/cumulative_study_report.json` under `hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/`. A bad run using nonexistent Ollama tag `qwen3.5:9.7b` returned HTTP 404 and is excluded.

### Current-code full continuation audit (2026-10-24)

Re-evaluating the former 387/405 checkpoint with current code gives 370/405. The current three-phase continuation trained ten facts and validated all ten against held-out answers; 17 validated/promoted facts persist in its checkpoint. It passes retention and reproduces all 405 predictions after reload. Current full-exam performance is 357/405 (88.1%), versus matched Qwen3.5 9.7B Q4_K_M with BM25 at 288/405 (71.1%). HPM answer-token F1 trails Qwen (0.808 vs 0.867), and HPM correction accuracy is 32/60. The automated audit passes task accuracy, knowledge growth, and retention, but overall acceptance remains false because correction transfer and human usefulness remain unresolved. Treat 387/405 as historical for its earlier code state. Current files: `hpm_ai_v2/experiments/results/2026-10-24-balanced-six-family-corrected/growth-validated-final-scoped/cumulative_study_report.json`, `full405_9b_comparison_current.json`, and `training_acceptance_audit_current.json`.
