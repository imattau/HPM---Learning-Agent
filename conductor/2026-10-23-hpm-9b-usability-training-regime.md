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

The current Phase 1 implementation in `hpm_ai_v2/training/study.py` reads JSONL sources and episodes, builds passage, sentence, source-linked fact, and user-scoped negative-pattern nodes, applies feedback to Observer weights, calibrates abstention on validation data, returns extractive answers with exact offsets, checkpoints its state, and evaluates test episodes read-only. Facts are candidates until held-out answer/provenance checks validate them; repeated support across sources can promote them. Learned fact nodes contribute a small, trust-weighted retrieval bonus to their source passages, making supervised L2 knowledge operational during evidence selection. A supervised abstain episode creates a persistent HFN negative pattern, which can reject a similar question for that user without reducing the global utility of its source. For the frozen semantic encoder, a configurable BM25 corroboration bonus supports retrieval; near-threshold lexical rescue is opt-in and remains subject to hard-negative evaluation. Correction episodes create `corrects` graph edges and use stronger reinforcement and rejection penalties. The runner reloads each saved checkpoint and reports whether held-out predictions remain identical. `fit_phase` measures a fixed retention suite before and after a phase; `fit_curriculum` runs ordered phases against one cumulative HFN state and reports knowledge growth and retention gates after every phase. The runner compares neutral-weight cosine retrieval, trained HPM, plain Ollama, and BM25 plus the same Ollama model. An optional fixed Ollama embedding substrate provides the semantic encoder control; its parameters are frozen while HPM trains its retrieval and fact weights. Reports capture per-task and per-episode metrics, validation resubstitution diagnostics, knowledge growth, retention, latency, model/configuration metadata, and BM25 recall/context size. C0–C2 chat code supplies extractive plans and basic fixed-format study responses; it does not yet learn procedures.

The hand-authored smoke fixtures were compared with pinned Qwen3.5 9.7B Q4_K_M on Ollama, both plain and with BM25 retrieval. Results by fixture (HPM default / plain Qwen / BM25+Qwen task accuracy) are: grounded QA 4/4 / 3/4 / 3/4; procedures 4/4 / 4/4 / 4/4; semantic paraphrase transfer 4/5 / 5/5 / 5/5; correction transfer 3/3 / 3/3 / 3/3. On default semantic paraphrase transfer HPM evidence accuracy is 0.75 and answer-token F1 is 0.75; plain and BM25+Qwen have answer F1 0.483 and 0.441. The curriculum now trains one user-scoped unsupported pattern (“food warmth”) and validates transfer to a near-negative temperature question. The pattern rejects that held-out validation query while preserving the adjacent supported distance query. With opt-in lexical rescue, HPM gets 5/5 test accuracy and 0.833 answer F1; its calibration resubstitution is 3/3. The default remains at 4/5; broader unseen negative queries are needed before enabling rescue by default. On correction transfer, HPM's answer F1 is 1.0 versus 0.779 and 0.602; all three abstain correctly on its unsupported item. HPM's extractive references are unusually close to the authored gold answers, so its perfect F1 on several fixtures is not a comparable free-form generation result. Median HPM retrieval latency is 0.02–0.12 ms versus 0.81–1.13 seconds for Ollama in these runs (semantic-encoder setup/precomputation is not included). Every run recorded the same model digest and GPU placement, and HPM predictions matched after checkpoint reload. The correction fixture creates one validated correction link. The sequential retention test passes across three cumulative curriculum phases, with one new fact node per phase and no loss on its single retained task. Per-episode scores, confidence intervals, model metadata, and limitations are archived in `hpm_ai_v2/experiments/results/2026-10-23-supervised-seed-benchmarks.json`. These diagnostics each have only 3–5 test episodes and are not comparative usability or 9B-capability evidence.

The answer baseline is Qwen3.5 9.7B Q4_K_M, digest `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`; Ollama reported GPU placement. The frozen encoder is Qwen3 embedding, 595.78M Q8_0, 1,024 dimensions, digest `ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d`. The host CPU is Intel Core i5-14600K on Linux x86_64; the GPU model could not be identified with the available driver tools.

Thirty-four focused supervised-study, chat, Reader, and Reader-scaling tests pass after the supervised-negative addition, with compile and diff checks clean. The full suite has known verification limits: collection reports 12 import errors because `tests/hpm_ai_v2/` imports the absent `hpm_ai_v2.patterns` package; a run excluding that directory showed unrelated failures and was interrupted at a slow controller test before completion.

Still outstanding from this design: independent and substantially larger task-family exams, semantic-encoder controls on a larger controlled set, generalized concept/relation learning beyond exact normalized answer patterns, learned rather than hand-authored procedures, synthesis and study-aid quality checks, clarification, scoped preference learning, broader correction transfer and collateral-regression measurement, realistic replay and retention checks across multiple curriculum phases, exact GPU-model capture, and blinded user preference evaluation. Treat the current work as infrastructure and smoke experiments only.

Do not begin with autonomous web crawling or unconstrained self-play. First establish trustworthy grounded learning from a user-selected corpus. Expand sources and autonomy only after evidence, correction, retention, and cost gates pass.

## 9. Delivery plan

1. **Benchmark contract:** select/pin the 9B baseline and freeze task families, data splits, scoring rubric, and system configurations.
2. **Episode and provenance layer:** standardize observations, answers, citations, corrections, outcomes, and persistence.
3. **Grounded study loop:** implement Phases 1–2 with source-linked retrieval, abstention, candidate concepts, and promotion checks.
4. **Composed study workflows:** implement Phase 3 and evaluate new-workflow acquisition.
5. **Personalization loop:** implement Phase 4 corrections and preferences with scope and rollback.
6. **Lifelong evaluation:** implement Phase 5 exams, replay, archive, and cost/retention reporting.
7. **Usability decision:** run blinded comparison against the pinned baseline; report whether each gate passed and revise the target only with explicit evidence.

The first milestone is the benchmark contract plus the grounded study loop. A 9B-usability claim is premature until the full acceptance target is measured.
