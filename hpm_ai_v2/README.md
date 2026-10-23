# hpm_ai_v2 — Domain-Agnostic HPM Agent Layer

A reusable, mixin-based agent framework built on top of the `hfn/` core. Replaces the experiment-by-experiment pattern from `hpm_fractal_node/experiments/` with a composable architecture that works across any domain (integers, strings, lists, graphs, etc.) without modification.

## Architecture Overview

```
hpm_ai_v2/
├── agents/
│   ├── base_agent.py       # BaseHFNAgent — foundation for all agents
│   ├── mixins/             # HPM level-specific capabilities
│   │   ├── l2_macro.py     # L2: Schema composition and decomposition
│   │   ├── l3_relational.py # L3: Meta-schema discovery
│   │   ├── l4_forward.py   # L4: Forward model + imaginative BFS
│   │   ├── social.py       # Social field + blackboard scaffolding
│   │   ├── recombination.py # Structural recombination (Appendix E)
│   │   └── sequential_composition.py # AST-based robust composition
│   └── agents.py           # Ready-to-use concrete agent classes
├── domains/
│   ├── base.py             # DomainConfig base class
│   ├── list_domain.py      # List transformation concepts (SP61-67)
│   └── math_physics_domain.py # Extended concepts for multi-domain (SP68+)
├── utils/
│   ├── executor.py         # PythonExecutor — sandboxed code execution
│   ├── oracle.py           # EmpiricalOracle, CountingOracle
│   ├── renderer.py         # ListRenderer — HFN node tree → Python source
│   ├── forward_model.py    # StateTransitionModel — per-node delta learning
│   └── meta_controller.py  # MetaStrategyController, SolveRecord
└── experiments/
    ├── README.md           # Experiments Index — validation suite documentation
    ├── run_sp67.py         # SP67: Multi-agent social recombination example
    └── experiment_sp80_comparative_benchmark.py # SOTA Comparative Benchmark
```

---

## Quick Start

```python
from hpm_ai_v2.agents.agents import InducedSchemaAgent
from hpm_ai_v2.domains.list_domain import ListDomainConfig

agent = InducedSchemaAgent(config=ListDomainConfig())

# Register a strategy (BFS is built-in)
success, code, strategy = agent.solve(
    inputs=[[1, 2, 3]],
    outputs=[[2, 3, 4]],
    goal_type="map",
    task_id="MAP_plus1",
)
print(success, code, strategy)
```

`solve()` returns `(success, code, strategy, path)`.

---

## Concrete Agents

All live in `agents/agents.py` and are ready to use without subclassing.

| Class | Mixins | Use when |
|---|---|---|
| `InducedSchemaAgent` | Base + L2 + L3 | Schema learning from I/O examples; meta-schema discovery |
| `ImaginativeAgent` | Base + L2 + L4 | Zero-oracle BFS via learned forward model |
| `AnalogicalAgent` | Base + L2 | Cross-domain transfer via structural substitution |
| `SocialAnalogicalAgent` | Base + L2 + L4 + Social + Recombination + Composition | Multi-agent experiments with shared forest, blackboard and sequential composition |

```python
from hpm_ai_v2.agents.agents import (
    InducedSchemaAgent,
    ImaginativeAgent,
    AnalogicalAgent,
    SocialAnalogicalAgent,
)
from hpm_ai_v2.agents.mixins.social import SocialForest

# Social experiment: two agents sharing a forest
shared_forest = SocialForest(D=54, cold_dir="/tmp/shared")
alice = SocialAnalogicalAgent("Alice", shared_forest)
bob   = SocialAnalogicalAgent("Bob",   shared_forest)
```

---

## BaseHFNAgent

`agents/base_agent.py` — the foundation every agent inherits from.

### Constructor parameters

| Parameter | Default | Description |
|---|---|---|
| `cold_dir` | `"data/knowledge_base/hpm_ai_v2"` | Persistent storage directory |
| `forest_class` | `TieredForest` | HFN forest implementation |
| `hot_cap` | `10_000` | Max nodes in hot (in-memory) tier |
| `tau` | `0.5` | Observer learning rate |
| `compression_cooccurrence_threshold` | `2` | Min co-occurrences before node compression |
| `use_density_tracker` | `False` | Enable pattern density tracking (HPM App A.8) |
| `use_affective_evaluator` | `False` | Enable affective state evaluator (HPM §9.3) |
| `n_workers` | `os.cpu_count()` | Workers for parallel BFS evaluation |

### Key attributes

| Attribute | Type | Description |
|---|---|---|
| `forest` | `TieredForest` | Pattern substrate — stores all HFN nodes |
| `observer` | `Observer` | Pattern dynamics — learning, compression, weights |
| `retriever` | `GoalConditionedRetriever` | Goal-conditioned nearest-node lookup |
| `renderer` | `ListRenderer` | Converts HFN node tree → executable Python |
| `executor` | `PythonExecutor` | Runs code strings against input batches |
| `oracle` | `EmpiricalOracle` | Computes empirical state vector from execution results |
| `meta` | `MetaStrategyController` | L5: learns which strategy works in which context |
| `patterns` | `dict[str, HFN]` | Named learned patterns (macros, schemas, etc.) |
| `n_workers` | `int` | CPU workers for parallel BFS |

### Key methods

```python
# Register a custom solve strategy
agent.add_strategy("my_strategy", fn)   # fn(inputs, outputs) -> Optional[List[HFN]]

# Solve a task (tries strategies in meta-controller ranked order)
success, code, strategy_used = agent.solve(inputs, outputs, goal_type="map", task_id="t1")

# Persist and restore
agent.save_state()
agent.load_state()
```

### Built-in strategies

| Strategy | Method | Description |
|---|---|---|
| `exact` | `_try_exact` | Retrieves top-5 nearest nodes, tests directly |
| `bfs` | `_try_bfs` | Beam BFS; evaluates each depth level in parallel |

---

## Mixins

Mixins add HPM level-specific capabilities. Combine them via multiple inheritance.

### L2MacroMixin (`mixins/l2_macro.py`)

Schema composition and decomposition.

```python
# Register a learned macro from a node path
node = agent.register_macro("MAP_plus1", path=[var_inp, add1, list_append], inputs=[[1,2]], outputs=[[2,3]])

# Decomposition strategy: substitute one constituent op
agent.add_strategy("decompose", agent._try_decompose)
```

### L3RelationalMixin (`mixins/l3_relational.py`)

Discovers a common structural prefix across macros — the L3 meta-schema.

```python
meta_node = agent.discover_meta_schema()
# Returns HFN node with id="meta_list_iteration" if ≥2 list macros share a prefix
```

### L4ForwardModelMixin (`mixins/l4_forward.py`)

Learns per-node state deltas and uses them for zero-oracle imaginative BFS.

```python
# Record transitions after solving (call after register_macro)
agent._record_transitions(path, training_inputs)

# Imaginative BFS: navigates predicted state space, oracle called only to verify
agent.add_strategy("imagine", agent._try_imagine)
```

### SocialMixin + SocialForest (`mixins/social.py`)

Shared forest with blackboard for institutional scaffolding (HPM §9.7).

```python
# SocialForest is a TieredForest with a blackboard
shared = SocialForest(D=54, cold_dir="/tmp/shared")

# Post and query failures
shared.post_failure("task_X", "rule_Y", "No solution found")
failures = shared.get_failures("task_X")   # list of {rule, reason, time}

# Exchange patterns between agents
agent.exchange_patterns(partner_agent)
```

### RecombinationMixin (`mixins/recombination.py`)

Structural recombination of two patterns into a novel node (HPM Appendix E).

```python
new_node = agent.recombine_patterns(macro_a, macro_b)
score = agent.insight_score(new_node, test_inputs, test_outputs)  # novelty + effectiveness

agent.add_strategy("recombine", agent._try_recombine)
```

### SequentialCompositionMixin (`mixins/sequential_composition.py`)

Robust sequential composition of two macros via AST transformation. Generates a wrapper function calling the first then the second.

```python
# Create a composite macro (f then g)
composite_node = agent.compose_sequential("macro_add1", "macro_mul2", "compound_add1_mul2")

# Composition strategy: try pairs of existing macros
agent.add_strategy("compose", agent._try_sequential_compose)
```

## Domains

The `hpm_ai_v2` framework is domain-agnostic. Domain semantics are encapsulated in `DomainConfig` objects.

### `domains/base.py`

Defines the `DomainConfig` base class which manages manifold dimensions and concept mappings.

```python
from hpm_ai_v2.domains.list_domain import ListDomainConfig

config = ListDomainConfig()
# config.S_DIM = 20      — empirical state vector length
# config.DIM    = 14     — number of concept dimensions
# config.m_dim  = 54     — total HFN node dimensionality
```

Available domains:
- `ListDomainConfig`: Standard 14-concept set for list transformations.
- `MathPhysicsDomainConfig`: Extended concept set for mathematical and physical reasoning.

---

## Utils

### `utils/executor.py`

```python
from hpm_ai_v2.utils.executor import PythonExecutor

executor = PythonExecutor()
results, errors = executor.run_batch("x = inp\nx += 1", inputs=[1, 2, 3])
# results: [2, 3, 4], errors: [None, None, None]
```

The module also exports `_eval_path_worker` — the picklable worker used by the parallel BFS. You should not need to call it directly.

### `utils/oracle.py`

```python
from hpm_ai_v2.utils.oracle import ListOracle, CountingOracle

oracle = ListOracle(config)
state = oracle.compute_state(results, errors, code_str)  # returns np.ndarray shape (20,)

# CountingOracle wraps any oracle and tracks call count
counting = CountingOracle(oracle)
counting.call_count  # int
```

### `utils/meta_controller.py`

```python
from hpm_ai_v2.utils.meta_controller import MetaStrategyController, SolveRecord

ctrl = MetaStrategyController()
ranked = ctrl.rank_strategies("map", n_macros=3)   # list of strategy names, best first
ctrl.record(SolveRecord(...))
```

### `utils/forward_model.py`

```python
from hpm_ai_v2.utils.forward_model import StateTransitionModel

model = StateTransitionModel()
model.record_path(path, state_sequence)            # learn deltas
predicted = model.predict(current_state, node)     # predict next state
```

---

## Parallel BFS

BFS candidate evaluation uses `ProcessPoolExecutor` by default. Each depth level is evaluated in parallel — the first path that matches exits the pool immediately.

```python
# Use all CPUs (default)
agent = InducedSchemaAgent()

# Limit workers
agent = InducedSchemaAgent(n_workers=2)

# Disable parallelism
agent = InducedSchemaAgent(n_workers=1)
```

Workers re-execute code strings in isolated processes — no shared mutable state, no GIL contention.

---

## Building New Agents with Mixins

The mixin architecture is designed to make creating new agents straightforward and modular. Each mixin adds a specific HPM capability. By combining different mixins you can build agents with any subset of these capabilities without duplicating code.

`BaseHFNAgent` provides the domain-agnostic core — pattern storage, strategy dispatch, BFS, persistence. Mixins then add:

- **New strategies** — `_try_decompose`, `_try_imagine`, `_try_social`, `_try_recombine`
- **New methods** — `register_macro`, `discover_meta_schema`, `exchange_patterns`
- **New state** — `forward_model`, `shared_forest`

### Agent variants

| Agent | Mixins | Capabilities |
|---|---|---|
| `InducedSchemaAgent` | L2, L3 | Macro learning + meta-schema discovery |
| `ImaginativeAgent` | L2, L4 | Macros + mental simulation |
| `AnalogicalAgent` | L2 | Macros + cross-domain analogy (add `_try_analogy` manually) |
| `SocialAnalogicalAgent` | L2, L4, Social, Recombination, Composition | Full L2–L5 + social + recombination + composition |
| Minimal (just `BaseHFNAgent`) | — | Exact match and BFS only |

### Creating a new agent

```python
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.agents.mixins.l2_macro import L2MacroMixin
from hpm_ai_v2.agents.mixins.l4_forward import L4ForwardModelMixin
from hpm_ai_v2.agents.mixins.social import SocialMixin, SocialForest

class MyAgent(BaseHFNAgent, L2MacroMixin, L4ForwardModelMixin, SocialMixin):
    def __init__(self, agent_id: str, shared_forest: SocialForest, **kwargs):
        super().__init__(**kwargs)
        L4ForwardModelMixin.__init__(self)
        SocialMixin.__init__(self, agent_id, shared_forest)
        self.add_strategy("decompose", self._try_decompose)
        self.add_strategy("imagine",   self._try_imagine)
        self.add_strategy("social",    self._try_social)
        self.add_strategy("my_custom", self._my_custom_strategy)

    def _my_custom_strategy(self, inputs, outputs):
        # Return List[HFN] path on success, None on failure
        ...
```

`register_macro`, `discover_meta_schema`, `exchange_patterns`, `_record_transitions` — all inherited from the mixins, no reimplementation needed.

### Why this matters

- **Separation of concerns** — each mixin focuses on exactly one HPM level or mechanism
- **Reusability** — `L2MacroMixin` is shared across `InducedSchemaAgent`, `ImaginativeAgent`, and `SocialAnalogicalAgent`
- **Testability** — each mixin can be tested in isolation
- **Extensibility** — to add a new HPM level, write a new mixin and combine it with existing ones
- **Experimentation** — compare agents with different capability sets by swapping mixins in or out

---

## Relationship to hpm_fractal_node

`hpm_ai_v2` is a **new layer alongside** `hpm_fractal_node/experiments/` — it does not replace or modify it. The experiments in `hpm_fractal_node/` remain the canonical validated implementations of SP61–SP67. `hpm_ai_v2` extracts their reusable patterns into a composable framework for future work.

The `hfn/` package is the shared core used by both.

---

## Running the SOTA Benchmark

```bash
cd /path/to/HPM---Learning-Agent
PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp80_comparative_benchmark.py
```

## Supervised grounded-study training

The first implemented training slice is a supervised evidence-retrieval and extractive-answer loop for `ReaderAgent`. It ingests source passages and sentence patterns, reinforces labeled evidence and answer sentences and penalizes distractors through Observer weights, calibrates abstention on a validation split, then evaluates held-out episodes without updating weights. Predictions include the exact source sentence and its character offsets.

Curricula are JSONL files with `source` and `episode` rows. Episode splits are `train`, `validation`, and `test`; answerable episodes name an `evidence_id`, while `abstain` episodes use `null`. The checked-in seed curriculum is deliberately small and is a smoke fixture, not a 9B usability benchmark.

For corpus pretraining, `run_cumulative_study.py` accepts repeatable `--pretraining-corpus FILE.jsonl` inputs containing **only** `source` rows. The runner ingests those documents into the HFN before the first held-out phase exam and records the corpus file list, source IDs, document count, and character count. This seeds a checkpoint with grounded source knowledge without manufacturing supervised labels; later train/validation phases can add supported facts and retrieval behavior. The current 9B-comparison checkpoint contains only 38 source documents, so it is a small experiment corpus, not the large pretrained HPM knowledge base required by the training target. A production pretraining corpus still needs broad domain coverage, source/license provenance, deduplication, and scale-specific held-out evaluation.

Prepare a source-only corpus from local files with `hpm_ai_v2.training.prepare_pretraining_corpus`. It reads plain text/Markdown, EPUB and PDF books (PDF requires `pypdf`), Wikipedia/wiki JSON or JSONL exports, conversation JSONL/JSON (`messages`, `turns`, or `conversation` fields), saved HTML/XML web pages, notebooks, and common programming-language files. It strips script/style content from HTML, chunks long documents, and stores source type, path, content hash, and license metadata. Example:

```bash
PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.training.prepare_pretraining_corpus \
  data/pretraining/books data/pretraining/wikipedia data/pretraining/conversations \
  data/pretraining/websites data/pretraining/code \
  --license "CC BY 4.0" --output /tmp/hpm-pretraining.jsonl
```

Pass the generated file to `run_cumulative_study.py` with `--pretraining-corpus /tmp/hpm-pretraining.jsonl`, or create a source-only HFN checkpoint first and continue it with supervised training:

```bash
PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.training.pretrain_hfn_corpus \
  --corpus /tmp/hpm-pretraining.jsonl --output-dir /tmp/hpm-pretrained-checkpoint
PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.experiments.run_cumulative_study \
  --phase hpm_ai_v2/training/curricula/grounded_study_seed.jsonl \
  --initial-checkpoint /tmp/hpm-pretrained-checkpoint \
  --output-dir /tmp/hpm-supervised-checkpoint
```

The pretraining checkpoint has no fabricated training labels or learned fact nodes; it stores the source passages and HFN patterns, ready for the supervised continuation. Both stages can use `--encoder ollama_embedding --embedding-model qwen3-embedding:0.6b`; that fixed-size, frozen encoder supports larger corpora without TF-IDF vocabulary growth, while HFN still stores and learns over the evidence patterns. The utilities prepare local or already-exported materials; they do not download books or crawl websites. Confirm reuse rights and preserve source metadata before adding external material.

The same utility can import a **pinned** Hugging Face dataset revision with `--hf-dataset ID --hf-revision COMMIT --hf-split train --source-type TYPE --license LICENSE`; `--hf-max-rows` bounds the import and `--hf-streaming` avoids a full local download where the dataset supports streaming. Conversation row/parent/tree IDs and language, dataset/config/revision/split, row IDs, URLs, and license are retained in each source's metadata. When both ends of a conversation link are imported, HFN stores a typed `conversation_reply_to` edge between their passage nodes. For example, [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) declares Apache-2.0 and includes conversational message rows. [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) carries per-file original licenses and additional access/training conditions, so it must not be bulk imported until those terms and provenance requirements are satisfied. Hugging Face hosting alone is not a license grant; inspect and record the exact dataset card and revision before import.

```bash
PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.experiments.run_supervised_study \
  hpm_ai_v2/training/curricula/grounded_study_seed.jsonl \
  --output-dir /tmp/hpm-study-run
```

The run writes `training_events.jsonl`, `study_training_state.json`, `study_report.json`, and the ReaderAgent checkpoint. Reports include Wilson 95% intervals for task and evidence accuracy, abstention quality, per-task and per-episode answer scores, and median latency. It compares supervised HFN-weighted retrieval with the same encoder at neutral weights. Pass `--ollama-model MODEL` to add both a plain Ollama chat baseline and BM25 retrieval plus the same model. The Ollama baseline retrieves up to eight relevant same-user demonstrations per test question by default, from the full available train/validation history; use `--baseline-history-mode all` as a diagnostic or `--max-prior-interactions N` to change the retrieval cap. Preference examples are always restricted to the same user and preference key. Use `--encoder ollama_embedding --embedding-model qwen3-embedding:0.6b` to compare a fixed semantic embedding substrate; its parameters are not trained, while HPM learns evidence and fact weights. For semantic retrieval, HPM also applies a small BM25 corroboration bonus and allows a near-threshold answer only when strong lexical support agrees with the top semantic source. These thresholds are included in the report/checkpoint. The report records encoder/model digests, runtime placement, host details, prompts, and retrieval settings. Use `--retention-curriculum FILE.jsonl` for held-out before/after retention scoring.

The `paraphrase_transfer_seed.jsonl` fixture probes transfer to changed wording. TF-IDF HPM scored 2/5 overall (0.25 answerable evidence accuracy). With frozen Qwen embeddings and BM25 corroboration, default HPM (rescue disabled) scored 4/5 on test (0.75 answerable evidence accuracy, 0.75 answer-token F1); opt-in near-threshold rescue scored 5/5 (0.833 answer-token F1). The learned user-scoped negative pattern rejects a held-out near-negative question about temperature, with 3/3 validation resubstitution accuracy. In a matched-history rerun, HPM and Qwen3.5 9.7B Q4_K_M both scored 5/5 task accuracy after receiving the same seven train/validation interactions; answer-token F1 was 0.833 for HPM and 0.791 for Qwen3.5 (plain and BM25-retrieval variants). The older zero-history baseline scores in the archived run entries are not directly comparable to this rerun. This five-item fixture is diagnostic only, with wide confidence intervals; it does not establish 9B capability parity. Answers remain extractive and broad task-family exams are still required.

`retrieval_discrimination_90_seed.jsonl` expands the corpus to 10 topic documents compiled from 90 source passages. It contains 10 training queries, 8 confusable validation queries, 8 paraphrase test queries, and one unsupported item in each split. On the 9-episode held-out test, HPM, plain Qwen3.5 9.7B, and BM25 plus Qwen3.5 each scored 9/9 task accuracy; HPM and plain Qwen had 1.0 answer-token F1, and BM25 plus Qwen had 0.912. Neutral-weight HPM already retrieved all eight answerable topics correctly, so this run does not show an evidence-retrieval gain from supervision. Training did add 10 source-linked fact nodes and one user-scoped negative pattern; HPM rejected the unsupported query and retained identical predictions after checkpoint reload. All systems received 20 train/validation interactions, and the report records the model and encoder digests. This is still a small within-corpus transfer experiment: query wording is based on hand-authored keyword cores, each topic occurs in train and test, and the extractive gold answer is the source sentence with maximum lexical overlap. See `experiments/results/2026-10-23-retrieval-discrimination-90-report.json` and `experiments/results/2026-10-23-supervised-seed-benchmarks.json`; do not treat the result as 9B-level general usability evidence.

Supervised `abstain` episodes create persistent, user-scoped HFN negative patterns. A sufficiently similar question from that user can trigger abstention without lowering a source passage's global utility. Reports include the learned pattern count, similarity threshold, validation behavior, and abstention reason; checkpoints restore negative patterns. The transfer test covers a paraphrased unsupported question, a supported neighboring question, another user's scope, and restart behavior.

Supervised `preference` episodes store explicit user-scoped values (for example `preference_key: "summary_format"`, `expected_answer: "bullets"`). Later episodes can update the same key; the current value and supersession history persist in a typed HFN node and checkpoint. `trainer.get_preference(user_id, preference_key)` is the retrieval interface for response planners. Preference test episodes are scored by exact value recall and are kept out of evidence calibration. The Ollama comparison receives only the matching user's preference history and gets the task/key in its request context. The chat planner applies learned `summary_format` values of `bullets` or `numbered` to summaries and outlines for the matching user. Other preference keys and unrecognized values remain stored but have no automatic response effect yet; preferences are learned only from explicit supervision.

Supervised `procedure` episodes learn user-scoped trigger patterns for the supported `define`, `summarize`, `outline`, and `flashcards` tasks. A workflow becomes available to chat only after it has at least two distinct training triggers and matches a held-out validation trigger; its typed HFN node, trigger representation (hashed lexical by default or frozen semantic when configured), and provenance persist in checkpoints. The planner uses this learned routing to select the existing evidence-grounded executor, which still verifies citations. This learns when to apply a supported workflow; it does not yet learn arbitrary new multi-step execution programs. A chat transfer test covers a held-out alias, citation preservation, user scoping, and reload.

The held-out `workflow_alias_transfer_seed.jsonl` experiment has 8 training examples, 4 validation examples, and 16 test phrases across the four supported tasks. TF-IDF HPM and HPM with the frozen `qwen3-embedding:0.6b` encoder each routed all 16 test prompts correctly after all four patterns passed validation; the matched `qwen3.5:latest` 9.7B Q4_K_M model also scored 16/16 plain and 15/16 with BM25 retrieval. HPM and plain Qwen both have a 95% Wilson interval of [0.806, 1.0], while BM25+Qwen is [0.717, 0.989]. All systems received the same 12 training/validation interactions. The run recorded model digest `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`; HPM predictions matched after checkpoint reload. The test phrases vary sentence framing and topic context, but retain known task words or explicit aliases such as “recap” and “definition.” This supports parity only for this narrow workflow-routing task: four hand-selected workflow classes, four topic sources, and 16 phrases do not establish general 9B-level capability. Full run details are in `experiments/results/2026-10-23-workflow-alias-tfidf/study_report.json` and `experiments/results/2026-10-23-workflow-alias-semantic-qwen35/study_report.json`.

A paired cumulative diagnostic combined the knowledge-acquisition, retrieval-discrimination, and workflow-alias curricula: 79 training episodes, 45 phase-test items, 14 source documents, and three sequential phases. The HFN added 70 source-linked fact nodes, four workflow nodes, and one negative pattern; all three retention gates passed and predictions matched after checkpoint reload. On the current code, HPM scored 43/45 overall (95.6%, Wilson 95% CI [0.852, 0.988]); plain Qwen3.5 9.7B Q4_K_M scored 44/45 (97.8%, [0.884, 0.996]); BM25+Qwen scored 43/45 (95.6%, [0.852, 0.988]). On grounded QA HPM scored 26/28 (92.9%) versus 27/28 (96.4%) for both 9B variants; workflow routing was 16/16 for HPM and plain Qwen, 15/16 for BM25+Qwen; unsupported rejection was 1/1 for each. HPM answer-token F1 was 0.944 versus 0.919 plain Qwen and 0.929 BM25+Qwen. Median measured HPM response latency was 0.32 ms versus 1.99 seconds plain Qwen and 1.44 seconds BM25+Qwen; startup/precompute costs are not included. Confidence intervals overlap, but this is far below the preregistered 300-item, six-family target, contains only one unsupported item, and has no blinded usefulness ratings. Sources are shared and available at evaluation, so it measures within-corpus learning and workflow transfer, not knowledge from documents hidden from the assistant. Full current-code results: `experiments/results/2026-10-23-cumulative-anchor-transfer-qwen35/cumulative_study_report.json`. The same 45-item run with the frozen `qwen3-embedding:0.6b` substrate also scored 43/45 task accuracy, with identical family accuracy and retention, but lower answer-token F1 (0.847 versus 0.944 for TF-IDF) and slightly higher measured HPM median latency (0.51 ms versus 0.32 ms; embedding precompute excluded). It did not improve this corpus, so the TF-IDF run remains the primary diagnostic. Semantic run: `experiments/results/2026-10-23-cumulative-anchor-qwen35-semantic/cumulative_study_report.json`.

The expanded four-family cumulative run combines knowledge acquisition, retrieval discrimination, workflow alias transfer, and cross-document comparison: 89 train, 28 validation, and 105 held-out episodes over four phases and 14 source documents. The initial run scored HPM 91/105 (86.7%, Wilson 95% CI [0.789, 0.919]); plain Qwen3.5 9.7B scored 92/105 (87.6%, [0.800, 0.926]); BM25+Qwen scored 75/105 (71.4%, [0.622, 0.792]). By family, the initial HPM run scored 26/28 grounded QA, 16/16 workflow, 48/60 comparison, and 1/1 unsupported rejection. Plain Qwen scored 19/28, 16/16, 56/60, and 1/1 respectively; BM25+Qwen scored 24/28, 15/16, 35/60, and 1/1. HPM comparison evidence precision and recall were both 0.90. Inspection showed that HPM sometimes preferred a related distractor even when both source titles were named in the comparison request. The comparison selector now prioritizes explicit source-title mentions, prefers the longest matching title (so “solar power” does not also select “solar”), and extracts a cited sentence from explicitly named sources whose titles do not occur in their text. Rescoring the same trained HPM checkpoint on the same 105 held-out episodes produced 103/105 overall (98.1%, Wilson 95% CI [0.933, 0.995]): 26/28 grounded QA, 16/16 workflow, 60/60 comparison, and 1/1 unsupported rejection, with comparison evidence precision/recall both 1.00. This is a post-training evaluator/selection-path fix and rescore; Ollama scores were not rerun. The HFN gained 70 fact nodes, four procedure nodes, one abstention pattern, and two promoted/validated fact nodes; retention gates passed and all 105 predictions matched after checkpoint reload. Plain Qwen had four malformed-output evaluation errors and BM25+Qwen had 14, counted as incorrect episodes, so baseline scores include these harness failures. Median measured HPM latency was 0.55 ms versus 2.88 s for plain Qwen and 2.15 s for BM25+Qwen; startup costs are excluded. This result is limited to a benchmark where sources are shared, comparison uses five fixed topic pairs and hand-authored prompts, the unsupported set has one item, and there are no blinded usefulness ratings. Full initial run and corrected HPM rescore summary: `experiments/results/2026-10-23-cumulative-four-family-qwen35/cumulative_study_report.json` and `experiments/results/2026-10-23-cumulative-four-family-hpm-explicit-source-v2/rescore_summary.json`.

The correction-transfer phase adds 12 user-scoped corrected facts and 60 held-out correction questions across 12 topics. A first pass scored 43/60. Storing corrected query patterns on the user-scoped fact and reinforcing those patterns from 24 additional supervised question forms raised HPM to 55/60 (91.7%; Wilson 95% CI [0.817, 0.969]); token F1 on this family was 0.698. Across the cumulative 165-item exam HPM scored 155/165 (93.9%) and grounded answer accuracy was 63/88 at the reported 0.8 F1 threshold. Correction transfer improved by 15 percentage points; grounded QA, workflow, comparison, and abstention retention deltas were all 0, and the per-family retention gate passed. The correction graph added eight new `corrects` links during the query-pattern continuation, each from an initially wrong prediction; correction influence is user-scoped and did not change shared source weights. On the same 60 held-out correction episodes with the same 38 sources and up to eight relevant prior interactions per request, Qwen3.5 9.7B Q4_K_M scored 1/60 (1.7%, [0.003, 0.089]) plain and 12/60 (20.0%, [0.118, 0.318]) with BM25. Plain Qwen had six malformed-output errors; BM25+Qwen had none. Their evidence accuracies were 1.7% and 20.0%; BM25+Qwen answer-token F1 was 0.757 despite failing to return the exact expected source on many answers. This is a narrow correction-memory diagnostic with one correcting user and hand-authored facts, not broad 9B capability evidence. Reports: `experiments/results/2026-10-24-correction-query-augmentation/correction_augmentation_report.json` and `correction_9b_baseline_comparison.json`.

The `knowledge_acquisition_90_seed.jsonl` curriculum supervises 60 exact facts, validates on 10, and tests 20 held-out facts from 10 source topics. Training created 60 source-linked fact nodes and 60 provenance edges; HPM produced exact cited answers for 20/20 test episodes and matched after checkpoint reload. The neutral HPM retrieval baseline also scored 20/20, so this split shows knowledge growth but no retrieval improvement from the supervised weights. Qwen3.5 9.7B scored 19/20 with no demonstration history. Passing all 70 train/validation interactions caused both plain Qwen and BM25+Qwen to drop to 2/20 by repeating a training answer. With the evaluator's BM25-retrieved history (at most 8 examples; mean 2 selected), both scored 19/20, matching zero-history Qwen. HPM and Qwen's 95% intervals overlap substantially, so this exam does not show a reliable accuracy advantage for HPM. The full sources were supplied to all systems, so this is not a memory-only comparison. Results and the HPM checkpoint are in `experiments/results/2026-10-23-knowledge-acquisition-90/`.

The `correction_transfer_seed.jsonl` fixture checks a user correction across restart. Before correction, the paraphrased held-out request retrieves an earlier incorrect source; after training the correction, HPM selects the corrected source, keeps an unrelated bat answer correct, and abstains on the unsupported request. Corrections create `corrects` links in the fact graph and use stronger reinforcement/penalty settings than ordinary feedback. The runner reloads each saved checkpoint and reports whether held-out predictions remain identical. For cumulative training, `SupervisedStudyTrainer.fit_curriculum(phases, retention_episodes=...)` accepts ordered phase mappings (`name`, `sources`, `train_episodes`, and optional `validation_episodes` and `test_episodes`), reuses the same HFN state, and measures each phase’s test exam after sources are ingested but before supervision and again after fitting, exposing observed acquisition deltas. Exams from completed phases are checked before and after every later phase as retention tests. Earlier validation examples also remain in the calibration pool, while test examples are read-only and never used to fit patterns or tune thresholds. This prevents a later phase from replacing an earlier abstention threshold using only its small local validation set. Learned negative patterns retain user scope and content-word anchors; their similarity cutoff is calibrated on accumulated validation data. An optional fixed `retention_episodes` suite is scored throughout. The report includes per-phase knowledge growth, pre/post exam results, acquisition deltas, and retention gate outcomes. These remain small synthetic checks, not longitudinal usability evidence.

An answerable supervised episode may include an explicit `concept_key`. HPM stores a user-scoped `study_concept` HFN node, links its fact nodes with `instance_of` relations, and records all supporting source IDs. Candidate concepts do not affect retrieval until validation shows a correct, source-grounded held-out answer and the concept has support from at least two sources; three-source support can promote it further. The report tracks candidate/validated/promoted counts, and concept state and observer weights survive checkpoint reload. Use this field only when the examples represent the same reusable concept; an incorrect concept label can connect unrelated facts. The concept-transfer test verifies two-source linking, held-out promotion, and checkpoint persistence.

To run cumulative training from separate session files, use `run_cumulative_study.py` with repeated `--phase session.jsonl` arguments. Each phase may contain train, validation, and test episodes; its test exam is evaluated before and after phase fitting, then joins later retention checks. The pre-fit exam is read-only; its labels never update patterns or calibration. An optional separate `--retention retained_exam.jsonl` adds a fixed exam whose evidence must be available before phase one. The runner reuses one HFN state, reports per-phase fact, concept, preference and abstention-pattern growth plus acquisition deltas and retention gate results, saves a checkpoint and report, and checks predictions after reload. Optional `--ollama-model` runs plain and BM25 9B comparisons on the combined exams with the same accumulated train/validation pool, retrieving a bounded relevant subset per request by default; `--baseline-history-mode all` passes the entire history for diagnostics. The runner rejects source-content collisions and reused episode IDs across sessions. Example: `PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.experiments.run_cumulative_study --phase session-1.jsonl --phase session-2.jsonl --retention retained-exam.jsonl --output-dir /tmp/hpm-cumulative --encoder ollama_embedding --ollama-model qwen3.5:latest`.

## Grounded chat API

`ChatSession` adds a small conversational wrapper around a fitted `SupervisedStudyTrainer`. It produces an extractive answer plan, verifies each citation against the stored source offsets, abstains when no supported sentence is retrieved, and appends replayable turn events. A saved session retains recent questions so simple pronoun references can be resolved after restart.

```python
from hpm_ai_v2.chat import ChatSession

chat = ChatSession(trainer, user_id="student-1", event_log="study-turns.jsonl")
reply = chat.send("What is the main idea?")
print(reply.text, reply.plan.claims, reply.verified)
chat.save("study-session.json")
```

This is the first chat slice: grounded QA and abstention, exact citation verification, event logging, rule-based reference carry-over, clarification between tied sources, extractive summaries/outlines/flashcards, two-source evidence comparisons, and learned user-scoped summary formatting. Comparisons present separately cited source claims without synthesizing new inferences. Interactive correction feedback is not yet connected to chat.

The sixth-family continuation adds 12 simulated users with five explicit preference keys each (summary format, explanation level, example style, quiz format, and review order). It trained 60 user/key records, validated on 60 differently phrased recalls, and scored 60/60 on the held-out recall set (Wilson 95% CI [0.940, 1.000]). All five earlier family retention gates passed against the exact same 165-item exam with unchanged accuracy, giving a 225-item, six-family held-out total. Every test pair had an explicit training example, so this establishes user-scoped storage and exact recall under phrasing changes; it does not establish generalization to unseen preferences or user-rated usefulness. At that checkpoint the abstention family still had one item and the exam was below 300; the balanced continuation below expands both. Report and checkpoint: `experiments/results/2026-10-24-six-family-preference/preference_phase_report.json` and `experiments/results/2026-10-24-six-family-preference/`.

The balanced six-family continuation adds 60 held-out grounded-QA questions, 60 unsupported requests matched to supported-neighbor validation prompts, and 60 varied workflow-routing prompts. HPM added six source-linked fact nodes and validated nine fact patterns, learned 12 additional negative patterns, and reinforced eight new workflow triggers. On the 405-item cumulative exam it scores 385/405 (95.1%); each family has at least 60 test items, and all three phase retention gates pass. By family: grounded QA 73/88, correction 55/60, workflow 76/76, cross-document comparison 60/60, preference recall 60/60, and abstention 61/61. The new 60-item QA phase has 49/60 correct sources and 46/60 sufficiently matched cited answers (F1 >= 0.8); mean answer-token F1 is 0.790. On the matched new 180 episodes, HPM scores 169/180, while Qwen3.5 9.7B Q4_K_M with BM25 scores 165/180. The Qwen BM25 QA answer-token F1 is higher on this slice (0.921 vs HPM's 0.790). Plain Qwen scores 52/180 under the exact source-ID and workflow-label protocol. A subsequent full-exam run gave HPM 385/405 (95.1%, Wilson 95% CI [0.925, 0.968]) and BM25+Qwen 288/405 (71.1%, [0.667, 0.753]), with the same 38 sources and 357 train/validation demonstrations available to both and a relevant-history cap of eight for Qwen. Per family, BM25+Qwen scored 67/88 grounded QA, 12/60 correction, 14/60 comparison, 74/76 workflow, 60/60 preferences, and 61/61 abstention. HPM’s aggregate answer-token F1 is 0.832 vs Qwen’s 0.885; its task accuracy advantage is concentrated in comparison and correction. The source corpus is shared and the questions are hand-authored, so this is evidence of strong performance on this supervised in-corpus benchmark, not general/open-world 9B capability or user-rated usefulness. Reloading the saved checkpoint reproduced all 405 predictions exactly after normalizing the JSON tuple/list representation, with identical 385/405 accuracy. A blinded usefulness packet is now prepared for 329 answer-bearing cases; the routing-only procedure items are excluded and no human ratings have been collected. The packet and separate private key are in `experiments/results/2026-10-24-balanced-six-family-corrected/blinded-usefulness-study/` and `private_blinding_key.json`. Full comparison: `experiments/results/2026-10-24-balanced-six-family-corrected/full405_9b_comparison.json`; detailed system reports: `balanced_six_family_report.json`, `qwen_bm25_full405.json`, and `restart_consistency.json` in the same directory.

### Knowledge acquisition validation and scaling

`knowledge_acquisition_growth_150_seed.jsonl` pairs every one of its 60 train facts with a differently worded validation query whose exact answer is source-supported; it retains ten other validation items and a disjoint 20-item test exam. HPM added and validated all 60 fact nodes. The 20-item test was already 20/20 before supervision and remained 20/20 after, so this run verifies stored fact growth and retention, not an accuracy increase. Matched Qwen3.5 9.7B scored 14/20 plain and 18/20 with BM25 retrieval; HPM answer-token F1 was 1.0 versus .933 and .908. This is a small in-corpus diagnostic, not broad capability parity. The cumulative report records acquisition coverage and restart consistency at `experiments/results/2026-10-24-knowledge-acquisition-growth-150-fixed/cumulative_study_report.json`.

The corrected six-family continuation restores the checkpoint's 100 prior validation examples during abstention recalibration. A subsequent source-label audit found and fixed eight held-out QA answers that had been paired with neighboring sentences. The matched 405-item gold-audited exam scores HPM 387/405 (95.6%), Qwen3.5 9.7B plain 125/405 (30.9%), and BM25+Qwen 299/405 (73.8%). HPM answer-token F1 is 0.860; BM25+Qwen's is 0.873. All three retention gates pass, and all 405 HPM predictions match after checkpoint reload. The new QA phase adds ten fact nodes, validates eight, and improves held-out source selection from 49/60 to 50/60; its answer-token F1 rises from 0.808 to 0.839. This hand-authored, shared-source benchmark does not establish broad 9B capability parity. Full matched summary: `experiments/results/2026-10-24-balanced-six-family-corrected/growth_validated_goldaudit_matched_405_summary.json`; detailed report: `experiments/results/2026-10-24-balanced-six-family-corrected/growth-validated-goldaudit/cumulative_study_report.json`.

The current-code acquisition rerun trained on 60 facts, validated all 60 against held-out paraphrases, and saved all 60 as validated fact nodes in the checkpoint. It reproduced its 20 held-out predictions after reload. HPM scored 20/20 before and after supervision because the sources were already available before fitting; this verifies validated, persistent fact acquisition but not an accuracy gain from training on this already-saturated exam. BM25+Qwen3.5 9.7B Q4_K_M scored 18/20, with answer-token F1 0.908 vs HPM's 1.0. The acceptance audit now separates `validated_knowledge_growth` from `knowledge_transfer_effect`: the former checks coverage, persisted facts, and restart consistency; the latter records held-out accuracy gain, regression, or a saturated/no-gain result. Run: `experiments/results/2026-10-24-knowledge-acquisition-growth-150-current/cumulative_study_report.json`.

### Current-code continuation audit (2026-10-24)

Re-evaluating the older 387/405 checkpoint under the current code gives 370/405, so that older score is not directly comparable to current results. The current three-phase continuation adds ten source-linked facts and validates all ten on held-out answers; 17 fact nodes are validated/promoted in the resulting checkpoint, and all 405 predictions match after restart. The cumulative exam is 357/405 (88.1%), versus 288/405 (71.1%) for matched Qwen3.5 9.7B Q4_K_M with BM25 top-three retrieval. HPM's answer-token F1 is 0.808 versus Qwen's 0.867. HPM leads task accuracy on this shared hand-authored benchmark, but correction transfer remains 32/60 and human usefulness is unmeasured. The audit therefore passes task completion, knowledge growth, and retention, while overall acceptance remains false. Detailed report: `experiments/results/2026-10-24-balanced-six-family-corrected/growth-validated-final-scoped/cumulative_study_report.json`; comparison: `experiments/results/2026-10-24-balanced-six-family-corrected/full405_9b_comparison_current.json`; gates: `experiments/results/2026-10-24-balanced-six-family-corrected/training_acceptance_audit_current.json`.

Correction-transfer follow-up tightened transfer matching so an accepted correction cannot alter answers to nearby but distinct collateral questions. On the latest v4 correction run, 13/16 correction queries transfer correctly (answer F1 0.945), all 8 collateral controls remain unchanged, and restart predictions match. Only 7/8 corrected facts pass held-out source validation, so the correction-learning gate remains failed. The acceptance audit now requires that validation gate to pass and reports the validated/total fact counts; the overall 9B usability decision remains not accepted because grounded answer-token F1 is below the comparator and blinded usefulness/user effort are unmeasured. See `experiments/results/2026-10-24-balanced-six-family-corrected/correction-transfer-v4-r6/cumulative_study_report.json` and `training_acceptance_audit.json`.

### Calibrating abstention for a pretrained checkpoint

A checkpoint that was only pretrained has no validation episodes, so its abstention threshold is 0.0 and the chat answers any question that shares a word with the corpus. Calibrate it from labelled questions (an answer is a quoted snippet, `null` for a question the corpus cannot answer):

```bash
PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.chat.calibrate /tmp/hpm-pretrained-checkpoint \
  --validation validation.jsonl --test held_out.jsonl --output calibration.json
PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.chat.server /tmp/hpm-pretrained-checkpoint \
  --data-dir /tmp/hpm-chat --calibration calibration.json
```

This runs the trainer's own `fit` calibration and stores the threshold in a small file bound to the checkpoint; the checkpoint is not modified. On the 76-source test corpus, the committed question sets (`training/eval_sets/pretrained_calibration_*.jsonl`) moved held-out out-of-scope abstentions from 7 of 24 to 19 of 24 while right-source in-scope answers went from 16 of 24 to 15 of 24. On held-out summary, outline, flashcard and comparison requests the same threshold held up, so no per-request-type threshold is needed. Details and limits are in `conductor/2026-10-23-hpm-v2-generation-chat-design.md`.
