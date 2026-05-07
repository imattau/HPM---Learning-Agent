# HPM v5 Adapter & Preprocessor Reference

All adapters implement the `Adapter` protocol: `name`, `requires`, `provides`, `run(packet) -> packet`.
All preprocessors implement `Preprocessor`: `name`, `run(raw) -> PreprocessedInput`.

---

## NLP Adapters (`hpm_ai_v5.adapter.nlp`)

| Class | requires | provides | Notes |
|-------|----------|----------|-------|
| `NLPTokenizer` | — | `tokens`, `pos_tags`, `lemmas` | spaCy `en_core_web_sm`; filters PUNCT/SPACE |
| `CanonicalPhraser` | `nlp_tokenizer` | `canonical_tokens`, `state` | Lemma → concept map (`WEATHER`, `FLIGHT`, etc.) |
| `SkeletonExtractor` | `nlp_tokenizer` | `skeleton`, `state` | POS → groups: `N`/`V`/`D`/`P`/`R`/`C`/`A`; `AUX→V`, `PART→R` |
| `SkeletonNgramAdapter` | `skeleton_extractor` | `skeleton_ngrams` | Bigrams in `context["skeleton_ngrams"]` only — NOT in `packet.states` |
| `DeltaEncoder` | `skeleton_extractor` | `delta`, `state` | First-differing-position delta between consecutive skeletons; has `reset()` |
| `KnowledgeBaseLookup` | `nlp_tokenizer` | `semantic_candidates` | WordNet synonyms, max 5; `context["semantic_candidates"]` |
| `NL2CodeBridgeAdapter` | `canonical_phraser` | `bridge_tokens`, `states` | NL tokens → `U_*` IDs (`U_IF`, `U_WHILE`, `U_TRY`, `U_FOR`, `U_ASSIGN`, `U_RETURN`, `U_CALL`, `U_THROW`) |
| `StartOfEpisodeAdapter` | — | — | Injects episode boundary signal into context |

---

## ATIS Adapter (`hpm_ai_v5.adapter.atis`)

| Class | Notes |
|-------|-------|
| `IntentLabelAdapter` | Injects `context["intent_label"]` during training; omit `label=None` for inference |

```python
def load_atis() -> tuple[list[dict], list[dict]]:
    """Returns (train, test) as lists of {text, intent} dicts. Requires `datasets` package."""
```

---

## CLT / Code Adapters (`hpm_ai_v5.adapter.clt`, `hpm_ai_v5.adapter.code`)

| Class | Notes |
|-------|-------|
| `UnifiedVocabulary` | Global string→int mapping. `get_id(token) -> int` |
| `UnifiedASTFlattener` | Linearises Python/Java AST to `U_*` universal sequence |
| `UnifiedStateAdapter` | Maps universal AST nodes to `UnifiedVocabulary` IDs |
| `CLTRefinementAdapter` | Post-processes CLT adapter output |
| `LanguageDetector` | Detects source language for routing |
| `ASTFlattener` | Python-specific AST flattener (`adapter.code`) |
| `CodeTokenizer` | Tokenises source code |
| `CodeStateAdapter` | Maps code tokens to states |
| `CanonicalRenamer` | Normalises variable/function names |
| `CodeRefinementAdapter` | Post-processes code adapter output |

---

## Physics / Control Adapters (`hpm_ai_v5.adapter.physics`)

| Class | Notes |
|-------|-------|
| `CartpoleStateAdapter` | Normalises CartPole obs + action history buffer to state tuple |
| `AcrobotStateAdapter` | Normalises Acrobot obs to state tuple |
| `RunningNormaliserAdapter` | Online mean/variance normalisation |
| `RewardToGoalAdapter` | Converts scalar reward to goal vector |
| `TDErrorAdapter` | Computes TD error signal for Q-learning |
| `RewardAdapter` (`adapter.reward`) | Wraps reward signal as a state |

---

## Buffer / Memory Adapters

| Class | Module | Notes |
|-------|--------|-------|
| `RecentBufferAdapter` | `adapter.recent_buffer` | Maintains sliding window of recent states |
| `DeltaBufferAdapter` | `adapter.delta_buffer` | Maintains sliding window of recent deltas |
| `TrajectoryBufferAdapter` | `adapter.trajectory_buffer` | Stores full episode trajectory for batch learning |
| `ActionSequenceUnpacker` | `adapter.action_sequence_unpacker` | Unpacks multi-step action sequences from state |

---

## Grid / ARC Adapters

| Class | Module | Notes |
|-------|--------|-------|
| `FlattenGridAdapter` | `adapter.flatten_grid` | Flattens 2D grid to 1D state tuple |
| `ConnectedComponentsAdapter` | `adapter.connected_components` | Extracts connected regions from grid |
| `GridPostprocessor` | `adapter.grid_postprocessor` | Validates and formats grid output |

---

## Utility Adapters

| Class | Module | Notes |
|-------|--------|-------|
| `NumericAdapter` | `adapter.numeric` | Normalises numeric observations |
| `ChangepointAdapter` | `adapter.changepoint` | Detects distribution shifts in reward/polygraph signals |
| `PatternStoreSizeAdapter` | `adapter.store_size` | Exposes store size as a state signal |
| `ValidationOnlyAdapter` | `adapter.validation_only` | Pass-through; used in benchmarks |
| `AdapterRegistry` | `adapter.registry` | Dynamic adapter lookup by name |

---

## Preprocessors (`hpm_ai_v5.preprocessors`)

Preprocessors run before the adapter pipeline on raw numeric/signal input.

| Class | Notes |
|-------|-------|
| `PrefixBufferPreprocessor` | Maintains last-N raw symbols for context disambiguation (fixes PDT) |
| `StateFusionPreprocessor` | Merges multiple state signals into one |
| `NormalisationPreprocessor` | Z-score or min-max normalisation |
| `DifferencingPreprocessor` | First-order differencing (stationarity) |
| `RollingStatsPreprocessor` | Rolling mean/std/min/max window |
| `AutocorrelationPreprocessor` | Lag-k autocorrelation features |
| `EntropyPreprocessor` | Shannon entropy of recent symbol distribution |
| `SymbolicDiscretiser` | Discretises continuous signal to symbolic alphabet |
| `NumericPreprocessor` | Generic numeric feature extraction |
