# Query, Converter, and Code Experiment — Design Spec

**Date:** 2026-03-29
**Status:** Approved

---

## 1. Motivation

The Observer can now detect gaps in its knowledge via `evaluator.coverage_gap()` and `evaluator.underrepresented_regions()`. The next step is to act on those gaps: query an external field (a library, an LLM, a web search) for information, encode the response as observations, and feed them back through the normal learning pipeline.

This implements the HPM **field/substrate** concept: the external world provides raw signal; the Observer processes it using the same mechanism it uses for everything else. The Query and Converter infrastructure makes this field interface reusable and swappable — `QueryStdlib` today, `QueryLLM` later, same Observer code.

The first experiment using this infrastructure is a **code token pattern experiment**: the Observer learns structural patterns of Python code, detects gaps, and fills them by querying the Python stdlib.

---

## 2. Architecture

```
Observer
  ├── Evaluator       — what do I know / not know?
  ├── Recombination   — absorb, compress
  └── Query (optional) — when gap detected, fetch + encode
        ↓
      Query HFN  (stored in Forest — what was asked)
        └── Knowledge Gap HFN  (child — what was learned)
```

**Query** and **Converter** are base classes in `hfn/` — domain-agnostic. Concrete implementations (`QueryStdlib`, `ConverterCode`) live in `hpm_fractal_node/code/` alongside the code experiment.

**Observer** gains `query` as an optional collaborator (defaults to `None` — existing experiments unaffected). When a gap is detected and `query` is set, the Observer calls the query pipeline and stores the result as a Query HFN + child Knowledge Gap HFN.

---

## 3. Query Base Class (`hfn/query.py`)

```python
class Query:
    def fetch(self, gap_mu: np.ndarray, context: dict | None = None) -> list[str]
```

Takes the mu-vector of the detected gap and an optional context dict. Returns a list of raw strings — the raw response from the external field. No encoding, no HFN logic — just retrieval.

**Context dict** (provided by Observer, Query uses what it needs):
- `explanation_tree`: list of HFN nodes that explained `gap_mu` at the current expansion depth
- `accuracy_scores`: dict of node_id → accuracy for those nodes
- `residual_surprise`: float — how much of gap_mu remains unexplained

Query subclasses use `context["explanation_tree"]` to understand what the Observer already knows near the gap before deciding what to fetch. If `context` is `None` or `explanation_tree` is empty, Query fetches based on `gap_mu` alone.

Returning an empty list signals insufficient information — Observer will deepen expansion and retry.

Subclasses: `QueryStdlib`, `QueryLLM`, `QueryWebSearch` (all in experiment layer).

---

## 4. Converter Base Class (`hfn/converter.py`)

```python
class Converter:
    def encode(self, raw: list[str], D: int) -> list[np.ndarray]
```

Takes raw strings returned by Query and a target dimensionality D. Returns a list of D-dimensional numpy arrays — observations ready to feed through `Observer.observe()`.

Subclasses: `ConverterCode`, `ConverterText` (all in experiment layer).

The base class provides no default implementation — subclasses must define `encode`. Each subclass pairs naturally with one or more Query subclasses (e.g. `QueryStdlib` + `ConverterCode`).

---

## 5. Query HFNs and Knowledge Gap HFNs

Every query event is stored as a pair of HFN nodes in the Observer's Forest:

**Query HFN**
- `id`: `query_{hash}` where hash is derived from the gap_mu (stable across repeated identical gaps)
- `mu`: the gap_mu vector — encodes what was asked
- Subject to normal Observer dynamics (not protected) — stale queries can be absorbed

**Knowledge Gap HFN** (child of Query HFN)
- `id`: `gap_{hash}_{i}` for each encoded observation returned
- `mu`: the encoded observation from the library response
- Wired as a child of its Query HFN via `query_node.add_child(gap_node)` before registration
- Also fed through `Observer.observe()` as a new observation — the Observer learns from it immediately

The parent-child relationship captures provenance: "this knowledge came from that query." Multiple queries about similar gaps may produce similar Knowledge Gap HFNs — Observer compression naturally handles deduplication over time.

**Query HFNs answer:** "Have I asked about this before?" — find nearest Query HFN to current gap_mu.
**Knowledge Gap children answer:** "What did I learn from that query?"

---

## 6. Observer Integration

Observer gains these new optional constructor parameters:

```python
query: Query | None = None
converter: Converter | None = None
gap_query_threshold: float = 0.7      # only query when coverage_gap exceeds this
max_expand_depth: int = 2             # max expansion retries before giving up
```

Both `query` and `converter` default to `None`. No behaviour changes for existing experiments.

`_check_gap_query` is called as a third step inside `_check_node_creation`, alongside (not inside) `_check_residual_surprise` and `_check_compression_candidates`:

```python
def _check_node_creation(self, x, result):
    self._check_residual_surprise(x, result)
    self._check_compression_candidates()
    self._check_gap_query(x)
```

**Reentrancy guard:** `_check_gap_query` feeds new observations into the Observer pipeline. To prevent recursion, Observer holds a `_in_gap_query: bool = False` flag. `_check_gap_query` returns immediately if already set.

**`_mu_hash(mu)`:** Returns `str(int(np.argmax(mu)))` — the index of the highest-weighted vocabulary dimension. Stable, cheap, and appropriate for the one-hot-ish vocabulary space. Collisions are intentional: two gaps near the same vocabulary token resolve to the same query.

**Expansion retry loop:** Observer runs `_expand(gap_mu)` and passes the `ExplanationResult` as context to `query.fetch()`. If fetch returns empty, Observer deepens expansion by one budget level and retries, up to `max_expand_depth` attempts. Observer owns this loop entirely — Query just receives richer context on each retry.

```python
def _check_gap_query(self, x: np.ndarray) -> None:
    if self.query is None or self.converter is None:
        return
    if self._in_gap_query:
        return  # reentrancy guard

    gap = self.evaluator.coverage_gap(x, self.forest.active_nodes(),
                                       self.lacunarity_creation_radius)
    if gap < self.gap_query_threshold:
        return

    gap_hash = str(int(np.argmax(x)))
    query_id = f"query_{gap_hash}"
    if query_id in self.forest:
        return  # already explored this region

    # Expansion retry loop — deepen until Query has enough context or limit reached
    raw = []
    for depth in range(self.max_expand_depth):
        result = self._expand_with_budget(x, budget=self.budget + depth)
        context = {
            "explanation_tree": result.explanation_tree,
            "accuracy_scores": result.accuracy_scores,
            "residual_surprise": result.residual_surprise,
        }
        raw = self.query.fetch(x, context=context)
        if raw:
            break  # Query has enough context

    if not raw:
        return  # exhausted expansion depth, nothing to query

    observations = self.converter.encode(raw, D=x.shape[0])
    if not observations:
        return

    D = x.shape[0]

    # Register Query HFN first — anchors the query event before any children
    query_node = HFN(mu=x.copy(), sigma=np.eye(D), id=query_id)
    self.register(query_node)

    # Knowledge Gap HFNs are priors: stable external knowledge, not subject to dynamics
    self._in_gap_query = True
    try:
        for i, obs_vec in enumerate(observations):
            gap_id = f"gap_{gap_hash}_{i}"
            gap_node = HFN(mu=obs_vec, sigma=np.eye(obs_vec.shape[0]), id=gap_id)
            query_node.add_child(gap_node)
            self.register(gap_node)
            self.protected_ids.add(gap_id)
            # Feed through pipeline without triggering another gap query
            inner = self._expand(obs_vec)
            self._update_weights(obs_vec, inner)
            self._update_scores(inner)
            self._track_cooccurrence(inner.explanation_tree)
            self._check_absorption()
            self._check_residual_surprise(obs_vec, inner)
            self._check_compression_candidates()
    finally:
        self._in_gap_query = False
```

**`_expand_with_budget(x, budget)`:** A thin variant of `_expand` that accepts an explicit budget override, used only in the expansion retry loop. Implemented as `_expand` with a temporary budget substitution.

**Knowledge Gap HFNs as priors:** Query responses come from a stable external source. They should not be absorbed or compressed by Observer dynamics. The Query HFN itself is NOT protected — it can be absorbed if the gap becomes well-covered.

---

## 7. Code Experiment Structure

### 7.1 Vocabulary (`hpm_fractal_node/code/code_loader.py`)

Python code token vocabulary — keywords, builtins, and common operators/punctuation. Approximately 60–80 tokens. Same structure as `nlp_loader.py`:

- `VOCAB`: ordered list of code tokens
- `VOCAB_INDEX`: dict mapping token → index
- `VOCAB_SIZE`: int (D)
- `compose_context_node(left2, left1, right1, right2) → np.ndarray`: same slot-weighted composition as NLP (0.2/0.35/0.35/0.1)
- `generate_code_snippets(seed) → list[(vec, true_token, category)]`: generates observations from synthetic Python snippets across categories: control_flow (`if`, `for`, `while`), functions (`def`, `return`, `lambda`), data (`list`, `dict`, `int`, `str`), builtins (`print`, `len`, `range`, `type`)

### 7.2 World Model (`hpm_fractal_node/code/code_world_model.py`)

Code structure priors — same pattern as `nlp_world_model.py`:

- 60–80 atomic token nodes (one-hot in D-space)
- **Syntax sub-tree**: keywords (`if`, `for`, `while`, `def`, `class`, `return`, `import`, `from`, `with`, `try`, `except`), operators (`=`, `==`, `!=`, `<`, `>`, `+`, `-`, `*`, `/`), punctuation (`(`, `)`, `:`, `,`, `.`, `[`, `]`)
- **Type sub-tree**: `int`, `str`, `float`, `bool`, `list`, `dict`, `tuple`, `None`
- **Builtin sub-tree**: `print`, `len`, `range`, `type`, `input`, `open`, `map`, `filter`, `zip`, `enumerate`
- **Pattern sub-tree**: common co-occurrence patterns — `for_loop` (recombine `for`, `in`, `range`), `if_condition` (recombine `if`, `==`, `!=`), `function_def` (recombine `def`, `(`, `)`:`), `assignment` (recombine token, `=`, value tokens)
- ~15 sentence-equivalent priors: short synthetic code snippets, mu = equal-weight recombination of constituent token mus

### 7.3 QueryStdlib (`hpm_fractal_node/code/query_stdlib.py`)

```python
class QueryStdlib(Query):
    def fetch(self, gap_mu: np.ndarray, context: dict | None = None) -> list[str]
```

Queries the Python stdlib **source code** for real token co-occurrence patterns near the gap.

1. `gap_mu → argmax → nearest vocab token` (e.g. `"range"`)
2. Optionally uses `context["explanation_tree"]` to refine the target token if the argmax is ambiguous
3. Scans stdlib `.py` files (via `sysconfig.get_paths()["stdlib"]`) for occurrences of that token using Python's `tokenize` module
4. For each occurrence, extracts the 4-token context window as a space-separated string: `"left2 left1 right1 right2"` — keeping only tokens present in VOCAB, substituting `"<unk>"` for out-of-vocabulary tokens
5. Returns up to `max_results` (default: 20) context window strings

Returns empty list if no occurrences found — Observer will deepen expansion and retry with broader context.

The stdlib source IS Python code with the same token patterns the experiment is learning. This produces observations in exactly the format the main training pipeline expects.

### 7.4 ConverterCode (`hpm_fractal_node/code/converter_code.py`)

```python
class ConverterCode(Converter):
    def encode(self, raw: list[str], D: int) -> list[np.ndarray]
```

Each raw string is a space-separated 4-token context window (`"left2 left1 right1 right2"`). Splits the string, looks up each token in `VOCAB_INDEX` (skips `"<unk>"`), calls `compose_context_node(left2, left1, right1, right2)` directly. Returns one D-dimensional observation per raw string.

This is the same encoding as `code_loader.py` — no separate tokenisation step. Observations from the stdlib query are identical in format to observations from the main training data. World model priors and Knowledge Gap HFNs live in the same representation space naturally.

### 7.5 Experiment (`hpm_fractal_node/experiments/experiment_code.py`)

Same structure as `experiment_nlp.py`:
- `N_SAMPLES = 2000`, `N_PASSES = 3`, `SEED = 42`
- Builds code world model, creates Observer with `query=QueryStdlib()`, `converter=ConverterCode()`
- Runs 3-pass observation loop
- Reports category purity for syntax / type / builtin / pattern categories
- Reports Query HFN count (how many gaps were explored)
- Reports Knowledge Gap HFN count (how many observations came from library)

---

## 8. File Changes

| File | Action |
|------|--------|
| `hfn/query.py` | New — base Query class |
| `hfn/converter.py` | New — base Converter class |
| `hfn/observer.py` | Add `query`, `converter` params; add `_check_gap_query`, `_mu_hash` |
| `hfn/__init__.py` | Export Query, Converter |
| `hpm_fractal_node/code/__init__.py` | New (empty) |
| `hpm_fractal_node/code/code_loader.py` | New |
| `hpm_fractal_node/code/code_world_model.py` | New |
| `hpm_fractal_node/code/query_stdlib.py` | New |
| `hpm_fractal_node/code/converter_code.py` | New |
| `hpm_fractal_node/experiments/experiment_code.py` | New |
| `tests/hfn/test_query.py` | New |
| `tests/hfn/test_converter.py` | New |
| `tests/hpm_fractal_node/code/test_code_loader.py` | New |
| `tests/hpm_fractal_node/code/test_query_stdlib.py` | New |
| `tests/hpm_fractal_node/code/test_converter_code.py` | New |

---

## 9. What Does Not Change

- `hfn/evaluator.py`, `hfn/recombination.py` — unchanged
- NLP experiment — unchanged (Observer `query=None` by default)
- All existing tests — must pass without modification

---

## 10. HPM Framework Alignment

| Component | HPM Role |
|-----------|----------|
| `Query` | Field interface — external world provides raw signal |
| `Converter` | Substrate mediator — translates field signal to observation space |
| Query HFN | Pattern substrate — the Observer's memory of what it has asked |
| Knowledge Gap HFN | Pattern substrate — the knowledge that filled the gap, with provenance |
| `_check_gap_query` | Pattern dynamics — Observer acts on meta-awareness to seek new patterns |

The Query/Converter loop is the first implementation of the full HPM cycle: Observer detects its own ignorance → queries the field → field returns signal → substrate encodes it → Observer learns from it.

---

## 11. Extension Points

- `QueryLLM`: replace `fetch()` with LLM API call — same storage pattern, richer responses
- `QueryWebSearch`: replace `fetch()` with web search — same storage pattern
- `ConverterText`: encode natural language responses — enables cross-domain gap filling
- Query HFN weighting: Observer could track which queries produced high-utility Knowledge Gap children (by monitoring child node weights) — future: deprioritise queries that consistently produce low-weight children
