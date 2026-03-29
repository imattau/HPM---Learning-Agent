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

Takes the mu-vector of the detected gap (what region of observation space is underrepresented) and optional context. Returns a list of raw strings — the raw response from the external field. No encoding, no HFN logic — just retrieval.

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

Observer gains two new optional constructor parameters:

```python
query: Query | None = None
converter: Converter | None = None
```

Both default to `None`. No behaviour changes for existing experiments.

When both are set, Observer calls the query pipeline at the end of `_check_residual_surprise` — after the normal node creation decision, only when a gap signal is high:

```python
def _check_gap_query(self, x: np.ndarray) -> None:
    if self.query is None or self.converter is None:
        return
    gap = self.evaluator.coverage_gap(x, self.forest.active_nodes(),
                                       self.lacunarity_creation_radius)
    if gap < self._gap_query_threshold:
        return

    # Check if we've queried this region before
    gap_hash = _mu_hash(x)
    query_id = f"query_{gap_hash}"
    if query_id in self.forest:
        return  # already explored this region

    # Fetch + encode
    raw = self.query.fetch(x)
    if not raw:
        return
    observations = self.converter.encode(raw, D=x.shape[0])
    if not observations:
        return

    # Store Query HFN
    query_node = HFN(mu=x.copy(), sigma=np.eye(x.shape[0]), id=query_id)

    # Store Knowledge Gap children + observe them
    for i, obs_vec in enumerate(observations):
        gap_node = HFN(mu=obs_vec, sigma=np.eye(obs_vec.shape[0]),
                       id=f"gap_{gap_hash}_{i}")
        query_node.add_child(gap_node)
        self.register(gap_node)
        self.observe(obs_vec)  # feed through normal pipeline

    self.register(query_node)
```

`_gap_query_threshold` is a new Observer config parameter (default: 0.7 — only query when the region is substantially underrepresented).

`_mu_hash(mu)` is a private helper that produces a stable short hash from a mu-vector for use in node ids.

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

Converts gap_mu to a token (nearest vocab token to gap_mu), then queries Python's `inspect` module for functions in the stdlib that involve that token. Returns a list of function signatures as strings.

Example: gap near `range` token → queries `inspect.getmembers(builtins)` for functions with `range`-related signatures → returns `["range(stop)", "range(start, stop[, step])"]`

### 7.4 ConverterCode (`hpm_fractal_node/code/converter_code.py`)

```python
class ConverterCode(Converter):
    def encode(self, raw: list[str], D: int) -> list[np.ndarray]
```

Tokenises each raw string using the code vocabulary. For each tokenised string, calls `compose_context_node` on the tokens to produce a D-dimensional observation. Returns list of observation vectors.

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
