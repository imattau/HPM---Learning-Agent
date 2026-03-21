# Linguistic and Math Substrate Design Specification

**Date:** 2026-03-21
**Status:** Draft v1

---

## Overview

Two new `ExternalSubstrate` implementations that give the HPM agent grounded baseline knowledge from real reference sources:

- **`LinguisticSubstrate`** — backed by NLTK/WordNet (required), Free Dictionary API (optional), and spaCy (optional)
- **`MathSubstrate`** — backed by SymPy (required), SciPy constants (optional), and Wolfram Alpha (optional)

Both follow the existing `ExternalSubstrate` protocol (`fetch`, `field_frequency`, `stream`) and the pattern established by `WikipediaSubstrate`. Both use `hash_vectorise` from `hpm/substrate/base.py` for text→vector conversion.

---

## 1. LinguisticSubstrate

### 1.1 File

`hpm/substrate/linguistic.py`

### 1.2 Constructor

```python
def __init__(
    self,
    feature_dim: int = 32,
    timeout: float = 5.0,
    use_api: bool = True,
    use_spacy: bool = True,
):
```

At construction:
- Import NLTK and ensure `wordnet` corpus is available (`nltk.data.find('corpora/wordnet')`). Raise `ImportError` if NLTK is not installed. Raise `LookupError` if the corpus is not downloaded (caller must run `nltk.download('wordnet')` and `nltk.download('words')`).
- If `use_spacy=True`, attempt `import spacy; spacy.load('en_core_web_sm')`. If this fails (not installed or model missing), set `self._nlp = None` silently.
- If `use_api=True`, set `self._use_api = True` (no network check at init; failures degrade per-call).

### 1.3 Protocol Implementation

#### `fetch(query: str) -> list[np.ndarray]`

Returns a list of fixed-dim float vectors representing the linguistic content of `query`. Results are cached per query.

All three components run independently. Results from each are collected and combined. Return `[]` only if all active components produce nothing.

**WordNet component (always active):**
- Look up synsets: `nltk.corpus.wordnet.synsets(query)`
- For each synset, collect its definition string and example sentences
- Apply `hash_vectorise(text, self.feature_dim)` to each non-empty string
- If no synsets found, this component contributes nothing (other components still run)

**Dictionary API component (if `use_api=True`):**
- GET `https://api.dictionaryapi.dev/api/v2/entries/en/{query}` with `timeout=self.timeout`
- On network error or non-200 response: skip silently
- Parse JSON: collect all `meanings[*].definitions[*].definition` and `example` strings
- Apply `hash_vectorise` to each non-empty string
- Append to results list

**spaCy component (if `self._nlp is not None`):**
- Parse query string: `doc = self._nlp(query)`
- Build POS-tag frequency vector: count tokens per Universal POS tag using the 17-tag UD v2 set: `NOUN`, `VERB`, `ADJ`, `ADV`, `PRON`, `DET`, `ADP`, `NUM`, `PUNCT`, `SYM`, `X`, `PART`, `INTJ`, `PROPN`, `AUX`, `CCONJ`, `SCONJ` — 17 tags total
- Normalise by total token count (frequency per tag); return zero vector if empty doc
- Zero-pad or truncate to `feature_dim`
- Append as one vector to results list

**Return:** combined list of all vectors from active components. Empty list only if all components produced nothing.

#### `field_frequency(pattern) -> float`

Same pattern as `WikipediaSubstrate`:
- Query = `str(getattr(pattern, 'label', None) or 'word')`
- Fetch vectors for query
- Compute mean cosine similarity of `pattern.mu[:feature_dim]` against each vector
- Rescale from `[-1, 1]` to `[0, 1]` via `(sim + 1) / 2`
- Return `float` in `[0, 1]`; return `0.0` if no vectors

#### `stream() -> Iterator[np.ndarray]`

- Load word list: `nltk.corpus.words.words()` — filtered to alphabetic-only words, length 4–12
- Cycle through this list indefinitely (use `itertools.cycle`)
- For each word, call `self.fetch(word)` and yield each vector in the result
- Skip words that return empty fetch results
- **Note:** when `use_api=True`, each word triggers an HTTP request to the dictionary API. Callers consuming the stream continuously should be aware of this. Use `LinguisticSubstrate(use_api=False)` for offline streaming.

### 1.4 Caching

`_cache: dict[str, list[np.ndarray]]` — per-query, in-memory, no TTL. Same as `WikipediaSubstrate`.

---

## 2. MathSubstrate

### 2.1 File

`hpm/substrate/math.py`

### 2.2 Constructor

```python
def __init__(
    self,
    feature_dim: int = 32,
    timeout: float = 5.0,
    use_scipy: bool = True,
    wolfram_app_id: str | None = None,
):
```

At construction:
- Import SymPy. Raise `ImportError` if not installed.
- If `use_scipy=True`, attempt `import scipy.constants`. If not installed, set `self._scipy_constants = None` silently.
- If `wolfram_app_id` is not None, store it; if None, check `os.environ.get('WOLFRAM_APP_ID')`; if still None, Wolfram component is inactive.

### 2.3 Topic Expression Map

A module-level dict mapping topic names to lists of SymPy expression strings, parsed once at module load:

```python
_TOPIC_EXPRS: dict[str, list[str]] = {
    "algebra":       ["x**2 + x + 1", "x**2 - 4", "2*x + 3", "x**3 - x", "a*x**2 + b*x + c"],
    "calculus":      ["sin(x)", "exp(x)", "log(x)", "x**2 * sin(x)", "1/(1 + x**2)"],
    "statistics":    ["(x - mu)/sigma", "exp(-x**2/2)", "x*(1-x)", "n*p", "p**k * (1-p)**(n-k)"],
    "geometry":      ["pi*r**2", "4*pi*r**3/3", "sqrt(a**2 + b**2)", "2*pi*r", "b*h/2"],
    "number_theory": ["2**n - 1", "n*(n+1)/2", "gcd(a, b)", "a**p % p", "phi(n)"],
    "trigonometry":  ["sin(x)**2 + cos(x)**2", "sin(2*x)", "cos(x - y)", "tan(x)", "2*sin(x)*cos(x)"],
}
```

### 2.4 Protocol Implementation

#### `fetch(query: str) -> list[np.ndarray]`

Results cached per query.

**SymPy component (always active):**

First, try to parse `query` as a SymPy expression:
```python
from sympy import sympify, SympifyError
try:
    expr = sympify(query)
    vecs = [hash_vectorise(str(expr), self.feature_dim)]
except (SympifyError, TypeError):
    expr = None
```

If parse failed, treat `query` as a topic name:
- Normalise: `query.lower().strip()`
- Look up in `_TOPIC_EXPRS` by exact key; if not found, try partial match: find the first topic key that is a substring of the normalised query string (e.g., query `"basic trigonometry"` matches topic key `"trigonometry"` because `"trigonometry" in "basic trigonometry"`)
- If still no match, return `[]`
- For each expression string in the topic's list, call `hash_vectorise(expr_str, self.feature_dim)` and collect results

**SciPy constants component (if `self._scipy_constants is not None`):**
- Search `scipy.constants.physical_constants` dict for keys containing `query.lower()`
- For each matching constant `(value, unit, uncertainty)`: apply `hash_vectorise(f"{name} {unit}", self.feature_dim)`
- Append matched vectors (max 5 matches to avoid flooding)

**Wolfram Alpha component (if Wolfram App ID active):**
- GET `https://api.wolframalpha.com/v1/result?appid={id}&i={query}` with `timeout=self.timeout`
- On error or non-200: skip silently
- On success: apply `hash_vectorise(response_text, self.feature_dim)`, append one vector

**Return:** combined list from all active components. Empty list if SymPy parse failed and no topic match.

#### `field_frequency(pattern) -> float`

- Query = `str(getattr(pattern, 'label', None) or 'algebra')`
- Same cosine similarity pattern as `WikipediaSubstrate` and `LinguisticSubstrate`
- Return `float` in `[0, 1]`; return `0.0` if no vectors

#### `stream() -> Iterator[np.ndarray]`

If SciPy constants are unavailable: cycle through `_TOPIC_EXPRS` topics indefinitely, calling `self.fetch(topic)` and yielding each vector in the result.

If SciPy constants are available: alternate one-for-one — yield one vector from `self.fetch(topic)` (cycling through topics in `_TOPIC_EXPRS`), then yield one vector from `hash_vectorise(constant_name, self.feature_dim)` (cycling through `scipy.constants.physical_constants` keys), then repeat. Each call to `self.fetch(topic)` may return multiple vectors; yield them all before advancing to the next constant. All keys in `_TOPIC_EXPRS` are guaranteed to return non-empty fetch results; if a topic fetch returns empty (should not occur), skip that topic and advance.

### 2.5 Caching

`_cache: dict[str, list[np.ndarray]]` — per-query, in-memory, no TTL.

---

## 3. File Changes

```
hpm/substrate/linguistic.py     # LinguisticSubstrate (new)
hpm/substrate/math.py           # MathSubstrate (new)
hpm/substrate/__init__.py       # add exports for both
tests/substrate/test_linguistic.py
tests/substrate/test_math.py
```

`hpm/substrate/__init__.py` currently exports `WikipediaSubstrate`, `PyPISubstrate`, `LocalFileSubstrate`. Add:
```python
from .linguistic import LinguisticSubstrate
from .math import MathSubstrate
```

---

## 4. Testing Strategy

### LinguisticSubstrate

- `test_fetch_returns_vectors_for_known_word`: `fetch("dog")` returns non-empty list; each vector shape `(32,)`, values in `[0, 1]`
- `test_fetch_unknown_word_returns_empty`: `fetch("xyzzy123")` returns `[]`
- `test_fetch_caches_results`: second call to `fetch("dog")` returns the identical list object (`result1 is result2`, not just equality) confirming the cache is hit with no re-computation
- `test_field_frequency_in_range`: `field_frequency(pattern)` returns float in `[0, 1]`
- `test_api_component_skipped_on_network_error`: with `requests` mocked to raise `ConnectionError`, fetch still returns WordNet vectors
- `test_spacy_disabled`: `LinguisticSubstrate(use_spacy=False)` works; fetch returns only WordNet + API vectors
- `test_api_disabled`: `LinguisticSubstrate(use_api=False)` works; fetch returns only WordNet vectors
- `test_stream_yields_arrays`: take 10 items from `stream()`, each shape `(32,)`
- `test_importerror_if_nltk_missing`: mock `import nltk` to fail; constructor raises `ImportError`
- `test_lookuperror_if_wordnet_corpus_missing`: mock `nltk.data.find` to raise `LookupError`; assert constructor raises `LookupError`
- `test_spacy_pos_vector_normalised`: with spaCy enabled and a multi-token query, the POS vector appended to results sums to approximately 1.0 (normalised by token count)

### MathSubstrate

- `test_fetch_sympy_expression`: `fetch("x**2 + 1")` returns non-empty list; each vector shape `(32,)`
- `test_fetch_topic_name`: `fetch("algebra")` returns non-empty list
- `test_fetch_unknown_returns_empty`: `fetch("not_a_topic_xyzzy")` returns `[]`
- `test_fetch_caches_results`: second call returns the identical list object (`result1 is result2`) confirming cache hit with no re-computation
- `test_field_frequency_in_range`: returns float in `[0, 1]`
- `test_wolfram_skipped_when_no_key`: with no `WOLFRAM_APP_ID` and `wolfram_app_id=None`, Wolfram component inactive; no HTTP call made
- `test_wolfram_used_when_key_set`: with mocked `requests.get`, Wolfram result appended to vectors
- `test_scipy_disabled`: `MathSubstrate(use_scipy=False)` works without scipy
- `test_stream_yields_arrays`: take 10 items from `stream()`, each shape `(32,)`
- `test_importerror_if_sympy_missing`: mock `import sympy` to fail; constructor raises `ImportError`

---

## 5. Dependencies

New optional dependencies (add to `requirements.txt` with comments):

```
nltk>=3.8          # LinguisticSubstrate core
sympy>=1.12        # MathSubstrate core
spacy>=3.7         # LinguisticSubstrate optional (+ python -m spacy download en_core_web_sm)
scipy>=1.11        # MathSubstrate optional
```

NLTK corpora required (user must download once):
```python
import nltk
nltk.download('wordnet')
nltk.download('words')
```

---

## 6. What Is NOT in Scope

- Embedding-based vectorisation (word2vec, sentence-transformers) — `hash_vectorise` is sufficient for HPM's pattern-matching use case
- Persistent caching (disk/Redis) — in-memory cache per-instance is sufficient
- OEIS integration — deferred; Wolfram Alpha covers math queries adequately
- Custom spaCy models or fine-tuned parsers
- Rate limiting / retry logic for external APIs
