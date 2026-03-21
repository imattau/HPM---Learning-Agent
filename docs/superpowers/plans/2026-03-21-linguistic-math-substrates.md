# Linguistic and Math Substrates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `LinguisticSubstrate` and `MathSubstrate` — two `ExternalSubstrate` implementations giving the HPM agent grounded baseline knowledge from NLTK/WordNet, dictionary API, spaCy, SymPy, SciPy constants, and Wolfram Alpha.

**Architecture:** Each substrate is a standalone class in `hpm/substrate/` implementing the `ExternalSubstrate` protocol (`fetch`, `field_frequency`, `stream`). Core dependencies (NLTK, SymPy) are required and raise `ImportError`/`LookupError` at construction; optional components (dictionary API, spaCy, SciPy, Wolfram) degrade silently if unavailable. All text→vector conversion uses `hash_vectorise` from `hpm/substrate/base.py`.

**Tech Stack:** Python, NLTK (WordNet + words corpus), SymPy, requests, spaCy (optional), SciPy (optional), Wolfram Alpha API (optional)

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `hpm/substrate/linguistic.py` | Create | `LinguisticSubstrate` class |
| `hpm/substrate/math.py` | Create | `MathSubstrate` class |
| `hpm/substrate/__init__.py` | Modify | Add exports for both classes |
| `tests/substrate/test_linguistic.py` | Create | Tests for `LinguisticSubstrate` |
| `tests/substrate/test_math.py` | Create | Tests for `MathSubstrate` |
| `requirements.txt` | Modify | Add nltk, sympy, spacy, scipy |

---

## Context for Implementers

### ExternalSubstrate protocol (hpm/substrate/base.py)

```python
@runtime_checkable
class ExternalSubstrate(Protocol):
    def fetch(self, query: str) -> list[np.ndarray]: ...
    def field_frequency(self, pattern) -> float: ...
    def stream(self) -> Iterator[np.ndarray]: ...
```

### hash_vectorise (hpm/substrate/base.py)

```python
def hash_vectorise(text: str, dim: int = 32) -> np.ndarray:
    """Hash-trick text vectoriser. Maps words to a fixed-dim float array.
    Returns a normalised word-frequency vector in [0, 1] per element, sum = 1."""
```

### Reference implementation pattern (hpm/substrate/wikipedia.py)

`WikipediaSubstrate.__init__` stores `self.feature_dim`, `self.timeout`, `self._cache: dict[str, list[np.ndarray]] = {}`. `fetch` checks cache first, calls API, stores result in cache, returns it. `field_frequency` does cosine similarity: normalise `pattern.mu`, dot with each fetched vector, mean, rescale from `[-1,1]` to `[0,1]` via `(sim + 1) / 2`, clip to `[0, 1]`. `stream` cycles topics indefinitely.

### hpm/substrate/__init__.py

Currently empty. Will need to export all substrate classes.

---

## Task 1: LinguisticSubstrate

**Files:**
- Create: `hpm/substrate/linguistic.py`
- Create: `tests/substrate/test_linguistic.py`

---

- [ ] **Step 1: Write the failing tests**

Create `tests/substrate/test_linguistic.py`:

```python
import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


def make_substrate(**kwargs):
    """Construct LinguisticSubstrate with minimal dependencies for fast tests."""
    from hpm.substrate.linguistic import LinguisticSubstrate
    defaults = dict(feature_dim=32, use_api=False, use_spacy=False)
    defaults.update(kwargs)
    return LinguisticSubstrate(**defaults)


def test_fetch_returns_vectors_for_known_word():
    s = make_substrate()
    vecs = s.fetch("dog")
    assert len(vecs) > 0
    for v in vecs:
        assert v.shape == (32,)
        assert np.all(v >= 0)


def test_fetch_unknown_word_returns_empty():
    s = make_substrate()
    vecs = s.fetch("xyzzy123qqqq")
    assert vecs == []


def test_fetch_caches_results():
    s = make_substrate()
    r1 = s.fetch("dog")
    r2 = s.fetch("dog")
    assert r1 is r2


def test_field_frequency_in_range():
    s = make_substrate()
    pattern = MagicMock()
    pattern.label = "dog"
    pattern.mu = np.ones(32) * 0.1
    freq = s.field_frequency(pattern)
    assert 0.0 <= freq <= 1.0


def test_field_frequency_zero_for_unknown_word():
    s = make_substrate()
    pattern = MagicMock()
    pattern.label = "xyzzy123qqqq"
    pattern.mu = np.ones(32) * 0.1
    freq = s.field_frequency(pattern)
    assert freq == 0.0


def test_api_component_skipped_on_network_error():
    s = make_substrate(use_api=True)
    with patch('requests.get', side_effect=ConnectionError):
        vecs = s.fetch("dog")
    # WordNet vecs still returned; API failure is silent
    assert len(vecs) > 0


def test_api_disabled():
    s = make_substrate(use_api=False)
    assert not s._use_api
    vecs = s.fetch("dog")
    assert isinstance(vecs, list)


def test_spacy_disabled():
    s = make_substrate(use_spacy=False)
    assert s._nlp is None
    vecs = s.fetch("cat")
    assert isinstance(vecs, list)


def test_stream_yields_arrays():
    s = make_substrate()
    gen = s.stream()
    items = [next(gen) for _ in range(10)]
    for v in items:
        assert isinstance(v, np.ndarray)
        assert v.shape == (32,)


def test_lookuperror_if_wordnet_corpus_missing():
    import nltk
    with patch.object(nltk.data, 'find', side_effect=LookupError("Corpora not found")):
        from hpm.substrate.linguistic import LinguisticSubstrate
        with pytest.raises(LookupError):
            LinguisticSubstrate(use_spacy=False, use_api=False)


def test_importerror_if_nltk_missing():
    # Simulate nltk being absent by blocking the import inside __init__
    import builtins
    import sys
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == 'nltk' or name.startswith('nltk.'):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    from hpm.substrate.linguistic import LinguisticSubstrate
    with patch('builtins.__import__', side_effect=mock_import):
        # Remove cached nltk from sys.modules so the constructor's import is not short-circuited
        sys.modules.pop('nltk', None)
        with pytest.raises(ImportError):
            LinguisticSubstrate(use_spacy=False, use_api=False)


def test_spacy_pos_vector_normalised():
    """spaCy POS frequency vector sums to ~1.0."""
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load('en_core_web_sm')
    except Exception:
        pytest.skip("en_core_web_sm model not available")
    from hpm.substrate.linguistic import LinguisticSubstrate
    s = LinguisticSubstrate(feature_dim=32, use_api=False, use_spacy=True)
    if s._nlp is None:
        pytest.skip("spaCy model not loaded")
    vecs = s.fetch("the quick brown fox jumps")
    # POS vector is the last one appended
    pos_vec = vecs[-1]
    # Values in [0, 1] and sum <= 1.0 (sum < 1 if feature_dim > 17 tags, zero-padded)
    assert np.all(pos_vec >= 0)
    assert pos_vec.sum() <= 1.01  # allow float tolerance
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
pytest tests/substrate/test_linguistic.py -v 2>&1 | head -30
```

Expected: ImportError or ModuleNotFoundError — `hpm.substrate.linguistic` does not exist.

- [ ] **Step 3: Implement LinguisticSubstrate**

Create `hpm/substrate/linguistic.py`:

```python
import itertools
from typing import Iterator

import numpy as np
import requests

from .base import hash_vectorise

_POS_TAGS = [
    'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM',
    'PUNCT', 'SYM', 'X', 'PART', 'INTJ', 'PROPN', 'AUX', 'CCONJ', 'SCONJ',
]  # 17 Universal Dependencies v2 POS tags


class LinguisticSubstrate:
    """
    ExternalSubstrate backed by NLTK/WordNet (required), Free Dictionary API
    (optional), and spaCy (optional).

    Core dependency: NLTK with wordnet + words corpora.
    Optional: requests (API), spacy with en_core_web_sm (POS vectors).

    All three components run independently; results are combined.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        timeout: float = 5.0,
        use_api: bool = True,
        use_spacy: bool = True,
    ):
        try:
            import nltk
            nltk.data.find('corpora/wordnet')
            nltk.data.find('corpora/words')
        except ImportError:
            raise ImportError(
                "nltk is required for LinguisticSubstrate. Install with: pip install nltk"
            )
        except LookupError:
            raise LookupError(
                "NLTK corpora not found. Run: "
                "nltk.download('wordnet'); nltk.download('words')"
            )

        import nltk
        from nltk.corpus import wordnet, words as nltk_words
        self._wordnet = wordnet
        self._word_list = [
            w for w in nltk_words.words()
            if w.isalpha() and 4 <= len(w) <= 12
        ]

        self.feature_dim = feature_dim
        self.timeout = timeout
        self._use_api = use_api
        self._cache: dict[str, list[np.ndarray]] = {}

        self._nlp = None
        if use_spacy:
            try:
                import spacy
                self._nlp = spacy.load('en_core_web_sm')
            except Exception:
                pass

    def fetch(self, query: str) -> list[np.ndarray]:
        if query in self._cache:
            return self._cache[query]

        results = []

        # WordNet component (always active)
        synsets = self._wordnet.synsets(query)
        for syn in synsets:
            if syn.definition():
                results.append(hash_vectorise(syn.definition(), self.feature_dim))
            for ex in syn.examples():
                if ex:
                    results.append(hash_vectorise(ex, self.feature_dim))

        # Free Dictionary API component (optional)
        if self._use_api:
            try:
                resp = requests.get(
                    f'https://api.dictionaryapi.dev/api/v2/entries/en/{query}',
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    for entry in resp.json():
                        for meaning in entry.get('meanings', []):
                            for defn in meaning.get('definitions', []):
                                if defn.get('definition'):
                                    results.append(
                                        hash_vectorise(defn['definition'], self.feature_dim)
                                    )
                                if defn.get('example'):
                                    results.append(
                                        hash_vectorise(defn['example'], self.feature_dim)
                                    )
            except Exception:
                pass

        # spaCy POS-tag frequency vector (optional)
        if self._nlp is not None:
            doc = self._nlp(query)
            total = len(doc)
            if total > 0:
                counts = {tag: 0 for tag in _POS_TAGS}
                for token in doc:
                    if token.pos_ in counts:
                        counts[token.pos_] += 1
                freq = np.array([counts[tag] / total for tag in _POS_TAGS])
                if len(freq) < self.feature_dim:
                    freq = np.pad(freq, (0, self.feature_dim - len(freq)))
                else:
                    freq = freq[:self.feature_dim]
                results.append(freq)

        self._cache[query] = results
        return results

    def field_frequency(self, pattern) -> float:
        query = str(getattr(pattern, 'label', None) or 'word')
        vecs = self.fetch(query)
        if not vecs:
            return 0.0
        dim = self.feature_dim
        mu = np.array(pattern.mu, dtype=float)
        if len(mu) > dim:
            mu = mu[:dim]
        elif len(mu) < dim:
            mu = np.pad(mu, (0, dim - len(mu)))
        mu_norm = np.linalg.norm(mu)
        if mu_norm == 0:
            return 0.0
        mu_unit = mu / mu_norm
        mean_sim = float(np.mean([np.dot(mu_unit, v) for v in vecs]))
        return float(np.clip((mean_sim + 1.0) / 2.0, 0.0, 1.0))

    def stream(self) -> Iterator[np.ndarray]:
        """Stream vectorised definitions from the NLTK word list.

        Note: when use_api=True, each word triggers an HTTP request to the
        dictionary API. Use LinguisticSubstrate(use_api=False) for offline streaming.
        """
        for word in itertools.cycle(self._word_list):
            vecs = self.fetch(word)
            for v in vecs:
                yield v
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/substrate/test_linguistic.py -v
```

Expected: All tests pass (tests requiring spaCy model will skip if model not installed).

- [ ] **Step 5: Commit**

```bash
git add hpm/substrate/linguistic.py tests/substrate/test_linguistic.py
git commit -m "feat: add LinguisticSubstrate (WordNet + Dictionary API + spaCy)"
```

---

## Task 2: MathSubstrate

**Files:**
- Create: `hpm/substrate/math.py`
- Create: `tests/substrate/test_math.py`

---

- [ ] **Step 1: Write the failing tests**

Create `tests/substrate/test_math.py`:

```python
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


def make_substrate(**kwargs):
    """Construct MathSubstrate with minimal optional dependencies."""
    from hpm.substrate.math import MathSubstrate
    defaults = dict(feature_dim=32, use_scipy=False, wolfram_app_id=None)
    defaults.update(kwargs)
    return MathSubstrate(**defaults)


def test_fetch_sympy_expression():
    s = make_substrate()
    vecs = s.fetch("x**2 + 1")
    assert len(vecs) > 0
    for v in vecs:
        assert v.shape == (32,)


def test_fetch_topic_name():
    s = make_substrate()
    vecs = s.fetch("algebra")
    assert len(vecs) > 0
    for v in vecs:
        assert v.shape == (32,)


def test_fetch_partial_topic_match():
    s = make_substrate()
    # "basic trigonometry" should match topic key "trigonometry"
    vecs = s.fetch("basic trigonometry")
    assert len(vecs) > 0


def test_fetch_unknown_returns_empty():
    s = make_substrate()
    vecs = s.fetch("not_a_topic_or_expression_xyzzy")
    assert vecs == []


def test_fetch_caches_results():
    s = make_substrate()
    r1 = s.fetch("algebra")
    r2 = s.fetch("algebra")
    assert r1 is r2


def test_field_frequency_in_range():
    s = make_substrate()
    pattern = MagicMock()
    pattern.label = "algebra"
    pattern.mu = np.ones(32) * 0.1
    freq = s.field_frequency(pattern)
    assert 0.0 <= freq <= 1.0


def test_field_frequency_uses_algebra_default():
    """field_frequency falls back to 'algebra' when pattern has no label."""
    s = make_substrate()
    pattern = MagicMock()
    pattern.label = None
    pattern.mu = np.ones(32) * 0.1
    freq = s.field_frequency(pattern)
    assert 0.0 <= freq <= 1.0


def test_wolfram_skipped_when_no_key():
    s = make_substrate(wolfram_app_id=None)
    with patch('requests.get') as mock_get:
        s.fetch("integral of sin(x)")
    mock_get.assert_not_called()


def test_wolfram_used_when_key_set():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "negative cosine of x plus constant"
    with patch('requests.get', return_value=mock_resp):
        s = make_substrate(wolfram_app_id="FAKE_KEY")
        vecs = s.fetch("integral of sin(x)")
    # Should have SymPy/topic vecs + Wolfram vec
    assert len(vecs) > 0


def test_wolfram_skipped_on_network_error():
    with patch('requests.get', side_effect=ConnectionError):
        s = make_substrate(wolfram_app_id="FAKE_KEY")
        # Should still work (no exception); returns SymPy/topic vecs
        vecs = s.fetch("algebra")
    assert isinstance(vecs, list)


def test_scipy_disabled():
    s = make_substrate(use_scipy=False)
    assert s._scipy_constants is None
    vecs = s.fetch("calculus")
    assert isinstance(vecs, list)


def test_scipy_enabled():
    pytest.importorskip("scipy")
    from hpm.substrate.math import MathSubstrate
    s = MathSubstrate(feature_dim=32, use_scipy=True, wolfram_app_id=None)
    assert s._scipy_constants is not None
    vecs = s.fetch("speed of light")
    assert len(vecs) > 0


def test_stream_yields_arrays():
    s = make_substrate()
    gen = s.stream()
    items = [next(gen) for _ in range(10)]
    for v in items:
        assert isinstance(v, np.ndarray)
        assert v.shape == (32,)


def test_importerror_if_sympy_missing():
    import builtins
    import sys
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == 'sympy' or name.startswith('sympy.'):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    from hpm.substrate.math import MathSubstrate
    with patch('builtins.__import__', side_effect=mock_import):
        # Remove cached sympy from sys.modules so the constructor's import is not short-circuited
        sys.modules.pop('sympy', None)
        with pytest.raises(ImportError):
            MathSubstrate(use_scipy=False, wolfram_app_id=None)


def test_wolfram_app_id_from_env():
    """Wolfram App ID is picked up from WOLFRAM_APP_ID env var."""
    with patch.dict(os.environ, {'WOLFRAM_APP_ID': 'ENV_KEY'}):
        s = make_substrate()
        assert s._wolfram_app_id == 'ENV_KEY'
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/substrate/test_math.py -v 2>&1 | head -20
```

Expected: ImportError — `hpm.substrate.math` does not exist.

- [ ] **Step 3: Implement MathSubstrate**

Create `hpm/substrate/math.py`:

```python
import itertools
import os
from typing import Iterator

import numpy as np
import requests

from .base import hash_vectorise

_TOPIC_EXPRS: dict[str, list[str]] = {
    "algebra":       ["x**2 + x + 1", "x**2 - 4", "2*x + 3", "x**3 - x", "a*x**2 + b*x + c"],
    "calculus":      ["sin(x)", "exp(x)", "log(x)", "x**2 * sin(x)", "1/(1 + x**2)"],
    "statistics":    ["(x - mu)/sigma", "exp(-x**2/2)", "x*(1-x)", "n*p", "p**k * (1-p)**(n-k)"],
    "geometry":      ["pi*r**2", "4*pi*r**3/3", "sqrt(a**2 + b**2)", "2*pi*r", "b*h/2"],
    "number_theory": ["2**n - 1", "n*(n+1)/2", "gcd(a, b)", "a**p % p", "phi(n)"],
    "trigonometry":  ["sin(x)**2 + cos(x)**2", "sin(2*x)", "cos(x - y)", "tan(x)", "2*sin(x)*cos(x)"],
}


class MathSubstrate:
    """
    ExternalSubstrate backed by SymPy (required), SciPy constants (optional),
    and Wolfram Alpha (optional).

    Core dependency: sympy.
    Optional: scipy (physical constants), Wolfram Alpha API key.

    fetch(query): tries to parse as SymPy expression; if that fails, treats
    as topic name and returns vectors for known math topic expressions.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        timeout: float = 5.0,
        use_scipy: bool = True,
        wolfram_app_id: str | None = None,
    ):
        try:
            from sympy import sympify, SympifyError
            self._sympify = sympify
            self._SympifyError = SympifyError
        except ImportError:
            raise ImportError(
                "sympy is required for MathSubstrate. Install with: pip install sympy"
            )

        self.feature_dim = feature_dim
        self.timeout = timeout
        self._cache: dict[str, list[np.ndarray]] = {}

        self._scipy_constants = None
        if use_scipy:
            try:
                import scipy.constants
                self._scipy_constants = scipy.constants.physical_constants
            except ImportError:
                pass

        self._wolfram_app_id = wolfram_app_id or os.environ.get('WOLFRAM_APP_ID')

    def fetch(self, query: str) -> list[np.ndarray]:
        if query in self._cache:
            return self._cache[query]

        results = []

        # SymPy component: try to parse as expression first
        try:
            expr = self._sympify(query)
            results.append(hash_vectorise(str(expr), self.feature_dim))
        except (Exception,):
            # Treat as topic name
            normalised = query.lower().strip()
            topic_exprs = _TOPIC_EXPRS.get(normalised)
            if topic_exprs is None:
                # Partial match: first topic key that is a substring of the query
                for key, exprs in _TOPIC_EXPRS.items():
                    if key in normalised:
                        topic_exprs = exprs
                        break
            if topic_exprs is None:
                self._cache[query] = []
                return []
            for expr_str in topic_exprs:
                results.append(hash_vectorise(expr_str, self.feature_dim))

        # SciPy constants component (optional)
        if self._scipy_constants is not None:
            q = query.lower()
            count = 0
            for name, (value, unit, uncertainty) in self._scipy_constants.items():
                if q in name.lower():
                    results.append(hash_vectorise(f"{name} {unit}", self.feature_dim))
                    count += 1
                    if count >= 5:
                        break

        # Wolfram Alpha component (optional)
        if self._wolfram_app_id:
            try:
                resp = requests.get(
                    'https://api.wolframalpha.com/v1/result',
                    params={'appid': self._wolfram_app_id, 'i': query},
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    results.append(hash_vectorise(resp.text, self.feature_dim))
            except Exception:
                pass

        self._cache[query] = results
        return results

    def field_frequency(self, pattern) -> float:
        query = str(getattr(pattern, 'label', None) or 'algebra')
        vecs = self.fetch(query)
        if not vecs:
            return 0.0
        dim = self.feature_dim
        mu = np.array(pattern.mu, dtype=float)
        if len(mu) > dim:
            mu = mu[:dim]
        elif len(mu) < dim:
            mu = np.pad(mu, (0, dim - len(mu)))
        mu_norm = np.linalg.norm(mu)
        if mu_norm == 0:
            return 0.0
        mu_unit = mu / mu_norm
        mean_sim = float(np.mean([np.dot(mu_unit, v) for v in vecs]))
        return float(np.clip((mean_sim + 1.0) / 2.0, 0.0, 1.0))

    def stream(self) -> Iterator[np.ndarray]:
        """Stream math expression vectors, interleaved with SciPy constant vectors if available."""
        topics = itertools.cycle(_TOPIC_EXPRS.keys())
        if self._scipy_constants is None:
            for topic in topics:
                for v in self.fetch(topic):
                    yield v
        else:
            const_names = itertools.cycle(self._scipy_constants.keys())
            for topic in topics:
                vecs = self.fetch(topic)
                if not vecs:
                    continue
                for v in vecs:
                    yield v
                yield hash_vectorise(next(const_names), self.feature_dim)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/substrate/test_math.py -v
```

Expected: All tests pass (scipy test skips if scipy not installed).

- [ ] **Step 5: Commit**

```bash
git add hpm/substrate/math.py tests/substrate/test_math.py
git commit -m "feat: add MathSubstrate (SymPy + SciPy constants + Wolfram Alpha)"
```

---

## Task 3: Exports and Requirements

**Files:**
- Modify: `hpm/substrate/__init__.py`
- Modify: `requirements.txt`

---

- [ ] **Step 1: Verify existing substrate files exist**

```bash
ls hpm/substrate/*.py
```

Expected output includes: `wikipedia.py`, `pypi.py`, `local_file.py`. If any are missing, only import the ones that exist in the next step.

- [ ] **Step 2: Update `hpm/substrate/__init__.py`**

The file is currently empty. Replace with:

```python
from .wikipedia import WikipediaSubstrate
from .pypi import PyPISubstrate
from .local_file import LocalFileSubstrate
from .linguistic import LinguisticSubstrate
from .math import MathSubstrate

__all__ = [
    'WikipediaSubstrate',
    'PyPISubstrate',
    'LocalFileSubstrate',
    'LinguisticSubstrate',
    'MathSubstrate',
]
```

- [ ] **Step 3: Verify existing substrate imports still work**

```bash
python -c "from hpm.substrate import WikipediaSubstrate, PyPISubstrate, LocalFileSubstrate, LinguisticSubstrate, MathSubstrate; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Check current requirements.txt**

```bash
cat requirements.txt
```

Note what's already there before adding new entries.

- [ ] **Step 5: Update requirements.txt**

Add the following entries (with comments) if not already present:

```
nltk>=3.8          # LinguisticSubstrate core (run: nltk.download('wordnet'); nltk.download('words'))
sympy>=1.12        # MathSubstrate core
spacy>=3.7         # LinguisticSubstrate optional (also: python -m spacy download en_core_web_sm)
scipy>=1.11        # MathSubstrate optional
```

- [ ] **Step 6: Run the full test suite**

```bash
pytest -q
```

Expected: All existing tests pass plus new substrate tests. The count should be higher than before (248 was the count before this feature).

- [ ] **Step 7: Commit**

```bash
git add hpm/substrate/__init__.py requirements.txt
git commit -m "feat: export LinguisticSubstrate and MathSubstrate; add dependencies to requirements.txt"
```

---

## Verification

After all three tasks:

```bash
# Confirm test count increased
pytest -q 2>&1 | tail -3

# Confirm both substrates importable and instantiable
python -c "
from hpm.substrate import LinguisticSubstrate, MathSubstrate
m = MathSubstrate(use_scipy=False)
vecs = m.fetch('algebra')
print(f'MathSubstrate OK: {len(vecs)} vectors for algebra')
"
```
