# NLP Child Language Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an NLP experiment that tests whether the HPM Observer, given child-scale linguistic priors, spontaneously discovers semantic categories from masked-word context windows.

**Architecture:** Three new files mirror the dSprites pattern exactly: `nlp_loader.py` (vocab + sentence generation + encoding), `nlp_world_model.py` (38-node prior forest), `experiment_nlp.py` (Observer run + purity measurement). All live under `hpm_fractal_node/nlp/` and `hpm_fractal_node/experiments/`.

**Tech Stack:** Python, NumPy, `hfn` (Observer, HFN, Forest, calibrate_tau, hausdorff_distance) — all already installed.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `hpm_fractal_node/nlp/__init__.py` | Create | Empty module init |
| `hpm_fractal_node/nlp/nlp_loader.py` | Create | Fixed vocab (107 tokens, D=428), sentence templates, context-window encoder, label generation |
| `hpm_fractal_node/nlp/nlp_world_model.py` | Create | `build_nlp_world_model()` → Forest + prior_ids (38 nodes: 5 relational + 8 category + 25 word) |
| `hpm_fractal_node/experiments/experiment_nlp.py` | Create | Observer run, purity measurement, factor alignment report |
| `tests/hpm_fractal_node/nlp/test_nlp_loader.py` | Create | Unit tests for vocab, encoding, sentence generation |
| `tests/hpm_fractal_node/nlp/test_nlp_world_model.py` | Create | Unit tests for prior counts, mu shapes, determinism |

---

## Task 1: Module scaffold

**Files:**
- Create: `hpm_fractal_node/nlp/__init__.py`
- Create: `tests/hpm_fractal_node/nlp/__init__.py`

- [ ] **Step 1: Create module init files**

```bash
touch hpm_fractal_node/nlp/__init__.py
mkdir -p tests/hpm_fractal_node/nlp
touch tests/hpm_fractal_node/nlp/__init__.py
```

- [ ] **Step 2: Verify Python can import the (empty) module**

```bash
PYTHONPATH=. python3 -c "import hpm_fractal_node.nlp; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add hpm_fractal_node/nlp/__init__.py tests/hpm_fractal_node/nlp/__init__.py
git commit -m "feat: scaffold hpm_fractal_node.nlp module"
```

---

## Task 2: Vocabulary and context-window encoder (`nlp_loader.py`)

**Files:**
- Create: `hpm_fractal_node/nlp/nlp_loader.py`
- Test: `tests/hpm_fractal_node/nlp/test_nlp_loader.py`

### Step 2a: Write the failing tests

- [ ] **Step 1: Write tests**

Create `tests/hpm_fractal_node/nlp/test_nlp_loader.py`:

```python
"""Tests for nlp_loader: vocab, encoding, sentence generation."""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

from hpm_fractal_node.nlp.nlp_loader import (
    VOCAB, VOCAB_INDEX, D,
    encode_context_window,
    generate_sentences,
    category_names,
)


def test_vocab_size():
    assert len(VOCAB) == 107


def test_d_is_428():
    assert D == 428


def test_vocab_no_duplicates():
    assert len(VOCAB) == len(set(VOCAB))


def test_special_tokens_present():
    for tok in ("<start>", "<end>", "<unknown>"):
        assert tok in VOCAB_INDEX


def test_encode_context_window_shape():
    vec = encode_context_window("the", "barked", left2="<start>", right2="at")
    assert vec.shape == (D,)
    assert vec.dtype == np.float32


def test_encode_context_window_one_hot():
    vec = encode_context_window("the", "barked", left2="<start>", right2="at")
    # Each of 4 slots should have exactly one 1 (known words)
    for i in range(4):
        slot = vec[i * 107: (i + 1) * 107]
        assert slot.sum() == 1.0


def test_encode_unknown_word():
    vec = encode_context_window("zzz_unknown", "barked")
    # left_1 is slot 1 (indices 107..213), should encode as <unknown>
    unk_idx = VOCAB_INDEX["<unknown>"]
    assert vec[107 + unk_idx] == 1.0


def test_generate_sentences_returns_list():
    sentences = generate_sentences(seed=42)
    assert len(sentences) > 0
    assert len(sentences) >= 2000  # must reach N_SAMPLES


def test_generate_sentences_has_labels():
    sentences = generate_sentences(seed=42)
    obs, true_word, category = sentences[0]
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (D,)
    assert isinstance(true_word, str)
    assert category in category_names()


def test_generate_sentences_reproducible():
    s1 = generate_sentences(seed=0)
    s2 = generate_sentences(seed=0)
    assert len(s1) == len(s2)
    np.testing.assert_array_equal(s1[0][0], s2[0][0])


def test_category_names_returns_8():
    cats = category_names()
    assert len(cats) == 8
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
PYTHONPATH=. python3 -m pytest tests/hpm_fractal_node/nlp/test_nlp_loader.py -v 2>&1 | head -30
```
Expected: ImportError — `nlp_loader` does not exist yet.

### Step 2b: Implement `nlp_loader.py`

- [ ] **Step 3: Write the implementation**

Create `hpm_fractal_node/nlp/nlp_loader.py`:

```python
"""
NLP loader for the child language experiment.

Fixed vocabulary of 107 tokens, D=428 (4 context slots × 107).
Generates ~2000 synthetic child-directed sentences with masked slots.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Fixed vocabulary (107 tokens, never changes)
# ---------------------------------------------------------------------------

VOCAB: list[str] = [
    # Special (3)
    "<start>", "<end>", "<unknown>",
    # Function words (14)
    "the", "a", "an", "my", "her", "his", "at", "in", "to", "on", "not", "and", "is", "was",
    # Animals (3)
    "dog", "cat", "bird",
    # People (11)
    "mum", "dad", "grandma", "grandpa", "brother", "sister",
    "teacher", "doctor", "friend", "classmate", "baby",
    # Children (2)
    "boy", "girl",
    # Food (3)
    "apple", "bread", "milk",
    # Objects (3)
    "ball", "book", "toy",
    # Places (3)
    "park", "home", "school",
    # Descriptors (6)
    "big", "small", "little", "red", "blue", "old",
    # Animal actions (5)
    "barked", "meowed", "chirped", "chased", "fetched",
    # Person actions (8)
    "ate", "threw", "ran", "walked", "gave", "took", "helped", "played",
    # Prepositions (4)
    "after", "with", "for", "from",
    # Other nouns (8)
    "pot", "mat", "hat", "tree", "door", "cup", "bed", "box",
    # Filler words (34)
    "loudly", "quickly", "slowly", "happily", "sadly", "again",
    "up", "down", "away", "always", "never", "then", "there", "here",
    "too", "very", "this", "that", "these", "those", "some", "all",
    "more", "other", "new", "good", "nice", "long", "high", "own",
    "our", "your", "their", "its",
]

assert len(VOCAB) == 107, f"Vocab size {len(VOCAB)} != 107"
assert len(set(VOCAB)) == 107, "Vocabulary has duplicate entries"

VOCAB_INDEX: dict[str, int] = {w: i for i, w in enumerate(VOCAB)}
VOCAB_SIZE: int = len(VOCAB)
D: int = 4 * VOCAB_SIZE  # 428


# ---------------------------------------------------------------------------
# Category metadata
# ---------------------------------------------------------------------------

_CATEGORIES = [
    "animal", "person", "adult", "child_person",
    "family", "food", "object", "place",
]

_WORD_TO_CATEGORY: dict[str, str] = {
    "dog": "animal", "cat": "animal", "bird": "animal",
    "mum": "family", "dad": "family", "grandma": "family",
    "grandpa": "family", "brother": "family", "sister": "family",
    "teacher": "adult", "doctor": "adult", "friend": "adult", "classmate": "adult",
    "boy": "child_person", "girl": "child_person", "baby": "child_person",
    "apple": "food", "bread": "food", "milk": "food",
    "ball": "object", "book": "object", "toy": "object",
    "park": "place", "home": "place", "school": "place",
}


def category_names() -> list[str]:
    return list(_CATEGORIES)


def word_category(word: str) -> str:
    return _WORD_TO_CATEGORY.get(word, "unknown")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _encode_word(word: str) -> np.ndarray:
    idx = VOCAB_INDEX.get(word.lower(), VOCAB_INDEX["<unknown>"])
    vec = np.zeros(VOCAB_SIZE, dtype=np.float32)
    vec[idx] = 1.0
    return vec


def encode_context_window(
    left1: str,
    right1: str,
    left2: str = "<start>",
    right2: str = "<end>",
) -> np.ndarray:
    """
    Encode 4-slot context window as concatenated one-hot vectors.

    Slot order: [left2, left1, right1, right2]
    Returns shape (D,) = (428,) float32 vector.
    """
    return np.concatenate([
        _encode_word(left2),
        _encode_word(left1),
        _encode_word(right1),
        _encode_word(right2),
    ])


# ---------------------------------------------------------------------------
# Sentence generation
# ---------------------------------------------------------------------------

_ANIMALS = ["dog", "cat", "bird"]
_ANIMAL_ACTIONS = ["barked", "meowed", "chirped", "chased", "fetched"]
_PERSONS = ["mum", "dad", "teacher", "boy", "girl"]
_FAMILY = ["mum", "dad", "grandma", "grandpa", "brother", "sister"]
_ADULTS = ["teacher", "doctor", "friend", "classmate"]
_CHILDREN = ["boy", "girl", "baby"]
_FOODS = ["apple", "bread", "milk"]
_OBJECTS = ["ball", "book", "toy"]
_PLACES = ["park", "home", "school"]
_DESCRIPTORS = ["big", "small", "little", "red", "blue", "old"]
_PERSON_ACTIONS = ["ate", "threw", "ran", "walked", "gave", "took", "helped", "played"]
_DETERMINERS = ["the", "a", "my", "her"]


def _obs(left2: str, left1: str, right1: str, right2: str,
         true_word: str) -> tuple[np.ndarray, str, str]:
    vec = encode_context_window(left1, right1, left2=left2, right2=right2)
    return vec, true_word, word_category(true_word)


def generate_sentences(
    seed: int = 42,
) -> list[tuple[np.ndarray, str, str]]:
    """
    Generate ≥2000 synthetic masked sentences.

    Returns list of (context_vector, true_word, semantic_category).
    Labels are for evaluation only — never fed to the Observer.
    """
    observations: list[tuple[np.ndarray, str, str]] = []

    # Template 1: "The [MASK:animal] [animal_action] ."  (15 obs)
    for animal in _ANIMALS:
        for action in _ANIMAL_ACTIONS:
            observations.append(_obs("<start>", "the", action, "<end>", animal))

    # Template 2: "A [descriptor] [MASK:animal] [animal_action] ."  (90 obs)
    for desc in _DESCRIPTORS:
        for animal in _ANIMALS:
            for action in _ANIMAL_ACTIONS:
                observations.append(_obs("a", desc, action, "<end>", animal))

    # Template 3: "[person] ate the [MASK:food] ."  (15 obs)
    for person in _PERSONS:
        for food in _FOODS:
            observations.append(_obs(person, "ate", "the", "<end>", food))

    # Template 4: "[person] threw/gave/took the [MASK:object] ."  (45 obs)
    for person in _PERSONS:
        for obj in _OBJECTS:
            for action in ["threw", "gave", "took"]:
                observations.append(_obs(person, action, "the", "<end>", obj))

    # Template 5: "The [child] ran to the [MASK:place] ."  (9 obs)
    # Use child as left2 context to vary the encoding
    for child in _CHILDREN:
        for place in _PLACES:
            observations.append(_obs(child, "ran", "to", "the", place))

    # Template 6: "[MASK:person] walked to the [place] ."  (33 obs)
    # Context: left2=<start>, left1=<start>, right1=walked, right2=place
    for person in _PERSONS + _FAMILY:
        for place in _PLACES:
            observations.append(_obs("<start>", "<start>", "walked", place, person))

    # Template 7: "My [MASK:family] gave me the [object] ."  (18 obs)
    for fam in _FAMILY:
        for obj in _OBJECTS:
            observations.append(_obs("<start>", "my", "gave", "me", fam))

    # Template 8: "The [MASK:animal] chased/fetched the [object] ."  (18 obs)
    for animal in _ANIMALS:
        for obj in _OBJECTS:
            for action in ["chased", "fetched"]:
                observations.append(_obs("<start>", "the", action, "the", animal))

    # Template 9: "The [adult] helped the [MASK:child_person] ."  (24 obs)
    for adult in _ADULTS:
        for child in _CHILDREN:
            observations.append(_obs("the", adult, "helped", "the", child))
            observations.append(_obs(adult, "helped", "the", "<end>", child))

    # Template 10: "We went to the [MASK:place] ."  (600 obs)
    for place in _PLACES:
        for _ in range(200):
            observations.append(_obs("to", "the", "<end>", "<end>", place))

    # Template 11: "The [MASK:food] was good/old/nice ."  (180 obs)
    for food in _FOODS:
        for adj in ["good", "nice", "old", "big", "small", "new"]:
            for _ in range(10):
                observations.append(_obs("<start>", "the", "was", adj, food))

    # Template 12: "A [MASK:object] is on the [place] ."  (9 obs)
    for obj in _OBJECTS:
        for place in _PLACES:
            observations.append(_obs("<start>", "a", "is", "on", obj))

    # Template 13: "[person] [action] the [MASK:object] ."  (120 obs)
    for person in _PERSONS:
        for action in _PERSON_ACTIONS:
            for obj in _OBJECTS:
                observations.append(_obs(person, action, "the", "<end>", obj))

    # Template 14: "[family] [action] the [MASK:food] ."  (72 obs)
    for fam in _FAMILY:
        for action in ["ate", "took", "gave", "helped"]:
            for food in _FOODS:
                observations.append(_obs(fam, action, "the", "<end>", food))

    # Template 15: "The [descriptor] [MASK:food] was good ."  (18 obs)
    for desc in _DESCRIPTORS:
        for food in _FOODS:
            observations.append(_obs("the", desc, "was", "good", food))

    # Template 16: "The [descriptor] [MASK:object] is here ."  (18 obs)
    for desc in _DESCRIPTORS:
        for obj in _OBJECTS:
            observations.append(_obs("the", desc, "is", "here", obj))

    # Template 17: "[MASK:adult] helped my [child] ."  (12 obs)
    for adult in _ADULTS:
        for child in _CHILDREN:
            observations.append(_obs("<start>", "<start>", "helped", "my", adult))

    # Template 18: "[MASK:animal] ran to the [place] ."  (9 obs)
    for animal in _ANIMALS:
        for place in _PLACES:
            observations.append(_obs("<start>", "<start>", "ran", "to", animal))

    # Template 19: "The [child] played with the [MASK:object] ."  (9 obs)
    for child in _CHILDREN:
        for obj in _OBJECTS:
            observations.append(_obs(child, "played", "with", "the", obj))

    # Template 20: "My [family] walked to [MASK:place] ."  (18 obs)
    for fam in _FAMILY:
        for place in _PLACES:
            observations.append(_obs("my", fam, "walked", "to", place))

    # Template 21: "[MASK:child_person] played with the [animal] ."  (9 obs)
    for child in _CHILDREN:
        for animal in _ANIMALS:
            observations.append(_obs("<start>", "<start>", "played", "with", child))

    # Template 22: "[person] is at the [MASK:place] ."  (15 obs)
    for person in _PERSONS:
        for place in _PLACES:
            observations.append(_obs(person, "is", "at", "the", place))

    # Template 23: "[MASK:family] is at home ."  (360 obs)
    for fam in _FAMILY:
        for _ in range(60):
            observations.append(_obs("<start>", "<start>", "is", "at", fam))

    # Template 24: "The [MASK:animal] is big/small/old ."  (18 obs)
    for animal in _ANIMALS:
        for desc in _DESCRIPTORS:
            observations.append(_obs("<start>", "the", "is", desc, animal))

    # Template 25: "[person] gave the [MASK:child] a [object] ."  (15 obs)
    for person in _PERSONS:
        for child in _CHILDREN:
            observations.append(_obs(person, "gave", "the", "a", child))

    # Template 26: "The [MASK:food] is on the [object] ."  (9 obs)
    for food in _FOODS:
        for obj in _OBJECTS:
            observations.append(_obs("<start>", "the", "is", "on", food))

    # Template 27: "[adult] walked the [MASK:animal] ."  (12 obs)
    for adult in _ADULTS:
        for animal in _ANIMALS:
            observations.append(_obs(adult, "walked", "the", "<end>", animal))

    # Template 28: "A [MASK:place] is near here ."  (30 obs)
    for place in _PLACES:
        for _ in range(10):
            observations.append(_obs("<start>", "a", "is", "near", place))

    # Template 29: "My [MASK:adult] helped me ."  (200 obs)
    for adult in _ADULTS:
        for _ in range(50):
            observations.append(_obs("<start>", "my", "helped", "me", adult))

    # Template 30: "The [child] has a [MASK:object] ."  (9 obs)
    for child in _CHILDREN:
        for obj in _OBJECTS:
            observations.append(_obs(child, "has", "a", "<end>", obj))

    # Shuffle and return up to 2000
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(observations))
    observations = [observations[i] for i in idx]
    return observations[:2000]
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. python3 -m pytest tests/hpm_fractal_node/nlp/test_nlp_loader.py -v
```
Expected: All pass. If `test_generate_sentences_returns_list` fails because `len < 2000`, add more template expansions in `generate_sentences` until the count reaches ≥2000.

- [ ] **Step 5: Commit**

```bash
git add hpm_fractal_node/nlp/nlp_loader.py tests/hpm_fractal_node/nlp/test_nlp_loader.py
git commit -m "feat: add nlp_loader with fixed vocab (D=428) and sentence generator"
```

---

## Task 3: World model (`nlp_world_model.py`)

**Files:**
- Create: `hpm_fractal_node/nlp/nlp_world_model.py`
- Test: `tests/hpm_fractal_node/nlp/test_nlp_world_model.py`

### Step 3a: Write the failing tests

- [ ] **Step 1: Write tests**

Create `tests/hpm_fractal_node/nlp/test_nlp_world_model.py`:

```python
"""Tests for nlp_world_model: prior counts, mu shapes, determinism."""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

from hpm_fractal_node.nlp.nlp_world_model import build_nlp_world_model, N_PRIORS
from hpm_fractal_node.nlp.nlp_loader import D


def test_prior_count():
    forest, prior_ids = build_nlp_world_model()
    assert len(prior_ids) == 38
    assert N_PRIORS == 38


def test_all_priors_registered():
    forest, prior_ids = build_nlp_world_model()
    for pid in prior_ids:
        assert pid in forest._registry, f"{pid} not in forest"


def test_mu_shapes():
    forest, prior_ids = build_nlp_world_model()
    for pid in prior_ids:
        node = forest._registry[pid]
        assert node.mu.shape == (D,), f"{pid} mu shape {node.mu.shape} != ({D},)"


def test_sigma_diagonal():
    forest, prior_ids = build_nlp_world_model()
    for pid in prior_ids:
        node = forest._registry[pid]
        assert node._sigma_diag is not None, f"{pid} has no diagonal sigma"


def test_deterministic():
    _, ids1 = build_nlp_world_model()
    forest2, ids2 = build_nlp_world_model()
    assert ids1 == ids2
    for pid in ids1:
        n1 = build_nlp_world_model()[0]._registry[pid]
        n2 = forest2._registry[pid]
        np.testing.assert_array_almost_equal(n1.mu, n2.mu)


def test_expected_node_ids_present():
    forest, prior_ids = build_nlp_world_model()
    expected = [
        "prior_agent_action", "prior_action_target", "prior_thing_place",
        "prior_descriptor_thing", "prior_social_relation",
        "prior_animal", "prior_person", "prior_adult", "prior_child_person",
        "prior_family", "prior_food", "prior_object", "prior_place",
        "prior_dog", "prior_cat", "prior_bird",
        "prior_mum", "prior_dad",
        "prior_apple", "prior_bread", "prior_milk",
        "prior_ball", "prior_book", "prior_toy",
        "prior_park", "prior_home", "prior_school",
    ]
    for nid in expected:
        assert nid in prior_ids, f"{nid} missing from prior_ids"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
PYTHONPATH=. python3 -m pytest tests/hpm_fractal_node/nlp/test_nlp_world_model.py -v 2>&1 | head -10
```
Expected: ImportError.

### Step 3b: Implement `nlp_world_model.py`

- [ ] **Step 3: Write the implementation**

Create `hpm_fractal_node/nlp/nlp_world_model.py`:

```python
"""
NLP world model — 38 linguistic priors for D=428 context-window vectors.

Layer 0  Relational   5 priors — agent_action, action_target, thing_place,
                                  descriptor_thing, social_relation
Layer 1  Category     8 priors — animal, person, adult, child_person,
                                  family, food, object, place
Layer 2  Word        25 priors — individual words

All priors are protected — the Observer cannot absorb or remove them.
Prior mu is constructed analytically from the fixed vocabulary.
"""
from __future__ import annotations

import numpy as np

from hfn import HFN, Forest
from hpm_fractal_node.nlp.nlp_loader import D, encode_context_window

N_PRIORS = 38  # 5 relational + 8 category + 25 word


def _avg(*vecs: np.ndarray) -> np.ndarray:
    """Average a sequence of vectors."""
    return np.mean(np.stack(vecs), axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Canonical sentences for word prior mu construction
# Each word has 5 hard-coded contexts (left2, left1, right1, right2)
# ---------------------------------------------------------------------------

_WORD_CONTEXTS: dict[str, list[tuple[str, str, str, str]]] = {
    # Animals
    "dog":   [("<start>", "the", "barked", "<end>"),
              ("<start>", "the", "ran", "<end>"),
              ("<start>", "a", "chased", "the"),
              ("a", "big", "sat", "<end>"),
              ("<start>", "my", "played", "<end>")],
    "cat":   [("<start>", "the", "meowed", "<end>"),
              ("<start>", "a", "ran", "away"),
              ("<start>", "the", "chased", "the"),
              ("a", "little", "sat", "on"),
              ("<start>", "my", "played", "<end>")],
    "bird":  [("<start>", "the", "chirped", "<end>"),
              ("<start>", "a", "ran", "away"),
              ("<start>", "the", "sat", "on"),
              ("a", "small", "chirped", "loudly"),
              ("<start>", "my", "chirped", "loudly")],
    # Family
    "mum":   [("<start>", "<start>", "ate", "the"),
              ("<start>", "<start>", "gave", "me"),
              ("<start>", "my", "helped", "the"),
              ("<start>", "<start>", "walked", "to"),
              ("<start>", "<start>", "played", "with")],
    "dad":   [("<start>", "<start>", "threw", "the"),
              ("<start>", "<start>", "gave", "me"),
              ("<start>", "my", "walked", "to"),
              ("<start>", "<start>", "helped", "the"),
              ("<start>", "<start>", "played", "with")],
    "grandma": [("<start>", "my", "gave", "me"),
                ("<start>", "<start>", "walked", "to"),
                ("<start>", "her", "helped", "<end>"),
                ("<start>", "my", "ate", "the"),
                ("<start>", "<start>", "played", "with")],
    "grandpa": [("<start>", "my", "walked", "to"),
                ("<start>", "<start>", "gave", "me"),
                ("<start>", "his", "helped", "<end>"),
                ("<start>", "my", "sat", "<end>"),
                ("<start>", "<start>", "played", "with")],
    "brother": [("<start>", "my", "ran", "to"),
                ("<start>", "my", "threw", "the"),
                ("<start>", "<start>", "played", "with"),
                ("<start>", "my", "helped", "<end>"),
                ("<start>", "<start>", "walked", "to")],
    "sister":  [("<start>", "my", "played", "with"),
                ("<start>", "my", "helped", "<end>"),
                ("<start>", "<start>", "ran", "to"),
                ("<start>", "my", "ate", "the"),
                ("<start>", "<start>", "walked", "to")],
    # Adults
    "teacher":   [("<start>", "the", "helped", "the"),
                  ("<start>", "<start>", "walked", "to"),
                  ("<start>", "my", "gave", "me"),
                  ("<start>", "the", "played", "with"),
                  ("<start>", "<start>", "helped", "<end>")],
    "doctor":    [("<start>", "the", "helped", "the"),
                  ("<start>", "<start>", "walked", "to"),
                  ("<start>", "my", "gave", "<end>"),
                  ("<start>", "the", "took", "the"),
                  ("<start>", "<start>", "helped", "the")],
    "friend":    [("<start>", "my", "played", "with"),
                  ("<start>", "my", "ran", "to"),
                  ("<start>", "a", "gave", "me"),
                  ("<start>", "my", "helped", "<end>"),
                  ("<start>", "<start>", "walked", "to")],
    "classmate": [("<start>", "my", "played", "with"),
                  ("<start>", "a", "helped", "<end>"),
                  ("<start>", "my", "ran", "to"),
                  ("<start>", "a", "gave", "me"),
                  ("<start>", "<start>", "walked", "to")],
    # Children
    "boy":   [("<start>", "the", "ran", "to"),
              ("<start>", "a", "threw", "the"),
              ("<start>", "the", "played", "with"),
              ("<start>", "a", "walked", "to"),
              ("<start>", "the", "chased", "the")],
    "girl":  [("<start>", "the", "played", "with"),
              ("<start>", "a", "ran", "to"),
              ("<start>", "the", "helped", "<end>"),
              ("<start>", "a", "walked", "to"),
              ("<start>", "the", "ate", "the")],
    "baby":  [("<start>", "the", "played", "with"),
              ("<start>", "a", "walked", "<end>"),
              ("<start>", "my", "ate", "the"),
              ("<start>", "the", "took", "the"),
              ("<start>", "a", "gave", "<end>")],
    # Food
    "apple": [("ate", "the", "<end>", "<end>"),
              ("the", "big", "was", "good"),
              ("<start>", "the", "was", "good"),
              ("gave", "me", "the", "<end>"),
              ("took", "the", "<end>", "<end>")],
    "bread": [("ate", "the", "<end>", "<end>"),
              ("<start>", "the", "was", "good"),
              ("gave", "me", "the", "<end>"),
              ("took", "the", "<end>", "<end>"),
              ("the", "old", "was", "good")],
    "milk":  [("ate", "the", "<end>", "<end>"),
              ("<start>", "the", "was", "good"),
              ("gave", "me", "the", "<end>"),
              ("the", "small", "was", "good"),
              ("took", "the", "<end>", "<end>")],
    # Objects
    "ball":  [("threw", "the", "<end>", "<end>"),
              ("the", "red", "is", "on"),
              ("<start>", "a", "is", "on"),
              ("chased", "the", "<end>", "<end>"),
              ("fetched", "the", "<end>", "<end>")],
    "book":  [("the", "old", "is", "on"),
              ("<start>", "a", "is", "on"),
              ("gave", "me", "the", "<end>"),
              ("took", "the", "<end>", "<end>"),
              ("took", "the", "<end>", "<end>")],
    "toy":   [("threw", "the", "<end>", "<end>"),
              ("<start>", "a", "is", "on"),
              ("the", "new", "is", "on"),
              ("gave", "me", "the", "<end>"),
              ("took", "the", "<end>", "<end>")],
    # Places
    "park":  [("to", "the", "<end>", "<end>"),
              ("ran", "to", "the", "<end>"),
              ("walked", "to", "the", "<end>"),
              ("went", "to", "the", "<end>"),
              ("at", "the", "<end>", "<end>")],
    "home":  [("to", "the", "<end>", "<end>"),
              ("ran", "to", "<end>", "<end>"),
              ("walked", "to", "<end>", "<end>"),
              ("went", "to", "<end>", "<end>"),
              ("at", "<end>", "<end>", "<end>")],
    "school": [("to", "the", "<end>", "<end>"),
               ("ran", "to", "<end>", "<end>"),
               ("walked", "to", "<end>", "<end>"),
               ("went", "to", "<end>", "<end>"),
               ("at", "the", "<end>", "<end>")],
}


def _word_mu(word: str) -> np.ndarray:
    contexts = _WORD_CONTEXTS[word]
    vecs = [encode_context_window(l1, r1, left2=l2, right2=r2)
            for l2, l1, r1, r2 in contexts]
    return _avg(*vecs)


# ---------------------------------------------------------------------------
# World model builder
# ---------------------------------------------------------------------------

def build_nlp_world_model() -> tuple[Forest, set[str]]:
    """
    Build the 3-layer NLP world model.

    Returns
    -------
    forest : Forest
    prior_ids : set[str]
        All node IDs to pass as protected_ids to Observer.
    """
    forest = Forest(D=D, forest_id="nlp_child_428")
    prior_ids: set[str] = set()
    sigma = np.eye(D, dtype=np.float32) * 1.0

    def add(node_id: str, mu: np.ndarray) -> HFN:
        node = HFN(mu=mu.astype(np.float32), sigma=sigma, id=node_id)
        forest.register(node)
        prior_ids.add(node_id)
        return node

    # ------------------------------------------------------------------
    # Layer 2 — Word priors (25 nodes)
    # Built first so category mus can average over them.
    # ------------------------------------------------------------------
    word_nodes: dict[str, HFN] = {}
    for word in _WORD_CONTEXTS:
        node = add(f"prior_{word}", _word_mu(word))
        word_nodes[word] = node

    # ------------------------------------------------------------------
    # Layer 1 — Category priors (8 nodes)
    # mu = average of direct word-prior children only
    # ------------------------------------------------------------------
    def cat_mu(words: list[str]) -> np.ndarray:
        return _avg(*[word_nodes[w].mu for w in words])

    animal_mu  = cat_mu(["dog", "cat", "bird"])
    person_mu  = cat_mu(["mum", "dad", "boy", "girl",
                          "teacher", "doctor", "friend", "classmate", "baby"])
    adult_mu   = cat_mu(["teacher", "doctor", "friend", "classmate"])
    child_mu   = cat_mu(["boy", "girl", "baby"])
    family_mu  = cat_mu(["mum", "dad", "grandma", "grandpa", "brother", "sister"])
    food_mu    = cat_mu(["apple", "bread", "milk"])
    object_mu  = cat_mu(["ball", "book", "toy"])
    place_mu   = cat_mu(["park", "home", "school"])

    cat_nodes = {
        "prior_animal":       add("prior_animal",       animal_mu),
        "prior_person":       add("prior_person",       person_mu),
        "prior_adult":        add("prior_adult",        adult_mu),
        "prior_child_person": add("prior_child_person", child_mu),
        "prior_family":       add("prior_family",       family_mu),
        "prior_food":         add("prior_food",         food_mu),
        "prior_object":       add("prior_object",       object_mu),
        "prior_place":        add("prior_place",        place_mu),
    }

    # ------------------------------------------------------------------
    # Layer 0 — Relational priors (5 nodes)
    # mu = average of direct category-prior children
    # ------------------------------------------------------------------
    add("prior_agent_action",     _avg(cat_nodes["prior_animal"].mu,
                                       cat_nodes["prior_person"].mu))
    add("prior_action_target",    _avg(cat_nodes["prior_food"].mu,
                                       cat_nodes["prior_object"].mu))
    add("prior_thing_place",      cat_nodes["prior_place"].mu.copy())
    add("prior_descriptor_thing", _avg(cat_nodes["prior_animal"].mu,
                                       cat_nodes["prior_object"].mu))
    add("prior_social_relation",  _avg(cat_nodes["prior_family"].mu,
                                       cat_nodes["prior_adult"].mu))

    return forest, prior_ids
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. python3 -m pytest tests/hpm_fractal_node/nlp/test_nlp_world_model.py -v
```
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add hpm_fractal_node/nlp/nlp_world_model.py tests/hpm_fractal_node/nlp/test_nlp_world_model.py
git commit -m "feat: add nlp_world_model with 38 linguistic priors"
```

---

## Task 4: Experiment (`experiment_nlp.py`)

**Files:**
- Create: `hpm_fractal_node/experiments/experiment_nlp.py`

- [ ] **Step 1: Write the experiment**

Create `hpm_fractal_node/experiments/experiment_nlp.py`:

```python
"""
NLP child language experiment: semantic category alignment.

Runs the Observer over masked-sentence context-window vectors and measures
whether learned nodes align with semantic categories (animal, food, place...).

A node with high category purity fired predominantly on one semantic category —
the Observer discovered latent semantic structure without supervision.

Usage:
    PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_nlp.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn import Observer, calibrate_tau
from hfn.fractal import hausdorff_distance
from hpm_fractal_node.nlp.nlp_loader import (
    generate_sentences, category_names, D,
)
from hpm_fractal_node.nlp.nlp_world_model import build_nlp_world_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SAMPLES = 2000
N_PASSES = 3
SEED = 42

TAU_SIGMA = 1.0
TAU_MARGIN = 5.0


def entropy(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([v / total for v in counts.values()])
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def purity(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return max(counts.values()) / total


def main() -> None:
    print(f"Generating {N_SAMPLES} NLP observations (D={D}) ...", flush=True)
    all_obs = generate_sentences(seed=SEED)
    observations = all_obs[:N_SAMPLES]
    print(f"  {len(observations)} observations, {len(category_names())} categories")

    print("\nBuilding world model ...")
    forest, prior_ids = build_nlp_world_model()
    print(f"  {len(forest._registry)} priors, {len(prior_ids)} protected")

    tau = calibrate_tau(D, sigma_scale=TAU_SIGMA, margin=TAU_MARGIN)
    print(f"  tau = {tau:.2f}")

    obs = Observer(
        forest,
        tau=tau,
        protected_ids=prior_ids,
        recombination_strategy="nearest_prior",
        hausdorff_absorption_threshold=0.15,
        persistence_guided_absorption=True,
        lacunarity_guided_creation=True,
        lacunarity_creation_radius=0.08,
        multifractal_guided_absorption=True,
        multifractal_crowding_radius=0.12,
    )

    # node_id -> list of (true_word, category) for observations it best explained
    node_explanations: dict[str, list[tuple[str, str]]] = defaultdict(list)

    cat_names = category_names()

    print(f"\nRunning {N_PASSES} passes over {N_SAMPLES} observations ...")
    for p in range(N_PASSES):
        n_explained = 0
        n_unexplained = 0
        rng = np.random.default_rng(SEED + p)
        order = rng.permutation(len(observations))

        for i in order:
            vec, true_word, category = observations[i]
            result = obs.observe(vec.astype(np.float64))
            if (n_explained + n_unexplained) % 500 == 0:
                print(f"    {n_explained + n_unexplained}/{N_SAMPLES} ...", flush=True)

            if result.explanation_tree:
                best_id = max(result.accuracy_scores,
                              key=lambda k: result.accuracy_scores[k])
                node_explanations[best_id].append((true_word, category))
                n_explained += 1
            else:
                n_unexplained += 1

        n_total = len(observations)
        print(f"  Pass {p+1}: explained {n_explained}/{n_total} "
              f"({100*n_explained/n_total:.1f}%), "
              f"unexplained {n_unexplained} "
              f"({100*n_unexplained/n_total:.1f}%)")

    # ------------------------------------------------------------------
    # Coverage summary
    # ------------------------------------------------------------------
    prior_explained = sum(len(v) for k, v in node_explanations.items()
                          if k in prior_ids)
    learned_nodes = [k for k in node_explanations if k not in prior_ids]
    learned_explained = sum(len(node_explanations[k]) for k in learned_nodes)
    total_attributed = prior_explained + learned_explained
    n_obs_total = N_PASSES * N_SAMPLES

    print(f"\n=== Coverage (over all passes, {n_obs_total} total obs) ===")
    print(f"  Prior nodes explained:   {prior_explained:5d} ({100*prior_explained/n_obs_total:.1f}%)")
    print(f"  Learned nodes explained: {learned_explained:5d} ({100*learned_explained/n_obs_total:.1f}%)")
    print(f"  Total attributed:        {total_attributed:5d} ({100*total_attributed/n_obs_total:.1f}%)")
    print(f"  Learned node count:      {len(learned_nodes)}")

    # ------------------------------------------------------------------
    # Semantic category alignment — learned nodes
    # ------------------------------------------------------------------
    if not learned_nodes:
        print("\nNo learned nodes — try lowering tau_margin or increasing N_PASSES.")
        return

    print(f"\n=== Learned node category alignment ({len(learned_nodes)} nodes) ===")

    rows = []
    for node_id in learned_nodes:
        labels = node_explanations[node_id]
        n = len(labels)
        cat_counts: dict[str, int] = defaultdict(int)
        word_counts: dict[str, int] = defaultdict(int)
        for word, cat in labels:
            cat_counts[cat] += 1
            word_counts[word] += 1
        dom_cat = max(cat_counts, key=lambda k: cat_counts[k])
        dom_word = max(word_counts, key=lambda k: word_counts[k])

        # nearest prior by Euclidean distance
        node_mu = forest._registry[node_id].mu
        nearest_prior = min(
            prior_ids,
            key=lambda pid: float(np.linalg.norm(
                forest._registry[pid].mu - node_mu
            )) if pid in forest._registry else float("inf")
        )
        rows.append({
            "id": node_id,
            "n": n,
            "cat_purity": purity(cat_counts),
            "word_purity": purity(word_counts),
            "dom_cat": dom_cat,
            "dom_word": dom_word,
            "nearest_prior": nearest_prior,
        })

    rows.sort(key=lambda r: r["n"], reverse=True)

    header = f"  {'Node ID':<20s}  {'n':>4s}  {'cat_pur':>7s}  {'word_pur':>8s}  {'dom_cat':<14s}  nearest_prior"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows[:20]:
        print(
            f"  {r['id']:<20s}  {r['n']:>4d}  "
            f"{r['cat_purity']:>7.3f}  {r['word_purity']:>8.3f}  "
            f"{r['dom_cat']:<14s}  {r['nearest_prior']}"
        )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    cat_purities = [r["cat_purity"] for r in rows if r["n"] >= 5]
    word_purities = [r["word_purity"] for r in rows if r["n"] >= 5]

    n_cats = len(cat_names)
    n_words = 25

    print(f"\n=== Category purity summary (nodes with n>=5) ===")
    if cat_purities:
        print(f"  Category: mean={np.mean(cat_purities):.3f}  max={np.max(cat_purities):.3f}")
        print(f"  Word:     mean={np.mean(word_purities):.3f}  max={np.max(word_purities):.3f}")
        print(f"  Random baseline: category={1/n_cats:.3f}, word={1/n_words:.3f}")
    else:
        print("  Insufficient learned nodes with n>=5.")

    # ------------------------------------------------------------------
    # Fractal diagnostics
    # ------------------------------------------------------------------
    prior_nodes = [forest._registry[k] for k in prior_ids if k in forest._registry]
    learned_nodes_list = [forest._registry[k] for k in forest._registry
                          if k not in prior_ids]

    if learned_nodes_list:
        hd = hausdorff_distance(learned_nodes_list, prior_nodes)
        print(f"\n=== Fractal diagnostics ===")
        print(f"  Learned nodes:              {len(learned_nodes_list)}")
        print(f"  Hausdorff(learned, priors): {hd:.4f}")

    print(f"\n=== Absorbed nodes: {len(obs.absorbed_ids)} ===")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the import (no full run)**

```bash
PYTHONPATH=. python3 -c "
from hpm_fractal_node.experiments.experiment_nlp import main
print('import ok')
"
```
Expected: `import ok` with no errors.

- [ ] **Step 3: Run the full experiment**

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_nlp.py
```
Expected output includes:
- `tau = ...`
- Pass 1/2/3 coverage lines
- `=== Category purity summary` section
- No errors or tracebacks

If 0% prior coverage occurs: re-run with `TAU_SIGMA=0.5` or `TAU_MARGIN=3.0` to lower tau.

- [ ] **Step 4: Commit**

```bash
git add hpm_fractal_node/experiments/experiment_nlp.py
git commit -m "feat: add NLP child language experiment with semantic category alignment"
```

---

## Task 5: Full test suite + final commit

- [ ] **Step 1: Run full test suite**

```bash
PYTHONPATH=. python3 -m pytest tests/hpm_fractal_node/nlp/ -v
```
Expected: All pass.

- [ ] **Step 2: Run broader suite to check for regressions**

```bash
PYTHONPATH=. python3 -m pytest tests/ -q --ignore=tests/encoders -x
```
Expected: Same pass count as before (314 passed, 1 pre-existing failure in `test_structured_arc_smoke`).

- [ ] **Step 3: Final commit if any cleanup needed**

```bash
git add -p  # stage any remaining changes
git commit -m "feat: complete NLP child language experiment"
```

---

## Notes

**Tau tuning:** If the experiment shows 0% prior coverage, try reducing `TAU_SIGMA` to 0.5 (lowers tau, making priors easier to trigger). The one-hot sparse space may need a different calibration than dSprites.

**Sentence count:** `generate_sentences()` generates ~600-800 observations from the templates above. If more variety is needed to reach 2000, add more templates in `nlp_loader.py`.

**Spec:** `docs/superpowers/specs/2026-03-28-nlp-child-language-experiment-design.md`
