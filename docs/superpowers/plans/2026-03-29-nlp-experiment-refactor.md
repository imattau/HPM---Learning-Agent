# NLP Experiment Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the NLP child language experiment to use HPM-aligned hierarchical HFN composition: D=107 slot-weighted context nodes replace the D=428 one-hot concatenation hack, and the world model is redesigned as four sub-trees (Objects, Grammar, Capabilities, Sentence Priors) built by uniform word HFN recombination.

**Architecture:** Word HFNs (D=107, one-hot mus) are atomic building blocks. All priors are composed from these atomics via weighted recombination. The Observer learns in D=107 space against a structured world model with explicit parent-child hierarchy encoding linguistic knowledge.

**Tech Stack:** Python, numpy, hfn (HFN, Forest, Observer, TieredForest)

---

## Pre-flight checks

- [ ] Confirm current test suite passes: `PYTHONPATH=. python3 -m pytest tests/ -v --tb=short`
- [ ] Note the current test count as baseline

---

## Task 1: nlp_loader.py — D=107, compose_context_node

**File:** `hpm_fractal_node/nlp/nlp_loader.py`

**What changes:**
- Remove `encode_context_window` function (D=428 concatenation)
- Add `compose_context_node(left2, left1, right1, right2)` returning D=107 float64
- Update `D = VOCAB_SIZE = 107` (was `4 * VOCAB_SIZE`)
- Update `_obs()` to call `compose_context_node`
- Keep all of: `VOCAB`, `VOCAB_INDEX`, `VOCAB_SIZE`, `_encode_word`, sentence generation code

**Slot weights:** `0.2*left2 + 0.35*left1 + 0.35*right1 + 0.10*right2`

### Step 1.1 — Write failing tests first

- [ ] Create file `tests/hpm_fractal_node/test_nlp_loader.py` with:

```python
"""Tests for nlp_loader.py — D=107 compose_context_node."""
import numpy as np
import pytest
from hpm_fractal_node.nlp.nlp_loader import (
    VOCAB_SIZE, D, compose_context_node, generate_sentences,
    VOCAB_INDEX,
)


def test_d_equals_vocab_size():
    """D must be 107, not 428."""
    assert D == 107
    assert D == VOCAB_SIZE


def test_compose_context_node_shape():
    """compose_context_node returns shape (107,)."""
    out = compose_context_node("the", "dog", "barked", "<end>")
    assert out.shape == (D,)


def test_compose_context_node_dtype():
    """compose_context_node returns float64."""
    out = compose_context_node("the", "dog", "barked", "<end>")
    assert out.dtype == np.float64


def test_compose_context_node_weights_sum_to_one():
    """Slot weights 0.2+0.35+0.35+0.10 sum to 1.0; so all-distinct output sums to 1.0."""
    # Use four distinct words so no index overlaps
    out = compose_context_node("the", "dog", "barked", "park")
    assert abs(out.sum() - 1.0) < 1e-10


def test_compose_context_node_repeated_word_accumulates():
    """Same word in left1 and right1 accumulates 0.35+0.35=0.70 at that index."""
    word = "dog"
    idx = VOCAB_INDEX["dog"]
    out = compose_context_node("<start>", word, word, "<end>")
    # left2=<start>=0.20, left1=dog=0.35, right1=dog=0.35, right2=<end>=0.10
    # dog index should hold 0.70
    assert abs(out[idx] - 0.70) < 1e-10


def test_compose_context_node_known_values():
    """Spot-check exact weight at a specific index."""
    # left2="the" (w=0.20), left1="the" (w=0.35), right1="cat" (w=0.35), right2="<end>" (w=0.10)
    out = compose_context_node("the", "the", "cat", "<end>")
    the_idx = VOCAB_INDEX["the"]
    cat_idx = VOCAB_INDEX["cat"]
    assert abs(out[the_idx] - 0.55) < 1e-10   # 0.20 + 0.35
    assert abs(out[cat_idx] - 0.35) < 1e-10


def test_generate_sentences_returns_d107_vectors():
    """generate_sentences() must produce D=107 vectors after refactor."""
    data = generate_sentences(seed=42)
    assert len(data) == 2000
    vec, word, cat = data[0]
    assert vec.shape == (D,)
    assert vec.dtype == np.float64


def test_generate_sentences_all_vectors_sum_to_one():
    """Every context vector from generate_sentences should sum to 1.0."""
    data = generate_sentences(seed=42)
    for vec, _, _ in data[:50]:  # check first 50 for speed
        assert abs(vec.sum() - 1.0) < 1e-9, f"Vector sum={vec.sum()}"
```

- [ ] Run tests and confirm they **fail**: `PYTHONPATH=. python3 -m pytest tests/hpm_fractal_node/test_nlp_loader.py -v`

### Step 1.2 — Implement changes in nlp_loader.py

- [ ] Replace `D: int = 4 * VOCAB_SIZE  # 428` with `D: int = VOCAB_SIZE  # 107`

- [ ] Remove the `encode_context_window` function entirely

- [ ] Add `compose_context_node` immediately after `_encode_word`:

```python
def compose_context_node(
    left2: str,
    left1: str,
    right1: str,
    right2: str,
) -> np.ndarray:
    """
    Compose a D=107 context node from four context slot words.

    Slot-weighted recombination of atomic one-hot mus:
        mu = 0.20*mu(left2) + 0.35*mu(left1) + 0.35*mu(right1) + 0.10*mu(right2)

    Weights encode proximity to the masked position (left1/right1 nearest,
    left2/right2 furthest). Weights sum to 1.0. Repeated words accumulate
    weight at their vocab index.

    Returns shape (D,) float64 vector.
    """
    mu_left2  = _encode_word(left2).astype(np.float64)
    mu_left1  = _encode_word(left1).astype(np.float64)
    mu_right1 = _encode_word(right1).astype(np.float64)
    mu_right2 = _encode_word(right2).astype(np.float64)
    return 0.20 * mu_left2 + 0.35 * mu_left1 + 0.35 * mu_right1 + 0.10 * mu_right2
```

- [ ] Update `_obs()` to call `compose_context_node` instead of `encode_context_window`:

```python
def _obs(left2: str, left1: str, right1: str, right2: str,
         true_word: str) -> tuple[np.ndarray, str, str]:
    vec = compose_context_node(left2, left1, right1, right2)
    return vec, true_word, word_category(true_word)
```

### Step 1.3 — Verify tests pass

- [ ] Run: `PYTHONPATH=. python3 -m pytest tests/hpm_fractal_node/test_nlp_loader.py -v`
- [ ] All 8 tests must pass; no `encode_context_window` references remain in loader

---

## Task 2: nlp_world_model.py — atomic word nodes + Objects sub-tree

**File:** `hpm_fractal_node/nlp/nlp_world_model.py`

**What changes:**
- Remove `_CANONICAL_SENTENCES`, `_word_mu`, `_avg_mus`
- Add `_one_hot(word)` helper returning D=107 float64 one-hot
- Add `_recombine(*mus)` helper returning equal-weight mean
- Register 107 atomic word nodes (`word_{vocab_word}`)
- Build Objects sub-tree bottom-up with `node.add_child()` wiring

### Step 2.1 — Write failing tests first

- [ ] Create file `tests/hpm_fractal_node/test_nlp_world_model_objects.py` with:

```python
"""Tests for nlp_world_model.py — atomic nodes + Objects sub-tree."""
import numpy as np
import pytest
from hpm_fractal_node.nlp.nlp_loader import VOCAB, VOCAB_SIZE
from hpm_fractal_node.nlp.nlp_world_model import build_nlp_world_model


def test_atomic_word_nodes_all_registered():
    """All 107 atomic word nodes must be in the forest."""
    forest, prior_ids = build_nlp_world_model()
    for word in VOCAB:
        node_id = f"word_{word}"
        assert node_id in prior_ids, f"Missing atomic node: {node_id}"
        assert forest.get(node_id) is not None, f"Not in forest: {node_id}"


def test_atomic_word_node_mu_is_one_hot():
    """Atomic word node mu must be one-hot at the word's vocab index."""
    from hpm_fractal_node.nlp.nlp_loader import VOCAB_INDEX
    forest, _ = build_nlp_world_model()
    node = forest.get("word_dog")
    assert node is not None
    dog_idx = VOCAB_INDEX["dog"]
    assert abs(node.mu[dog_idx] - 1.0) < 1e-10
    assert abs(node.mu.sum() - 1.0) < 1e-10
    assert node.mu.dtype == np.float64


def test_atomic_word_node_mu_shape():
    """Atomic word node mu must be shape (107,)."""
    forest, _ = build_nlp_world_model()
    node = forest.get("word_the")
    assert node.mu.shape == (VOCAB_SIZE,)


def test_objects_leaf_nodes_present():
    """All 25 object leaf nodes must be registered."""
    expected_leaves = [
        "obj_dog", "obj_cat", "obj_bird",
        "obj_mum", "obj_dad", "obj_grandma", "obj_grandpa",
        "obj_brother", "obj_sister",
        "obj_teacher", "obj_doctor", "obj_friend", "obj_classmate",
        "obj_boy", "obj_girl", "obj_baby",
        "obj_apple", "obj_bread", "obj_milk",
        "obj_ball", "obj_book", "obj_toy",
        "obj_park", "obj_home", "obj_school",
    ]
    forest, prior_ids = build_nlp_world_model()
    for leaf_id in expected_leaves:
        assert leaf_id in prior_ids, f"Missing object leaf: {leaf_id}"
        assert forest.get(leaf_id) is not None


def test_objects_intermediate_nodes_present():
    """Intermediate Objects sub-tree nodes must exist."""
    expected = [
        "obj_animal", "obj_person", "obj_food", "obj_object", "obj_place",
        "obj_animate", "obj_inanimate",
        "obj_family", "obj_adult", "obj_child",
        "obj_noun",
    ]
    forest, prior_ids = build_nlp_world_model()
    for node_id in expected:
        assert node_id in prior_ids, f"Missing intermediate: {node_id}"
        assert forest.get(node_id) is not None


def test_obj_animal_has_three_children():
    """obj_animal must have exactly 3 children: dog, cat, bird."""
    forest, _ = build_nlp_world_model()
    node = forest.get("obj_animal")
    assert node is not None
    assert len(node._children) == 3
    child_ids = {c.id for c in node._children}
    assert child_ids == {"obj_dog", "obj_cat", "obj_bird"}


def test_obj_food_has_three_children():
    """obj_food must have 3 children: apple, bread, milk."""
    forest, _ = build_nlp_world_model()
    node = forest.get("obj_food")
    assert node is not None
    assert len(node._children) == 3
    child_ids = {c.id for c in node._children}
    assert child_ids == {"obj_apple", "obj_bread", "obj_milk"}


def test_obj_place_has_three_children():
    """obj_place must have 3 children: park, home, school."""
    forest, _ = build_nlp_world_model()
    node = forest.get("obj_place")
    assert node is not None
    assert len(node._children) == 3
    child_ids = {c.id for c in node._children}
    assert child_ids == {"obj_park", "obj_home", "obj_school"}


def test_obj_animate_has_two_children():
    """obj_animate must have children: obj_animal, obj_person."""
    forest, _ = build_nlp_world_model()
    node = forest.get("obj_animate")
    assert node is not None
    child_ids = {c.id for c in node._children}
    assert "obj_animal" in child_ids
    assert "obj_person" in child_ids


def test_obj_noun_mu_shape():
    """obj_noun mu must be shape (107,) float64."""
    forest, _ = build_nlp_world_model()
    node = forest.get("obj_noun")
    assert node is not None
    assert node.mu.shape == (VOCAB_SIZE,)
    assert node.mu.dtype == np.float64


def test_obj_dog_mu_orthogonal_to_obj_place():
    """Dog context words (barked, fetched) vs place context words (to, at, park) are distinct."""
    forest, _ = build_nlp_world_model()
    dog_node = forest.get("obj_dog")
    place_node = forest.get("obj_place")
    assert dog_node is not None
    assert place_node is not None
    # Cosine similarity should be low — they encode orthogonal vocabularies
    dot = float(np.dot(dog_node.mu, place_node.mu))
    norm_dog = float(np.linalg.norm(dog_node.mu))
    norm_place = float(np.linalg.norm(place_node.mu))
    cosine = dot / (norm_dog * norm_place + 1e-12)
    assert cosine < 0.5, f"Dog and place priors too similar: cosine={cosine:.3f}"
```

- [ ] Run tests and confirm they **fail**: `PYTHONPATH=. python3 -m pytest tests/hpm_fractal_node/test_nlp_world_model_objects.py -v`

### Step 2.2 — Rewrite nlp_world_model.py header and helpers

- [ ] Replace the entire file with the new implementation. Begin with:

```python
"""
NLP world model — HPM-aligned hierarchical HFN composition.

One Forest, four sub-trees (all D=107):
  1. Atomic word nodes (107): one-hot mus for every vocab word
  2. Objects sub-tree: semantic object hierarchy
  3. Grammar sub-tree: grammatical structure patterns
  4. Capabilities sub-tree: entity-specific action patterns
  5. Sentence priors (~20): exemplar sentence mus

All composed node mus are equal-weight recombinations of atomic word mus
or other composed node mus. Parent-child relationships are wired via
HFN.add_child() before Forest registration. Nodes registered bottom-up.
"""
from __future__ import annotations

import numpy as np

from hfn import HFN, Forest
from hpm_fractal_node.nlp.nlp_loader import D, VOCAB, VOCAB_INDEX, VOCAB_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _one_hot(word: str) -> np.ndarray:
    """Return D=107 float64 one-hot for word (falls back to <unknown>)."""
    idx = VOCAB_INDEX.get(word.lower(), VOCAB_INDEX["<unknown>"])
    vec = np.zeros(VOCAB_SIZE, dtype=np.float64)
    vec[idx] = 1.0
    return vec


def _recombine(*mus: np.ndarray) -> np.ndarray:
    """Equal-weight mean of one or more D=107 mu vectors. Returns float64."""
    return np.mean(np.stack(mus, axis=0), axis=0).astype(np.float64)
```

### Step 2.3 — Register 107 atomic word nodes

- [ ] Inside `build_nlp_world_model`, add atomic word node registration:

```python
def build_nlp_world_model(forest_cls=None, **tiered_kwargs) -> tuple[Forest, set[str]]:
    """
    Build the NLP world model with four HPM-aligned sub-trees.

    Returns
    -------
    forest : Forest
        Contains all prior nodes (atomic + composed).
    prior_ids : set[str]
        All node IDs to pass as protected_ids to Observer.
    """
    from hfn.forest import Forest as _Forest
    if forest_cls is None:
        forest_cls = _Forest
    kwargs = tiered_kwargs if forest_cls is not _Forest else {}
    forest = forest_cls(D=D, forest_id="nlp_child_language", **kwargs)
    prior_ids: set[str] = set()

    sigma = np.eye(D, dtype=np.float64) * 1.0

    def add(node: HFN) -> None:
        forest.register(node)
        prior_ids.add(node.id)

    # ------------------------------------------------------------------
    # Sub-tree 1: Atomic word nodes (107 nodes, one per vocab word)
    # mu = one-hot at word's vocab index. Fixed building blocks.
    # ------------------------------------------------------------------
    atomic: dict[str, HFN] = {}
    for word in VOCAB:
        node = HFN(mu=_one_hot(word), sigma=sigma, id=f"word_{word}")
        add(node)
        atomic[word] = node
```

### Step 2.4 — Build Objects sub-tree

- [ ] Add Objects sub-tree code after atomic registration. Leaf mus are recombinations of the atomic one-hot mus of context-characterising words (words that appear AROUND the entity, not the entity itself):

```python
    # ------------------------------------------------------------------
    # Sub-tree 2: Objects sub-tree
    # Leaf mus = recombine of atomic one-hot mus of context words
    # (words that characterise what appears AROUND that entity)
    # ------------------------------------------------------------------

    # --- Leaves: animals ---
    # Dog context: animal action verbs + determiners
    obj_dog = HFN(
        mu=_recombine(atomic["barked"].mu, atomic["fetched"].mu,
                      atomic["chased"].mu, atomic["the"].mu,
                      atomic["a"].mu, atomic["my"].mu),
        sigma=sigma, id="obj_dog",
    )
    add(obj_dog)

    obj_cat = HFN(
        mu=_recombine(atomic["meowed"].mu, atomic["chased"].mu,
                      atomic["chirped"].mu, atomic["the"].mu,
                      atomic["a"].mu, atomic["my"].mu),
        sigma=sigma, id="obj_cat",
    )
    add(obj_cat)

    obj_bird = HFN(
        mu=_recombine(atomic["chirped"].mu, atomic["chased"].mu,
                      atomic["the"].mu, atomic["a"].mu,
                      atomic["little"].mu, atomic["my"].mu),
        sigma=sigma, id="obj_bird",
    )
    add(obj_bird)

    # --- Leaves: family ---
    _family_ctx = _recombine(atomic["my"].mu, atomic["her"].mu,
                             atomic["his"].mu, atomic["walked"].mu,
                             atomic["helped"].mu, atomic["gave"].mu)
    obj_mum      = HFN(mu=_recombine(_family_ctx, atomic["ate"].mu), sigma=sigma, id="obj_mum")
    obj_dad      = HFN(mu=_recombine(_family_ctx, atomic["ate"].mu), sigma=sigma, id="obj_dad")
    obj_grandma  = HFN(mu=_recombine(_family_ctx, atomic["her"].mu), sigma=sigma, id="obj_grandma")
    obj_grandpa  = HFN(mu=_recombine(_family_ctx, atomic["his"].mu), sigma=sigma, id="obj_grandpa")
    obj_brother  = HFN(mu=_recombine(_family_ctx, atomic["ran"].mu), sigma=sigma, id="obj_brother")
    obj_sister   = HFN(mu=_recombine(_family_ctx, atomic["ran"].mu), sigma=sigma, id="obj_sister")
    for n in [obj_mum, obj_dad, obj_grandma, obj_grandpa, obj_brother, obj_sister]:
        add(n)

    # --- Leaves: adults ---
    _adult_ctx = _recombine(atomic["the"].mu, atomic["my"].mu,
                            atomic["helped"].mu, atomic["walked"].mu,
                            atomic["gave"].mu)
    obj_teacher   = HFN(mu=_recombine(_adult_ctx, atomic["my"].mu), sigma=sigma, id="obj_teacher")
    obj_doctor    = HFN(mu=_recombine(_adult_ctx, atomic["the"].mu), sigma=sigma, id="obj_doctor")
    obj_friend    = HFN(mu=_recombine(_adult_ctx, atomic["my"].mu), sigma=sigma, id="obj_friend")
    obj_classmate = HFN(mu=_recombine(_adult_ctx, atomic["my"].mu), sigma=sigma, id="obj_classmate")
    for n in [obj_teacher, obj_doctor, obj_friend, obj_classmate]:
        add(n)

    # --- Leaves: children ---
    _child_ctx = _recombine(atomic["the"].mu, atomic["played"].mu,
                            atomic["ran"].mu, atomic["walked"].mu,
                            atomic["helped"].mu)
    obj_boy   = HFN(mu=_recombine(_child_ctx, atomic["ran"].mu),    sigma=sigma, id="obj_boy")
    obj_girl  = HFN(mu=_recombine(_child_ctx, atomic["played"].mu), sigma=sigma, id="obj_girl")
    obj_baby  = HFN(mu=_recombine(_child_ctx, atomic["took"].mu),   sigma=sigma, id="obj_baby")
    for n in [obj_boy, obj_girl, obj_baby]:
        add(n)

    # --- Leaves: food ---
    _food_ctx = _recombine(atomic["the"].mu, atomic["a"].mu,
                           atomic["ate"].mu, atomic["was"].mu,
                           atomic["good"].mu, atomic["is"].mu)
    obj_apple = HFN(mu=_recombine(_food_ctx, atomic["red"].mu),   sigma=sigma, id="obj_apple")
    obj_bread = HFN(mu=_recombine(_food_ctx, atomic["old"].mu),   sigma=sigma, id="obj_bread")
    obj_milk  = HFN(mu=_recombine(_food_ctx, atomic["good"].mu),  sigma=sigma, id="obj_milk")
    for n in [obj_apple, obj_bread, obj_milk]:
        add(n)

    # --- Leaves: objects ---
    _obj_ctx = _recombine(atomic["the"].mu, atomic["a"].mu,
                          atomic["is"].mu, atomic["here"].mu,
                          atomic["on"].mu)
    obj_ball = HFN(mu=_recombine(_obj_ctx, atomic["big"].mu),    sigma=sigma, id="obj_ball")
    obj_book = HFN(mu=_recombine(_obj_ctx, atomic["old"].mu),    sigma=sigma, id="obj_book")
    obj_toy  = HFN(mu=_recombine(_obj_ctx, atomic["little"].mu), sigma=sigma, id="obj_toy")
    for n in [obj_ball, obj_book, obj_toy]:
        add(n)

    # --- Leaves: places ---
    _place_ctx = _recombine(atomic["to"].mu, atomic["at"].mu,
                            atomic["went"].mu if "went" in atomic else atomic["walked"].mu,
                            atomic["ran"].mu, atomic["the"].mu)
    obj_park   = HFN(mu=_recombine(_place_ctx, atomic["big"].mu),  sigma=sigma, id="obj_park")
    obj_home   = HFN(mu=_recombine(_place_ctx, atomic["old"].mu),  sigma=sigma, id="obj_home")
    obj_school = HFN(mu=_recombine(_place_ctx, atomic["my"].mu),   sigma=sigma, id="obj_school")
    for n in [obj_park, obj_home, obj_school]:
        add(n)

    # --- Intermediate: obj_animal (parent of dog, cat, bird) ---
    obj_animal = HFN(
        mu=_recombine(obj_dog.mu, obj_cat.mu, obj_bird.mu),
        sigma=sigma, id="obj_animal",
    )
    obj_animal.add_child(obj_dog)
    obj_animal.add_child(obj_cat)
    obj_animal.add_child(obj_bird)
    add(obj_animal)

    # --- Intermediate: obj_family, obj_adult, obj_child ---
    obj_family = HFN(
        mu=_recombine(obj_mum.mu, obj_dad.mu, obj_grandma.mu,
                      obj_grandpa.mu, obj_brother.mu, obj_sister.mu),
        sigma=sigma, id="obj_family",
    )
    for n in [obj_mum, obj_dad, obj_grandma, obj_grandpa, obj_brother, obj_sister]:
        obj_family.add_child(n)
    add(obj_family)

    obj_adult = HFN(
        mu=_recombine(obj_teacher.mu, obj_doctor.mu, obj_friend.mu, obj_classmate.mu),
        sigma=sigma, id="obj_adult",
    )
    for n in [obj_teacher, obj_doctor, obj_friend, obj_classmate]:
        obj_adult.add_child(n)
    add(obj_adult)

    obj_child = HFN(
        mu=_recombine(obj_boy.mu, obj_girl.mu, obj_baby.mu),
        sigma=sigma, id="obj_child",
    )
    for n in [obj_boy, obj_girl, obj_baby]:
        obj_child.add_child(n)
    add(obj_child)

    # --- Intermediate: obj_person (parent of family, adult, child) ---
    obj_person = HFN(
        mu=_recombine(obj_family.mu, obj_adult.mu, obj_child.mu),
        sigma=sigma, id="obj_person",
    )
    obj_person.add_child(obj_family)
    obj_person.add_child(obj_adult)
    obj_person.add_child(obj_child)
    add(obj_person)

    # --- Intermediate: obj_animate (parent of animal, person) ---
    obj_animate = HFN(
        mu=_recombine(obj_animal.mu, obj_person.mu),
        sigma=sigma, id="obj_animate",
    )
    obj_animate.add_child(obj_animal)
    obj_animate.add_child(obj_person)
    add(obj_animate)

    # --- Intermediate: obj_food, obj_object, obj_place (already leaves, now wrap) ---
    obj_food_node = HFN(
        mu=_recombine(obj_apple.mu, obj_bread.mu, obj_milk.mu),
        sigma=sigma, id="obj_food",
    )
    for n in [obj_apple, obj_bread, obj_milk]:
        obj_food_node.add_child(n)
    add(obj_food_node)

    obj_object_node = HFN(
        mu=_recombine(obj_ball.mu, obj_book.mu, obj_toy.mu),
        sigma=sigma, id="obj_object",
    )
    for n in [obj_ball, obj_book, obj_toy]:
        obj_object_node.add_child(n)
    add(obj_object_node)

    obj_place_node = HFN(
        mu=_recombine(obj_park.mu, obj_home.mu, obj_school.mu),
        sigma=sigma, id="obj_place",
    )
    for n in [obj_park, obj_home, obj_school]:
        obj_place_node.add_child(n)
    add(obj_place_node)

    # --- Intermediate: obj_inanimate (parent of food, object, place) ---
    obj_inanimate = HFN(
        mu=_recombine(obj_food_node.mu, obj_object_node.mu, obj_place_node.mu),
        sigma=sigma, id="obj_inanimate",
    )
    obj_inanimate.add_child(obj_food_node)
    obj_inanimate.add_child(obj_object_node)
    obj_inanimate.add_child(obj_place_node)
    add(obj_inanimate)

    # --- Root: obj_noun (parent of animate, inanimate) ---
    obj_noun = HFN(
        mu=_recombine(obj_animate.mu, obj_inanimate.mu),
        sigma=sigma, id="obj_noun",
    )
    obj_noun.add_child(obj_animate)
    obj_noun.add_child(obj_inanimate)
    add(obj_noun)
```

### Step 2.5 — Verify Task 2 tests pass

- [ ] Run: `PYTHONPATH=. python3 -m pytest tests/hpm_fractal_node/test_nlp_world_model_objects.py -v`
- [ ] All tests must pass

---

## Task 3: nlp_world_model.py — Grammar + Capabilities + Sentence priors

**File:** `hpm_fractal_node/nlp/nlp_world_model.py` (continue from Task 2)

### Step 3.1 — Write failing tests first

- [ ] Create file `tests/hpm_fractal_node/test_nlp_world_model_full.py` with:

```python
"""Tests for nlp_world_model.py — Grammar, Capabilities, Sentence priors."""
import numpy as np
import pytest
from hpm_fractal_node.nlp.nlp_loader import VOCAB_SIZE
from hpm_fractal_node.nlp.nlp_world_model import build_nlp_world_model


def test_grammar_nodes_present():
    """All grammar sub-tree nodes must be present."""
    expected = [
        "gram_determiner", "gram_preposition", "gram_descriptor",
        "gram_word_class",
        "gram_noun_phrase", "gram_verb_phrase", "gram_prep_phrase",
        "gram_phrase_structure",
        "gram_agent_action", "gram_action_patient", "gram_motion_to_place",
        "gram_sentence_pattern",
        "gram_root",
    ]
    forest, prior_ids = build_nlp_world_model()
    for node_id in expected:
        assert node_id in prior_ids, f"Missing grammar node: {node_id}"
        assert forest.get(node_id) is not None


def test_gram_word_class_has_three_children():
    """gram_word_class must have 3 children: determiner, preposition, descriptor."""
    forest, _ = build_nlp_world_model()
    node = forest.get("gram_word_class")
    assert len(node._children) == 3
    child_ids = {c.id for c in node._children}
    assert child_ids == {"gram_determiner", "gram_preposition", "gram_descriptor"}


def test_gram_root_has_three_children():
    """gram_root must have 3 children: word_class, phrase_structure, sentence_pattern."""
    forest, _ = build_nlp_world_model()
    node = forest.get("gram_root")
    assert len(node._children) == 3
    child_ids = {c.id for c in node._children}
    assert child_ids == {"gram_word_class", "gram_phrase_structure", "gram_sentence_pattern"}


def test_capability_nodes_present():
    """All capabilities sub-tree nodes must be present."""
    expected = [
        "cap_dog_barks", "cap_dog_fetches", "cap_dog",
        "cap_cat_meows", "cap_cat_chases", "cap_cat",
        "cap_bird_chirps", "cap_bird",
        "cap_animal",
        "cap_person_walks", "cap_person_eats", "cap_person_gives",
        "cap_general_person",
        "cap_family_helps", "cap_family",
        "cap_child_plays", "cap_child",
        "cap_person",
        "cap_root",
    ]
    forest, prior_ids = build_nlp_world_model()
    for node_id in expected:
        assert node_id in prior_ids, f"Missing capability node: {node_id}"
        assert forest.get(node_id) is not None


def test_cap_dog_has_two_children():
    """cap_dog must have 2 children: cap_dog_barks, cap_dog_fetches."""
    forest, _ = build_nlp_world_model()
    node = forest.get("cap_dog")
    assert len(node._children) == 2
    child_ids = {c.id for c in node._children}
    assert child_ids == {"cap_dog_barks", "cap_dog_fetches"}


def test_cap_root_has_two_children():
    """cap_root must have 2 children: cap_animal, cap_person."""
    forest, _ = build_nlp_world_model()
    node = forest.get("cap_root")
    assert len(node._children) == 2
    child_ids = {c.id for c in node._children}
    assert child_ids == {"cap_animal", "cap_person"}


def test_sentence_priors_present():
    """At least 15 sentence prior nodes must be present (sent_ prefix)."""
    forest, prior_ids = build_nlp_world_model()
    sent_nodes = [nid for nid in prior_ids if nid.startswith("sent_")]
    assert len(sent_nodes) >= 15, f"Only {len(sent_nodes)} sentence priors"


def test_sentence_prior_mu_shape():
    """Sentence prior mus must be shape (107,) float64."""
    forest, prior_ids = build_nlp_world_model()
    sent_ids = [nid for nid in prior_ids if nid.startswith("sent_")]
    for sid in sent_ids[:3]:
        node = forest.get(sid)
        assert node is not None
        assert node.mu.shape == (VOCAB_SIZE,)
        assert node.mu.dtype == np.float64


def test_sentence_prior_mu_sums_to_one():
    """Sentence prior mu = mean of one-hot mus, so must sum to 1.0."""
    forest, prior_ids = build_nlp_world_model()
    sent_ids = [nid for nid in prior_ids if nid.startswith("sent_")]
    for sid in sent_ids:
        node = forest.get(sid)
        assert abs(node.mu.sum() - 1.0) < 1e-9, f"{sid} mu sum={node.mu.sum()}"


def test_total_node_count_reasonable():
    """Total prior node count: 107 atomic + ~60 composed + ~20 sentence >= 180."""
    _, prior_ids = build_nlp_world_model()
    assert len(prior_ids) >= 180, f"Too few prior nodes: {len(prior_ids)}"


def test_all_node_mus_are_float64():
    """Every node mu in the forest must be float64."""
    forest, prior_ids = build_nlp_world_model()
    for node_id in prior_ids:
        node = forest.get(node_id)
        assert node is not None
        assert node.mu.dtype == np.float64, f"{node_id} has dtype {node.mu.dtype}"


def test_all_node_mus_correct_shape():
    """Every node mu must be shape (107,)."""
    forest, prior_ids = build_nlp_world_model()
    for node_id in prior_ids:
        node = forest.get(node_id)
        assert node.mu.shape == (VOCAB_SIZE,), (
            f"{node_id} has shape {node.mu.shape}"
        )
```

- [ ] Run tests and confirm they **fail**: `PYTHONPATH=. python3 -m pytest tests/hpm_fractal_node/test_nlp_world_model_full.py -v`

### Step 3.2 — Add Grammar sub-tree

- [ ] Continue inside `build_nlp_world_model()`, after the Objects sub-tree:

```python
    # ------------------------------------------------------------------
    # Sub-tree 3: Grammar rules sub-tree
    # ------------------------------------------------------------------

    # --- Word class leaves ---
    gram_determiner = HFN(
        mu=_recombine(atomic["the"].mu, atomic["a"].mu, atomic["an"].mu,
                      atomic["my"].mu, atomic["her"].mu, atomic["his"].mu),
        sigma=sigma, id="gram_determiner",
    )
    add(gram_determiner)

    gram_preposition = HFN(
        mu=_recombine(atomic["to"].mu, atomic["at"].mu, atomic["in"].mu,
                      atomic["on"].mu, atomic["with"].mu,
                      atomic["for"].mu, atomic["from"].mu),
        sigma=sigma, id="gram_preposition",
    )
    add(gram_preposition)

    gram_descriptor = HFN(
        mu=_recombine(atomic["big"].mu, atomic["small"].mu, atomic["little"].mu,
                      atomic["red"].mu, atomic["blue"].mu, atomic["old"].mu),
        sigma=sigma, id="gram_descriptor",
    )
    add(gram_descriptor)

    # --- gram_word_class: parent of determiner, preposition, descriptor ---
    gram_word_class = HFN(
        mu=_recombine(gram_determiner.mu, gram_preposition.mu, gram_descriptor.mu),
        sigma=sigma, id="gram_word_class",
    )
    gram_word_class.add_child(gram_determiner)
    gram_word_class.add_child(gram_preposition)
    gram_word_class.add_child(gram_descriptor)
    add(gram_word_class)

    # --- Action verb atomic mus for phrase structure ---
    _action_verb_mus = _recombine(
        atomic["ate"].mu, atomic["threw"].mu, atomic["ran"].mu, atomic["walked"].mu,
        atomic["gave"].mu, atomic["took"].mu, atomic["helped"].mu, atomic["played"].mu,
    )

    # --- Phrase structure leaves ---
    gram_noun_phrase = HFN(
        mu=_recombine(gram_determiner.mu, obj_animate.mu, obj_inanimate.mu),
        sigma=sigma, id="gram_noun_phrase",
    )
    add(gram_noun_phrase)

    gram_verb_phrase = HFN(
        mu=_recombine(_action_verb_mus, obj_inanimate.mu),
        sigma=sigma, id="gram_verb_phrase",
    )
    add(gram_verb_phrase)

    gram_prep_phrase = HFN(
        mu=_recombine(gram_preposition.mu, obj_place_node.mu),
        sigma=sigma, id="gram_prep_phrase",
    )
    add(gram_prep_phrase)

    # --- gram_phrase_structure: parent of noun_phrase, verb_phrase, prep_phrase ---
    gram_phrase_structure = HFN(
        mu=_recombine(gram_noun_phrase.mu, gram_verb_phrase.mu, gram_prep_phrase.mu),
        sigma=sigma, id="gram_phrase_structure",
    )
    gram_phrase_structure.add_child(gram_noun_phrase)
    gram_phrase_structure.add_child(gram_verb_phrase)
    gram_phrase_structure.add_child(gram_prep_phrase)
    add(gram_phrase_structure)

    # --- Sentence pattern leaves ---
    gram_agent_action = HFN(
        mu=_recombine(obj_animate.mu, _action_verb_mus),
        sigma=sigma, id="gram_agent_action",
    )
    add(gram_agent_action)

    gram_action_patient = HFN(
        mu=_recombine(_action_verb_mus, obj_inanimate.mu),
        sigma=sigma, id="gram_action_patient",
    )
    add(gram_action_patient)

    gram_motion_to_place = HFN(
        mu=_recombine(atomic["walked"].mu, atomic["ran"].mu,
                      gram_preposition.mu, obj_place_node.mu),
        sigma=sigma, id="gram_motion_to_place",
    )
    add(gram_motion_to_place)

    # --- gram_sentence_pattern: parent of agent_action, action_patient, motion_to_place ---
    gram_sentence_pattern = HFN(
        mu=_recombine(gram_agent_action.mu, gram_action_patient.mu, gram_motion_to_place.mu),
        sigma=sigma, id="gram_sentence_pattern",
    )
    gram_sentence_pattern.add_child(gram_agent_action)
    gram_sentence_pattern.add_child(gram_action_patient)
    gram_sentence_pattern.add_child(gram_motion_to_place)
    add(gram_sentence_pattern)

    # --- gram_root: parent of word_class, phrase_structure, sentence_pattern ---
    gram_root = HFN(
        mu=_recombine(gram_word_class.mu, gram_phrase_structure.mu, gram_sentence_pattern.mu),
        sigma=sigma, id="gram_root",
    )
    gram_root.add_child(gram_word_class)
    gram_root.add_child(gram_phrase_structure)
    gram_root.add_child(gram_sentence_pattern)
    add(gram_root)
```

### Step 3.3 — Add Capabilities sub-tree

- [ ] Continue inside `build_nlp_world_model()`:

```python
    # ------------------------------------------------------------------
    # Sub-tree 4: Capabilities sub-tree
    # ------------------------------------------------------------------

    # --- Dog capabilities ---
    cap_dog_barks = HFN(
        mu=_recombine(obj_dog.mu, atomic["barked"].mu),
        sigma=sigma, id="cap_dog_barks",
    )
    add(cap_dog_barks)

    cap_dog_fetches = HFN(
        mu=_recombine(obj_dog.mu, atomic["fetched"].mu),
        sigma=sigma, id="cap_dog_fetches",
    )
    add(cap_dog_fetches)

    cap_dog = HFN(
        mu=_recombine(cap_dog_barks.mu, cap_dog_fetches.mu),
        sigma=sigma, id="cap_dog",
    )
    cap_dog.add_child(cap_dog_barks)
    cap_dog.add_child(cap_dog_fetches)
    add(cap_dog)

    # --- Cat capabilities ---
    cap_cat_meows = HFN(
        mu=_recombine(obj_cat.mu, atomic["meowed"].mu),
        sigma=sigma, id="cap_cat_meows",
    )
    add(cap_cat_meows)

    cap_cat_chases = HFN(
        mu=_recombine(obj_cat.mu, atomic["chased"].mu),
        sigma=sigma, id="cap_cat_chases",
    )
    add(cap_cat_chases)

    cap_cat = HFN(
        mu=_recombine(cap_cat_meows.mu, cap_cat_chases.mu),
        sigma=sigma, id="cap_cat",
    )
    cap_cat.add_child(cap_cat_meows)
    cap_cat.add_child(cap_cat_chases)
    add(cap_cat)

    # --- Bird capabilities ---
    cap_bird_chirps = HFN(
        mu=_recombine(obj_bird.mu, atomic["chirped"].mu),
        sigma=sigma, id="cap_bird_chirps",
    )
    add(cap_bird_chirps)

    cap_bird = HFN(
        mu=cap_bird_chirps.mu.copy(),
        sigma=sigma, id="cap_bird",
    )
    cap_bird.add_child(cap_bird_chirps)
    add(cap_bird)

    # --- cap_animal: parent of cap_dog, cap_cat, cap_bird ---
    cap_animal = HFN(
        mu=_recombine(cap_dog.mu, cap_cat.mu, cap_bird.mu),
        sigma=sigma, id="cap_animal",
    )
    cap_animal.add_child(cap_dog)
    cap_animal.add_child(cap_cat)
    cap_animal.add_child(cap_bird)
    add(cap_animal)

    # --- General person capabilities ---
    cap_person_walks = HFN(
        mu=_recombine(obj_person.mu, atomic["walked"].mu),
        sigma=sigma, id="cap_person_walks",
    )
    add(cap_person_walks)

    cap_person_eats = HFN(
        mu=_recombine(obj_person.mu, atomic["ate"].mu),
        sigma=sigma, id="cap_person_eats",
    )
    add(cap_person_eats)

    cap_person_gives = HFN(
        mu=_recombine(obj_person.mu, atomic["gave"].mu),
        sigma=sigma, id="cap_person_gives",
    )
    add(cap_person_gives)

    cap_general_person = HFN(
        mu=_recombine(cap_person_walks.mu, cap_person_eats.mu, cap_person_gives.mu),
        sigma=sigma, id="cap_general_person",
    )
    cap_general_person.add_child(cap_person_walks)
    cap_general_person.add_child(cap_person_eats)
    cap_general_person.add_child(cap_person_gives)
    add(cap_general_person)

    # --- Family capabilities ---
    cap_family_helps = HFN(
        mu=_recombine(obj_family.mu, atomic["helped"].mu),
        sigma=sigma, id="cap_family_helps",
    )
    add(cap_family_helps)

    cap_family = HFN(
        mu=cap_family_helps.mu.copy(),
        sigma=sigma, id="cap_family",
    )
    cap_family.add_child(cap_family_helps)
    add(cap_family)

    # --- Child capabilities ---
    cap_child_plays = HFN(
        mu=_recombine(obj_child.mu, atomic["played"].mu),
        sigma=sigma, id="cap_child_plays",
    )
    add(cap_child_plays)

    cap_child = HFN(
        mu=cap_child_plays.mu.copy(),
        sigma=sigma, id="cap_child",
    )
    cap_child.add_child(cap_child_plays)
    add(cap_child)

    # --- cap_person: parent of general_person, family, child ---
    cap_person = HFN(
        mu=_recombine(cap_general_person.mu, cap_family.mu, cap_child.mu),
        sigma=sigma, id="cap_person",
    )
    cap_person.add_child(cap_general_person)
    cap_person.add_child(cap_family)
    cap_person.add_child(cap_child)
    add(cap_person)

    # --- cap_root: parent of cap_animal, cap_person ---
    cap_root = HFN(
        mu=_recombine(cap_animal.mu, cap_person.mu),
        sigma=sigma, id="cap_root",
    )
    cap_root.add_child(cap_animal)
    cap_root.add_child(cap_person)
    add(cap_root)
```

### Step 3.4 — Add Sentence priors

- [ ] Continue inside `build_nlp_world_model()`:

```python
    # ------------------------------------------------------------------
    # Sub-tree 5: Sentence priors (~20 exemplar sentences)
    # mu = (1/N) * sum(one_hot(w) for w in sentence_words)
    # ------------------------------------------------------------------

    def _sentence_mu(words: list[str]) -> np.ndarray:
        """Equal-weight recombination of one-hot mus for all words in sentence."""
        return _recombine(*[atomic[w].mu if w in atomic else _one_hot(w)
                            for w in words])

    _exemplar_sentences = [
        ("sent_dog_barked_cat",    ["the", "dog", "barked", "at", "the", "cat"]),
        ("sent_cat_chased_bird",   ["the", "cat", "chased", "the", "bird"]),
        ("sent_bird_chirped",      ["a", "small", "bird", "chirped", "loudly"]),
        ("sent_mum_walked_park",   ["mum", "walked", "to", "the", "park"]),
        ("sent_dad_ate_bread",     ["dad", "ate", "the", "bread"]),
        ("sent_teacher_helped",    ["my", "teacher", "helped", "me"]),
        ("sent_boy_played_ball",   ["the", "boy", "played", "with", "the", "ball"]),
        ("sent_we_went_school",    ["we", "went", "to", "school"]),
        ("sent_apple_was_good",    ["the", "big", "apple", "was", "good"]),
        ("sent_book_on_mat",       ["a", "book", "is", "on", "the", "mat"]),
        ("sent_friend_ran_park",   ["my", "friend", "ran", "to", "the", "park"]),
        ("sent_girl_helped_baby",  ["the", "girl", "helped", "the", "baby"]),
        ("sent_grandma_gave_toy",  ["grandma", "gave", "the", "toy", "to", "the", "boy"]),
        ("sent_old_dog_chased",    ["the", "old", "dog", "chased", "the", "ball"]),
        ("sent_cat_at_home",       ["a", "little", "cat", "is", "at", "home"]),
        ("sent_brother_took_milk", ["my", "brother", "took", "the", "milk"]),
        ("sent_doctor_at_school",  ["the", "doctor", "is", "at", "school"]),
        ("sent_bird_at_park",      ["a", "bird", "is", "at", "the", "park"]),
        ("sent_toy_is_here",       ["the", "red", "toy", "is", "here"]),
        ("sent_classmate_ran",     ["my", "classmate", "ran", "to", "school"]),
    ]

    for sent_id, words in _exemplar_sentences:
        # Filter to known vocab words; skip unknown (we treat them as absent)
        known_words = [w for w in words if w in atomic or w in VOCAB_INDEX]
        if not known_words:
            continue
        sent_node = HFN(
            mu=_sentence_mu(known_words),
            sigma=sigma, id=sent_id,
        )
        add(sent_node)

    return forest, prior_ids
```

### Step 3.5 — Verify Task 3 tests pass

- [ ] Run: `PYTHONPATH=. python3 -m pytest tests/hpm_fractal_node/test_nlp_world_model_full.py -v`
- [ ] Also re-run Task 2 tests: `PYTHONPATH=. python3 -m pytest tests/hpm_fractal_node/test_nlp_world_model_objects.py -v`
- [ ] All tests pass

---

## Task 4: experiment_nlp.py — update for D=107

**File:** `hpm_fractal_node/experiments/experiment_nlp.py`

**What changes:**
- Update import: `compose_context_node` instead of (removed) `encode_context_window`
- `D` is now 107 — no other experiment config change needed
- Replace `forest._registry[node_id]` accesses with `forest.get(node_id)` (public API)
- Replace `for k in forest._registry` iteration with `forest.active_nodes()`

### Step 4.1 — Write failing tests first

- [ ] Create file `tests/hpm_fractal_node/test_experiment_nlp_smoke.py` with:

```python
"""Smoke tests for experiment_nlp.py integration with D=107 refactor."""
import numpy as np
import pytest
from hpm_fractal_node.nlp.nlp_loader import D, generate_sentences
from hpm_fractal_node.nlp.nlp_world_model import build_nlp_world_model


def test_d_is_107():
    """D must be 107 in the refactored loader."""
    assert D == 107


def test_observation_vector_shape():
    """Observations returned by generate_sentences must be shape (107,)."""
    data = generate_sentences(seed=42)
    vec, _, _ = data[0]
    assert vec.shape == (107,)
    assert vec.dtype == np.float64


def test_world_model_d_matches_loader():
    """World model forest D must match loader D."""
    forest, _ = build_nlp_world_model()
    assert forest.D == D


def test_observer_can_observe_d107_vector():
    """Observer accepts D=107 input without error."""
    from hfn import Observer, calibrate_tau
    from hfn.tiered_forest import TieredForest
    from pathlib import Path
    import tempfile

    data = generate_sentences(seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        forest, prior_ids = build_nlp_world_model(
            forest_cls=TieredForest,
            cold_dir=Path(tmpdir),
            max_hot=100,
        )
        forest.set_protected(prior_ids)
        tau = calibrate_tau(D, sigma_scale=1.0, margin=5.0)
        obs = Observer(forest, tau=tau, protected_ids=prior_ids)
        # Observe 10 samples without error
        for vec, _, _ in data[:10]:
            x = vec.astype(np.float64)
            result = obs.observe(x)
            forest._on_observe()
        assert True  # no exception raised


def test_no_encode_context_window_in_loader():
    """encode_context_window must not exist in the refactored loader."""
    import hpm_fractal_node.nlp.nlp_loader as loader_module
    assert not hasattr(loader_module, "encode_context_window"), (
        "encode_context_window should be removed in the refactor"
    )


def test_compose_context_node_importable():
    """compose_context_node must be importable from nlp_loader."""
    from hpm_fractal_node.nlp.nlp_loader import compose_context_node
    assert callable(compose_context_node)
```

- [ ] Run tests and confirm they **fail**: `PYTHONPATH=. python3 -m pytest tests/hpm_fractal_node/test_experiment_nlp_smoke.py -v`

### Step 4.2 — Update experiment_nlp.py imports

- [ ] Update the import line in `experiment_nlp.py`:

```python
# BEFORE:
from hpm_fractal_node.nlp.nlp_loader import generate_sentences, category_names, D

# AFTER (no change needed — compose_context_node is used internally by generate_sentences,
# not called directly in the experiment):
from hpm_fractal_node.nlp.nlp_loader import generate_sentences, category_names, D
```

Note: `encode_context_window` is not imported directly in `experiment_nlp.py` — only `D` and `generate_sentences` are used. The import line does not need changing. Verify with:

```bash
grep "encode_context_window" hpm_fractal_node/experiments/experiment_nlp.py
```

### Step 4.3 — Replace forest._registry usages with public API

The experiment uses `forest._registry` in three places. Replace all with public API:

- [ ] **Line ~174** (check node still in forest before analysis):

```python
# BEFORE:
if node_id not in forest._registry:
    continue  # absorbed during a later pass

# AFTER:
if forest.get(node_id) is None:
    continue  # absorbed during a later pass
```

- [ ] **Line ~192** (get node mu for nearest-prior computation):

```python
# BEFORE:
node_mu = forest._registry[node_id].mu

# AFTER:
node_mu = forest.get(node_id).mu
```

- [ ] **Line ~196** (iterate prior nodes for distance computation):

```python
# BEFORE:
for pid in prior_ids:
    d = float(np.linalg.norm(node_mu - forest._registry[pid].mu))

# AFTER:
for pid in prior_ids:
    prior_node = forest.get(pid)
    if prior_node is None:
        continue
    d = float(np.linalg.norm(node_mu - prior_node.mu))
```

- [ ] **Line ~242** (fractal diagnostics — iterate forest):

```python
# BEFORE:
prior_node_list = [forest._registry[k] for k in prior_ids if k in forest._registry]
learned_node_list = [forest._registry[k] for k in forest._registry if k not in prior_ids]

# AFTER:
prior_node_list = [forest.get(k) for k in prior_ids if forest.get(k) is not None]
learned_node_list = [n for n in forest.active_nodes() if n.id not in prior_ids]
```

### Step 4.4 — Verify Task 4 tests pass

- [ ] Run: `PYTHONPATH=. python3 -m pytest tests/hpm_fractal_node/test_experiment_nlp_smoke.py -v`
- [ ] All tests pass

---

## Task 5: Full verification

### Step 5.1 — Run complete test suite

- [ ] Run all tests: `PYTHONPATH=. python3 -m pytest tests/ -v --tb=short`
- [ ] All existing tests continue to pass (no regressions)
- [ ] All new tests pass

### Step 5.2 — Run the full experiment

- [ ] Run 3-pass experiment:

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_nlp.py
```

Expected outcomes:
- Completes 3 passes without OOM (D=107 sigma matrices are 16× smaller than D=428)
- Prints prior node category alignment table
- Prints learned node category alignment (if any learned nodes exist)
- Prints category purity summary

### Step 5.3 — Verify category purity improvement

- [ ] Record category purity mean from the run output
- [ ] Compare against the previous baseline (D=428, averaged priors):
  - Previous issue: `prior_animal` had `purity=0.65, dom=place` — animal/place confusion
  - Expected improvement: animal nodes should have `dom=animal`, place nodes `dom=place`
  - Target: mean category purity > previous baseline

### Step 5.4 — Check for no encode_context_window references

- [ ] Confirm no remaining references to old API:

```bash
grep -r "encode_context_window" hpm_fractal_node/ tests/
```

Expected: no matches.

### Step 5.5 — Commit final

- [ ] Stage files:

```bash
git add hpm_fractal_node/nlp/nlp_loader.py
git add hpm_fractal_node/nlp/nlp_world_model.py
git add hpm_fractal_node/experiments/experiment_nlp.py
git add tests/hpm_fractal_node/test_nlp_loader.py
git add tests/hpm_fractal_node/test_nlp_world_model_objects.py
git add tests/hpm_fractal_node/test_nlp_world_model_full.py
git add tests/hpm_fractal_node/test_experiment_nlp_smoke.py
```

- [ ] Commit:

```bash
git commit -m "Refactor NLP experiment to D=107 HPM-aligned hierarchical world model

- Replace encode_context_window (D=428 concat) with compose_context_node
  (D=107 slot-weighted recombination: 0.2*L2 + 0.35*L1 + 0.35*R1 + 0.10*R2)
- Redesign world model as four sub-trees: 107 atomic word nodes (one-hot
  mus) + Objects hierarchy + Grammar rules + Capabilities + 20 sentence priors
- All composed nodes built via _recombine() equal-weight mean of atomic mus
- Parent-child wired via HFN.add_child() before Forest registration (bottom-up)
- Replace forest._registry accesses with forest.get() / forest.active_nodes()
- Sigma matrices: 107x107 (~91KB) vs 428x428 (~1.47MB) — 16x smaller"
```

---

## Appendix: Key design decisions for implementer reference

### Slot weight rationale
Weights `0.2*left2 + 0.35*left1 + 0.35*right1 + 0.10*right2` encode proximity to the masked position. Left1 and right1 are the nearest neighbours and carry equal weight. Right2 receives less weight than left2 because right2 in many templates is `<end>` (low signal). The asymmetry intentionally down-weights the far-right slot.

### Why atomic one-hot mus
One-hot mus are orthonormal basis vectors in R^107. Their recombination is a weighted histogram over the vocabulary — interpretable, lossless (no information collapse from averaging multiple context windows), and guaranteed to be in the same space as the observation vectors produced by `compose_context_node`.

### Why bottom-up registration
HFN.add_child() sets `node._children` but does not register with Forest. Registering parent before children would create a node that has child references to unregistered nodes, which could confuse Forest traversal. Bottom-up (leaves first) ensures all referenced nodes are already in the Forest when the parent is registered.

### forest._registry vs public API
`forest._registry` is a private dict. The public API is `forest.get(node_id)` (returns `HFN | None`) and `forest.active_nodes()` (returns `list[HFN]`). Both exist in `hfn/forest.py`. The experiment should use public API only — this makes the experiment robust to future Forest implementation changes (e.g. TieredForest may not use `_registry` directly).

### Sentence prior mu formula
`mu = (1/N) * sum(one_hot(w) for w in sentence_words)` is identical to `_recombine(*[atomic[w].mu for w in words])` because `_recombine` uses equal-weight mean. The explicit formula is stated in the spec (Section 5.4) for clarity; the implementation uses `_sentence_mu(words)` which delegates to `_recombine`.
