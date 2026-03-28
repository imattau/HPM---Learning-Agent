# NLP Experiment Refactor — Design Spec

**Date:** 2026-03-29
**Status:** Approved

---

## 1. Motivation

The current NLP experiment has two structural problems:

1. **Encoding hack:** `encode_context_window` concatenates one-hot word vectors into a D=428 flat vector. The HFN machinery plays no role in constructing the observation — it is bypassed at the encoding layer.

2. **World model misalignment:** Priors are averages-of-averages of context window encodings. Category priors collapse discriminative information. `prior_animal` explains place observations (purity 0.65, dom=place) because animal and place canonical sentences share overlapping context patterns ("ran", "to", "the").

The refactor removes the encoding hack and redesigns the world model to be genuinely HPM-aligned: lower-level patterns (word HFNs) compose into higher-level patterns (context nodes, sentence priors, grammatical/semantic structure) using the same mechanism at every level.

---

## 2. Core Principle

**Uniform composition.** Everything — context observations, sentence priors, category priors, grammar priors, relationship priors — is built by the same operation: weighted recombination of word HFN mus. Scale (2 words vs 4 words vs 6 words) does not change the mechanism. Equal weights are used unless specified otherwise.

---

## 3. Two Layers of Node

There are two distinct types of HFN node, both at D=107:

**Atomic word nodes** — one per vocabulary word (107 nodes). mu = one-hot vector (1.0 at the word's vocab index, 0.0 elsewhere). These are fixed, used only as building blocks for composition. They are registered in the Forest but are not the primary targets for Observer matching.

**Composed nodes** — all other prior nodes (Objects, Grammar, Relationships, Sentences). mu = weighted recombination of atomic word node mus, or of other composed node mus. These represent patterns at higher levels of abstraction.

> **Important:** The existing `_CANONICAL_SENTENCES` and `_word_mu()` approach is replaced entirely. Leaf object node mus are no longer averages of context window encodings — they are recombinations of the atomic one-hot mus of the words that characterise that node's context.

---

## 4. Encoding: D=107

**Context node (observation):**
For each masked sentence observation (4 context slots), compose the 4 atomic word node mus into a single D=107 context node using slot-weighted recombination:

```python
context_mu = 0.2·mu(left2) + 0.35·mu(left1) + 0.35·mu(right1) + 0.1·mu(right2)
```

Slot weights encode proximity to the masked position (left1/right1 are nearest neighbours, weighted higher). The result is a weighted histogram over the vocabulary. Weights sum to 1.0. When the same word appears in multiple slots, its index accumulates weight — this is intentional (repeated words signal strong context).

This replaces `encode_context_window` entirely. The observation passed to the Observer is `context_mu`.

**Memory benefit:** Sigma matrices drop from 428×428 (~1.47MB) to 107×107 (~91KB) — 16× smaller. 3-pass runs without TieredForest become feasible.

---

## 5. World Model: One Forest, Four Sub-Trees

All nodes live in one Forest at D=107. The Forest tree encodes structural hierarchy — parent-child relationships are registered via HFN node children lists when building the world model. Parent node mus are recombinations of their child node mus (equal weights unless stated).

### 5.1 Objects Sub-Tree

Encodes *what things are* — semantic and conceptual hierarchy, independent of grammar.

```
noun (root)
  ├── animate
  │     ├── animal        → dog, cat, bird
  │     └── person
  │           ├── family  → mum, dad, grandma, grandpa, brother, sister
  │           ├── adult   → teacher, doctor, friend, classmate
  │           └── child   → boy, girl, baby
  └── inanimate
        ├── food          → apple, bread, milk
        ├── object        → ball, book, toy
        └── place         → park, home, school
```

Leaf node mus (dog, cat, park, etc.) = equal-weight recombination of the atomic one-hot mus of words that characterise that entity's typical context (the words that appear AROUND that entity, not the entity word itself). For example, `dog` leaf mu = recombine(barked, meowed, chirped, chased, fetched, the, a, my) — words that appear in dog contexts.

In D=107 space, animal context words ("barked", "meowed") are orthogonal to place context words ("to", "went", "at") — the misalignment bug is structurally eliminated.

### 5.2 Grammar Rules Sub-Tree

Encodes *structural patterns of language* — grammatical roles and phrase structure, independent of semantic content. All leaf node mus are equal-weight recombinations of the atomic one-hot mus of the function/action words in that class.

```
grammar (root)
  ├── word_class
  │     ├── determiner    → the, a, an, my, her, his
  │     ├── preposition   → to, at, in, on, with, for, from
  │     └── descriptor    → big, small, little, red, blue, old
  ├── phrase_structure
  │     ├── noun_phrase   → recombine(determiner mu, animate mu, inanimate mu)
  │     ├── verb_phrase   → recombine(action verb mus, object mu)
  │     └── prep_phrase   → recombine(preposition mu, place mu)
  └── sentence_pattern
        ├── agent_action      → recombine(animate mu, action verb mus)
        ├── action_patient    → recombine(action verb mus, inanimate mu)
        └── motion_to_place   → recombine(walked/ran/went mu, preposition mu, place mu)
```

`phrase_structure` and `sentence_pattern` nodes reference composed node mus (from Objects or word_class), not raw atomic mus. This is intentional — these nodes encode cross-tree patterns.

### 5.3 Relationship Sub-Tree

Encodes *entity-action relationships* — which specific entities perform which specific actions. These are cross-cutting: a relationship node connects an Objects node to a Grammar node.

```
relationships (root)
  ├── dog_barks     → recombine(dog mu, barked mu)
  ├── cat_meows     → recombine(cat mu, meowed mu)
  ├── bird_chirps   → recombine(bird mu, chirped mu)
  ├── animal_chases → recombine(animal mu, chased mu)
  ├── person_walks  → recombine(person mu, walked mu)
  ├── person_eats   → recombine(person mu, ate mu)
  ├── person_gives  → recombine(person mu, gave mu)
  ├── family_helps  → recombine(family mu, helped mu)
  └── child_plays   → recombine(child mu, played mu)
```

These nodes encode that *barking is a capability of dogs specifically* (not just that barking is an animal action). The Observer can match an observation to `dog_barks` rather than just `animal_action` — more specific and more discriminative.

Note: relationship node mus reference composed node mus from Objects (e.g., `dog mu` = the dog leaf node mu from 5.1), not atomic word one-hots directly.

### 5.4 Sentence Priors (~20 nodes)

Concrete exemplar sentences grounding the abstract structure in real usage patterns. Similar-but-not-identical: enough diversity to cover the pattern space without redundancy.

Each sentence prior mu = equal-weight recombination of the atomic one-hot mus of all words in that sentence. For an N-word sentence: `mu = (1/N) * sum(one_hot(w) for w in sentence_words)`.

Example sentences (implementer selects final list, covering diverse grammar + object combinations):
- "The dog barked at the cat" (animal subject, animal-action verb, animal object)
- "The cat chased the bird"
- "A small bird chirped loudly"
- "Mum walked to the park"
- "Dad ate the bread"
- "My teacher helped me"
- "The boy played with the ball"
- "We went to school"
- "The big apple was good"
- "A book is on the mat"
- "My friend ran to the park"
- "The girl helped the baby"
- "Grandma gave the toy to the boy"
- "The old dog chased the ball"
- "A little cat is at home"
- ... (~5 more, implementer's choice, covering food/object/place patterns)

---

## 6. Forest Parent-Child Registration

The existing `Forest` class supports `HFN._children` (list of child HFN nodes). The world model builder registers nodes bottom-up: leaf nodes first, then parent nodes with their children passed at construction time. No changes to `hfn/` are needed — the existing HFN/Forest API supports this.

---

## 7. Changes to Existing Code

| File | Change |
|------|--------|
| `hpm_fractal_node/nlp/nlp_loader.py` | Replace `encode_context_window` (D=428) with `compose_context_node(left2, left1, right1, right2)` (D=107). Update `D = VOCAB_SIZE = 107`. Keep `VOCAB`, `VOCAB_INDEX`, `VOCAB_SIZE`. Remove `_CANONICAL_SENTENCES`. |
| `hpm_fractal_node/nlp/nlp_world_model.py` | Redesign all priors using four sub-tree structure. All nodes in D=107. Register parent-child relationships bottom-up. Remove `_word_mu`, `_CANONICAL_SENTENCES` usage. |
| `hpm_fractal_node/experiments/experiment_nlp.py` | Update for D=107. Replace `encode_context_window` call with `compose_context_node`. |

---

## 8. What Does Not Change

- `hfn/` — HFN, Forest, Observer, TieredForest unchanged
- The experiment's evaluation logic (category purity, word purity, Hausdorff diagnostics)
- The Observer's learning dynamics and configuration
- N_SAMPLES=2000, N_PASSES=3, SEED=42

---

## 9. What We Will Learn From Testing

This design is structurally principled but empirically unproven. The key questions:

1. Does D=107 provide sufficient resolution for the Observer to distinguish semantic categories?
2. Do relationship priors (dog_barks, cat_meows) improve specificity over general animal_action priors?
3. Do grammar rule priors help or add noise to the Observer's explanations?
4. Does the slot-weighted composition (0.2/0.35/0.35/0.1) capture enough structural information, or does it lose too much by not preserving slot identity?
5. Do sentence priors improve coverage and category alignment?

Results will drive the next iteration.
