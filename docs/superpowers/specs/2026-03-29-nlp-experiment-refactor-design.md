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

**Uniform composition.** Everything — context observations, sentence priors, category priors, grammar priors — is built by the same operation: weighted recombination of word HFN mus. Scale (4 words vs 6 words vs 20 words) does not change the mechanism.

---

## 3. Encoding: D=107

**Word HFNs (atomic layer, fixed):**
- One HFN node per vocabulary word (107 nodes)
- mu = one-hot vector in D=107 vocab space
- sigma = identity × 1.0
- These are fixed priors used for composition only — the Observer does not learn over them directly

**Context node (observation):**
For each masked sentence observation (4 context slots), compose the 4 word HFN mus into a single D=107 context node:

```python
context_mu = 0.2·mu(left2) + 0.35·mu(left1) + 0.35·mu(right1) + 0.1·mu(right2)
```

Slot weights encode proximity to the masked position (left1/right1 are nearest neighbours, weighted higher). The result is a soft histogram over the vocabulary — a D=107 vector that represents "what words appear near the blank."

This replaces `encode_context_window` entirely. The observation passed to the Observer is `context_mu`.

**Memory benefit:** Sigma matrices drop from 428×428 (~1.47MB) to 107×107 (~91KB) — 16× smaller. 3-pass runs without TieredForest become feasible.

---

## 4. World Model: One Forest, Three Sub-Trees

All nodes live in one Forest at D=107. All node mus are built by word HFN recombination. The Forest tree encodes the structural hierarchy explicitly — parent-child relationships are registered, not just implied by mu averaging.

### 4.1 Objects Sub-Tree

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

Each leaf node mu = recombine(canonical context words for that word).
Each parent node mu = recombine(child node mus).

Crucially, in D=107 space, animal context words ("barked", "meowed", "chirped") are orthogonal to place context words ("to", "went", "at") — the misalignment bug is structurally impossible.

### 4.2 Grammar Rules Sub-Tree

Encodes *structural patterns of language* — grammatical roles and phrase structure, independent of semantic content.

```
grammar (root)
  ├── word_class
  │     ├── determiner    → the, a, an, my, her, his
  │     ├── preposition   → to, at, in, on, with, for, from
  │     └── descriptor    → big, small, little, red, blue, old
  ├── phrase_structure
  │     ├── noun_phrase   → recombine(determiners + content nouns)
  │     ├── verb_phrase   → recombine(verbs + direct objects)
  │     └── prep_phrase   → recombine(prepositions + places)
  └── sentence_pattern
        ├── agent_action      → recombine(animate words + action verbs)
        ├── action_patient    → recombine(verbs + inanimate targets)
        └── motion_to_place   → recombine(motion verbs + prepositions + places)
```

Grammar rule nodes are subnodes of `grammar`. Word class nodes are subnodes of `word_class`. This makes grammatical structure explicit in the Forest — the Observer can explain an observation via a grammar node ("this context matches the agent_action pattern") independently of which specific words are involved.

### 4.3 Sentence Priors (~20 nodes)

Concrete exemplar sentences that ground the abstract grammar + object knowledge in real usage patterns. Similar-but-not-identical: enough diversity to cover the pattern space without being redundant.

Each sentence prior mu = recombine(all words in that sentence), same composition operation as context nodes.

Example sentences (one per structural pattern):
- "The dog barked at the cat"
- "The cat chased the bird"
- "A small bird chirped loudly"
- "Mum walked to the park"
- "Dad ate the bread"
- "My teacher helped me"
- "The boy played with the ball"
- "We went to school"
- "The big apple was good"
- "A book is on the mat"
- ... (~10 more covering diverse grammar + object combinations)

Sentence priors give the Observer reference patterns that look like language — not just abstract categories or grammatical slots.

---

## 5. Observer Configuration

Unchanged from current experiment. Observer operates in D=107 space with:
- `recombination_strategy="nearest_prior"`
- `hausdorff_absorption_threshold=0.15`
- `persistence_guided_absorption=True`
- `lacunarity_guided_creation=True`
- `multifractal_guided_absorption=True`

TieredForest can be used as before (now with much smaller sigma matrices, making hot cache far more affordable).

---

## 6. Changes to Existing Code

| File | Change |
|------|--------|
| `hpm_fractal_node/nlp/nlp_loader.py` | Replace `encode_context_window` (D=428) with `compose_context_node` (D=107). Add slot weights. Keep VOCAB, VOCAB_INDEX, VOCAB_SIZE. Update D=107. |
| `hpm_fractal_node/nlp/nlp_world_model.py` | Redesign all priors using three sub-tree structure. All nodes in D=107. Register parent-child relationships explicitly. |
| `hpm_fractal_node/experiments/experiment_nlp.py` | Update for D=107. Remove any reference to old `encode_context_window`. |

---

## 7. What Does Not Change

- `hfn/` — HFN, Forest, Observer, TieredForest unchanged
- The experiment's evaluation logic (category purity, word purity, Hausdorff diagnostics)
- The Observer's learning dynamics and configuration
- N_SAMPLES=2000, N_PASSES=3, SEED=42

---

## 8. What We Will Learn From Testing

This design is structurally principled but empirically unproven. The key questions the experiment will answer:

1. Does D=107 provide sufficient resolution for the Observer to distinguish semantic categories?
2. Do grammar rule priors help or add noise to the Observer's explanations?
3. Do sentence priors improve coverage and category alignment compared to the current 38-node world model?
4. Does the slot-weighted composition (0.2/0.35/0.35/0.1) capture enough structural information, or does it lose too much by not preserving slot identity?

Results will drive the next iteration.
