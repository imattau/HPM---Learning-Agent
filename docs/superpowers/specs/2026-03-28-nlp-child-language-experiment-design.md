# NLP Child Language Experiment — Design Spec

**Date:** 2026-03-28
**Status:** Approved

## Overview

An NLP experiment for the HPM Fractal Node (Observer) framework, modelled on a child's early language acquisition. The Observer is given a masked sentence and must match it against a world model encoding child-scale linguistic knowledge. We measure whether nodes that form correspond to meaningful semantic categories — analogous to the dSprites generative factor alignment experiment.

---

## 1. Representation

Each observation is a masked sentence. We extract a fixed **4-word context window** around the blank (2 left, 2 right):

```
"The ___ barked at the cat"
→ left_2=<start>, left_1="the", right_1="barked", right_2="at"
```

Each of the 4 slots is encoded as a **one-hot vector** over a **fixed, explicitly enumerated vocabulary** of exactly 107 tokens:

```
Special tokens (3):  <start>, <end>, <unknown>
Function words (14): the, a, an, my, her, his, at, in, to, on, not, and, is, was
Animals (3):         dog, cat, bird
People (11):         mum, dad, grandma, grandpa, brother, sister, teacher, doctor, friend, classmate, baby
Children (2):        boy, girl
Food (3):            apple, bread, milk
Objects (3):         ball, book, toy
Places (3):          park, home, school
Descriptors (6):     big, small, little, red, blue, old
Animal actions (5):  barked, meowed, chirped, chased, fetched
Person actions (8):  ate, threw, ran, walked, gave, took, helped, played
Prepositions (4):    after, with, for, from
Other nouns (8):     pot, mat, hat, tree, door, cup, bed, box
Filler words (34):   loudly, quickly, slowly, happily, sadly, again, up, down, away,
                     always, never, then, there, here, too, very, this, that, these,
                     those, some, all, more, other, new, good, nice, long, high, own,
                     our, your, their, its
```

`vocab_size = 107`, so `D = 4 × 107 = 428`.

The vocabulary is **fixed at definition time** in `nlp_loader.py` as a constant list. It never changes based on the dataset. If a word is not in the vocabulary, it maps to `<unknown>`.

The true word filling the gap and its semantic category are stored as labels, used **only for evaluation** — never fed to the Observer.

---

## 2. World Model (Prior Forest)

38 protected HFN nodes encoding child-like linguistic knowledge across three levels (5 relational + 8 category + 25 word).

### Prior `mu` construction

Each prior's `mu` is constructed **analytically** from the fixed vocabulary — independent of the runtime dataset. Specifically:

- For **word priors**: `mu` is the average of the context window vectors for the 5 canonical sentences that word appears in (hard-coded in `nlp_world_model.py`). Example: `prior_dog` is derived from averaging the encodings of `["The dog barked .", "The dog ran .", "A dog chased the cat .", "The big dog sat .", "My dog played ."]`.
- For **category priors**: `mu` is the average of the `mu` vectors of **direct word-prior children only** (not grandchildren). `prior_person`'s mu is computed from `prior_mum`, `prior_dad`, `prior_boy`, `prior_girl`, `prior_teacher`, `prior_doctor`, `prior_friend`, `prior_classmate`, `prior_baby` — not from `prior_grandma` etc., which are children of `prior_family`, itself a child of `prior_person`.
- For **relational priors**: `mu` is the average of the `mu` vectors of all category priors that are direct children of that relational prior.

This ensures priors are fully deterministic and independent of the generated dataset.

### Multi-parent nodes

The HFN Forest does not support multiple structural parents. Nodes listed with two parents (`prior_person`, `prior_family`) are registered once under their **primary structural parent**, with the secondary relationship noted as a semantic annotation only:

- `prior_person` → primary parent: `prior_agent_action`
- `prior_family` → primary parent: `prior_person`
- `prior_social_relation` is a **parallel relational prior** (sibling to `prior_agent_action`), not a structural ancestor of `prior_family`

### Relational priors (top level, 5 nodes)

| ID | Meaning | Example context |
|----|---------|----------------|
| `prior_agent_action` | something doing something | "the ___ ran/barked/ate" |
| `prior_action_target` | something receiving an action | "chased/ate/threw the ___" |
| `prior_thing_place` | something located somewhere | "at/in/to the ___" |
| `prior_descriptor_thing` | something being described | "the big/red/little ___" |
| `prior_social_relation` | person in relation to child | "my/her/his ___" |

### Semantic category priors (mid level, 8 nodes)

| ID | Primary parent | Members |
|----|----------------|---------|
| `prior_animal` | agent_action | dog, cat, bird |
| `prior_person` | agent_action | mum, dad, boy, girl, teacher, doctor, friend, classmate, baby |
| `prior_adult` | person | mum, dad, teacher, doctor |
| `prior_child_person` | person | boy, girl, baby |
| `prior_family` | person | mum, dad, grandma, grandpa, brother, sister |
| `prior_food` | action_target | apple, bread, milk |
| `prior_object` | action_target | ball, book, toy |
| `prior_place` | thing_place | park, home, school |

### Word priors (leaf level, 25 nodes)

Explicit parent mapping (structural parent in Forest):

| Word prior | Structural parent |
|------------|------------------|
| `prior_dog`, `prior_cat`, `prior_bird` | `prior_animal` |
| `prior_mum`, `prior_dad`, `prior_grandma`, `prior_grandpa`, `prior_brother`, `prior_sister` | `prior_family` |
| `prior_teacher`, `prior_doctor`, `prior_friend`, `prior_classmate` | `prior_adult` |
| `prior_boy`, `prior_girl`, `prior_baby` | `prior_child_person` |
| `prior_apple`, `prior_bread`, `prior_milk` | `prior_food` |
| `prior_ball`, `prior_book`, `prior_toy` | `prior_object` |
| `prior_park`, `prior_home`, `prior_school` | `prior_place` |

And the category-level structural parents:

| Category prior | Structural parent |
|----------------|------------------|
| `prior_animal` | `prior_agent_action` |
| `prior_person` | `prior_agent_action` |
| `prior_adult` | `prior_person` |
| `prior_child_person` | `prior_person` |
| `prior_family` | `prior_person` |
| `prior_food` | `prior_action_target` |
| `prior_object` | `prior_action_target` |
| `prior_place` | `prior_thing_place` |

All priors are **protected** — the Observer cannot absorb or remove them.

---

## 3. Dataset

**Synthetic child-directed sentences**, generated programmatically from ~20 templates. Each template has **exactly one masked slot** — the slot labelled `[MASK]`. Other slots are filled from word lists during generation.

Example templates (bracket = fill from list, `[MASK]` = the gap):

```python
"The [MASK] [animal_action] ."              # animal in subject position
"[person] ate the [MASK] ."                 # food in object position
"The [child_word] ran to the [MASK] ."      # place in object position
"[person] threw the [MASK] ."               # object in object position
"The [descriptor] [MASK] [animal_action] ." # animal after descriptor
```

~20 templates × ~100 word fill combinations = **~2000 observations**.

Each observation carries a label: `(true_word, semantic_category)` where `semantic_category` is one of the 8 mid-level categories. Labels are used only for evaluation.

---

## 4. Experiment

### Configuration
- N_SAMPLES = 2000
- N_PASSES = 3
- D = 428
- Tau calibrated with `calibrate_tau(D=428, sigma_scale=1.0, margin=5.0)`

### Tau/sigma note

The observation vectors are sparse one-hot concatenations in `{0,1}^428`, not dense floats. This differs from dSprites. The priors use `sigma = np.eye(D) * 1.0` as in the dSprites world model. The same `calibrate_tau` formula applies, but `sigma_scale` may require empirical tuning if 0% prior coverage is observed. Start with `sigma_scale=1.0, margin=5.0` and adjust if needed.

### Procedure
Same structure as `experiment_dsprites.py`:
1. Build world model → Forest + protected prior IDs
2. Instantiate Observer with full fractal strategy stack
3. Run N_PASSES over shuffled observations
4. Track best-explaining node per observation (max accuracy score)
5. Compute semantic category purity per node

### Measurement
For each learned node with n≥5 observations:
- **Category purity**: fraction of observations matching the dominant semantic category (8 categories)
- **Word purity**: fraction matching the dominant specific word (30 words)
- **Nearest prior**: the prior node with the smallest Euclidean distance to the learned node's `mu` (computed per-node, not via Hausdorff)
- **Hausdorff(learned, priors)**: overall structural distance, computed via `hfn.fractal.hausdorff_distance`

### Success criteria
- Category purity significantly above chance (random baseline = 1/8 = 0.125 for 8 semantic categories)
- Learned nodes cluster near the correct word-level prior (nearest prior matches expected category)
- Sub-clusters emerge within categories (e.g. dog-sound vs dog-movement contexts)

---

## 5. File Structure

```
hpm_fractal_node/
  nlp/
    __init__.py
    nlp_loader.py           — fixed vocab, sentence generator, context window encoder
    nlp_world_model.py      — build_nlp_world_model() → Forest, prior_ids
  experiments/
    experiment_nlp.py       — main experiment
```

---

## 6. HPM Interpretation

This experiment tests the HPM principle that a learner with child-scale priors (coarse, relational, hierarchical) will spontaneously develop finer-grained internal representations through exposure to structured input — without supervision.

Success would demonstrate:
- The Observer carves semantic space at its joints (animal vs food vs person)
- Sub-structure emerges within categories (dog contexts vs cat contexts)
- The prior hierarchy shapes but does not determine the learned structure
