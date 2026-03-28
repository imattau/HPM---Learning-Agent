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

Each of the 4 slots is encoded as a **one-hot vector** over the vocabulary (~110 tokens including special tokens `<start>`, `<end>`, `<unknown>`).

The 4 one-hot vectors are concatenated:

```
D = 4 × vocab_size ≈ 4 × 110 = 440
```

The true word filling the gap and its semantic category are stored as labels, used **only for evaluation** — never fed to the Observer.

---

## 2. World Model (Prior Forest)

~43 protected HFN nodes encoding child-like linguistic knowledge across three levels. Each prior's `mu` is constructed by averaging context windows of sentences where that word/category fills the gap.

### Relational priors (top level, 5 nodes)

| ID | Meaning | Example context |
|----|---------|----------------|
| `prior_agent_action` | something doing something | "the ___ ran/barked/ate" |
| `prior_action_target` | something receiving an action | "chased/ate/threw the ___" |
| `prior_thing_place` | something located somewhere | "at/in/to the ___" |
| `prior_descriptor_thing` | something being described | "the big/red/little ___" |
| `prior_social_relation` | person in relation to child | "my/her/his ___" |

### Semantic category priors (mid level, ~8 nodes)

| ID | Parent | Members |
|----|--------|---------|
| `prior_animal` | agent_action | dog, cat, bird |
| `prior_person` | agent_action + social_relation | mum, dad, boy, girl, teacher... |
| `prior_adult` | person | mum, dad, teacher, doctor |
| `prior_child_person` | person | boy, girl, baby |
| `prior_family` | person + social_relation | mum, dad, grandma, grandpa, brother, sister |
| `prior_food` | action_target | apple, bread, milk |
| `prior_object` | action_target | ball, book, toy |
| `prior_place` | thing_place | park, home, school |

### Word priors (leaf level, ~30 nodes)

Animals: `prior_dog`, `prior_cat`, `prior_bird`
Family: `prior_mum`, `prior_dad`, `prior_grandma`, `prior_grandpa`, `prior_brother`, `prior_sister`
Authority/peers: `prior_teacher`, `prior_doctor`, `prior_friend`, `prior_classmate`
Children: `prior_boy`, `prior_girl`, `prior_baby`
Food: `prior_apple`, `prior_bread`, `prior_milk`
Objects: `prior_ball`, `prior_book`, `prior_toy`
Places: `prior_park`, `prior_home`, `prior_school`

All priors are **protected** — the Observer cannot absorb or remove them.

---

## 3. Dataset

**Synthetic child-directed sentences**, generated programmatically from ~20 templates. Each template has one masked slot.

Example templates:
```
"The [animal] [animal_action] ."          → "The ___ barked ."
"[person] ate the [food] ."               → "Mum ate the ___ ."
"The [child] ran to the [place] ."        → "The boy ran to the ___ ."
"[person] threw the [object] ."           → "Dad threw the ___ ."
"The [descriptor] [animal] [action] ."    → "The big ___ ran ."
```

~20 templates × ~100 word combinations = **~2000 observations**.

Each observation carries a label: `(true_word, semantic_category)`. Labels are used only for evaluation.

---

## 4. Experiment

### Configuration
- N_SAMPLES = 2000
- N_PASSES = 3
- Tau calibrated for D≈440 with sigma_scale=1.0

### Procedure
Same structure as `experiment_dsprites.py`:
1. Build world model → Forest + protected prior IDs
2. Instantiate Observer with full fractal strategy stack
3. Run N_PASSES over shuffled observations
4. Track best-explaining node per observation (max accuracy score)
5. Compute semantic category purity per node

### Measurement
For each learned node with n≥5 observations:
- **Category purity**: fraction of observations matching the dominant semantic category
- **Word purity**: fraction matching the dominant specific word
- **Proximity to prior**: which prior node is geometrically nearest

### Success criteria
- Category purity significantly above chance (random baseline = 1/5 = 0.20 for 5 categories)
- Learned nodes cluster near the correct word-level prior
- Sub-clusters emerge within categories (e.g. dog-sound vs dog-movement contexts)

---

## 5. File Structure

```
hpm_fractal_node/
  nlp/
    __init__.py
    nlp_loader.py           — sentence generator + context window encoder
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
