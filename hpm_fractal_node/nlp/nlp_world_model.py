"""
NLP world model — four sub-tree HPM-aligned world model for child language experiment.

Four sub-trees, all nodes at D=107:
  Atomic word nodes (107): one per vocabulary word, mu = one-hot
  Objects sub-tree: semantic/conceptual hierarchy (noun root → animate/inanimate → leaves)
  Grammar sub-tree: grammatical structure (gram_root → word_class/phrase_structure/sentence_pattern)
  Capabilities sub-tree: entity-action associations (cap_root → animal/person capabilities)
  Sentence priors (~20): exemplar sentences, mu = (1/N)*sum(one_hot(w))

All composed node mus are equal-weight recombinations of child mus unless noted.
Parent-child wiring via node.add_child() before forest registration.
"""
from __future__ import annotations

import numpy as np

from hfn import HFN, Forest
from hpm_fractal_node.nlp.nlp_loader import D, VOCAB_INDEX, VOCAB_SIZE, VOCAB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _one_hot(word: str) -> np.ndarray:
    """One-hot vector at D=107 for a vocabulary word."""
    idx = VOCAB_INDEX.get(word.lower(), VOCAB_INDEX["<unknown>"])
    vec = np.zeros(D, dtype=np.float64)
    vec[idx] = 1.0
    return vec


def _recombine(*mus: np.ndarray) -> np.ndarray:
    """Equal-weight recombination of mus. Returns float64, sums to ~1.0."""
    stacked = np.stack(mus, axis=0)
    return np.mean(stacked, axis=0).astype(np.float64)


_SIGMA = np.eye(D, dtype=np.float64)


def _node(node_id: str, mu: np.ndarray) -> HFN:
    return HFN(mu=mu, sigma=_SIGMA.copy(), id=node_id)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_nlp_world_model(forest_cls=None, **tiered_kwargs) -> tuple[Forest, set[str]]:
    """
    Build the NLP world model with four sub-trees.

    Returns
    -------
    forest : Forest
        Contains all prior nodes.
    prior_ids : set[str]
        All node IDs to pass as protected_ids to Observer.
    """
    from hfn.forest import Forest as _Forest
    if forest_cls is None:
        forest_cls = _Forest
    kwargs = tiered_kwargs if forest_cls is not _Forest else {}
    forest = forest_cls(D=D, forest_id="nlp_child_language", **kwargs)
    prior_ids: set[str] = set()

    def add(node: HFN) -> None:
        forest.register(node)
        prior_ids.add(node.id)

    # ==================================================================
    # ATOMIC WORD NODES (107 nodes): word_<token>
    # ==================================================================
    word_nodes: dict[str, HFN] = {}
    for word in VOCAB:
        nid = f"word_{word}"
        n = _node(nid, _one_hot(word))
        word_nodes[word] = n
        add(n)

    def wmu(word: str) -> np.ndarray:
        """Get atomic word node mu."""
        return word_nodes[word].mu

    # ==================================================================
    # OBJECTS SUB-TREE
    # Leaf mus = recombination of context-characterising words
    # (words that appear AROUND that entity in typical sentences)
    # ==================================================================

    # --- Animal leaves ---
    # dog: typically surrounded by barked, fetched, chased, the, a, my, big
    obj_dog = _node("obj_dog", _recombine(
        wmu("barked"), wmu("fetched"), wmu("chased"),
        wmu("the"), wmu("a"), wmu("my"), wmu("big"),
    ))
    # cat: typically surrounded by meowed, chased, the, a, my, small
    obj_cat = _node("obj_cat", _recombine(
        wmu("meowed"), wmu("chased"), wmu("the"), wmu("a"), wmu("my"), wmu("small"),
    ))
    # bird: typically surrounded by chirped, chased, the, a, little
    obj_bird = _node("obj_bird", _recombine(
        wmu("chirped"), wmu("chased"), wmu("the"), wmu("a"), wmu("little"),
    ))
    for n in [obj_dog, obj_cat, obj_bird]:
        add(n)

    # --- Family leaves ---
    obj_mum = _node("obj_mum", _recombine(
        wmu("my"), wmu("walked"), wmu("ate"), wmu("helped"), wmu("gave"),
    ))
    obj_dad = _node("obj_dad", _recombine(
        wmu("my"), wmu("walked"), wmu("ate"), wmu("helped"), wmu("gave"),
    ))
    obj_grandma = _node("obj_grandma", _recombine(
        wmu("my"), wmu("her"), wmu("helped"), wmu("gave"), wmu("walked"),
    ))
    obj_grandpa = _node("obj_grandpa", _recombine(
        wmu("my"), wmu("his"), wmu("helped"), wmu("gave"), wmu("walked"),
    ))
    obj_brother = _node("obj_brother", _recombine(
        wmu("my"), wmu("ran"), wmu("played"), wmu("helped"), wmu("walked"),
    ))
    obj_sister = _node("obj_sister", _recombine(
        wmu("my"), wmu("ran"), wmu("played"), wmu("helped"), wmu("walked"),
    ))
    for n in [obj_mum, obj_dad, obj_grandma, obj_grandpa, obj_brother, obj_sister]:
        add(n)

    # --- Adult leaves ---
    obj_teacher = _node("obj_teacher", _recombine(
        wmu("the"), wmu("my"), wmu("helped"), wmu("walked"), wmu("gave"),
    ))
    obj_doctor = _node("obj_doctor", _recombine(
        wmu("the"), wmu("helped"), wmu("walked"), wmu("gave"), wmu("took"),
    ))
    obj_friend = _node("obj_friend", _recombine(
        wmu("my"), wmu("played"), wmu("ran"), wmu("helped"), wmu("walked"),
    ))
    obj_classmate = _node("obj_classmate", _recombine(
        wmu("my"), wmu("played"), wmu("ran"), wmu("helped"), wmu("walked"),
    ))
    for n in [obj_teacher, obj_doctor, obj_friend, obj_classmate]:
        add(n)

    # --- Child leaves ---
    obj_boy = _node("obj_boy", _recombine(
        wmu("the"), wmu("ran"), wmu("played"), wmu("walked"), wmu("helped"),
    ))
    obj_girl = _node("obj_girl", _recombine(
        wmu("the"), wmu("ran"), wmu("played"), wmu("walked"), wmu("helped"),
    ))
    obj_baby = _node("obj_baby", _recombine(
        wmu("the"), wmu("played"), wmu("walked"), wmu("helped"), wmu("my"),
    ))
    for n in [obj_boy, obj_girl, obj_baby]:
        add(n)

    # --- Food leaves ---
    obj_apple = _node("obj_apple", _recombine(
        wmu("the"), wmu("an"), wmu("is"), wmu("was"), wmu("good"), wmu("red"),
    ))
    obj_bread = _node("obj_bread", _recombine(
        wmu("the"), wmu("is"), wmu("was"), wmu("good"), wmu("old"),
    ))
    obj_milk = _node("obj_milk", _recombine(
        wmu("the"), wmu("is"), wmu("was"), wmu("good"),
    ))
    for n in [obj_apple, obj_bread, obj_milk]:
        add(n)

    # --- Object leaves ---
    obj_ball = _node("obj_ball", _recombine(
        wmu("a"), wmu("the"), wmu("is"), wmu("big"), wmu("red"), wmu("here"),
    ))
    obj_book = _node("obj_book", _recombine(
        wmu("a"), wmu("the"), wmu("is"), wmu("big"), wmu("old"), wmu("here"),
    ))
    obj_toy = _node("obj_toy", _recombine(
        wmu("a"), wmu("the"), wmu("is"), wmu("little"), wmu("old"), wmu("here"),
    ))
    for n in [obj_ball, obj_book, obj_toy]:
        add(n)

    # --- Place leaves ---
    obj_park = _node("obj_park", _recombine(
        wmu("to"), wmu("at"), wmu("the"), wmu("ran"), wmu("walked"),
    ))
    obj_home = _node("obj_home", _recombine(
        wmu("to"), wmu("at"), wmu("the"), wmu("ran"), wmu("walked"),
    ))
    obj_school = _node("obj_school", _recombine(
        wmu("to"), wmu("at"), wmu("the"), wmu("ran"), wmu("walked"),
    ))
    for n in [obj_park, obj_home, obj_school]:
        add(n)

    # --- Objects parents (bottom-up) ---
    obj_animal = _node("obj_animal", _recombine(obj_dog.mu, obj_cat.mu, obj_bird.mu))
    obj_animal.add_child(obj_dog)
    obj_animal.add_child(obj_cat)
    obj_animal.add_child(obj_bird)
    add(obj_animal)

    obj_family = _node("obj_family", _recombine(
        obj_mum.mu, obj_dad.mu, obj_grandma.mu, obj_grandpa.mu,
        obj_brother.mu, obj_sister.mu,
    ))
    obj_family.add_child(obj_mum)
    obj_family.add_child(obj_dad)
    obj_family.add_child(obj_grandma)
    obj_family.add_child(obj_grandpa)
    obj_family.add_child(obj_brother)
    obj_family.add_child(obj_sister)
    add(obj_family)

    obj_adult = _node("obj_adult", _recombine(
        obj_teacher.mu, obj_doctor.mu, obj_friend.mu, obj_classmate.mu,
    ))
    obj_adult.add_child(obj_teacher)
    obj_adult.add_child(obj_doctor)
    obj_adult.add_child(obj_friend)
    obj_adult.add_child(obj_classmate)
    add(obj_adult)

    obj_child = _node("obj_child", _recombine(obj_boy.mu, obj_girl.mu, obj_baby.mu))
    obj_child.add_child(obj_boy)
    obj_child.add_child(obj_girl)
    obj_child.add_child(obj_baby)
    add(obj_child)

    obj_person = _node("obj_person", _recombine(
        obj_family.mu, obj_adult.mu, obj_child.mu,
    ))
    obj_person.add_child(obj_family)
    obj_person.add_child(obj_adult)
    obj_person.add_child(obj_child)
    add(obj_person)

    obj_animate = _node("obj_animate", _recombine(obj_animal.mu, obj_person.mu))
    obj_animate.add_child(obj_animal)
    obj_animate.add_child(obj_person)
    add(obj_animate)

    obj_food = _node("obj_food", _recombine(obj_apple.mu, obj_bread.mu, obj_milk.mu))
    obj_food.add_child(obj_apple)
    obj_food.add_child(obj_bread)
    obj_food.add_child(obj_milk)
    add(obj_food)

    obj_object = _node("obj_object", _recombine(obj_ball.mu, obj_book.mu, obj_toy.mu))
    obj_object.add_child(obj_ball)
    obj_object.add_child(obj_book)
    obj_object.add_child(obj_toy)
    add(obj_object)

    obj_place = _node("obj_place", _recombine(obj_park.mu, obj_home.mu, obj_school.mu))
    obj_place.add_child(obj_park)
    obj_place.add_child(obj_home)
    obj_place.add_child(obj_school)
    add(obj_place)

    obj_inanimate = _node("obj_inanimate", _recombine(
        obj_food.mu, obj_object.mu, obj_place.mu,
    ))
    obj_inanimate.add_child(obj_food)
    obj_inanimate.add_child(obj_object)
    obj_inanimate.add_child(obj_place)
    add(obj_inanimate)

    obj_noun = _node("obj_noun", _recombine(obj_animate.mu, obj_inanimate.mu))
    obj_noun.add_child(obj_animate)
    obj_noun.add_child(obj_inanimate)
    add(obj_noun)

    # ==================================================================
    # GRAMMAR SUB-TREE
    # ==================================================================

    # --- Word class leaves ---
    gram_determiner = _node("gram_determiner", _recombine(
        wmu("the"), wmu("a"), wmu("an"), wmu("my"), wmu("her"), wmu("his"),
    ))
    gram_preposition = _node("gram_preposition", _recombine(
        wmu("to"), wmu("at"), wmu("in"), wmu("on"), wmu("with"),
        wmu("for"), wmu("from"), wmu("after"),
    ))
    gram_descriptor = _node("gram_descriptor", _recombine(
        wmu("big"), wmu("small"), wmu("little"), wmu("red"), wmu("blue"), wmu("old"),
    ))
    for n in [gram_determiner, gram_preposition, gram_descriptor]:
        add(n)

    gram_word_class = _node("gram_word_class", _recombine(
        gram_determiner.mu, gram_preposition.mu, gram_descriptor.mu,
    ))
    gram_word_class.add_child(gram_determiner)
    gram_word_class.add_child(gram_preposition)
    gram_word_class.add_child(gram_descriptor)
    add(gram_word_class)

    # --- Phrase structure nodes ---
    gram_noun_phrase = _node("gram_noun_phrase", _recombine(
        gram_determiner.mu, obj_animate.mu, obj_inanimate.mu,
    ))
    gram_verb_phrase = _node("gram_verb_phrase", _recombine(
        wmu("barked"), wmu("ran"), wmu("walked"), wmu("ate"), wmu("played"),
        wmu("helped"), wmu("gave"), wmu("took"),
    ))
    gram_prep_phrase = _node("gram_prep_phrase", _recombine(
        gram_preposition.mu, obj_place.mu,
    ))
    for n in [gram_noun_phrase, gram_verb_phrase, gram_prep_phrase]:
        add(n)

    gram_phrase_structure = _node("gram_phrase_structure", _recombine(
        gram_noun_phrase.mu, gram_verb_phrase.mu, gram_prep_phrase.mu,
    ))
    gram_phrase_structure.add_child(gram_noun_phrase)
    gram_phrase_structure.add_child(gram_verb_phrase)
    gram_phrase_structure.add_child(gram_prep_phrase)
    add(gram_phrase_structure)

    # --- Sentence pattern nodes ---
    gram_agent_action = _node("gram_agent_action", _recombine(
        obj_animate.mu, gram_verb_phrase.mu,
    ))
    gram_action_patient = _node("gram_action_patient", _recombine(
        gram_verb_phrase.mu, obj_inanimate.mu,
    ))
    gram_motion_to_place = _node("gram_motion_to_place", _recombine(
        wmu("walked"), wmu("ran"), gram_preposition.mu, obj_place.mu,
    ))
    for n in [gram_agent_action, gram_action_patient, gram_motion_to_place]:
        add(n)

    gram_sentence_pattern = _node("gram_sentence_pattern", _recombine(
        gram_agent_action.mu, gram_action_patient.mu, gram_motion_to_place.mu,
    ))
    gram_sentence_pattern.add_child(gram_agent_action)
    gram_sentence_pattern.add_child(gram_action_patient)
    gram_sentence_pattern.add_child(gram_motion_to_place)
    add(gram_sentence_pattern)

    gram_root = _node("gram_root", _recombine(
        gram_word_class.mu, gram_phrase_structure.mu, gram_sentence_pattern.mu,
    ))
    gram_root.add_child(gram_word_class)
    gram_root.add_child(gram_phrase_structure)
    gram_root.add_child(gram_sentence_pattern)
    add(gram_root)

    # ==================================================================
    # CAPABILITIES SUB-TREE
    # Leaf: recombine(entity_obj_mu, action_atomic_mu)
    # ==================================================================

    # --- Animal capabilities ---
    cap_dog_barks = _node("cap_dog_barks", _recombine(obj_dog.mu, wmu("barked")))
    cap_dog_fetches = _node("cap_dog_fetches", _recombine(obj_dog.mu, wmu("fetched")))
    cap_cat_meows = _node("cap_cat_meows", _recombine(obj_cat.mu, wmu("meowed")))
    cap_cat_chases = _node("cap_cat_chases", _recombine(obj_cat.mu, wmu("chased")))
    cap_bird_chirps = _node("cap_bird_chirps", _recombine(obj_bird.mu, wmu("chirped")))
    for n in [cap_dog_barks, cap_dog_fetches, cap_cat_meows, cap_cat_chases, cap_bird_chirps]:
        add(n)

    cap_dog = _node("cap_dog", _recombine(cap_dog_barks.mu, cap_dog_fetches.mu))
    cap_dog.add_child(cap_dog_barks)
    cap_dog.add_child(cap_dog_fetches)
    add(cap_dog)

    cap_cat = _node("cap_cat", _recombine(cap_cat_meows.mu, cap_cat_chases.mu))
    cap_cat.add_child(cap_cat_meows)
    cap_cat.add_child(cap_cat_chases)
    add(cap_cat)

    cap_bird = _node("cap_bird", _recombine(cap_bird_chirps.mu,))
    cap_bird.add_child(cap_bird_chirps)
    add(cap_bird)

    cap_animal = _node("cap_animal", _recombine(cap_dog.mu, cap_cat.mu, cap_bird.mu))
    cap_animal.add_child(cap_dog)
    cap_animal.add_child(cap_cat)
    cap_animal.add_child(cap_bird)
    add(cap_animal)

    # --- Person capabilities ---
    cap_person_walks = _node("cap_person_walks", _recombine(obj_person.mu, wmu("walked")))
    cap_person_eats = _node("cap_person_eats", _recombine(obj_person.mu, wmu("ate")))
    cap_person_gives = _node("cap_person_gives", _recombine(obj_person.mu, wmu("gave")))
    for n in [cap_person_walks, cap_person_eats, cap_person_gives]:
        add(n)

    cap_general_person = _node("cap_general_person", _recombine(
        cap_person_walks.mu, cap_person_eats.mu, cap_person_gives.mu,
    ))
    cap_general_person.add_child(cap_person_walks)
    cap_general_person.add_child(cap_person_eats)
    cap_general_person.add_child(cap_person_gives)
    add(cap_general_person)

    cap_family_helps = _node("cap_family_helps", _recombine(obj_family.mu, wmu("helped")))
    add(cap_family_helps)

    cap_family = _node("cap_family", _recombine(cap_family_helps.mu,))
    cap_family.add_child(cap_family_helps)
    add(cap_family)

    cap_child_plays = _node("cap_child_plays", _recombine(obj_child.mu, wmu("played")))
    add(cap_child_plays)

    cap_child = _node("cap_child", _recombine(cap_child_plays.mu,))
    cap_child.add_child(cap_child_plays)
    add(cap_child)

    cap_person = _node("cap_person", _recombine(
        cap_general_person.mu, cap_family.mu, cap_child.mu,
    ))
    cap_person.add_child(cap_general_person)
    cap_person.add_child(cap_family)
    cap_person.add_child(cap_child)
    add(cap_person)

    cap_root = _node("cap_root", _recombine(cap_animal.mu, cap_person.mu))
    cap_root.add_child(cap_animal)
    cap_root.add_child(cap_person)
    add(cap_root)

    # ==================================================================
    # SENTENCE PRIORS (~20 nodes)
    # mu = (1/N) * sum(one_hot(w) for w in sentence_words)
    # ==================================================================

    def sent_mu(*words: str) -> np.ndarray:
        """Equal-weight average of atomic one-hot mus for sentence words."""
        vecs = [wmu(w) for w in words]
        return np.mean(np.stack(vecs), axis=0).astype(np.float64)

    sentence_defs = [
        ("sent_dog_barked_at_cat",    ("the", "dog", "barked", "at", "the", "cat")),
        ("sent_cat_chased_bird",      ("the", "cat", "chased", "the", "bird")),
        ("sent_bird_chirped_loudly",  ("a", "small", "bird", "chirped", "loudly")),
        ("sent_mum_walked_to_park",   ("mum", "walked", "to", "the", "park")),
        ("sent_dad_ate_bread",        ("dad", "ate", "the", "bread")),
        ("sent_teacher_helped",       ("my", "teacher", "helped", "the")),
        ("sent_boy_played_ball",      ("the", "boy", "played", "with", "the", "ball")),
        ("sent_mum_walked_school",     ("mum", "walked", "to", "school")),
        ("sent_apple_was_good",       ("the", "big", "apple", "was", "good")),
        ("sent_book_on_mat",          ("a", "book", "is", "on", "the", "mat")),
        ("sent_friend_ran_park",      ("my", "friend", "ran", "to", "the", "park")),
        ("sent_girl_helped_baby",     ("the", "girl", "helped", "the", "baby")),
        ("sent_grandma_gave_toy",     ("grandma", "gave", "the", "toy", "to", "the", "boy")),
        ("sent_old_dog_chased_ball",  ("the", "old", "dog", "chased", "the", "ball")),
        ("sent_little_cat_at_home",   ("a", "little", "cat", "is", "at", "home")),
        ("sent_milk_is_good",         ("the", "milk", "is", "good")),
        ("sent_ball_is_here",         ("the", "red", "ball", "is", "here")),
        ("sent_doctor_walked",        ("the", "doctor", "walked", "to", "school")),
        ("sent_sister_played",        ("my", "sister", "played", "with", "the", "toy")),
        ("sent_brother_ran_school",   ("my", "brother", "ran", "to", "school")),
    ]

    for nid, words in sentence_defs:
        n = _node(nid, sent_mu(*words))
        add(n)

    return forest, prior_ids
