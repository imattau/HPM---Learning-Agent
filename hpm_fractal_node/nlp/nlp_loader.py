"""
NLP loader for the child language experiment.

Fixed vocabulary of 107 tokens, D=428 (4 context slots × 107).
Generates ≥2000 synthetic child-directed sentences with masked slots.
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
