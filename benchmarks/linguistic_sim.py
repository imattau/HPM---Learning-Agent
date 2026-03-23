"""Linguistic Simulator for Register Shift Benchmark (SP14).
Mocks semantic word features and register-based transformations.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class Word:
    text: str
    features: np.ndarray # Simulated semantic vector
    
# Root verb semantic features (Mock embeddings)
ROOT_FEATURES = {
    "ask": np.array([1.0, 0.0, 0.0, 0.0, 0.5]),
    "help": np.array([0.0, 1.0, 0.0, 0.0, 0.5]),
    "tell": np.array([0.0, 0.0, 1.0, 0.0, 0.5]),
    "want": np.array([0.0, 0.0, 0.0, 1.0, 0.5]),
}

# Register transformations (Deltas added to root)
FORMAL_DELTA = np.array([0.5, 0.5, 0.5, 0.5, 1.0])
INFORMAL_DELTA = np.array([-0.5, -0.5, -0.5, -0.5, -1.0])

REGISTER_MAP = {
    "formal": {
        "ask": "inquire",
        "help": "assist",
        "tell": "inform",
        "want": "desire",
    },
    "informal": {
        "ask": "hit up",
        "help": "back up",
        "tell": "fill in",
        "want": "be down for",
    }
}

def get_word(text: str, is_root: bool = False, register: str = "formal") -> Word:
    """Mock word vectorizer."""
    if is_root:
        base = ROOT_FEATURES.get(text.lower(), np.zeros(5))
        return Word(text, base)
    
    # Find which root this word belongs to
    root = None
    for r, t in REGISTER_MAP[register].items():
        if t == text.lower():
            root = r
            break
            
    if root:
        base = ROOT_FEATURES[root]
        delta = FORMAL_DELTA if register == "formal" else INFORMAL_DELTA
        return Word(text, base + delta)
    
    return Word(text, np.zeros(5))

def generate_register_tasks(n_train: int = 3, seed: int = 42) -> list[dict]:
    """Generate training (Formal) and test (Informal trap) tasks."""
    rng = np.random.default_rng(seed)
    tasks = []
    
    verbs = list(ROOT_FEATURES.keys())
    
    # 1. Training tasks: All Formal
    for _ in range(n_train):
        verb = rng.choice(verbs)
        reactant = get_word(verb, is_root=True)
        product = get_word(REGISTER_MAP["formal"][verb], register="formal")
        
        # Candidates
        candidates = [product]
        for _ in range(4):
            # Wrong candidate: random register or random verb
            v_wrong = rng.choice(verbs)
            reg_wrong = rng.choice(["formal", "informal"])
            candidates.append(get_word(REGISTER_MAP[reg_wrong][v_wrong], register=reg_wrong))
            
        rng.shuffle(candidates)
        correct_idx = next(i for i, c in enumerate(candidates) if c.text == product.text)
        
        tasks.append({
            "register": "formal",
            "reactant": reactant,
            "product": product,
            "candidates": candidates,
            "correct_idx": correct_idx
        })
        
    # 2. Test task: Informal Trap
    verb = rng.choice(verbs)
    reactant = get_word(verb, is_root=True)
    product = get_word(REGISTER_MAP["informal"][verb], register="informal")
    
    candidates = [product]
    # Lure: The Formal version (The Trap)
    lure = get_word(REGISTER_MAP["formal"][verb], register="formal")
    candidates.append(lure)
    
    for _ in range(3):
        v_wrong = rng.choice(verbs)
        candidates.append(get_word(REGISTER_MAP["formal"][v_wrong], register="formal"))
        
    rng.shuffle(candidates)
    correct_idx = next(i for i, c in enumerate(candidates) if c.text == product.text)
    
    tasks.append({
        "register": "informal",
        "reactant": reactant,
        "product": product,
        "candidates": candidates,
        "correct_idx": correct_idx,
        "is_trap": True
    })
    
    return tasks
