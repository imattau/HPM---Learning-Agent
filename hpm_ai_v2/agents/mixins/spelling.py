"""SpellingMixin: adds character-level spelling induction and misspelling detection."""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from hfn.hfn import HFN


class SpellingMixin:
    """
    Mixin for ReaderAgent that adds character-level spelling induction.
    Learns to map words to sequences of character nodes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_spellings: Dict[str, str] = {} # word -> word_macro_id

    def _get_char_node(self, char: str, case_sensitive: bool = True) -> HFN:
        """Get or create an HFN node for a single character."""
        c_str = char if case_sensitive else char.lower()
        char_id = f"CHAR_{c_str}"
        if char.isdigit():
            char_id = f"CHAR_DIGIT_{char}"
        
        # Check if node already in patterns
        if char_id in self.patterns:
            return self.patterns[char_id]
        
        # Otherwise create a primitive node
        mu = np.zeros(self.m_dim)
        if char_id in self.config.concepts:
            idx = self.config.concepts.index(char_id)
            mu[self.config.S_DIM + idx] = 1.0
        
        node = HFN(mu=mu, sigma=np.ones(self.m_dim)*0.01, id=char_id, use_diag=True)
        node.metadata = {"type": "character", "char": c_str}
        node.relation_type = "character"
        self.observer.register(node, protected=True)
        self.patterns[char_id] = node
        return node

    def learn_word_spelling(self, word: str, case_sensitive: bool = True) -> HFN:
        """
        Create a macro node for a word as a sequence of character nodes.
        """
        chars = list(word)
        char_nodes = [self._get_char_node(ch, case_sensitive) for ch in chars]
        
        # mu is average of char nodes for semantic retrieval, but children store order
        mu = np.mean([n.mu for n in char_nodes], axis=0)
        word_id = f"word_spelling_{word}"
        
        word_macro = HFN(mu=mu, sigma=np.ones(self.m_dim)*0.05, id=word_id, use_diag=True)
        word_macro.metadata = {"type": "word_spelling", "word": word, "case_sensitive": case_sensitive}
        word_macro.relation_type = "spelling"
        
        for cn in char_nodes:
            word_macro.add_child(cn)
            
        self.observer.register(word_macro, protected=True)
        self.patterns[word_id] = word_macro
        self.word_spellings[word] = word_id
        return word_macro

    def detect_misspelling(self, word: str, known_words: List[str]) -> Tuple[bool, str, int]:
        """
        Return (is_correct, closest_word, edit_distance).
        """
        closest = None
        min_dist = float('inf')
        
        for kw in known_words:
            dist = self._edit_distance(word, kw)
            if dist < min_dist:
                min_dist = dist
                closest = kw
                
        is_correct = (min_dist == 0)
        return is_correct, closest, min_dist

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]
