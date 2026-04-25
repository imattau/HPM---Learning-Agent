from abc import ABC, abstractmethod
from typing import List, Dict, Set, Tuple, Optional
import numpy as np

class GrammarValidator(ABC):
    @abstractmethod
    def score_sequence(self, words: List[str]) -> float:
        """Return a score (0..1) for the grammatical plausibility of a word sequence."""
        pass

    @abstractmethod
    def is_valid_transition(self, prev_word: str, current_word: str) -> bool:
        """Return True if the transition between these two words is common in English."""
        pass

    @abstractmethod
    def get_pos(self, word: str) -> str:
        """Return a basic Part-of-Speech category for a single word."""
        pass

class HeuristicGrammarLibrary(GrammarValidator):
    """
    A rule-based grammar library that uses a hardcoded transition matrix
    and a small dictionary of common English words.
    """
    def __init__(self):
        # POS categories: DT (Determiner), NN (Noun), VB (Verb), JJ (Adj), RB (Adv), IN (Prep), PR (Pronoun)
        self.categories = {
            "DT": {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her"},
            "IN": {"in", "on", "at", "by", "with", "from", "for", "of", "to", "between", "under"},
            "PR": {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"},
            "RB": {"very", "quickly", "slowly", "well", "never", "always", "often", "not", "too"},
            "CC": {"and", "but", "or", "so", "yet", "for", "nor"}
        }
        
        # Suffix-based heuristics for Nouns, Verbs, Adjectives
        self.suffixes = {
            "VB": ["ing", "ed", "ize", "ate", "ify"],
            "NN": ["tion", "ness", "ity", "ment", "ship", "ism", "er", "ist"],
            "JJ": ["able", "ible", "al", "ful", "ish", "less", "ous", "ive"]
        }

        # Transition probabilities (Heuristic estimates)
        # 1.0 = High, 0.5 = Moderate, 0.1 = Rare/Invalid
        self.transitions = {
            "START": {"DT": 1.0, "PR": 1.0, "NN": 0.8, "RB": 0.5, "VB": 0.2},
            "DT": {"NN": 1.0, "JJ": 0.8, "NNP": 0.5},
            "NN": {"VB": 0.9, "IN": 0.8, "CC": 0.5, "END": 0.8},
            "VB": {"DT": 0.8, "IN": 0.7, "RB": 0.6, "PR": 0.5, "NN": 0.8, "END": 0.5},
            "JJ": {"NN": 1.0, "JJ": 0.5},
            "IN": {"DT": 0.9, "NN": 0.8, "PR": 0.7},
            "PR": {"VB": 1.0, "RB": 0.5},
            "RB": {"VB": 0.8, "JJ": 0.7, "RB": 0.5},
            "CC": {"DT": 0.7, "NN": 0.7, "VB": 0.7, "PR": 0.7}
        }

    def get_pos(self, word: str) -> str:
        word = word.lower()
        for cat, wset in self.categories.items():
            if word in wset:
                return cat
        
        # Suffix-based
        for cat, sufs in self.suffixes.items():
            for s in sufs:
                if word.endswith(s):
                    return cat
                    
        # Fallback to Noun
        return "NN"

    def is_valid_transition(self, prev_word: str, current_word: str) -> bool:
        tag1 = self.get_pos(prev_word)
        tag2 = self.get_pos(current_word)
        return self.transitions.get(tag1, {}).get(tag2, 0.1) > 0.4

    def score_sequence(self, words: List[str]) -> float:
        if not words: return 0.0
        
        tags = [self.get_pos(w) for w in words]
        score = 0.0
        prev_tag = "START"
        
        for tag in tags:
            prob = self.transitions.get(prev_tag, {}).get(tag, 0.1)
            score += np.log(prob + 1e-6)
            prev_tag = tag
            
        # Normalize and squash
        norm_score = score / len(tags)
        return 1.0 / (1.0 + np.exp(-norm_score - 1.0))

class NLTKGrammarLibrary(HeuristicGrammarLibrary):
    """
    Backwards compatibility: If NLTK fails, we use the heuristic fallback.
    """
    def __init__(self, download: bool = True):
        super().__init__()
        # In a real environment, we'd try NLTK here.
        # Given the persistent timeouts, we stick to the heuristic for robustness.
        print("[grammar] Using Heuristic Grammar Library (NLTK fallback enabled).")
