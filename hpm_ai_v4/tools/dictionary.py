from abc import ABC, abstractmethod
from typing import List, Set, Optional
import nltk

class DictionaryValidator(ABC):
    @abstractmethod
    def contains(self, word: str) -> bool:
        """Return True if the exact word is in the dictionary."""
        pass

    @abstractmethod
    def is_prefix(self, prefix: str) -> bool:
        """Return True if any word in the dictionary starts with `prefix`."""
        pass

    @abstractmethod
    def completions(self, prefix: str, max_suggestions: int = 5) -> List[str]:
        """Return up to `max_suggestions` words that start with `prefix`."""
        pass

    @abstractmethod
    def score_word(self, word: str) -> float:
        """Return a confidence score (0..1) for the word being valid. Default 1.0 for exact match."""
        pass

class NLTKWordList(DictionaryValidator):
    """
    Dictionary implementation using NLTK's word corpus (approx 235k words).
    Uses a trie for fast prefix checking.
    """
    def __init__(self, download: bool = True):
        if download:
            try:
                nltk.data.find('corpora/words')
            except LookupError:
                print("[nltk] Downloading 'words' corpus...")
                nltk.download('words', quiet=True)
        
        from nltk.corpus import words
        print("[nltk] Loading word list into trie...")
        all_words = words.words()
        
        self.words = set(w.lower() for w in all_words)
        
        # Build a trie for fast prefix operations
        self._trie = {}
        for w in self.words:
            node = self._trie
            for ch in w:
                node = node.setdefault(ch, {})
            node['$'] = True   # end marker
        print(f"[nltk] Trie complete. Indexed {len(self.words)} words.")

    def contains(self, word: str) -> bool:
        return word.lower() in self.words

    def is_prefix(self, prefix: str) -> bool:
        if not prefix: return True
        node = self._trie
        for ch in prefix.lower():
            if ch not in node:
                return False
            node = node[ch]
        return True

    def completions(self, prefix: str, max_suggestions: int = 5) -> List[str]:
        if not prefix: return []
        node = self._trie
        for ch in prefix.lower():
            if ch not in node:
                return []
            node = node[ch]
            
        results = []
        def _dfs(cur_node, cur_word):
            if len(results) >= max_suggestions:
                return
            if '$' in cur_node:
                results.append(prefix.lower() + cur_word)
            for ch, nxt in cur_node.items():
                if ch != '$':
                    _dfs(nxt, cur_word + ch)
                    
        _dfs(node, "")
        return results

    def score_word(self, word: str) -> float:
        return 1.0 if self.contains(word) else 0.0
