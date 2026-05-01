from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional, Sequence
import numpy as np


@dataclass(frozen=True)
class ParsedSentence:
    text: str
    tokens: Tuple[str, ...]
    pos_tags: Tuple[str, ...]
    lemmas: Tuple[str, ...]
    dependencies: Tuple[Tuple[str, str, str], ...] = ()

    def token_data(self) -> List[Dict[str, str]]:
        return [
            {
                "text": token,
                "pos": pos,
                "lemma": lemma,
            }
            for token, pos, lemma in zip(self.tokens, self.pos_tags, self.lemmas)
        ]

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

    def normalize_lemma(self, word: str) -> str:
        """Return a normalized lexical form for a single word."""
        return word.lower().strip()

    def parse_sentence(self, sentence: str) -> Optional[ParsedSentence]:
        """Return a cached sentence-level parse when the implementation supports it."""
        return None

    def extract_svo(self, sentence: str) -> Dict[str, str]:
        """Return a light subject/verb/object frame for the sentence."""
        return {"subject": "", "predicate": "", "object": "", "voice": "active"}

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
        self._sentence_cache: Dict[str, ParsedSentence] = {}

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

    def normalize_lemma(self, word: str) -> str:
        word = word.lower().strip()
        irregulars = {
            "gave": "give",
            "given": "give",
            "gives": "give",
            "bought": "buy",
            "buys": "buy",
            "broke": "break",
            "broken": "break",
            "receives": "receive",
            "received": "receive",
            "gets": "get",
            "got": "get",
            "having": "have",
            "has": "have",
            "had": "have",
            "owns": "own",
            "owned": "own",
            "possesses": "possess",
            "possessed": "possess",
        }
        if word in irregulars:
            return irregulars[word]
        if word.endswith("ing") and len(word) > 4:
            return word[:-3]
        if word.endswith("ed") and len(word) > 3:
            return word[:-2]
        if word.endswith("s") and len(word) > 3:
            return word[:-1]
        return word

    def parse_sentence(self, sentence: str) -> Optional[ParsedSentence]:
        text = (sentence or "").strip()
        if not text:
            return ParsedSentence(text="", tokens=(), pos_tags=(), lemmas=(), dependencies=())
        cached = self._sentence_cache.get(text)
        if cached is not None:
            return cached
        tokens = tuple(w.lower() for w in text.split() if w.strip())
        pos_tags = tuple(self.get_pos(tok) for tok in tokens)
        lemmas = tuple(self.normalize_lemma(tok) for tok in tokens)
        parsed = ParsedSentence(text=text, tokens=tokens, pos_tags=pos_tags, lemmas=lemmas, dependencies=())
        self._sentence_cache[text] = parsed
        return parsed

    def extract_svo(self, sentence: str) -> Dict[str, str]:
        parsed = self.parse_sentence(sentence)
        if parsed is None or not parsed.tokens:
            return super().extract_svo(sentence)
        subject = ""
        predicate = ""
        obj = ""
        canonical_verbs = {"give", "buy", "receive", "get", "have", "own", "possess", "contain", "hold", "carry", "break"}
        for idx, (tok, pos) in enumerate(zip(parsed.tokens, parsed.pos_tags)):
            if not subject and pos in {"PR", "NN", "NNP"}:
                subject = tok
            if not predicate and (pos == "VB" or parsed.lemmas[idx] in canonical_verbs):
                predicate = parsed.lemmas[idx]
            if pos in {"NN", "NNP", "PR"}:
                obj = tok
        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "voice": "active",
        }

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


class SpacyGrammarLibrary(GrammarValidator):
    """Optional spaCy-backed grammar validator with sentence-level caching."""

    def __init__(self, model: str = "en_core_web_sm", fallback: Optional[GrammarValidator] = None):
        self._fallback = fallback or HeuristicGrammarLibrary()
        self._nlp = None
        self._sentence_cache: Dict[str, ParsedSentence] = {}
        try:
            import spacy  # type: ignore

            self._nlp = spacy.load(model, disable=["ner"])
        except Exception:
            self._nlp = None

    def score_sequence(self, words: List[str]) -> float:
        return self._fallback.score_sequence(words)

    def is_valid_transition(self, prev_word: str, current_word: str) -> bool:
        return self._fallback.is_valid_transition(prev_word, current_word)

    def get_pos(self, word: str) -> str:
        if self._nlp is None:
            return self._fallback.get_pos(word)
        parsed = self.parse_sentence(word)
        if parsed and parsed.pos_tags:
            return parsed.pos_tags[0]
        return self._fallback.get_pos(word)

    def normalize_lemma(self, word: str) -> str:
        if self._nlp is None:
            return self._fallback.normalize_lemma(word)
        parsed = self.parse_sentence(word)
        if parsed and parsed.lemmas:
            return parsed.lemmas[0]
        return self._fallback.normalize_lemma(word)

    def parse_sentence(self, sentence: str) -> Optional[ParsedSentence]:
        text = (sentence or "").strip()
        if not text:
            return ParsedSentence(text="", tokens=(), pos_tags=(), lemmas=(), dependencies=())
        cached = self._sentence_cache.get(text)
        if cached is not None:
            return cached
        if self._nlp is None:
            parsed = self._fallback.parse_sentence(text)
            if parsed is None:
                parsed = ParsedSentence(text=text, tokens=(), pos_tags=(), lemmas=(), dependencies=())
            self._sentence_cache[text] = parsed
            return parsed

        doc = self._nlp(text)
        tokens = tuple(token.text.lower() for token in doc)
        pos_tags = tuple(token.pos_ for token in doc)
        lemmas = tuple((token.lemma_ or token.text).lower() for token in doc)
        dependencies = tuple((token.text.lower(), token.dep_, token.head.text.lower()) for token in doc)
        parsed = ParsedSentence(text=text, tokens=tokens, pos_tags=pos_tags, lemmas=lemmas, dependencies=dependencies)
        self._sentence_cache[text] = parsed
        return parsed

    def extract_svo(self, sentence: str) -> Dict[str, str]:
        parsed = self.parse_sentence(sentence)
        if parsed is None or not parsed.tokens:
            return {"subject": "", "predicate": "", "object": "", "voice": "active"}
        if self._nlp is None:
            return self._fallback.extract_svo(sentence)
        subject = ""
        predicate = ""
        obj = ""
        voice = "active"
        for token, dep, _ in parsed.dependencies:
            if dep in {"nsubj", "nsubjpass"} and not subject:
                subject = token
                if dep == "nsubjpass":
                    voice = "passive"
            elif dep == "ROOT" and not predicate:
                predicate = parsed.lemmas[parsed.tokens.index(token)]
            elif dep in {"dobj", "obj", "attr"} and not obj:
                obj = token
        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "voice": voice,
        }
