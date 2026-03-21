import itertools
import warnings
from typing import Iterator

import numpy as np
import requests

from .base import hash_vectorise

_POS_TAGS = [
    'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM',
    'PUNCT', 'SYM', 'X', 'PART', 'INTJ', 'PROPN', 'AUX', 'CCONJ', 'SCONJ',
]  # 17 Universal Dependencies v2 POS tags


class LinguisticSubstrate:
    """
    ExternalSubstrate backed by NLTK/WordNet (required), Free Dictionary API
    (optional), and spaCy (optional).

    Core dependency: NLTK with wordnet + words corpora.
    Optional: requests (API), spacy with en_core_web_sm (POS vectors).

    All three components run independently; results are combined.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        timeout: float = 5.0,
        use_api: bool = True,
        use_spacy: bool = True,
    ):
        try:
            import nltk
            nltk.data.find('corpora/wordnet')
            nltk.data.find('corpora/words')
        except ImportError:
            raise ImportError(
                "nltk is required for LinguisticSubstrate. Install with: pip install nltk"
            )
        except LookupError:
            raise LookupError(
                "NLTK corpora not found. Run: "
                "nltk.download('wordnet'); nltk.download('words')"
            )

        from nltk.corpus import wordnet, words as nltk_words
        self._wordnet = wordnet
        self._word_list = [
            w for w in nltk_words.words()
            if w.isalpha() and 4 <= len(w) <= 12
        ]

        self.feature_dim = feature_dim
        self.timeout = timeout
        self._use_api = use_api
        self._cache: dict[str, list[np.ndarray]] = {}

        self._nlp = None
        if use_spacy:
            try:
                import spacy
                self._nlp = spacy.load('en_core_web_sm')
            except (ImportError, OSError):
                warnings.warn(
                    "spaCy or en_core_web_sm model not available; POS vectors disabled. "
                    "Install with: pip install spacy && python -m spacy download en_core_web_sm",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def fetch(self, query: str) -> list[np.ndarray]:
        if query in self._cache:
            return self._cache[query]

        results = []

        # WordNet component (always active)
        synsets = self._wordnet.synsets(query)
        for syn in synsets:
            if syn.definition():
                results.append(hash_vectorise(syn.definition(), self.feature_dim))
            for ex in syn.examples():
                if ex:
                    results.append(hash_vectorise(ex, self.feature_dim))

        # Free Dictionary API component (optional)
        if self._use_api:
            try:
                resp = requests.get(
                    f'https://api.dictionaryapi.dev/api/v2/entries/en/{query}',
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    for entry in resp.json():
                        for meaning in entry.get('meanings', []):
                            for defn in meaning.get('definitions', []):
                                if defn.get('definition'):
                                    results.append(
                                        hash_vectorise(defn['definition'], self.feature_dim)
                                    )
                                if defn.get('example'):
                                    results.append(
                                        hash_vectorise(defn['example'], self.feature_dim)
                                    )
            except Exception:
                pass

        # spaCy POS-tag frequency vector (optional)
        if self._nlp is not None:
            doc = self._nlp(query)
            total = len(doc)
            if total > 0:
                counts = {tag: 0 for tag in _POS_TAGS}
                for token in doc:
                    if token.pos_ in counts:
                        counts[token.pos_] += 1
                freq = np.array([counts[tag] / total for tag in _POS_TAGS])
                if len(freq) < self.feature_dim:
                    freq = np.pad(freq, (0, self.feature_dim - len(freq)))
                else:
                    freq = freq[:self.feature_dim]
                results.append(freq)

        self._cache[query] = results
        return results

    def field_frequency(self, pattern) -> float:
        query = str(getattr(pattern, 'label', None) or 'word')
        vecs = self.fetch(query)
        if not vecs:
            return 0.0
        dim = self.feature_dim
        mu = np.array(pattern.mu, dtype=float)
        if len(mu) > dim:
            mu = mu[:dim]
        elif len(mu) < dim:
            mu = np.pad(mu, (0, dim - len(mu)))
        mu_norm = np.linalg.norm(mu)
        if mu_norm == 0:
            return 0.0
        mu_unit = mu / mu_norm
        mean_sim = float(np.mean([np.dot(mu_unit, v) for v in vecs]))
        return float(np.clip((mean_sim + 1.0) / 2.0, 0.0, 1.0))

    def stream(self) -> Iterator[np.ndarray]:
        """Stream vectorised definitions from the NLTK word list.

        Note: when use_api=True, each word triggers an HTTP request to the
        dictionary API. Use LinguisticSubstrate(use_api=False) for offline streaming.
        """
        if self._use_api:
            warnings.warn(
                "LinguisticSubstrate.stream() with use_api=True will make one HTTP "
                "request per word. Consider use_api=False for offline streaming.",
                RuntimeWarning,
                stacklevel=2,
            )
        for word in itertools.cycle(self._word_list):
            vecs = self.fetch(word)
            for v in vecs:
                yield v
