"""Extract (agent, predicate, patient) propositions from a sentence.

A sentence's relations are what let compressed concepts say something about
each other rather than only co-occur. The default extractor is conservative
and dictionary-based (closed-class words plus WordNet's morphology and sense
frequencies): it prefers to emit nothing over emitting a wrong relation, and
it never invents an argument. A stronger parser can replace it by providing
the same ``extract`` method.

Roles are AGENT and PATIENT only. Passives promote the surface subject to
PATIENT; a ``by`` phrase gives the AGENT. Copular definitions ("X is a Y") use
the predicate ``be``. Pronoun subjects are not resolved, so they yield no role.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Protocol

from hpm_ai_v2.agents.text_structure import MIN_STEM_LENGTH, stem

ROLE_ORDER = ("AGENT", "PATIENT")
COPULA = "be"

_DET = {"the", "a", "an", "this", "that", "these", "those", "its", "their", "his", "her", "our",
        "my", "your", "each", "every", "some", "any", "no", "all", "both", "such", "another"}
_PREP = {"in", "on", "at", "by", "with", "from", "to", "of", "for", "into", "onto", "over", "under",
         "between", "among", "through", "during", "as", "about", "than", "via", "per", "within",
         "without", "across", "after", "before", "against", "toward", "towards", "upon", "using",
         "along", "around", "beside", "near", "inside", "outside", "behind", "beneath", "beyond",
         "throughout", "past", "like", "unlike", "despite", "except", "round", "underneath", "alongside"}
_CONJ = {"and", "or", "but", "nor", "so", "yet", "because", "although", "while", "when", "where",
         "if", "unless", "since", "until", "whereas", "then"}
_REL = {"which", "who", "whom", "whose", "that"}
_PRON = {"it", "they", "them", "he", "she", "we", "you", "i", "there", "here", "what", "why", "how"}
_BE = {"is", "are", "was", "were", "be", "been", "being", "am"}
_AUX = {"has", "have", "had", "do", "does", "did", "will", "would", "can", "could", "may", "might",
        "shall", "should", "must"}
_ADV = {"then", "also", "often", "usually", "not", "never", "only", "just", "still", "thus", "always",
        "typically", "generally", "mainly", "mostly", "together", "later", "first", "next", "very",
        "more", "most", "less", "even", "again", "fast", "well", "far", "hard", "early", "late",
        "soon", "now", "today"}
_BOUNDARY = {",", ";", ":"} | _CONJ | {"which", "who", "whom", "whose"}
_PARTICLE = {"down", "up", "out", "off", "apart", "away", "back"}
_TOKEN = re.compile(r"[A-Za-z][A-Za-z\-']*|[,;:]")


@dataclass(frozen=True)
class Proposition:
    predicate: str                      # Porter stem of the predicate word ("be" for copula)
    lemma: str                          # dictionary form, for display
    voice: str                          # "active", "passive" or "copula"
    roles: dict                         # role -> (head stem, noun-phrase text)

    def key(self) -> tuple:
        return (self.predicate, tuple(sorted((r, v[0]) for r, v in self.roles.items())))


class RoleExtractor(Protocol):
    def extract(self, sentence: str) -> list[Proposition]: ...


@lru_cache(maxsize=None)
def _profile(word: str):
    """(verb lemma, verb freq, noun freq, adj freq, is plain-noun-form) from WordNet."""
    from nltk.corpus import wordnet as wn

    def weight(lemma, pos):
        """SemCor sense frequency plus half the number of senses (frequencies are sparse)."""
        if lemma is None:
            return 0.0
        senses = wn.synsets(lemma, pos)
        return sum(l.count() for s in senses for l in s.lemmas() if l.name().lower() == lemma) + 0.5 * len(senses)

    verb, noun, adj = wn.morphy(word, "v"), wn.morphy(word, "n"), wn.morphy(word, "a")
    if verb in {"be", "have", "do"} and word not in _BE | _AUX:
        verb = None                                   # "bees" is not an inflection of "be"
    return verb, weight(verb, "v"), weight(noun, "n"), weight(adj, "a"), noun is not None


def _is_participle(word: str) -> bool:
    """Past/passive participle form: -ed, -en, or irregular (built, shown) but not a noun form."""
    verb, _v, _n, _a, noun_form = _profile(word)
    if verb is None or noun_form or word.endswith(("ing", "s")):
        return False
    return word.endswith(("ed", "en")) or verb != word


def _verb_score(word: str, prev: Optional[str], nxt: Optional[str]) -> float:
    """How likely ``word`` is a verb here (0..1); WordNet sense frequency plus local context."""
    if "-" in word or word.endswith("ing") or prev in _DET:
        return 0.0                                    # modifiers, gerunds, "a waggle dance"
    if word in _DET | _PREP | _CONJ | _PRON | _ADV | _BE | _AUX:
        return 0.0
    verb, v, n, a, _noun_form = _profile(word)
    if verb is None:
        return 0.0
    score = (v + 0.1) / (v + n + a + 0.3)
    if _is_participle(word):
        score = max(score, 0.9)
    if nxt in _DET | _PRON and prev not in _DET | _PREP | {None}:
        score = max(score, 0.55)                      # "... limit the rate": verb before a determiner
    if nxt in _PARTICLE and prev not in _DET | _PREP | {None}:
        score = max(score, 0.6)                       # "... speed up ...": verb before a particle
    return score


class HeuristicRoleExtractor:
    """Conservative clause-level extractor; see the module docstring."""

    def extract(self, sentence: str) -> list[Proposition]:
        if sentence.rstrip().endswith("?"):
            return []                                 # a question asserts nothing
        w = [m.group(0).lower() for m in _TOKEN.finditer(sentence)]
        out: list[Proposition] = []
        seen: set = set()
        previous: Optional[tuple[int, int]] = None
        last_agent: Optional[tuple[str, str]] = None
        for lo, hi in self._segments(w):
            props, agent = self._clause(w, lo, hi, previous, last_agent)
            for prop in props:
                if prop.key() not in seen:
                    seen.add(prop.key())
                    out.append(prop)
            if agent is not None:
                last_agent = agent
            previous = (lo, hi)
        return out

    # -- segmentation ------------------------------------------------------------
    def _segments(self, w: list[str]) -> list[tuple[int, int]]:
        cuts, start = [], 0
        for i, token in enumerate(w):
            if token in _BOUNDARY:
                if i > start:
                    cuts.append((start, i))
                start = i
        cuts.append((start, len(w)))
        # a clause without a verb is part of the previous one (coordinated noun phrases)
        merged: list[tuple[int, int]] = []
        for lo, hi in cuts:
            if merged and self._predicate(w, lo, hi) is None:
                merged[-1] = (merged[-1][0], hi)
            else:
                merged.append((lo, hi))
        return merged

    # -- predicate ----------------------------------------------------------------
    def _predicate(self, w: list[str], lo: int, hi: int):
        """Return (index, kind) of the main predicate in ``w[lo:hi]`` or None."""
        i = lo
        while i < hi:
            token = w[i]
            if token in _BE:
                j = i + 1
                while j < hi and w[j] in _ADV:
                    j += 1
                if j < hi and w[j] not in _DET and self._has_left_nominal(w, lo, i):
                    after = w[j + 1] if j + 1 < hi else None
                    if _verb_score(w[j], w[j - 1], after) >= 0.5 and (
                        _is_participle(w[j]) or after is None or after in _PREP | _ADV | _BOUNDARY
                    ):
                        return j, "passive"
                if j < hi and w[j] in _DET and self._has_left_nominal(w, lo, i):
                    return i, "copula"
                i += 1
                continue
            if token in _AUX or token in _BOUNDARY or (i > lo and w[i - 1] == "to"):
                i += 1                                # infinitives are not clauses of their own
                continue
            prev = w[i - 1] if i > lo else None
            nxt = w[i + 1] if i + 1 < len(w) else None
            if i > lo and _verb_score(token, prev, nxt) >= 0.5:
                if self._has_left_nominal(w, lo, i):
                    return i, "active"
                if w[lo] in {"and", "or", "but", ","} and i == lo + 1:
                    return i, "coordinated"          # shares the previous clause's subject
            i += 1
        return None

    @staticmethod
    def _has_left_nominal(w: list[str], lo: int, i: int) -> bool:
        return any(t not in _DET | _PREP | _CONJ | _BE | _AUX | _ADV | {",", ";", ":"} for t in w[lo:i])

    # -- noun phrases -------------------------------------------------------------
    @staticmethod
    def _is_content(w: list[str], j: int, threshold: float = 0.5) -> bool:
        token = w[j]
        if token in _DET | _PREP | _CONJ | _REL | _PRON | _BE | _AUX | _ADV | {",", ";", ":"}:
            return False
        prev = w[j - 1] if j > 0 else None
        nxt = w[j + 1] if j + 1 < len(w) else None
        return _verb_score(token, prev, nxt) < threshold or "-" in token

    def _np_left(self, w: list[str], lo: int, end: int) -> Optional[tuple[str, str]]:
        """Noun phrase ending just before ``end`` (skipping adverbs and auxiliaries)."""
        j = end - 1
        while j >= lo and (w[j] in _ADV or w[j] in _BE or w[j] in _AUX):
            j -= 1
        tokens = []
        while j >= lo and self._is_content(w, j):
            tokens.append(w[j])
            j -= 1
        return self._np(list(reversed(tokens)))

    def _np_right(self, w: list[str], start: int, hi: int) -> tuple[list[tuple[str, str]], int]:
        """Coordinated noun phrases starting at ``start``; returns (phrases, end index)."""
        phrases, j = [], start
        while True:
            while j < hi and (w[j] in _ADV or w[j] in _PARTICLE):
                j += 1
            determined = False
            while j < hi and w[j] in _DET:
                j += 1
                determined = True
            tokens = []
            # The first word after the verb starts the object ("absorb water", "erupt molten
            # rock"); after a determiner the phrase continues to its head, so only a clear
            # verb ends it ("a waggle dance to ...": dance is the head, not a verb).
            while j < hi and (
                self._is_content(w, j, 0.8 if determined else 0.5)
                or (not tokens and self._is_content(w, j, 1.01))
            ):
                tokens.append(w[j])
                j += 1
            if not tokens:
                break
            # "process plants use": a noun phrase directly followed by an active verb is
            # a reduced relative clause, whose subject is the last noun
            if len(tokens) >= 2 and j < hi and _verb_score(w[j], w[j - 1], w[j + 1] if j + 1 < len(w) else None) >= 0.5 \
                    and not _is_participle(w[j]):
                tokens.pop()
                j -= 1
            phrase = self._np(tokens)
            if phrase is not None:
                phrases.append(phrase)
            if j + 1 < hi and w[j] in {"and", "or"} and phrases:
                j += 1
                continue
            break
        return phrases, j

    @staticmethod
    def _np(tokens: list[str]) -> Optional[tuple[str, str]]:
        if not tokens:
            return None
        head = re.split(r"[-']", tokens[-1])[-1] if "-" in tokens[-1] else tokens[-1].rstrip("'")
        key = stem(head)
        if len(key) < MIN_STEM_LENGTH:
            return None
        return key, " ".join(tokens)

    # -- one clause ---------------------------------------------------------------
    def _clause(self, w, lo, hi, previous, last_agent):
        found = self._predicate(w, lo, hi)
        if found is None:
            return [], None
        i, kind = found
        left_lo = lo + 1 if w[lo] in _BOUNDARY else lo
        if w[lo] in _REL:
            subject = self._antecedent(w, previous)
        elif kind == "coordinated":
            subject = last_agent
        else:
            subject = self._np_left(w, left_lo, i)
        if kind == "passive":
            roles: dict = {}
            if subject:
                roles["PATIENT"] = subject
            for j in range(i + 1, hi):
                if w[j] == "by":
                    agents, _ = self._np_right(w, j + 1, hi)
                    if agents:
                        roles["AGENT"] = agents[0]
                    break
            return self._make(w[i], "passive", [roles]), roles.get("AGENT")
        if kind == "copula":
            complements, _ = self._np_right(w, i + 1, hi)
            if subject and complements:
                return self._make(COPULA, "copula", [{"AGENT": subject, "PATIENT": complements[0]}]), subject
            return [], subject
        objects, _ = self._np_right(w, i + 1, hi)
        if not (subject and objects):
            return [], subject
        return self._make(w[i], "active", [{"AGENT": subject, "PATIENT": o} for o in objects]), subject

    def _antecedent(self, w, previous) -> Optional[tuple[str, str]]:
        """Noun a relative clause modifies: the last noun phrase of the preceding clause."""
        if previous is None:
            return None
        lo, hi = previous
        j = hi
        while j > lo and not self._is_content(w, j - 1):
            j -= 1
        return self._np_left(w, lo, j)

    @staticmethod
    def _make(word: str, voice: str, role_sets: list[dict]) -> list[Proposition]:
        role_sets = [roles for roles in role_sets if roles]
        if voice == "copula":
            return [Proposition(COPULA, COPULA, voice, roles) for roles in role_sets]
        lemma = _profile(word)[0] or word
        return [Proposition(stem(word), lemma, voice, roles) for roles in role_sets]
