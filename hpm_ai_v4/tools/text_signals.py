"""Lightweight text signal extraction for HPM text/control loops.

This module stays optional-dependency friendly:
- spaCy, rapidfuzz, and wordfreq are used if installed.
- Otherwise it falls back to simple local heuristics.

The goal is not to replace dictionary/grammar checks.
It is to provide a few additional, orthogonal signals for
structure, repetition, and lexical plausibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:  # Optional accelerator / better similarity.
    from rapidfuzz import fuzz as _rapidfuzz_fuzz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _rapidfuzz_fuzz = None

try:  # Optional lexical prior.
    from wordfreq import zipf_frequency as _zipf_frequency  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _zipf_frequency = None

try:  # Optional syntactic signal.
    import spacy  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    spacy = None

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _similarity(a: str, b: str) -> float:
    """Return a normalized similarity score in [0, 1]."""
    a = a.strip().lower()
    b = b.strip().lower()
    if not a or not b:
        return 0.0
    if _rapidfuzz_fuzz is not None:
        return float(_rapidfuzz_fuzz.ratio(a, b) / 100.0)
    return float(SequenceMatcher(None, a, b).ratio())


@dataclass
class TextSignalPack:
    """Compact auxiliary signal bundle for generated or observed text."""

    structure_score: float = 0.0
    repeat_score: float = 0.0
    commonness_score: float = 0.0
    sentence_confidence: float = 0.0
    target_alignment: float = 0.0
    source: str = "fallback"

    def combined_score(self) -> float:
        """Composite scalar used by ranking and reward shaping."""
        return (
            0.32 * self.structure_score
            + 0.24 * self.commonness_score
            + 0.18 * self.sentence_confidence
            + 0.14 * self.target_alignment
            - 0.38 * self.repeat_score
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TextSignalExtractor:
    """Compute thin text signals for ranking and feedback."""

    def __init__(self, use_spacy: bool = True):
        self._nlp = self._load_spacy() if use_spacy else None

    def _load_spacy(self):
        if spacy is None:  # pragma: no cover - optional dependency
            return None
        try:
            if hasattr(spacy, "util") and spacy.util.is_package("en_core_web_sm"):
                return spacy.load("en_core_web_sm", disable=["ner"])
        except Exception:
            pass
        try:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp
        except Exception:
            return None

    def analyze(
        self,
        text: str,
        *,
        context_texts: Optional[Sequence[str]] = None,
        target_text: Optional[str] = None,
        dictionary: Any = None,
        grammar: Any = None,
    ) -> TextSignalPack:
        text = text or ""
        context_texts = [t for t in (context_texts or []) if t]
        tokens = self._tokenize(text)
        alpha_tokens = [tok for tok in tokens if tok.isalpha()]

        structure_score = self._structure_score(text, tokens, alpha_tokens)
        commonness_score = self._commonness_score(alpha_tokens, dictionary=dictionary)
        sentence_confidence = self._sentence_confidence(text, tokens, alpha_tokens, grammar=grammar)
        repeat_score = self._repeat_score(text, context_texts)
        target_alignment = _similarity(text, target_text) if target_text else 0.0

        source = "heuristic"
        if self._nlp is not None:
            source = "spacy+heuristic" if getattr(self._nlp, "pipe_names", None) else "heuristic"

        return TextSignalPack(
            structure_score=_clamp(structure_score),
            repeat_score=_clamp(repeat_score),
            commonness_score=_clamp(commonness_score),
            sentence_confidence=_clamp(sentence_confidence),
            target_alignment=_clamp(target_alignment),
            source=source,
        )

    def score(
        self,
        text: str,
        *,
        context_texts: Optional[Sequence[str]] = None,
        target_text: Optional[str] = None,
        dictionary: Any = None,
        grammar: Any = None,
    ) -> float:
        return self.analyze(
            text,
            context_texts=context_texts,
            target_text=target_text,
            dictionary=dictionary,
            grammar=grammar,
        ).combined_score()

    def metadata(
        self,
        text: str,
        *,
        context_texts: Optional[Sequence[str]] = None,
        target_text: Optional[str] = None,
        dictionary: Any = None,
        grammar: Any = None,
        prefix: str = "text_signal",
    ) -> Dict[str, Any]:
        pack = self.analyze(
            text,
            context_texts=context_texts,
            target_text=target_text,
            dictionary=dictionary,
            grammar=grammar,
        )
        data = pack.to_dict()
        data["combined_score"] = pack.combined_score()
        return {f"{prefix}_{key}": value for key, value in data.items()}

    def _tokenize(self, text: str) -> List[str]:
        return _WORD_RE.findall(text)

    def _structure_score(
        self,
        text: str,
        tokens: Sequence[str],
        alpha_tokens: Sequence[str],
    ) -> float:
        if not text.strip():
            return 0.0
        sentence_hits = len(re.findall(r"[.!?]", text))
        ends_cleanly = 1.0 if text.rstrip().endswith((".", "!", "?")) else 0.0
        has_enough_tokens = 1.0 if len(tokens) >= 4 else 0.0
        alpha_ratio = len(alpha_tokens) / max(1, len(tokens))
        uppercase_start = 1.0 if text[:1].isupper() else 0.0
        spaced = 1.0 if " " in text.strip() else 0.0

        spacy_bonus = 0.0
        if self._nlp is not None:
            try:
                doc = self._nlp(text)
                sents = list(doc.sents) if getattr(doc, "has_annotation", lambda *_: False)("SENT_START") else []
                if sents:
                    spacy_bonus += 0.08
                pos_tags = [tok.pos_ for tok in doc if getattr(tok, "pos_", "") and tok.pos_ not in {"", "X"}]
                if pos_tags:
                    nouns = sum(1 for pos in pos_tags if pos.startswith("N"))
                    verbs = sum(1 for pos in pos_tags if pos.startswith("V"))
                    spacy_bonus += min(0.18, 0.06 * (nouns > 0) + 0.06 * (verbs > 0))
                if any(getattr(tok, "is_punct", False) for tok in doc):
                    spacy_bonus += 0.03
            except Exception:
                pass

        return (
            0.20 * min(1.0, sentence_hits / 2.0)
            + 0.20 * ends_cleanly
            + 0.20 * has_enough_tokens
            + 0.20 * alpha_ratio
            + 0.10 * uppercase_start
            + 0.10 * spaced
            + spacy_bonus
        )

    def _commonness_score(self, alpha_tokens: Sequence[str], dictionary: Any = None) -> float:
        if not alpha_tokens:
            return 0.0

        scores: List[float] = []
        for tok in alpha_tokens:
            low = tok.lower()
            if _zipf_frequency is not None:
                try:
                    # Zipf frequency roughly ranges 0..7+.
                    scores.append(_clamp(_zipf_frequency(low, "en") / 7.0))
                    continue
                except Exception:
                    pass
            if dictionary is not None and hasattr(dictionary, "score_word"):
                try:
                    scores.append(float(dictionary.score_word(low)))
                    continue
                except Exception:
                    pass
            if low.isalpha():
                scores.append(0.7 if 3 <= len(low) <= 10 else 0.45)
            else:
                scores.append(0.1)
        return float(sum(scores) / len(scores)) if scores else 0.0

    def _sentence_confidence(
        self,
        text: str,
        tokens: Sequence[str],
        alpha_tokens: Sequence[str],
        grammar: Any = None,
    ) -> float:
        if not text.strip():
            return 0.0
        if len(tokens) < 2:
            return 0.0

        end_punct = 1.0 if text.rstrip().endswith((".", "!", "?")) else 0.0
        alpha_ratio = len(alpha_tokens) / max(1, len(tokens))
        punctuation_balance = 1.0 - min(1.0, max(0, text.count("(") + text.count("[") + text.count("{") - text.count(")") - text.count("]") - text.count("}")) / 3.0)
        grammar_bonus = 0.0
        if grammar is not None and len(alpha_tokens) >= 2:
            try:
                grammar_score = float(grammar.score_sequence(list(alpha_tokens[:8])))
                grammar_bonus = 0.35 * _clamp(grammar_score)
            except Exception:
                grammar_bonus = 0.0
        return (
            0.34 * end_punct
            + 0.28 * alpha_ratio
            + 0.20 * punctuation_balance
            + 0.18 * grammar_bonus
        )

    def _repeat_score(self, text: str, context_texts: Sequence[str]) -> float:
        if not text.strip():
            return 1.0

        tokens = self._tokenize(text.lower())
        if len(tokens) < 2:
            token_repeat = 0.0
        else:
            unique_tokens = len(set(tokens))
            token_repeat = 1.0 - (unique_tokens / len(tokens))

        contextual = 0.0
        if context_texts:
            contextual = max(_similarity(text, ctx) for ctx in context_texts)

        return min(1.0, 0.55 * token_repeat + 0.45 * contextual)
