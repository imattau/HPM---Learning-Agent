"""Input ingest helpers for novelty checks and exact reuse gates."""
from __future__ import annotations

import hashlib
import json
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class IngestSummary:
    seen_text_signatures: int = 0
    seen_pattern_signatures: int = 0
    duplicate_texts: int = 0
    duplicate_patterns: int = 0


@dataclass
class TextIngestGate:
    """Exact novelty gate for text chunks and exact pattern fingerprints.

    The gate keeps the expensive training loop from re-processing identical
    chunks and exact duplicate patterns. It is intentionally conservative:
    approximate semantic reuse still flows through the learner; only exact
    canonical reuse is short-circuited here.
    """

    adapter: Any = None
    lowercase: bool = True
    seen_text_signatures: set[str] = field(default_factory=set)
    seen_pattern_signatures: set[str] = field(default_factory=set)
    seen_window_signatures: set[str] = field(default_factory=set)
    window_profiles: Dict[str, List[str]] = field(default_factory=dict)
    duplicate_texts: int = 0
    duplicate_patterns: int = 0
    duplicate_windows: int = 0
    near_duplicate_windows: int = 0
    max_window_profiles: int = 4096

    @classmethod
    def from_snapshot(cls, snapshot: Optional[Dict[str, Any]] = None, *, adapter: Any = None, lowercase: bool = True) -> "TextIngestGate":
        gate = cls(adapter=adapter, lowercase=lowercase)
        gate.load_snapshot(snapshot or {})
        return gate

    def _canonical_text(self, text: str) -> str:
        raw = str(text or "")
        return raw.lower() if self.lowercase else raw

    def _surface_tokens(self, text: str) -> List[str]:
        raw = str(text or "")
        adapter = self.adapter
        if adapter is not None and hasattr(adapter, "tokenize"):
            try:
                tokens = [str(tok) for tok in adapter.tokenize(raw) if str(tok)]
                if tokens:
                    return tokens
            except Exception:
                pass
        projected = []
        for ch in raw:
            if ch == "\n" or 32 <= ord(ch) <= 126:
                projected.append(ch.lower() if self.lowercase else ch)
        return projected

    @staticmethod
    def _digest(parts: Sequence[str]) -> str:
        joined = "\u241f".join(str(part) for part in parts)
        return hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()

    def text_signature(self, text: str) -> str:
        return self._digest(self._surface_tokens(self._canonical_text(text)))

    def register_text(self, text: str) -> bool:
        key = self.text_signature(text)
        if key in self.seen_text_signatures:
            self.duplicate_texts += 1
            return False
        self.seen_text_signatures.add(key)
        return True

    def pattern_signature(self, pattern: Any, *, decimals: int = 4) -> str:
        pieces: List[np.ndarray] = []
        for name in ("A", "B", "pi"):
            arr = getattr(pattern, name, None)
            if arr is None:
                continue
            pieces.append(np.asarray(arr, dtype=np.float32).ravel())
        if not pieces:
            return self._digest([str(getattr(pattern, "id", "")), str(getattr(pattern, "latent_dim", "")), str(getattr(pattern, "obs_dim", ""))])
        fingerprint = np.concatenate(pieces)
        rounded = np.round(fingerprint, decimals=decimals)
        return hashlib.sha1(rounded.tobytes()).hexdigest()

    def register_pattern(self, pattern: Any, *, decimals: int = 4) -> bool:
        key = self.pattern_signature(pattern, decimals=decimals)
        if key in self.seen_pattern_signatures:
            self.duplicate_patterns += 1
            return False
        self.seen_pattern_signatures.add(key)
        return True

    def filter_new_patterns(self, patterns: Sequence[Any], *, decimals: int = 4) -> List[Any]:
        kept: List[Any] = []
        for pattern in patterns:
            if self.register_pattern(pattern, decimals=decimals):
                kept.append(pattern)
        return kept

    def snapshot(self) -> Dict[str, Any]:
        return {
            "seen_text_signatures": len(self.seen_text_signatures),
            "seen_pattern_signatures": len(self.seen_pattern_signatures),
            "seen_window_signatures": len(self.seen_window_signatures),
            "duplicate_texts": int(self.duplicate_texts),
            "duplicate_patterns": int(self.duplicate_patterns),
            "duplicate_windows": int(self.duplicate_windows),
            "near_duplicate_windows": int(self.near_duplicate_windows),
            "text_signature_list": sorted(self.seen_text_signatures),
            "pattern_signature_list": sorted(self.seen_pattern_signatures),
            "window_signature_list": sorted(self.seen_window_signatures),
            "window_profile_list": [
                {"signature": sig, "tokens": list(tokens)}
                for sig, tokens in sorted(self.window_profiles.items(), key=lambda item: item[0])
            ],
        }

    def load_snapshot(self, snapshot: Dict[str, Any]) -> None:
        if not isinstance(snapshot, dict):
            return
        duplicate_texts = int(snapshot.get("duplicate_texts", 0) or 0)
        duplicate_patterns = int(snapshot.get("duplicate_patterns", 0) or 0)
        duplicate_windows = int(snapshot.get("duplicate_windows", 0) or 0)
        near_duplicate_windows = int(snapshot.get("near_duplicate_windows", 0) or 0)
        text_signatures = snapshot.get("text_signature_list", []) or []
        pattern_signatures = snapshot.get("pattern_signature_list", []) or []
        window_signatures = snapshot.get("window_signature_list", []) or []
        window_profiles = snapshot.get("window_profile_list", []) or []

        self.seen_text_signatures.update(str(v) for v in text_signatures if str(v))
        self.seen_pattern_signatures.update(str(v) for v in pattern_signatures if str(v))
        self.seen_window_signatures.update(str(v) for v in window_signatures if str(v))
        if isinstance(window_profiles, list):
            for item in window_profiles:
                if not isinstance(item, dict):
                    continue
                signature = str(item.get("signature", "") or "").strip()
                tokens = [str(v) for v in (item.get("tokens", []) or []) if str(v)]
                if signature and tokens:
                    self.window_profiles[signature] = tokens
        self.duplicate_texts += max(0, duplicate_texts)
        self.duplicate_patterns += max(0, duplicate_patterns)
        self.duplicate_windows += max(0, duplicate_windows)
        self.near_duplicate_windows += max(0, near_duplicate_windows)

    @classmethod
    def load_snapshot_from_path(cls, path: str, *, adapter: Any = None, lowercase: bool = True) -> "TextIngestGate":
        gate = cls(adapter=adapter, lowercase=lowercase)
        try:
            with open(path, "r", encoding="utf-8") as f:
                gate.load_snapshot(json.load(f))
        except FileNotFoundError:
            pass
        except Exception:
            pass
        return gate

    @staticmethod
    def window_signature(text: str, *, adapter: Any = None, lowercase: bool = True) -> str:
        raw = str(text or "")
        if adapter is not None and hasattr(adapter, "tokenize"):
            try:
                tokens = [str(tok) for tok in adapter.tokenize(raw) if str(tok)]
            except Exception:
                tokens = []
        else:
            tokens = []
        if not tokens:
            tokens = [ch.lower() if lowercase else ch for ch in raw if ch == "\n" or 32 <= ord(ch) <= 126]
        return TextIngestGate._digest(tokens)

    def window_tokens(self, text: str) -> List[str]:
        raw = str(text or "")
        adapter = self.adapter
        tokens: List[str] = []
        if adapter is not None and hasattr(adapter, "tokenize"):
            try:
                tokens = [str(tok) for tok in adapter.tokenize(raw) if str(tok)]
            except Exception:
                tokens = []
        if not tokens:
            tokens = [ch.lower() if self.lowercase else ch for ch in raw if ch.isalnum()]
        return [tok for tok in tokens if tok and tok != "\n"]

    @staticmethod
    def _token_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
        sa = {str(tok) for tok in a if str(tok)}
        sb = {str(tok) for tok in b if str(tok)}
        if not sa or not sb:
            return 0.0
        return float(len(sa & sb) / max(1, len(sa | sb)))

    def window_similarity(self, text: str, other_tokens: Sequence[str]) -> float:
        tokens = self.window_tokens(text)
        token_jaccard = self._token_jaccard(tokens, other_tokens)
        text_a = " ".join(tokens)
        text_b = " ".join(str(tok) for tok in other_tokens if str(tok))
        seq_ratio = SequenceMatcher(None, text_a, text_b).ratio() if text_a and text_b else 0.0
        return float(0.55 * token_jaccard + 0.45 * seq_ratio)

    def match_window(self, text: str, *, near_threshold: float = 0.64) -> Tuple[str, float]:
        exact_sig = self.window_signature(text, adapter=self.adapter, lowercase=self.lowercase)
        if exact_sig in self.seen_window_signatures:
            return "known", 1.0

        tokens = self.window_tokens(text)
        if not tokens or not self.window_profiles:
            return "novel", 0.0

        best = 0.0
        for profile_tokens in self.window_profiles.values():
            sim = self.window_similarity(text, profile_tokens)
            if sim > best:
                best = sim
                if best >= near_threshold:
                    break
        if best >= near_threshold:
            return "near", best
        return "novel", best

    def register_window(self, text: str) -> bool:
        key = self.window_signature(text, adapter=self.adapter, lowercase=self.lowercase)
        if key in self.seen_window_signatures:
            self.duplicate_windows += 1
            return False
        self.seen_window_signatures.add(key)
        if len(self.window_profiles) < self.max_window_profiles:
            self.window_profiles[key] = self.window_tokens(text)
        return True

    def window_status(self, text: str) -> str:
        state, _ = self.match_window(text)
        return state

    def to_json(self) -> str:
        return json.dumps(self.snapshot(), sort_keys=True)
