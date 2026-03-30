"""
QueryStdlib — scans Python stdlib source files for context windows around a target token.

Given a coverage gap (gap_mu), identifies the most likely token (argmax), then
tokenizes stdlib .py files to find 4-token context windows around that token.

Returns up to max_results context window strings (space-separated: "l2 l1 r1 r2").
Function definitions found near the token are returned prefixed with "sig: ".

scan_tokens() scans stdlib once for multiple tokens simultaneously and caches
results to disk — use this for world model construction to avoid repeated scans.
scan_token() is a convenience wrapper for single-token queries.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tokenize
import sysconfig
from pathlib import Path

import numpy as np

from hfn.query import Query
from hpm_fractal_node.code.code_loader import VOCAB, VOCAB_INDEX

# Disk cache location — alongside this file
_CACHE_FILE = Path(__file__).parent / "_stdlib_cache.json"


# ---------------------------------------------------------------------------
# Module-level scan — usable without a QueryStdlib instance
# ---------------------------------------------------------------------------

def _tokenize_file(path: Path) -> list[str]:
    """Tokenize a Python source file; return NAME and OP tokens only."""
    source = path.read_text(encoding="utf-8", errors="replace")
    tokens: list[str] = []
    try:
        reader = io.StringIO(source).readline
        for tok_type, tok_str, _, _, _ in tokenize.generate_tokens(reader):
            if tok_type in (tokenize.NAME, tokenize.OP):
                tokens.append(tok_str)
    except tokenize.TokenError:
        pass
    return tokens


def scan_tokens(
    target_tokens: list[str],
    max_results: int = 20,
    use_cache: bool = True,
) -> dict[str, list[str]]:
    """
    Scan stdlib .py files once for all target_tokens simultaneously.

    Returns a dict mapping token → list of context window strings (up to
    max_results each). Results are cached to disk; subsequent calls with the
    same Python version return instantly.

    Cache key includes Python version so upgrades invalidate automatically.
    """
    cache_key = f"py{sys.version_info.major}.{sys.version_info.minor}_max{max_results}"

    if use_cache and _CACHE_FILE.exists():
        try:
            cached = json.loads(_CACHE_FILE.read_text())
            if cached.get("_key") == cache_key:
                result = {t: cached.get(t, []) for t in target_tokens}
                # All requested tokens present → full cache hit
                if all(t in cached for t in target_tokens):
                    return result
        except Exception:
            pass

    stdlib_path = sysconfig.get_paths().get("stdlib", "")
    if not stdlib_path or not os.path.isdir(stdlib_path):
        return {t: [] for t in target_tokens}

    target_set = set(target_tokens)
    cap = max_results * 3

    # Per-token accumulators
    ctx: dict[str, list[str]] = {t: [] for t in target_tokens}
    sig: dict[str, list[str]] = {t: [] for t in target_tokens}

    for py_file in Path(stdlib_path).rglob("*.py"):
        # Stop when all tokens have enough results
        if all(len(ctx[t]) + len(sig[t]) >= cap for t in target_tokens):
            break
        try:
            tokens = _tokenize_file(py_file)
        except Exception:
            continue

        for i, tok in enumerate(tokens):
            if tok not in target_set:
                continue
            if len(ctx[tok]) + len(sig[tok]) >= cap:
                continue

            l2 = tokens[i - 2] if i >= 2 else "<unk>"
            l1 = tokens[i - 1] if i >= 1 else "<unk>"
            r1 = tokens[i + 1] if i + 1 < len(tokens) else "<unk>"
            r2 = tokens[i + 2] if i + 2 < len(tokens) else "<unk>"

            l2 = l2 if l2 in VOCAB_INDEX else "<unk>"
            l1 = l1 if l1 in VOCAB_INDEX else "<unk>"
            r1 = r1 if r1 in VOCAB_INDEX else "<unk>"
            r2 = r2 if r2 in VOCAB_INDEX else "<unk>"

            is_sig = (i >= 1 and tokens[i - 1] == "def") or (
                i >= 2 and tokens[i - 2] == "def"
            )
            if is_sig:
                sig[tok].append(f"sig: def {tok} {r1} {r2}")
            else:
                ctx[tok].append(f"{l2} {l1} {r1} {r2}")

    # Deduplicate and trim per token
    combined: dict[str, list[str]] = {}
    for t in target_tokens:
        seen: set[str] = set()
        deduped: list[str] = []
        for s in sig[t] + ctx[t]:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        combined[t] = deduped[:max_results]

    # Write disk cache
    if use_cache:
        try:
            payload = {"_key": cache_key, **combined}
            # Merge with any existing cached tokens
            if _CACHE_FILE.exists():
                try:
                    existing = json.loads(_CACHE_FILE.read_text())
                    if existing.get("_key") == cache_key:
                        existing.update(payload)
                        payload = existing
                except Exception:
                    pass
            _CACHE_FILE.write_text(json.dumps(payload))
        except Exception:
            pass

    return combined


def scan_token(target_token: str, max_results: int = 20) -> list[str]:
    """Convenience wrapper: scan stdlib for a single token."""
    return scan_tokens([target_token], max_results=max_results).get(target_token, [])


# ---------------------------------------------------------------------------
# QueryStdlib — Observer-facing Query subclass with per-token caching
# ---------------------------------------------------------------------------

class QueryStdlib(Query):
    """
    Query the Python standard library for context windows around a target token.

    Results are cached per token index so each token is scanned at most once
    per QueryStdlib instance.

    Parameters
    ----------
    max_results : int
        Maximum number of context window strings to return per token (default: 20).
    """

    def __init__(self, max_results: int = 20) -> None:
        self.max_results = max_results
        self._cache: dict[int, list[str]] = {}

    def fetch(self, gap_mu: np.ndarray, context=None) -> list[str]:
        """
        Map gap_mu -> argmax -> VOCAB token, return cached or freshly scanned results.
        """
        idx = int(np.argmax(gap_mu))
        if idx >= len(VOCAB):
            return []
        if idx in self._cache:
            return self._cache[idx]
        result = scan_token(VOCAB[idx], self.max_results)
        self._cache[idx] = result
        return result
