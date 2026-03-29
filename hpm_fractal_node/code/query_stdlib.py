"""
QueryStdlib — scans Python stdlib source files for context windows around a target token.

Given a coverage gap (gap_mu), identifies the most likely token (argmax), then
tokenizes stdlib .py files to find 4-token context windows around that token.

Returns up to max_results context window strings (space-separated: "l2 l1 r1 r2").
Function definitions found near the token are returned prefixed with "sig: ".
"""
from __future__ import annotations

import io
import os
import tokenize
import sysconfig
from pathlib import Path

import numpy as np

from hfn.query import Query
from hpm_fractal_node.code.code_loader import VOCAB, VOCAB_INDEX


class QueryStdlib(Query):
    """
    Query the Python standard library for context windows around a target token.

    Parameters
    ----------
    max_results : int
        Maximum number of context window strings to return (default: 20).
    """

    def __init__(self, max_results: int = 20) -> None:
        self.max_results = max_results

    def fetch(self, gap_mu: np.ndarray, context=None) -> list[str]:
        """
        Map gap_mu -> argmax -> VOCAB token, scan stdlib for occurrences,
        return up to max_results context window strings.

        Strings prefixed with "sig: " represent function definitions.
        Context window format: "left2 left1 right1 right2"
        OOV tokens are replaced with "<unk>".
        """
        idx = int(np.argmax(gap_mu))
        if idx >= len(VOCAB):
            return []
        target_token = VOCAB[idx]

        stdlib_path = sysconfig.get_paths().get("stdlib", "")
        if not stdlib_path or not os.path.isdir(stdlib_path):
            return []

        results: list[str] = []
        sig_results: list[str] = []

        for py_file in Path(stdlib_path).rglob("*.py"):
            if len(results) + len(sig_results) >= self.max_results * 3:
                break
            try:
                tokens = self._tokenize_file(py_file)
            except Exception:
                continue

            for i, tok in enumerate(tokens):
                if tok != target_token:
                    continue

                # Extract 4-token context window
                l2 = tokens[i - 2] if i >= 2 else "<unk>"
                l1 = tokens[i - 1] if i >= 1 else "<unk>"
                r1 = tokens[i + 1] if i + 1 < len(tokens) else "<unk>"
                r2 = tokens[i + 2] if i + 2 < len(tokens) else "<unk>"

                # Normalize: replace OOV with <unk>
                l2 = l2 if l2 in VOCAB_INDEX else "<unk>"
                l1 = l1 if l1 in VOCAB_INDEX else "<unk>"
                r1 = r1 if r1 in VOCAB_INDEX else "<unk>"
                r2 = r2 if r2 in VOCAB_INDEX else "<unk>"

                # Check if this is a function definition context
                # A function def occurs when "def" appears near the target token
                is_sig = (
                    (i >= 1 and tokens[i - 1] == "def")
                    or (i >= 2 and tokens[i - 2] == "def")
                )

                ctx_str = f"{l2} {l1} {r1} {r2}"
                if is_sig:
                    sig_results.append(f"sig: def {target_token} {r1} {r2}")
                else:
                    results.append(ctx_str)

                if len(results) + len(sig_results) >= self.max_results * 3:
                    break

        # Deduplicate
        seen: set[str] = set()
        deduped_sig: list[str] = []
        for s in sig_results:
            if s not in seen:
                seen.add(s)
                deduped_sig.append(s)
        deduped_ctx: list[str] = []
        for s in results:
            if s not in seen:
                seen.add(s)
                deduped_ctx.append(s)

        combined = deduped_sig + deduped_ctx
        return combined[: self.max_results]

    def _tokenize_file(self, path: Path) -> list[str]:
        """Tokenize a Python file, returning a list of string tokens (keywords, ops, etc.)."""
        source = path.read_text(encoding="utf-8", errors="replace")
        tokens: list[str] = []
        try:
            reader = io.StringIO(source).readline
            for tok_type, tok_str, _, _, _ in tokenize.generate_tokens(reader):
                if tok_type in (
                    tokenize.NAME,
                    tokenize.OP,
                ):
                    tokens.append(tok_str)
        except tokenize.TokenError:
            pass
        return tokens
