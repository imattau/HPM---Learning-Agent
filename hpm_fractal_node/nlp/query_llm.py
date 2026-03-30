"""
QueryLLM — gap-driven knowledge queries via ollama HTTP API.

Identifies the nearest vocabulary token to the gap_mu vector, then asks
the LLM to generate related words in a structured 3-group format:

    similar: [near-synonyms]
    related: [semantically related words]
    context: [words that appear near the token in text]

Returns the 3 lines as raw strings for ConverterLLM to encode.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error

import numpy as np

from hfn.query import Query
from hpm_fractal_node.nlp.nlp_loader import VOCAB

_DEFAULT_HOST = "http://127.0.0.1:11434"
_DEFAULT_MODEL = "tinyllama"

_PROMPT_TEMPLATE = (
    "<|system|>\nYou are a helpful assistant.\n</s>\n"
    "<|user|>\n"
    "List words related to '{token}' in exactly this format:\n"
    "similar: [3-5 near-synonyms]\n"
    "related: [3-5 semantically related words]\n"
    "context: [3-5 words that appear near '{token}' in sentences]\n"
    "Only output the 3 lines. No explanation.\n"
    "</s>\n"
    "<|assistant|>\n"
)


class QueryLLM(Query):
    """
    Gap query backed by ollama HTTP API.

    Parameters
    ----------
    host : str
        Ollama server URL (default: http://localhost:11435).
    model : str
        Model name to use (default: tinyllama).
    """

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._cache: dict[str, list[str]] = {}
        self.current_target: str | None = None

    def _token_for(self, gap_mu: np.ndarray) -> str:
        """Return vocab word nearest to gap_mu (argmax of the vector)."""
        idx = int(np.argmax(gap_mu))
        if idx < len(VOCAB):
            return VOCAB[idx]
        return "<unknown>"

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self._host}/api/generate"
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 80, "temperature": 0.2},
        }).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data.get("response", "").strip()
        except (urllib.error.URLError, json.JSONDecodeError, KeyError):
            return ""

    def fetch(self, gap_mu: np.ndarray, context=None) -> list[str]:
        # Use current_target if set by experiment (for corpus unknown words)
        token = self.current_target or self._token_for(gap_mu)
        if token in ("<unknown>", "<start>", "<end>", None):
            token = self._token_for(gap_mu)
        if token in ("<unknown>", "<start>", "<end>"):
            return []

        if token in self._cache:
            return self._cache[token]

        prompt = _PROMPT_TEMPLATE.format(token=token)
        text = self._call_ollama(prompt)
        if not text:
            return []

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        result = []
        for line in lines:
            lower = line.lower()
            if lower.startswith(("similar:", "related:", "context:")):
                result.append(line)
            if len(result) == 3:
                break

        self._cache[token] = result
        return result
