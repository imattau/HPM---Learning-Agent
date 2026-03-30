"""
Download a small public-domain corpus to data/corpus/.

Uses "The Tale of Peter Rabbit" by Beatrix Potter (Project Gutenberg).
~5KB, simple English, fits the child language theme.

Run once:
    PYTHONPATH=. python3 hpm_fractal_node/nlp/download_corpus.py
"""
from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

CORPUS_URL = (
    "https://www.gutenberg.org/cache/epub/14838/pg14838.txt"
)
CORPUS_PATH = Path(__file__).parents[2] / "data" / "corpus" / "peter_rabbit.txt"


def download() -> Path:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CORPUS_PATH.exists():
        print(f"Corpus already exists at {CORPUS_PATH}")
        return CORPUS_PATH

    print(f"Downloading Peter Rabbit corpus ...")
    try:
        with urllib.request.urlopen(CORPUS_URL, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    # Strip Gutenberg header/footer
    start = raw.find("Once upon a time")
    end = raw.find("End of the Project Gutenberg")
    if start == -1:
        start = 0
    if end == -1:
        end = len(raw)
    text = raw[start:end].strip()

    CORPUS_PATH.write_text(text, encoding="utf-8")
    print(f"Saved {len(text)} chars to {CORPUS_PATH}")
    return CORPUS_PATH


def corpus_path() -> Path:
    return CORPUS_PATH


if __name__ == "__main__":
    download()
