"""Fetch and parse webpage or plain text into passage chunks."""
from __future__ import annotations
import re
import urllib.request
from html.parser import HTMLParser
from typing import List, Optional


class _StripHTMLParser(HTMLParser):
    SKIP_TAGS = {"script", "style", "head", "nav", "footer", "header"}

    def __init__(self):
        super().__init__()
        self._skip = 0
        self._parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def strip_html(html: str) -> str:
    """Remove HTML tags and script/style content, return plain text."""
    parser = _StripHTMLParser()
    parser.feed(html)
    text = parser.get_text()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_url(url: str, timeout: int = 10) -> str:
    """Fetch URL and return stripped plain text."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return strip_html(raw)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences on .!? boundaries."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def fetch_passages(
    url: Optional[str] = None,
    text: Optional[str] = None,
    min_length: int = 40,
    mode: str = "paragraph",
) -> List[str]:
    """Fetch text from URL or accept raw text, split into passages."""
    if url is not None:
        text = fetch_url(url)
    if text is None:
        raise ValueError("Provide url or text")
    if mode == "sentence":
        chunks = split_sentences(text)
    else:
        chunks = [p.strip() for p in re.split(r"\n\n+", text)]
    return [c for c in chunks if len(c) >= min_length]
