"""Shared tokenizer and special-token constants for language training."""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*|[^\w\s]", re.UNICODE)
BOUNDARY = "<eos>"
UNK = "<unk>"
