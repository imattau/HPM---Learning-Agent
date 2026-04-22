# hpm_ai_v3/neural_lm_pattern.py
"""
LanguageModelPattern — self-supervised neural linguistic substrate for HPM v3.
CharLevelLSTM: lightweight next-char prediction. No external deps beyond PyTorch.
"""
from __future__ import annotations
import re
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class CharLevelLSTM(nn.Module):
    """
    Character-level LSTM language model.
    vocab_size=128 (printable ASCII), embed_dim=64, hidden_dim=128, n_layers=2.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers, batch_first=True
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """x: (batch, seq) int64. Returns (logits (batch, seq, vocab), hidden)."""
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.head(out)
        return logits, hidden

    def embed_sequence(self, text: str) -> torch.Tensor:
        """Return final hidden state as (hidden_dim,) tensor for a string."""
        indices = [min(ord(c), self.vocab_size - 1) for c in text]
        if not indices:
            return torch.zeros(self.hidden_dim)
        x = torch.tensor([indices], dtype=torch.long)
        with torch.no_grad():
            _, (h, _) = self.forward(x)
        return h[-1, 0, :]  # last layer, batch 0
