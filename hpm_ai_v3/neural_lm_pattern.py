# hpm_ai_v3/neural_lm_pattern.py
"""
LanguageModelPattern — self-supervised neural linguistic substrate for HPM v3.
CharLevelLSTM: lightweight next-char prediction. No external deps beyond PyTorch.
"""
from __future__ import annotations
import re
import os
import json
import importlib
import inspect
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import List, Optional, Tuple, Any, Dict
from hpm_ai_v3.pattern import HPMPattern


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


class LanguageModelPattern(HPMPattern):
    """
    Self-supervised neural linguistic substrate.
    Wraps CharLevelLSTM. Exposes tokenize, extract_numbers, embed via sample().
    Pretrains offline via next-char prediction.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = CharLevelLSTM(vocab_size, embed_dim, hidden_dim, n_layers)
        self.last_loss: float = float("inf")
        self.accuracy: float = 0.0
        self._pretrained: bool = False

    # ── sample() ──────────────────────────────────────────────────────────────

    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Any:
        """
        Dispatch on context["action"]:
          "tokenize"        → whitespace split
          "extract_numbers" → number extraction (LM-guided post-train, regex pre-train)
          "embed"           → LSTM hidden state as list of floats
        """
        action = context.get("action", "")
        text = context.get("text", "")

        if action == "tokenize":
            return text.split()

        if action == "extract_numbers":
            return self._extract_numbers(text)

        if action == "embed":
            return self._embed(text)

        return {"error": f"Unknown action: {action}"}

    def _extract_numbers(self, text: str) -> List[float]:
        """Regex-based extraction (upgraded post-training if needed)."""
        matches = re.findall(r"-?\d+\.?\d*", str(text))
        result = []
        for m in matches:
            try:
                result.append(float(m))
            except ValueError:
                pass
        return result

    def _embed(self, text: str) -> List[float]:
        """Return LSTM final hidden state as list of floats."""
        vec = self.model.embed_sequence(text)
        return vec.tolist()

    # ── pretrain() ────────────────────────────────────────────────────────────

    def pretrain(
        self,
        corpus_file: str,
        epochs: int = 5,
        lr: float = 1e-3,
        seq_len: int = 64,
        device: str = "cpu",
    ) -> None:
        """Self-supervised next-char prediction on corpus_file."""
        with open(corpus_file, "r", encoding="utf-8") as f:
            text = f.read()

        model = self.model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        indices = [min(ord(c), model.vocab_size - 1) for c in text]
        n = len(indices) - 1
        if n < seq_len:
            return

        for epoch in range(epochs):
            total_loss = 0.0
            steps = 0
            for start in range(0, n - seq_len, seq_len):
                x = torch.tensor(
                    [indices[start : start + seq_len]], dtype=torch.long
                ).to(device)
                y = torch.tensor(
                    [indices[start + 1 : start + seq_len + 1]], dtype=torch.long
                ).to(device)
                opt.zero_grad()
                logits, _ = model(x)
                loss = loss_fn(logits.view(-1, model.vocab_size), y.view(-1))
                loss.backward()
                opt.step()
                total_loss += loss.item()
                steps += 1

            if steps > 0:
                self.last_loss = total_loss / steps

        self._pretrained = True
        self.model = model

    # ── HPMPattern interface ──────────────────────────────────────────────────

    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reward signal based on result non-emptiness."""
        # Note: In discovery, reward is passed in observations['reward']
        if "reward" in observations:
            return observations["reward"].to(self._device)
            
        result = observations.get("result", None)
        if result is None:
            return torch.tensor(-1.0, device=self._device)
        
        # Simple heuristic reward for active results
        if isinstance(result, list) and len(result) > 0:
            val = min(1.0, len(result) * 0.1)
        elif isinstance(result, str) and result:
            val = 0.5
        else:
            val = 0.0
        return torch.tensor(val, device=self._device)

    def intervene(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Wrap sample for intervention."""
        return self.sample(context)

    def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        """No-op: pretraining is offline. Online updates not supported."""
        pass

    def structural_distance(self, other: 'HPMPattern') -> float:
        """0.0 if same hidden_dim, 0.5 if different hidden_dim, 1.0 if not LM."""
        if not isinstance(other, LanguageModelPattern):
            return 1.0
        if self.hidden_dim != other.hidden_dim:
            return 0.5
        return 0.0

    def extract_causal_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_node(self.id, type="neural_lm", hidden_dim=self.hidden_dim)
        return g

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model weights and metadata."""
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hidden_dim": self.hidden_dim,
                "vocab_size": self.model.vocab_size,
                "embed_dim": self.model.embed_dim,
                "n_layers": self.model.n_layers,
                "last_loss": self.last_loss,
                "accuracy": self.accuracy,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model weights and metadata."""
        checkpoint = torch.load(path, map_location="cpu")
        self.hidden_dim = checkpoint["hidden_dim"]
        self.model = CharLevelLSTM(
            vocab_size=checkpoint["vocab_size"],
            embed_dim=checkpoint["embed_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            n_layers=checkpoint["n_layers"],
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.last_loss = checkpoint.get("last_loss", float("inf"))
        self.accuracy = checkpoint.get("accuracy", 0.0)
        self._pretrained = True
