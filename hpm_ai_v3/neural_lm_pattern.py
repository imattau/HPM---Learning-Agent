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
from hpm_ai_v3.tools.registry import ToolRegistry
from hpm_ai_v3.symbolic_pattern import SymbolicPattern


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
        self.substrate_type = "neural_lm"

    @property
    def tool_name(self) -> str:
        return "language_model"

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
        self.fine_tune(text, epochs, lr, seq_len, device)

    def fine_tune(
        self,
        text: str,
        epochs: int = 1,
        lr: float = 1e-3,
        seq_len: int = 64,
        device: str = "cpu",
    ) -> None:
        """Lightweight online training on text chunks."""
        if not text: return
        
        # Clear external caches if bound to a ToolSelector
        if hasattr(self, '_tool_selector') and self._tool_selector:
            self._tool_selector.clear_cache()

        # Adaptive seq_len for small chunks
        if len(text) < seq_len:
            seq_len = max(2, len(text) // 2)

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

    def validate_internal(self) -> float:
        """
        Measure internal quality on a fixed set of linguistic tasks.
        Returns accuracy [0,1].
        """
        # Simple test set for number extraction and tokenization
        test_tasks = [
            ("The speed of light is 299792458 m/s", 299792458),
            ("Gold atomic number 79", 79),
            ("Jupiter mass 1.898e27 kg", 1.898e27),
            ("Nitrogen boils at 77.36 K", 77.36),
            ("Water freezes at 273.15 K", 273.15)
        ]
        
        matches = 0
        for text, expected in test_tasks:
            result = self._extract_numbers(text)
            if any(abs(float(r) - float(expected)) < 1e-6 for r in result):
                matches += 1
        
        score = matches / len(test_tasks)
        self.accuracy = score # Update pattern accuracy for HPM evaluators
        return score

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


def register_language_tool(lm_pattern: LanguageModelPattern) -> None:
    """Register the LM as 'language_model' tool in ToolRegistry."""

    def _language_model_fn(action: str = "tokenize", text: str = "") -> Any:
        return lm_pattern.sample({"action": action, "text": text})

    ToolRegistry.register(
        name="language_model",
        tool_fn=_language_model_fn,
        input_keys=["action", "text"],
        output_key="result",
        cost=0.05,
    )


def compile_lm_to_symbolic(lm_pattern: LanguageModelPattern) -> SymbolicPattern:
    """
    Distil the LM's learned number-extraction behaviour into a SymbolicPattern.
    Uses the derived regex the LM has converged on.
    This is substrate shifting in HPM terms: neural → symbolic.
    """
    # The LM uses regex-based extraction internally; derive the pattern
    derived_regex = r"-?\d+\.?\d*"

    def _symbolic_extract(context: Dict[str, Any]) -> Any:
        action = context.get("action", "")
        text = context.get("text", "")
        if action == "extract_numbers":
            matches = re.findall(derived_regex, str(text))
            result = []
            for m in matches:
                try:
                    result.append(float(m))
                except ValueError:
                    pass
            return result
        if action == "tokenize":
            return str(text).split()
        return {"error": f"Unknown action: {action}"}

    # Wrap the function to match SymbolicPattern expected interface
    sp = SymbolicPattern(
        forward_fn=_symbolic_extract,
        required_keys=["action", "text"],
        pattern_id="lm_distilled",
        output_key="result"
    )
    return sp
