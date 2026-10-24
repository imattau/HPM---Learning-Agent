"""A compact NumPy feed-forward neural n-gram language model.

This is a deliberately small, train-from-scratch baseline for comparing the
count-based language pilot on identical documents and vocabulary.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np

from hpm_ai_v2.training.language_prediction import BOUNDARY, UNK, split_sentences, tokenize


class NeuralNGramBaseline:
    def __init__(self, *, context_size: int = 2, embedding_dim: int = 24,
                 hidden_dim: int = 48, min_word_count: int = 2, seed: int = 17,
                 batch_size: int = 128, learning_rate: float = 0.002):
        if context_size < 1 or embedding_dim < 1 or hidden_dim < 1 or min_word_count < 1:
            raise ValueError("model sizes and min_word_count must be positive")
        if batch_size < 1 or learning_rate <= 0:
            raise ValueError("batch_size and learning_rate must be positive")
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.min_word_count = min_word_count
        self.seed = seed
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.word_counts: Counter[str] = Counter()
        self.vocab: list[str] = []
        self.index: dict[str, int] = {}
        self.parameters: dict[str, np.ndarray] = {}
        self.training_history: list[dict] = []
        self.best_epoch: int | None = None

    def fit(self, documents: Iterable[str], validation_documents: Iterable[str] = (), *,
            epochs: int = 2) -> "NeuralNGramBaseline":
        if epochs < 1:
            raise ValueError("epochs must be positive")
        train = list(documents)
        validation = list(validation_documents)
        raw_sequences = [tokenize(sentence) + [BOUNDARY]
                         for text in train for sentence in split_sentences(text)]
        self.word_counts = Counter(token for sequence in raw_sequences for token in sequence
                                   if token != BOUNDARY)
        self.vocab = [BOUNDARY, UNK] + sorted(
            (word for word, count in self.word_counts.items()
             if count >= self.min_word_count and word not in (BOUNDARY, UNK)),
            key=lambda word: (-self.word_counts[word], word),
        )
        self.index = {word: i for i, word in enumerate(self.vocab)}
        samples = [self._encode_sequence(sequence) for sequence in raw_sequences]
        if not samples:
            raise ValueError("training documents contain no tokens")
        x = np.concatenate([sample[0] for sample in samples], axis=0)
        y = np.concatenate([sample[1] for sample in samples], axis=0)
        self._initialize_parameters()
        rng = np.random.default_rng(self.seed)
        first_moments = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        second_moments = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        step = 0
        best_validation = math.inf
        best_parameters = None
        stale_epochs = 0
        for epoch in range(1, epochs + 1):
            permutation = rng.permutation(len(y))
            epoch_loss = 0.0
            for start in range(0, len(y), self.batch_size):
                rows = permutation[start:start + self.batch_size]
                batch_x, batch_y = x[rows], y[rows]
                logits, cache = self._forward(batch_x)
                logits -= logits.max(axis=1, keepdims=True)
                probs = np.exp(logits)
                probs /= probs.sum(axis=1, keepdims=True)
                epoch_loss -= float(np.log(np.maximum(probs[np.arange(len(rows)), batch_y], 1e-12)).sum())
                probs[np.arange(len(rows)), batch_y] -= 1.0
                probs /= len(rows)
                gradients = self._backward(batch_x, probs, cache)
                step += 1
                beta1, beta2 = 0.9, 0.999
                for key, grad in gradients.items():
                    first_moments[key] = beta1 * first_moments[key] + (1 - beta1) * grad
                    second_moments[key] = beta2 * second_moments[key] + (1 - beta2) * grad * grad
                    mhat = first_moments[key] / (1 - beta1 ** step)
                    vhat = second_moments[key] / (1 - beta2 ** step)
                    self.parameters[key] -= self.learning_rate * mhat / (np.sqrt(vhat) + 1e-8)
            row = {"epoch": epoch, "training_cross_entropy_nats": epoch_loss / len(y)}
            if validation:
                val_loss = self.score_cross_entropy(validation)["cross_entropy_nats"]
                row["validation_cross_entropy_nats"] = val_loss
                if val_loss < best_validation:
                    best_validation = val_loss
                    self.best_epoch = epoch
                    best_parameters = {key: value.copy() for key, value in self.parameters.items()}
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            self.training_history.append(row)
            if validation and stale_epochs >= 1:
                break
        if best_parameters is not None:
            self.parameters = best_parameters
        return self

    def _map(self, token: str) -> int:
        if token == BOUNDARY:
            return self.index[BOUNDARY]
        if token == UNK or self.word_counts.get(token, 0) < self.min_word_count:
            return self.index[UNK]
        return self.index.get(token, self.index[UNK])

    def _encode_sequence(self, sequence: list[str]) -> tuple[np.ndarray, np.ndarray]:
        encoded = [self._map(token) for token in sequence]
        padded = [self.index[BOUNDARY]] * self.context_size + encoded
        x = np.asarray([
            padded[index - self.context_size:index]
            for index in range(self.context_size, len(padded))
        ], dtype=np.int32)
        y = np.asarray(padded[self.context_size:], dtype=np.int32)
        return x, y

    def _initialize_parameters(self) -> None:
        rng = np.random.default_rng(self.seed)
        v, e, h = len(self.vocab), self.embedding_dim, self.hidden_dim
        scale = 1.0 / math.sqrt(e * self.context_size)
        self.parameters = {
            "embedding": rng.normal(0, .08, size=(v, e)).astype(np.float32),
            "hidden": rng.normal(0, scale, size=(self.context_size * e, h)).astype(np.float32),
            "hidden_bias": np.zeros(h, dtype=np.float32),
            "output": rng.normal(0, 1 / math.sqrt(h), size=(h, v)).astype(np.float32),
            "output_bias": np.zeros(v, dtype=np.float32),
        }

    def _forward(self, x: np.ndarray):
        p = self.parameters
        embedded = p["embedding"][x]
        flat = embedded.reshape(len(x), -1)
        pre = flat @ p["hidden"] + p["hidden_bias"]
        hidden = np.tanh(pre)
        logits = hidden @ p["output"] + p["output_bias"]
        return logits, (x, embedded, flat, hidden)

    def _backward(self, x: np.ndarray, grad_logits: np.ndarray, cache) -> dict[str, np.ndarray]:
        _x, embedded, flat, hidden = cache
        p = self.parameters
        grad_output = hidden.T @ grad_logits
        grad_output_bias = grad_logits.sum(axis=0)
        grad_hidden = grad_logits @ p["output"].T
        grad_pre = grad_hidden * (1 - hidden * hidden)
        grad_hidden_weights = flat.T @ grad_pre
        grad_hidden_bias = grad_pre.sum(axis=0)
        grad_flat = grad_pre @ p["hidden"].T
        grad_embedded = grad_flat.reshape(embedded.shape)
        grad_embedding = np.zeros_like(p["embedding"])
        np.add.at(grad_embedding, x.reshape(-1), grad_embedded.reshape(-1, self.embedding_dim))
        return {"embedding": grad_embedding, "hidden": grad_hidden_weights,
                "hidden_bias": grad_hidden_bias, "output": grad_output,
                "output_bias": grad_output_bias}

    def score_cross_entropy(self, documents: Iterable[str]) -> dict:
        if not self.parameters:
            raise RuntimeError("fit() must be called before scoring")
        nll = token_count = 0
        for text in documents:
            sequences = [tokenize(sentence) + [BOUNDARY] for sentence in split_sentences(text)]
            for sequence in sequences:
                x, y = self._encode_sequence(sequence)
                for start in range(0, len(y), self.batch_size):
                    batch_x, batch_y = x[start:start + self.batch_size], y[start:start + self.batch_size]
                    logits, _ = self._forward(batch_x)
                    logits -= logits.max(axis=1, keepdims=True)
                    log_norm = np.log(np.exp(logits).sum(axis=1))
                    nll += float((log_norm - logits[np.arange(len(batch_y)), batch_y]).sum())
                    token_count += len(batch_y)
        mean = nll / max(token_count, 1)
        return {"tokens": token_count, "cross_entropy_nats": mean, "perplexity": math.exp(mean),
                "vocabulary_size": len(self.vocab), "context_size": self.context_size,
                "embedding_dim": self.embedding_dim, "hidden_dim": self.hidden_dim,
                "min_word_count": self.min_word_count}
