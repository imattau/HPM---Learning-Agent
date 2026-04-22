# hpm_ai_v3/tests/test_lm_pattern.py
import pytest
import torch
from hpm_ai_v3.neural_lm_pattern import CharLevelLSTM

@pytest.fixture
def model():
    return CharLevelLSTM(vocab_size=128, embed_dim=64, hidden_dim=128, n_layers=2)

def test_char_lstm_forward_shape(model):
    """Forward pass produces logits of shape (batch, seq, vocab)."""
    x = torch.randint(0, 128, (2, 10))
    logits, hidden = model(x)
    assert logits.shape == (2, 10, 128)

def test_char_lstm_hidden_shape(model):
    """Hidden state has expected shape (n_layers, batch, hidden_dim)."""
    x = torch.randint(0, 128, (1, 5))
    _, (h, c) = model(x)
    assert h.shape == (2, 1, 128)
    assert c.shape == (2, 1, 128)

def test_char_lstm_embed_shape(model):
    """embed_sequence returns tensor of shape (hidden_dim,) for a string."""
    vec = model.embed_sequence("hello world")
    assert vec.shape == (128,)

def test_char_lstm_trains_one_step(model):
    """A single gradient step reduces the loss."""
    import torch.nn as nn
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randint(0, 128, (1, 20))
    targets = torch.randint(0, 128, (1, 20))
    logits, _ = model(x)
    loss_before = nn.CrossEntropyLoss()(logits.view(-1, 128), targets.view(-1)).item()
    for _ in range(5):
        opt.zero_grad()
        logits, _ = model(x)
        loss = nn.CrossEntropyLoss()(logits.view(-1, 128), targets.view(-1))
        loss.backward()
        opt.step()
    logits, _ = model(x)
    loss_after = nn.CrossEntropyLoss()(logits.view(-1, 128), targets.view(-1)).item()
    assert loss_after < loss_before
