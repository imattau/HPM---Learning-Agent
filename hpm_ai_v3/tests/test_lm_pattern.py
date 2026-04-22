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

from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
import pathlib

CORPUS = pathlib.Path(__file__).parent.parent / "data" / "lm_corpus" / "sample.txt"

@pytest.fixture
def lm():
    return LanguageModelPattern()

def test_lm_sample_tokenize(lm):
    result = lm.sample({"action": "tokenize", "text": "hello world foo"})
    assert result == ["hello", "world", "foo"]

def test_lm_sample_extract_numbers_regex_fallback(lm):
    """Before pretrain, regex fallback returns numbers."""
    result = lm.sample({"action": "extract_numbers", "text": "Speed is 20.5 m/s"})
    assert 20.5 in result

def test_lm_sample_embed_shape(lm):
    result = lm.sample({"action": "embed", "text": "hello"})
    assert isinstance(result, list)
    assert len(result) == 128

def test_lm_log_prob_returns_tensor(lm):
    obs = {"text": "hello world", "result": ["hello", "world"]}
    lp = lm.log_prob(obs)
    assert isinstance(lp, torch.Tensor)
    assert lp.item() > 0

def test_lm_structural_distance_same(lm):
    other = LanguageModelPattern()
    assert lm.structural_distance(other) == 0.0

def test_lm_structural_distance_different_hidden():
    a = LanguageModelPattern(hidden_dim=128)
    b = LanguageModelPattern(hidden_dim=64)
    assert a.structural_distance(b) == 0.5

def test_lm_structural_distance_non_lm(lm):
    class Dummy:
        pass
    assert lm.structural_distance(Dummy()) == 1.0

def test_lm_pretrain_reduces_loss(lm, tmp_path):
    """pretrain() for 3 epochs on sample corpus reduces loss below 3.5."""
    if not CORPUS.exists():
        pytest.skip("corpus not found")
    lm.pretrain(str(CORPUS), epochs=3)
    assert lm.last_loss < 3.5

def test_lm_save_load_roundtrip(lm, tmp_path):
    checkpoint = str(tmp_path / "lm.pt")
    lm.pretrain(str(CORPUS), epochs=1)
    lm.save(checkpoint)
    lm2 = LanguageModelPattern()
    lm2.load(checkpoint)
    r1 = lm.sample({"action": "embed", "text": "test"})
    r2 = lm2.sample({"action": "embed", "text": "test"})
    assert r1 == r2
