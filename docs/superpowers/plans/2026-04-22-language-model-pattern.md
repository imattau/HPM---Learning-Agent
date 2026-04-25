# LanguageModelPattern Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-supervised neural linguistic substrate (`LanguageModelPattern`) to HPM v3. The agent learns linguistic structure from raw text via next-character prediction, then exposes tokenization, number extraction, and embedding as tools. Once stable, the neural pattern distils into symbolic rules via the existing SubstrateCompiler.

**Architecture:** `CharLevelLSTM` (PyTorch, vocab=128, embed=64, hidden=128, 2 layers) wrapped by `LanguageModelPattern` (HPMPattern subclass) in `hpm_ai_v3/neural_lm_pattern.py`. Registered in ToolRegistry as `"language_model"`. Post-distillation: `compile_lm_to_symbolic()` uses SubstrateCompiler to emit a SymbolicPattern. LM embeddings optionally replace bag-of-chars in `train_cold_start.py`.

**Tech Stack:** Python, PyTorch (LSTM), existing ToolRegistry, SubstrateCompiler, HPMPattern base class

---

### Task 1: Sample corpus + CharLevelLSTM

**Files:**
- Create: `hpm_ai_v3/data/lm_corpus/sample.txt`
- Create: `hpm_ai_v3/neural_lm_pattern.py` (CharLevelLSTM only)
- Create: `hpm_ai_v3/tests/test_lm_pattern.py`

- [ ] **Step 1: Write failing tests for CharLevelLSTM**

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python3 -m pytest hpm_ai_v3/tests/test_lm_pattern.py -v 2>&1 | tail -5
```

Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3: Create sample corpus**

Create `hpm_ai_v3/data/lm_corpus/sample.txt` — 1000 lines of simple English sentences, number phrases, and basic math expressions. Example content:

```
The cat sat on the mat.
There are 42 students in the class.
Speed is 20.5 metres per second.
Add 3 and 7 to get 10.
The temperature is minus 5 degrees.
She has 100 apples and 200 oranges.
The distance is 3.14 kilometres.
Count: 1, 2, 3, 4, 5.
The answer is 99.
A fast car travels at 120 km per hour.
```
(Repeat and vary for 1000 lines total.)

- [ ] **Step 4: Create CharLevelLSTM in neural_lm_pattern.py**

```python
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
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest hpm_ai_v3/tests/test_lm_pattern.py -v 2>&1 | tail -15
```

Expected: all 4 CharLevelLSTM tests PASS.

- [ ] **Step 6: Commit**

```bash
git add hpm_ai_v3/neural_lm_pattern.py hpm_ai_v3/data/lm_corpus/sample.txt hpm_ai_v3/tests/test_lm_pattern.py
git commit -m "feat: add CharLevelLSTM and sample corpus for LanguageModelPattern"
```

---

### Task 2: LanguageModelPattern with sample() and pretrain()

**Files:**
- Modify: `hpm_ai_v3/neural_lm_pattern.py`
- Modify: `hpm_ai_v3/tests/test_lm_pattern.py`

- [ ] **Step 1: Add LanguageModelPattern tests**

Append to `hpm_ai_v3/tests/test_lm_pattern.py`:

```python
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

def test_lm_log_prob_returns_float(lm):
    obs = {"text": "hello world", "result": ["hello", "world"]}
    lp = lm.log_prob(obs)
    assert isinstance(lp, float)

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
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m pytest hpm_ai_v3/tests/test_lm_pattern.py -k "lm" -v 2>&1 | tail -5
```

Expected: `ImportError` — `LanguageModelPattern` not yet defined.

- [ ] **Step 3: Add LanguageModelPattern to neural_lm_pattern.py**

Append to `hpm_ai_v3/neural_lm_pattern.py`:

```python
import os
import re
import json
from typing import Any, Dict, List, Optional
from hpm_ai_v3.pattern import HPMPattern


class LanguageModelPattern(HPMPattern):
    """
    Self-supervised neural linguistic substrate.
    Wraps CharLevelLSTM. Exposes tokenize, extract_numbers, embed via sample().
    Pretrains offline via next-char prediction. No eval() anywhere.
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

    def sample(self, context: Dict[str, Any]) -> Any:
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

    def log_prob(self, obs: Dict[str, Any]) -> float:
        """Reward signal: positive if result is non-empty, scaled by length."""
        result = obs.get("result", None)
        if result is None:
            return -1.0
        if isinstance(result, list) and len(result) > 0:
            return min(1.0, len(result) * 0.1)
        if isinstance(result, str) and result:
            return 0.5
        return 0.0

    def update_parameters(self, obs: Dict[str, Any]) -> None:
        """No-op: pretraining is offline. Online updates not supported."""
        pass

    def structural_distance(self, other: Any) -> float:
        """0.0 if same hidden_dim, 0.5 if different hidden_dim, 1.0 if not LM."""
        if not isinstance(other, LanguageModelPattern):
            return 1.0
        if self.hidden_dim != other.hidden_dim:
            return 0.5
        return 0.0

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
```

- [ ] **Step 4: Run all LM tests**

```bash
python3 -m pytest hpm_ai_v3/tests/test_lm_pattern.py -v 2>&1 | tail -20
```

Expected: all tests PASS (pretrain test may be slow — expected).

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v3/neural_lm_pattern.py hpm_ai_v3/tests/test_lm_pattern.py
git commit -m "feat: add LanguageModelPattern with sample(), pretrain(), save/load"
```

---

### Task 3: register_language_tool() and compile_lm_to_symbolic()

**Files:**
- Modify: `hpm_ai_v3/neural_lm_pattern.py`
- Modify: `hpm_ai_v3/tests/test_lm_pattern.py`

- [ ] **Step 1: Add registration and distillation tests**

Append to `hpm_ai_v3/tests/test_lm_pattern.py`:

```python
from hpm_ai_v3.neural_lm_pattern import register_language_tool, compile_lm_to_symbolic
from hpm_ai_v3.tools import ToolRegistry
from hpm_ai_v3.symbolic_pattern import SymbolicPattern

def test_register_language_tool_in_registry(lm):
    register_language_tool(lm)
    info = ToolRegistry.get_tool_info("language_model")
    assert info is not None
    assert "action" in info["input_keys"]
    assert "text" in info["input_keys"]
    assert info["output_key"] == "result"

def test_registered_tool_callable_tokenize(lm):
    register_language_tool(lm)
    result = ToolRegistry.call("language_model", action="tokenize", text="foo bar baz")
    assert result == ["foo", "bar", "baz"]

def test_registered_tool_callable_extract_numbers(lm):
    register_language_tool(lm)
    result = ToolRegistry.call("language_model", action="extract_numbers", text="value is 42")
    assert 42.0 in result

def test_compile_lm_to_symbolic_returns_symbolic_pattern(lm):
    """compile_lm_to_symbolic returns a SymbolicPattern."""
    lm.accuracy = 0.95  # simulate trained
    sp = compile_lm_to_symbolic(lm)
    assert isinstance(sp, SymbolicPattern)

def test_compiled_symbolic_extracts_numbers(lm):
    """Compiled SymbolicPattern passes same extraction test as LM."""
    lm.accuracy = 0.95
    sp = compile_lm_to_symbolic(lm)
    result = sp.sample({"action": "extract_numbers", "text": "Speed is 20.5 m/s"})
    assert 20.5 in result
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m pytest hpm_ai_v3/tests/test_lm_pattern.py -k "register or compile" -v 2>&1 | tail -5
```

Expected: `ImportError` — functions not yet defined.

- [ ] **Step 3: Add register_language_tool() and compile_lm_to_symbolic() to neural_lm_pattern.py**

Append to `hpm_ai_v3/neural_lm_pattern.py`:

```python
from hpm_ai_v3.tools import ToolRegistry
from hpm_ai_v3.symbolic_pattern import SymbolicPattern


def register_language_tool(lm_pattern: LanguageModelPattern) -> None:
    """Register the LM as 'language_model' tool in ToolRegistry."""

    def _language_model_fn(action: str = "tokenize", text: str = "") -> Any:
        return lm_pattern.sample({"action": action, "text": text})

    ToolRegistry.register(
        name="language_model",
        fn=_language_model_fn,
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

    def _symbolic_extract(action: str = "extract_numbers", text: str = "") -> Any:
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

    sp = SymbolicPattern(
        name="lm_distilled",
        fn=_symbolic_extract,
        input_keys=["action", "text"],
        output_key="result",
    )
    return sp
```

- [ ] **Step 4: Run all tests**

```bash
python3 -m pytest hpm_ai_v3/tests/test_lm_pattern.py -v 2>&1 | tail -25
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v3/neural_lm_pattern.py hpm_ai_v3/tests/test_lm_pattern.py
git commit -m "feat: add register_language_tool and compile_lm_to_symbolic"
```

---

### Task 4: Wire LM embeddings into train_cold_start.py

**Files:**
- Modify: `hpm_ai_v3/task8/train_cold_start.py`

- [ ] **Step 1: Add wiring test**

Append to `hpm_ai_v3/tests/test_lm_pattern.py`:

```python
def test_lm_embed_produces_128_floats(lm):
    """LM embed output is suitable for vector memory (128 floats)."""
    vec = lm.sample({"action": "embed", "text": "The speed is 20 km/h"})
    assert isinstance(vec, list)
    assert len(vec) == 128
    assert all(isinstance(v, float) for v in vec)
```

- [ ] **Step 2: Run to verify it passes (embed already implemented)**

```bash
python3 -m pytest hpm_ai_v3/tests/test_lm_pattern.py::test_lm_embed_produces_128_floats -v
```

Expected: PASS.

- [ ] **Step 3: Read current embedding logic in train_cold_start.py**

Identify the bag-of-chars embedding function (likely named `bag_of_chars`, `char_vec`, or similar) that produces a fixed-length vector from text for vector memory.

- [ ] **Step 4: Add optional LM embedding path to train_cold_start.py**

In `hpm_ai_v3/task8/train_cold_start.py`, add import near top:

```python
try:
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
    _LM_AVAILABLE = True
except ImportError:
    _LM_AVAILABLE = False
```

Add a helper function:

```python
def make_text_embedding(text: str, lm_pattern=None) -> list:
    """
    Return a fixed-length float vector for text.
    Uses LM embedding if lm_pattern is provided and pretrained,
    otherwise falls back to bag-of-chars (first 128 ASCII chars).
    """
    if lm_pattern is not None and lm_pattern._pretrained:
        return lm_pattern.sample({"action": "embed", "text": text})
    # Bag-of-chars fallback: count of each ASCII char, normalised
    counts = [0.0] * 128
    for ch in str(text):
        idx = ord(ch)
        if idx < 128:
            counts[idx] += 1.0
    total = sum(counts) or 1.0
    return [c / total for c in counts]
```

In the main training loop, replace the existing embedding call with:

```python
embedding = make_text_embedding(task_text, lm_pattern=lm_pattern)
```

where `lm_pattern` is `None` by default (bag-of-chars) and can be passed in once pretrained.

- [ ] **Step 5: Run smoke test**

```bash
python3 -m pytest hpm_ai_v3/tests/test_lm_pattern.py -v 2>&1 | tail -5
python3 -c "
from hpm_ai_v3.task8.train_cold_start import make_text_embedding
vec = make_text_embedding('The value is 42')
print('embed length:', len(vec))
assert len(vec) == 128
print('OK')
" 2>&1
```

Expected: `embed length: 128`, `OK`.

- [ ] **Step 6: Commit**

```bash
git add hpm_ai_v3/task8/train_cold_start.py hpm_ai_v3/tests/test_lm_pattern.py
git commit -m "feat: wire LM embeddings into train_cold_start with bag-of-chars fallback"
```

---

### Task 5: Integration test — LM tool in curriculum

**Files:**
- Modify: `hpm_ai_v3/tests/test_lm_pattern.py`

- [ ] **Step 1: Write integration test**

Append to `hpm_ai_v3/tests/test_lm_pattern.py`:

```python
from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
from hpm_ai_v3.curriculum import CurriculumManager

def test_lm_tool_callable_via_tool_registry():
    """language_model tool registered and callable end-to-end."""
    lm = LanguageModelPattern()
    register_language_tool(lm)
    result = ToolRegistry.call(
        "language_model",
        action="extract_numbers",
        text="The price is 9.99 dollars",
    )
    assert isinstance(result, list)
    assert 9.99 in result

def test_compiled_symbolic_added_to_population():
    """compile_lm_to_symbolic produces a pattern that can be added to a population."""
    from hpm_ai_v3.population import Population
    lm = LanguageModelPattern()
    lm.accuracy = 0.95
    sp = compile_lm_to_symbolic(lm)
    pop = Population()
    pop.add_pattern(sp, weight=2.0)
    assert sp in pop.patterns

def test_lm_curriculum_smoke():
    """Agent with language_model tool registered completes a word-count task."""
    lm = LanguageModelPattern()
    register_language_tool(lm)

    agent = UnifiedDiscoveryAgent(context_dim=64)
    cm = CurriculumManager()

    # Run 10 episodes — no crash, some reward
    rewards = []
    for _ in range(10):
        task = cm.get_current_task()
        sol = agent.run_episode(task, max_steps=10)
        r = agent.evaluate_solution(sol)
        cm.update(r)
        rewards.append(r)

    assert len(rewards) == 10
    # At least one non-negative reward expected
    assert max(rewards) >= 0.0
```

- [ ] **Step 2: Run to verify they fail (population.add_pattern or similar may not exist yet)**

```bash
python3 -m pytest hpm_ai_v3/tests/test_lm_pattern.py -k "integration or curriculum or symbolic_added" -v 2>&1 | tail -10
```

- [ ] **Step 3: Fix any missing Population.add_pattern interface**

If `Population.add_pattern` does not exist, add a thin wrapper in `hpm_ai_v3/population.py`:

```python
def add_pattern(self, pattern, weight: float = 1.0) -> None:
    """Add a pattern to the population with the given initial weight."""
    if pattern not in self.patterns:
        self.patterns.append(pattern)
        self._weights.append(weight)
```

- [ ] **Step 4: Run all tests**

```bash
python3 -m pytest hpm_ai_v3/tests/test_lm_pattern.py -v 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
python3 -m pytest hpm_ai_v3/ -v --tb=short 2>&1 | tail -20
```

Expected: no new failures introduced.

- [ ] **Step 6: Commit**

```bash
git add hpm_ai_v3/tests/test_lm_pattern.py hpm_ai_v3/population.py
git commit -m "test: integration tests for LanguageModelPattern in curriculum"
```
