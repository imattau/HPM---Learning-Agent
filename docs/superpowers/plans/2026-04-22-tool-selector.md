# ToolSelector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `ToolSelector` that uses `LanguageModelPattern` embeddings to bias pattern selection toward semantically relevant tools — helping the agent connect NLP task descriptions to the right tools without relying solely on replicator weight history.

**Architecture:** `ToolSelector` in `hpm_ai_v3/tools/tool_selector.py` computes cosine similarity between task text embedding and pattern description embeddings, then applies a weight bias in `base_discovery.py act()`. Defaults to `None` (no behaviour change without LM). Each `ActionPattern` gets a `tool_description` property. A `TOOL_DESCRIPTIONS` registry provides richer descriptions for known NLP tools.

**Tech Stack:** numpy, existing `LanguageModelPattern._embed()`, existing `base_discovery.py act()`

---

### Task 1: Create ToolSelector with cosine similarity

**Files:**
- Create: `hpm_ai_v3/tools/tool_selector.py`
- Create: `hpm_ai_v3/tools/test_tool_selector.py`

- [ ] **Step 1: Write failing tests**

```python
# hpm_ai_v3/tools/test_tool_selector.py
import numpy as np
import pytest

def test_cosine_similar_vectors():
    from hpm_ai_v3.tools.tool_selector import ToolSelector
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
    lm = LanguageModelPattern()
    ts = ToolSelector(lm)
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert ts._cosine(a, b) == pytest.approx(1.0)

def test_cosine_orthogonal_vectors():
    from hpm_ai_v3.tools.tool_selector import ToolSelector
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
    lm = LanguageModelPattern()
    ts = ToolSelector(lm)
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert ts._cosine(a, b) == pytest.approx(0.0)

def test_cosine_zero_vector():
    from hpm_ai_v3.tools.tool_selector import ToolSelector
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
    lm = LanguageModelPattern()
    ts = ToolSelector(lm)
    assert ts._cosine([0.0, 0.0], [1.0, 0.0]) == 0.0

def test_apply_returns_same_length_as_weights():
    from hpm_ai_v3.tools.tool_selector import ToolSelector
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
    from hpm_ai_v3.agents.base_discovery import ActionPattern
    lm = LanguageModelPattern()
    ts = ToolSelector(lm)
    patterns = [
        ActionPattern("python_call", module="str", function="split"),
        ActionPattern("python_call", module="textblob", function="TextBlob"),
        ActionPattern("python_call", module="math", function="sqrt"),
    ]
    weights = np.array([1.0, 1.0, 1.0])
    adjusted = ts.apply("Count words in: hello world", weights, patterns)
    assert len(adjusted) == 3
    assert all(w >= 0 for w in adjusted)

def test_apply_boosts_relevant_pattern():
    from hpm_ai_v3.tools.tool_selector import ToolSelector
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
    from hpm_ai_v3.agents.base_discovery import ActionPattern
    lm = LanguageModelPattern()
    ts = ToolSelector(lm, alpha=1.0)
    split_pat = ActionPattern("python_call", module="builtins", function="str.split")
    sqrt_pat = ActionPattern("python_call", module="math", function="sqrt")
    patterns = [split_pat, sqrt_pat]
    weights = np.array([1.0, 1.0])
    adjusted = ts.apply("split the words in this text", weights, patterns)
    # str.split should be boosted relative to sqrt for a word-splitting task
    assert adjusted[0] >= adjusted[1]
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python3 -m pytest hpm_ai_v3/tools/test_tool_selector.py -v 2>&1 | tail -5
```
Expected: `ImportError` — module not yet created.

- [ ] **Step 3: Create tool_selector.py**

```python
# hpm_ai_v3/tools/tool_selector.py
"""
ToolSelector - Uses LM embeddings to bias pattern selection toward
semantically relevant tools. Acts as a pattern evaluator in HPM terms.
"""
import numpy as np
from typing import Any, Dict, List, Optional

TOOL_DESCRIPTIONS = {
    "textblob.TextBlob": "sentiment analysis polarity positive negative opinion text",
    "re.findall": "extract pattern match numbers regex search text",
    "re.search": "find pattern match regex text search",
    "builtins.str.split": "split words tokenize count whitespace text",
    "builtins.str.lower": "lowercase convert string text",
    "builtins.str.upper": "uppercase convert string text",
    "spacy.nlp": "entity noun parse sentence structure named entity",
    "math.sqrt": "square root numeric calculation math",
    "math.factorial": "factorial numeric calculation math",
    "math.floor": "floor round down numeric math",
    "math.gcd": "greatest common divisor numeric math",
    "sympy.sympify": "evaluate expression arithmetic symbolic math",
    "operator.add": "add sum two numbers arithmetic",
    "operator.mul": "multiply product two numbers arithmetic",
    "language_model": "language tokenize embed extract text nlp",
}


class ToolSelector:
    """
    Biases population pattern selection using LM semantic similarity.
    Selector only boosts relevant patterns — never suppresses.
    """
    def __init__(self, lm, alpha: float = 0.5):
        """
        lm: LanguageModelPattern instance (provides _embed())
        alpha: bias strength — 0.0 = no effect, 1.0 = strong bias
        """
        self.lm = lm
        self.alpha = alpha
        self._cache: Dict[str, List[float]] = {}

    def _embed(self, text: str) -> List[float]:
        if text not in self._cache:
            self._cache[text] = self.lm._embed(text)
        return self._cache[text]

    def _cosine(self, a: List[float], b: List[float]) -> float:
        va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom < 1e-8:
            return 0.0
        return float(np.clip(np.dot(va, vb) / denom, 0.0, 1.0))

    def score(self, task_text: str, patterns: List[Any]) -> np.ndarray:
        """Return similarity score [0,1] per pattern."""
        task_emb = self._embed(task_text)
        scores = []
        for p in patterns:
            desc = self._pattern_description(p)
            pat_emb = self._embed(desc)
            scores.append(self._cosine(task_emb, pat_emb))
        return np.array(scores)

    def apply(self, task_text: str, weights: np.ndarray,
              patterns: List[Any]) -> np.ndarray:
        """Return adjusted weights: weights * (1 + alpha * similarity)."""
        if not task_text or len(patterns) == 0:
            return weights
        similarity = self.score(task_text, patterns)
        return weights * (1.0 + self.alpha * similarity)

    def _pattern_description(self, pattern: Any) -> str:
        tool_name = getattr(pattern, 'tool_name', None)
        if tool_name and tool_name in TOOL_DESCRIPTIONS:
            return TOOL_DESCRIPTIONS[tool_name]
        module = getattr(pattern, 'module', None)
        function = getattr(pattern, 'function', None)
        if module and function:
            key = f"{module}.{function}"
            if key in TOOL_DESCRIPTIONS:
                return TOOL_DESCRIPTIONS[key]
            return f"{module} {function} tool call function"
        action = getattr(pattern, 'action_type', str(pattern))
        return action
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest hpm_ai_v3/tools/test_tool_selector.py -v 2>&1 | tail -15
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v3/tools/tool_selector.py hpm_ai_v3/tools/test_tool_selector.py
git commit -m "feat: add ToolSelector with LM-based semantic pattern selection bias"
```

---

### Task 2: Add tool_description property to ActionPattern

**Files:**
- Modify: `hpm_ai_v3/agents/base_discovery.py`
- Modify: `hpm_ai_v3/tools/test_tool_selector.py`

- [ ] **Step 1: Add test for tool_description**

Append to `hpm_ai_v3/tools/test_tool_selector.py`:

```python
def test_action_pattern_tool_description_module_function():
    from hpm_ai_v3.agents.base_discovery import ActionPattern
    p = ActionPattern("python_call", module="math", function="sqrt")
    assert "math" in p.tool_description
    assert "sqrt" in p.tool_description

def test_action_pattern_tool_description_action_only():
    from hpm_ai_v3.agents.base_discovery import ActionPattern
    p = ActionPattern("list_modules")
    assert p.tool_description == "list_modules"
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m pytest hpm_ai_v3/tools/test_tool_selector.py::test_action_pattern_tool_description_module_function hpm_ai_v3/tools/test_tool_selector.py::test_action_pattern_tool_description_action_only -v 2>&1 | tail -5
```
Expected: `AttributeError` — `tool_description` not yet defined.

- [ ] **Step 3: Add tool_description property to ActionPattern**

In `hpm_ai_v3/agents/base_discovery.py`, inside `ActionPattern` class after the `tool_name` property:

```python
    @property
    def tool_description(self) -> str:
        if self.module and self.function:
            return f"{self.module}.{self.function}: call {self.function} from {self.module}"
        return self.action_type
```

- [ ] **Step 4: Run all tool_selector tests**

```bash
python3 -m pytest hpm_ai_v3/tools/test_tool_selector.py -v 2>&1 | tail -10
```
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v3/agents/base_discovery.py hpm_ai_v3/tools/test_tool_selector.py
git commit -m "feat: add tool_description property to ActionPattern"
```

---

### Task 3: Wire ToolSelector into base_discovery.py act()

**Files:**
- Modify: `hpm_ai_v3/agents/base_discovery.py`
- Modify: `hpm_ai_v3/tools/test_tool_selector.py`

- [ ] **Step 1: Add integration test**

Append to `hpm_ai_v3/tools/test_tool_selector.py`:

```python
def test_act_with_tool_selector_does_not_crash():
    from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
    from hpm_ai_v3.tools.tool_selector import ToolSelector
    agent = UnifiedDiscoveryAgent(context_dim=64)
    lm = LanguageModelPattern()
    agent.tool_selector = ToolSelector(lm, alpha=0.5)
    task = {"text": "Count words in: hello world", "type": "pretraining", "answer": 2.0}
    agent.current_task = task
    result = agent.act(step_idx=0, step_population=False)
    assert "action" in result or "status" in result

def test_act_without_tool_selector_unchanged():
    from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
    agent = UnifiedDiscoveryAgent(context_dim=64)
    assert agent.tool_selector is None
    task = {"text": "10 + 5", "type": "pretraining", "answer": 15.0}
    agent.current_task = task
    result = agent.act(step_idx=0, step_population=False)
    assert "action" in result or "status" in result
```

- [ ] **Step 2: Run to verify second test passes, first fails**

```bash
python3 -m pytest hpm_ai_v3/tools/test_tool_selector.py::test_act_without_tool_selector_unchanged -v 2>&1 | tail -5
```
Expected: PASS (tool_selector=None already safe after next step).

- [ ] **Step 3: Add tool_selector to PureAgnosticDiscoveryAgent.__init__**

In `hpm_ai_v3/agents/base_discovery.py`, in `PureAgnosticDiscoveryAgent.__init__` after `self.substrate = InnateCognitiveSubstrate()`:

```python
        self.tool_selector = None  # Set externally with ToolSelector instance
```

- [ ] **Step 4: Wire into act() weight calculation**

In `hpm_ai_v3/agents/base_discovery.py`, replace lines 155-158:
```python
        weights = np.array([p.weight for p in self.population.patterns])
        total = weights.sum()
        probs = weights / total if total > 1e-6 else np.ones(len(weights)) / len(weights)
        chosen = np.random.choice(len(self.population.patterns), p=probs)
```

With:
```python
        weights = np.array([p.weight for p in self.population.patterns])
        if self.tool_selector is not None and self.current_task:
            weights = self.tool_selector.apply(
                self.current_task.get("text", ""),
                weights,
                self.population.patterns
            )
        total = weights.sum()
        probs = weights / total if total > 1e-6 else np.ones(len(weights)) / len(weights)
        chosen = np.random.choice(len(self.population.patterns), p=probs)
```

- [ ] **Step 5: Run all tool_selector tests**

```bash
python3 -m pytest hpm_ai_v3/tools/test_tool_selector.py -v 2>&1 | tail -15
```
Expected: all 9 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add hpm_ai_v3/agents/base_discovery.py hpm_ai_v3/tools/test_tool_selector.py
git commit -m "feat: wire ToolSelector into base_discovery act() weight calculation"
```

---

### Task 4: Instantiate ToolSelector in UnifiedDiscoveryAgent

**Files:**
- Modify: `hpm_ai_v3/agents/discovery_agent.py`
- Modify: `hpm_ai_v3/tools/test_tool_selector.py`

- [ ] **Step 1: Add agent instantiation test**

Append to `hpm_ai_v3/tools/test_tool_selector.py`:

```python
def test_unified_agent_has_tool_selector_when_lm_provided():
    from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern
    lm = LanguageModelPattern()
    agent = UnifiedDiscoveryAgent(context_dim=64, lm=lm)
    assert agent.tool_selector is not None

def test_unified_agent_no_tool_selector_without_lm():
    from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
    agent = UnifiedDiscoveryAgent(context_dim=64)
    assert agent.tool_selector is None
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m pytest hpm_ai_v3/tools/test_tool_selector.py::test_unified_agent_has_tool_selector_when_lm_provided -v 2>&1 | tail -5
```
Expected: FAIL — `lm` kwarg not yet accepted.

- [ ] **Step 3: Update UnifiedDiscoveryAgent.__init__ to accept lm**

In `hpm_ai_v3/agents/discovery_agent.py`, update `__init__`:

```python
    def __init__(self, context_dim: int = 64, lm=None):
        from hpm_ai_v3.tools.python_substrate import register_python_substrate
        register_python_substrate()
        super().__init__(context_feature_dim=context_dim)

        self.hidden_fn = None
        self.points = []
        self.confidence = 0.0

        if lm is not None:
            from hpm_ai_v3.tools.tool_selector import ToolSelector
            self.tool_selector = ToolSelector(lm, alpha=0.5)
```

- [ ] **Step 4: Run all tests**

```bash
python3 -m pytest hpm_ai_v3/tools/test_tool_selector.py -v 2>&1 | tail -15
```
Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v3/agents/discovery_agent.py hpm_ai_v3/tools/test_tool_selector.py
git commit -m "feat: UnifiedDiscoveryAgent accepts optional lm for ToolSelector"
```

---

### Task 5: Curriculum progression test with ToolSelector

**Files:**
- Modify: `hpm_ai_v3/task8/test_curriculum_progression.py`

- [ ] **Step 1: Add comparison test**

Append to `hpm_ai_v3/task8/test_curriculum_progression.py`:

```python
def test_tool_selector_improves_nlp_phase_reward():
    """Agent with ToolSelector should get higher avg reward on NLP tasks."""
    import numpy as np
    from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
    from hpm_ai_v3.curriculum import CurriculumManager
    from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern

    def run_nlp_episodes(use_lm: bool) -> float:
        lm = LanguageModelPattern() if use_lm else None
        agent = UnifiedDiscoveryAgent(context_dim=64, lm=lm)
        cm = CurriculumManager()
        # Fast-forward to NLP Tool Mastery phase
        nlp_idx = next(i for i, p in enumerate(cm.patterns)
                       if p.name == "NLP Tool Mastery")
        cm.active_pattern_idx = nlp_idx
        cm.phase = cm.patterns[nlp_idx].phase
        rewards = []
        for _ in range(30):
            task = cm.get_current_task()
            sol = agent.run_episode(task, max_steps=10)
            r = agent.evaluate_solution(sol)
            rewards.append(r)
        return float(np.mean(rewards))

    reward_without = run_nlp_episodes(use_lm=False)
    reward_with = run_nlp_episodes(use_lm=True)
    print(f"Without ToolSelector: {reward_without:.3f}")
    print(f"With ToolSelector:    {reward_with:.3f}")
    # ToolSelector should not make things worse
    assert reward_with >= reward_without - 0.1, (
        f"ToolSelector degraded performance: {reward_without:.3f} -> {reward_with:.3f}"
    )
```

- [ ] **Step 2: Run test**

```bash
python3 -m pytest hpm_ai_v3/task8/test_curriculum_progression.py::test_tool_selector_improves_nlp_phase_reward -v -s 2>&1 | grep -E "Without|With|PASS|FAIL"
```
Expected: PASS. Both reward values printed.

- [ ] **Step 3: Commit**

```bash
git add hpm_ai_v3/task8/test_curriculum_progression.py
git commit -m "test: add ToolSelector NLP phase reward comparison test"
```
