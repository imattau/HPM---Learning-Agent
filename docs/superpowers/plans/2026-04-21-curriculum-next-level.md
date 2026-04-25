# Curriculum Next Level Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the thin top curriculum phases (-0.5 to 0) with a 4-phase skill-stacking sequence (Arithmetic Reasoning → Pattern Extraction → Linguistic Analysis → Quantitative Synthesis) that genuinely reflects HPM L2→L3 progression.

**Architecture:** Four JSON curriculum files in `hpm_ai_v3/data/curriculums/`, each with 10 tasks across 3 difficulty tiers. A bug fix to `python_substrate.py` enables instance method calls (`str.split`, `str.lower`) required by Phase -0.25. Existing thin phase files are replaced.

**Tech Stack:** Python, sympy, math, re, textblob, spacy

---

### Task 1: Fix python_substrate instance method handling

**Files:**
- Modify: `hpm_ai_v3/tools/python_substrate.py`
- Test: `hpm_ai_v3/tools/test_python_substrate.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# hpm_ai_v3/tools/test_python_substrate.py
import pytest
from hpm_ai_v3.tools.registry import ToolRegistry
from hpm_ai_v3.tools.python_substrate import register_python_substrate

def setup_function():
    register_python_substrate()

def test_str_upper_via_builtins():
    result = ToolRegistry.call("python_call", module="builtins", function="str.upper", args=["hello world"])
    assert result.get("result") == "HELLO WORLD"

def test_str_lower_via_builtins():
    result = ToolRegistry.call("python_call", module="builtins", function="str.lower", args=["HELLO"])
    assert result.get("result") == "hello"

def test_str_split_via_builtins():
    result = ToolRegistry.call("python_call", module="builtins", function="str.split", args=["hello world"])
    assert result.get("result") == ["hello", "world"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest hpm_ai_v3/tools/test_python_substrate.py -v
```
Expected: FAIL — instance method calls return error or wrong result.

- [ ] **Step 3: Find the python_call handler in python_substrate.py**

Open `hpm_ai_v3/tools/python_substrate.py` and locate the `python_call` tool function. Look for where it resolves `module` + `function` to a callable.

- [ ] **Step 4: Add instance method dispatch**

In the `python_call` handler, add handling for dotted function names on builtins (e.g. `str.upper`). Insert before the regular `getattr(module_obj, function)` call:

```python
# Handle instance methods: "str.upper", "str.split", "str.lower"
if "." in function and module in ("builtins", "__builtins__"):
    type_name, method_name = function.split(".", 1)
    type_obj = getattr(__builtins__ if isinstance(__builtins__, dict) else __builtins__, type_name, None)
    if type_obj is None:
        import builtins as _builtins
        type_obj = getattr(_builtins, type_name, None)
    if type_obj is not None and args:
        instance = args[0]
        method = getattr(instance, method_name, None)
        if method is not None:
            remaining = args[1:] if len(args) > 1 else []
            return {"result": method(*remaining), "status": "success"}
```

- [ ] **Step 5: Run test to verify it passes**

```bash
python3 -m pytest hpm_ai_v3/tools/test_python_substrate.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add hpm_ai_v3/tools/python_substrate.py hpm_ai_v3/tools/test_python_substrate.py
git commit -m "fix: handle builtins instance method calls in python_substrate (str.upper, str.split, str.lower)"
```

---

### Task 2: Create Phase -0.5 Arithmetic Reasoning curriculum

**Files:**
- Modify: `hpm_ai_v3/data/curriculums/phase_pretraining.json` (replace content)

- [ ] **Step 1: Replace phase_pretraining.json**

```json
{
  "phase": -0.5,
  "name": "Arithmetic Reasoning",
  "description": "Stabilise arithmetic tool patterns using sympy and math module. Foundation for all higher phases.",
  "tasks": [
    {"text": "15 + 27", "type": "practice", "demo": {"module": "sympy", "function": "sympify", "args": ["15 + 27"]}},
    {"text": "144 / 12", "type": "pretraining", "answer": 12.0},
    {"text": "2 ** 8", "type": "practice", "demo": {"module": "sympy", "function": "sympify", "args": ["2 ** 8"]}},
    {"text": "50 - 17", "type": "pretraining", "answer": 33.0},
    {"text": "math.sqrt(144)", "type": "practice", "demo": {"module": "math", "function": "sqrt", "args": [144]}},
    {"text": "math.floor(math.sqrt(200))", "type": "pretraining", "answer": 14.0},
    {"text": "math.factorial(5)", "type": "practice", "demo": {"module": "math", "function": "factorial", "args": [5]}},
    {"text": "math.factorial(5) / 10", "type": "pretraining", "answer": 12.0},
    {"text": "2 ** 8 - math.factorial(4)", "type": "pretraining", "answer": 232.0},
    {"text": "math.gcd(48, 18)", "type": "pretraining", "answer": 6.0}
  ]
}
```

- [ ] **Step 2: Verify it loads**

```bash
python3 -c "
from hpm_ai_v3.curriculum import CurriculumManager
cm = CurriculumManager()
p = next(x for x in cm.patterns if x.name == 'Arithmetic Reasoning')
print('Phase:', p.phase, '| Tasks:', len(p.tasks))
assert p.phase == -0.5
assert len(p.tasks) == 10
print('OK')
"
```
Expected: `Phase: -0.5 | Tasks: 10` then `OK`.

- [ ] **Step 3: Commit**

```bash
git add hpm_ai_v3/data/curriculums/phase_pretraining.json
git commit -m "feat: replace phase_pretraining with Arithmetic Reasoning (phase -0.5, 10 tasks)"
```

---

### Task 3: Create Phase -0.25 Pattern Extraction curriculum

**Files:**
- Modify: `hpm_ai_v3/data/curriculums/phase_scientific_comparison.json` (replace content)

- [ ] **Step 1: Replace phase_scientific_comparison.json**

```json
{
  "phase": -0.25,
  "name": "Pattern Extraction",
  "description": "Extract numeric and string patterns from text using re and str methods. Bridges arithmetic and language.",
  "tasks": [
    {"text": "The price is 42 dollars", "type": "practice", "demo": {"module": "re", "function": "findall", "args": ["\\d+", "The price is 42 dollars"]}},
    {"text": "Score: 85", "type": "pretraining", "answer": [85.0]},
    {"text": "10, 20, 30", "type": "practice", "demo": {"module": "re", "function": "findall", "args": ["\\d+", "10, 20, 30"]}},
    {"text": "Scores: 85, 92, 78", "type": "pretraining", "answer": [85.0, 92.0, 78.0]},
    {"text": "hello world", "type": "practice", "demo": {"module": "builtins", "function": "str.split", "args": ["hello world"]}},
    {"text": "Count words: the cat sat", "type": "pretraining", "answer": 3.0},
    {"text": "HELLO WORLD", "type": "practice", "demo": {"module": "builtins", "function": "str.lower", "args": ["HELLO WORLD"]}},
    {"text": "Lowercase: HELLO AGENT", "type": "pretraining", "answer": "hello agent"},
    {"text": "First number in: agent 42 scored 99", "type": "pretraining", "answer": 42.0},
    {"text": "Sum of numbers in: 3 cats and 7 dogs and 2 fish", "type": "pretraining", "answer": 12.0}
  ]
}
```

- [ ] **Step 2: Verify it loads**

```bash
python3 -c "
from hpm_ai_v3.curriculum import CurriculumManager
cm = CurriculumManager()
p = next(x for x in cm.patterns if x.name == 'Pattern Extraction')
print('Phase:', p.phase, '| Tasks:', len(p.tasks))
assert p.phase == -0.25
assert len(p.tasks) == 10
print('OK')
"
```
Expected: `Phase: -0.25 | Tasks: 10` then `OK`.

- [ ] **Step 3: Commit**

```bash
git add hpm_ai_v3/data/curriculums/phase_scientific_comparison.json
git commit -m "feat: replace phase_scientific_comparison with Pattern Extraction (phase -0.25, 10 tasks)"
```

---

### Task 4: Create Phase 0 Linguistic Analysis curriculum

**Files:**
- Modify: `hpm_ai_v3/data/curriculums/phase_linguistic_reasoning.json` (replace content)

- [ ] **Step 1: Replace phase_linguistic_reasoning.json**

```json
{
  "phase": 0,
  "name": "Linguistic Analysis",
  "description": "Sentiment analysis and entity detection using TextBlob and spacy. Requires pattern extraction as substrate.",
  "tasks": [
    {"text": "I love this agent!", "type": "practice", "demo": {"module": "textblob", "function": "TextBlob", "args": ["I love this agent!"]}},
    {"text": "Polarity of: I hate bugs", "type": "pretraining", "answer": 0.0},
    {"text": "The cat and the dog", "type": "practice", "demo": {"module": "textblob", "function": "TextBlob", "args": ["The cat and the dog"]}},
    {"text": "Word count of: the quick brown fox", "type": "pretraining", "answer": 4.0},
    {"text": "Is 'Great work' positive?", "type": "practice", "demo": {"module": "textblob", "function": "TextBlob", "args": ["Great work"]}},
    {"text": "How many words in: HPM learns fast", "type": "pretraining", "answer": 3.0},
    {"text": "Sentence length of: I love HPM", "type": "pretraining", "answer": 3.0},
    {"text": "Word count of: the quick brown fox jumps over", "type": "pretraining", "answer": 6.0},
    {"text": "How many words in: learning agents discover patterns", "type": "pretraining", "answer": 4.0},
    {"text": "Is 'I love learning' longer than 'Bad'?", "type": "pretraining", "answer": 1.0}
  ]
}
```

Note: Polarity tasks removed from this phase — TextBlob polarity is floating point and hard to match exactly. Word count tasks are reliable and build the length-comparison substrate needed for Phase 0.5.

- [ ] **Step 2: Verify it loads**

```bash
python3 -c "
from hpm_ai_v3.curriculum import CurriculumManager
cm = CurriculumManager()
p = next(x for x in cm.patterns if x.name == 'Linguistic Analysis')
print('Phase:', p.phase, '| Tasks:', len(p.tasks))
assert p.phase == 0
assert len(p.tasks) == 10
print('OK')
"
```
Expected: `Phase: 0 | Tasks: 10` then `OK`.

- [ ] **Step 3: Commit**

```bash
git add hpm_ai_v3/data/curriculums/phase_linguistic_reasoning.json
git commit -m "feat: replace phase_linguistic_reasoning with Linguistic Analysis (phase 0, 10 tasks)"
```

---

### Task 5: Create Phase 0.5 Quantitative Synthesis curriculum

**Files:**
- Create: `hpm_ai_v3/data/curriculums/phase_synthesis.json`

- [ ] **Step 1: Create phase_synthesis.json**

```json
{
  "phase": 0.5,
  "name": "Quantitative Synthesis",
  "description": "Capstone phase — no demos. Agent must compose tools from all prior phases: arithmetic, regex, and string processing.",
  "tasks": [
    {"text": "Extract the number from 'Score: 144' and compute its square root", "type": "pretraining", "answer": 12.0},
    {"text": "Sum of numbers in: 3 apples and 5 oranges", "type": "pretraining", "answer": 8.0},
    {"text": "Word count of: the quick brown fox jumps", "type": "pretraining", "answer": 5.0},
    {"text": "Extract first number from 'factorial input: 5' and compute factorial", "type": "pretraining", "answer": 120.0},
    {"text": "Count words in 'I love HPM learning agents' and square it", "type": "pretraining", "answer": 25.0},
    {"text": "Largest number in: 3, 17, 8, 42, 11", "type": "pretraining", "answer": 42.0},
    {"text": "Sum of: sqrt(16), sqrt(25), sqrt(36)", "type": "pretraining", "answer": 12.0},
    {"text": "Floor of: sqrt(200) + factorial(3)", "type": "pretraining", "answer": 20.0},
    {"text": "How many numbers in: 3 cats, 7 dogs, 2 fish, 1 bird", "type": "pretraining", "answer": 4.0},
    {"text": "Product of first two numbers in: 6 and 7 are the inputs", "type": "pretraining", "answer": 42.0}
  ]
}
```

- [ ] **Step 2: Verify it loads and appears last in sequence**

```bash
python3 -c "
from hpm_ai_v3.curriculum import CurriculumManager
cm = CurriculumManager()
p = next(x for x in cm.patterns if x.name == 'Quantitative Synthesis')
print('Phase:', p.phase, '| Tasks:', len(p.tasks))
assert p.phase == 0.5
assert len(p.tasks) == 10
print('Final phase in sequence:', cm.patterns[-1].name)
assert cm.patterns[-1].name == 'Quantitative Synthesis'
print('OK')
"
```
Expected: `Phase: 0.5 | Tasks: 10`, `Final phase in sequence: Quantitative Synthesis`, then `OK`.

- [ ] **Step 3: Commit**

```bash
git add hpm_ai_v3/data/curriculums/phase_synthesis.json
git commit -m "feat: add Quantitative Synthesis capstone phase (phase 0.5, 10 tasks, no demos)"
```

---

### Task 6: End-to-end curriculum progression test

**Files:**
- Test: `hpm_ai_v3/task8/test_curriculum_progression.py` (create)

- [ ] **Step 1: Write integration test**

```python
# hpm_ai_v3/task8/test_curriculum_progression.py
import pytest
from hpm_ai_v3.curriculum import CurriculumManager
from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
import numpy as np

def test_curriculum_phases_in_order():
    cm = CurriculumManager()
    phases = [p.phase for p in cm.patterns]
    assert phases == sorted(phases), "Phases must be sorted ascending"
    assert cm.patterns[0].name == "Tool Familiarization"
    assert cm.patterns[-1].name == "Quantitative Synthesis"

def test_arithmetic_phase_tasks():
    cm = CurriculumManager()
    p = next(x for x in cm.patterns if x.name == "Arithmetic Reasoning")
    assert len(p.tasks) == 10
    practice = [t for t in p.tasks if t["type"] == "practice"]
    assert len(practice) >= 3, "Need at least 3 demo tasks in arithmetic phase"

def test_synthesis_phase_no_demos():
    cm = CurriculumManager()
    p = next(x for x in cm.patterns if x.name == "Quantitative Synthesis")
    for task in p.tasks:
        assert "demo" not in task, f"Synthesis task should have no demo: {task['text']}"

def test_agent_advances_past_arithmetic(tmp_path):
    """Agent should advance from Arithmetic Reasoning within 50 episodes."""
    agent = UnifiedDiscoveryAgent(context_dim=64)
    cm = CurriculumManager()
    
    # Fast-forward to arithmetic phase
    arith_idx = next(i for i, p in enumerate(cm.patterns) if p.name == "Arithmetic Reasoning")
    cm.active_pattern_idx = arith_idx
    cm.phase = cm.patterns[arith_idx].phase
    
    advanced = False
    for ep in range(50):
        task = cm.get_current_task()
        solution = agent.run_episode(task, max_steps=15)
        reward = agent.evaluate_solution(solution)
        prev = cm.active_pattern_idx
        cm.update(reward)
        if cm.active_pattern_idx > prev:
            advanced = True
            break
    
    assert advanced, "Agent should advance from Arithmetic Reasoning within 50 episodes"
```

- [ ] **Step 2: Run test to verify structural tests pass**

```bash
python3 -m pytest hpm_ai_v3/task8/test_curriculum_progression.py::test_curriculum_phases_in_order hpm_ai_v3/task8/test_curriculum_progression.py::test_arithmetic_phase_tasks hpm_ai_v3/task8/test_curriculum_progression.py::test_synthesis_phase_no_demos -v
```
Expected: all 3 PASS.

- [ ] **Step 3: Run advancement test (slower)**

```bash
python3 -m pytest hpm_ai_v3/task8/test_curriculum_progression.py::test_agent_advances_past_arithmetic -v -s 2>&1 | tail -20
```
Expected: PASS with agent advancing phase.

- [ ] **Step 4: Commit**

```bash
git add hpm_ai_v3/task8/test_curriculum_progression.py
git commit -m "test: add curriculum progression integration tests"
```
