# Quiz AI Answering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace human-clickable quiz answer buttons with automatic AI answering — the AI reasons through each question using `ReasoningAgent` and the human watches the result.

**Architecture:** A new `POST /api/quiz/ai_answer` endpoint in `web_demo.py` receives a `question_id`, calls `reader.reasoning_agent.reason()` with a structured prompt over the question and its four options, parses the first letter (A–D) from the response to determine `answer_index`, and returns `{answer_index, reasoning}`. The quiz UI JS is updated to render options as non-clickable labels, call this endpoint automatically after each question is displayed, highlight the AI's chosen option, then show the existing `/api/quiz/submit` feedback. The completion screen and "Train on gaps" button are unchanged.

**Tech Stack:** Python 3, Flask, existing `ReasoningAgent` (`reader.reasoning_agent`), vanilla JS in the inline HTML template.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `hpm_ai_v6/web/web_demo.py` | Modify | Add `POST /api/quiz/ai_answer` endpoint; replace `showQuestion` / `submitAnswer` JS |

---

### Task 1: Add `POST /api/quiz/ai_answer` endpoint

**Files:**
- Modify: `hpm_ai_v6/web/web_demo.py` — insert new Flask route before the existing `/api/quiz/train_gaps` route (~line 2175)

---

- [ ] **Step 1: Locate the insertion point**

Open `hpm_ai_v6/web/web_demo.py` and find the line:

```
@app.route("/api/quiz/train_gaps", methods=["POST"])
```

The new endpoint block goes immediately before this decorator.

- [ ] **Step 2: Insert the endpoint**

Add the following block immediately before `@app.route("/api/quiz/train_gaps", ...)`:

```python
@app.route("/api/quiz/ai_answer", methods=["POST"])
def api_quiz_ai_answer():
    """Use ReasoningAgent to answer a quiz question automatically."""
    data = request.get_json(force=True)
    question_id = data.get("question_id")
    if not question_id or question_id not in _quiz_state:
        return jsonify({"error": "Unknown question_id"}), 400

    q = _quiz_state[question_id]
    reasoning_agent = getattr(reader, "reasoning_agent", None)
    if reasoning_agent is None:
        return jsonify({"error": "ReasoningAgent not available"}), 503

    prompt = (
        "Given this question and these 4 options, which is most likely correct based on "
        "what you know? "
        f"Question: {q.question}. "
        f"Options: A) {q.options[0]} B) {q.options[1]} C) {q.options[2]} D) {q.options[3]}. "
        "Reply with ONLY the letter (A, B, C, or D) followed by a brief explanation."
    )
    try:
        response = reasoning_agent.reason(prompt)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    first_letter = next(
        (ch for ch in response.strip().upper() if ch in letter_map), None
    )
    answer_index = letter_map.get(first_letter, 0)
    return jsonify({"answer_index": answer_index, "reasoning": response})
```

- [ ] **Step 3: Verify the module imports without errors**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -c "from hpm_ai_v6.web.web_demo import app; print('OK')"
```

Expected:
```
OK
```

- [ ] **Step 4: Smoke-test the endpoint with Flask test client**

Create `/tmp/test_ai_answer_endpoint.py`:

```python
import sys
sys.path.insert(0, '/home/mattthomson/workspace/HPM---Learning-Agent')

from hpm_ai_v6.web.web_demo import app, _quiz_state
from hpm_ai_v6.agents.quiz_agent import QuizQuestion

fake_q = QuizQuestion(
    id="test-q-1",
    question="What is the capital of France?",
    options=["Berlin", "Madrid", "Paris", "Rome"],
    correct_index=2,
    topic="Geography",
    explanation="Paris is the capital of France.",
    difficulty="easy",
    source="bank",
)
_quiz_state["test-q-1"] = fake_q

with app.test_client() as client:
    resp = client.post(
        "/api/quiz/ai_answer",
        json={"question_id": "test-q-1"},
    )
    print("Status:", resp.status_code)
    data = resp.get_json()
    assert resp.status_code in (200, 503), f"Unexpected status {resp.status_code}"
    if resp.status_code == 200:
        assert isinstance(data["answer_index"], int)
        assert 0 <= data["answer_index"] <= 3
        print("answer_index:", data["answer_index"])
        print("reasoning preview:", str(data.get("reasoning", ""))[:80])
    print("PASS")
```

Run:

```bash
python /tmp/test_ai_answer_endpoint.py
```

Expected (exact `answer_index` varies; 503 is acceptable when no reader is initialised):
```
Status: 200
answer_index: 2
reasoning preview: C) Paris is the capital of France...
PASS
```
or
```
Status: 503
PASS
```

- [ ] **Step 5: Commit**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
git add hpm_ai_v6/web/web_demo.py
git commit -m "feat: add POST /api/quiz/ai_answer endpoint using ReasoningAgent"
```

---

### Task 2: Update quiz UI JS to auto-answer with AI

**Files:**
- Modify: `hpm_ai_v6/web/web_demo.py` — replace the `showQuestion` and `submitAnswer` JS functions inside `HTML_TEMPLATE`

---

- [ ] **Step 1: Locate the JS functions to replace**

In `web_demo.py`, inside `HTML_TEMPLATE`, find:

```javascript
    function showQuestion() {
```
and
```javascript
    async function submitAnswer(answerIndex) {
```

Both will be fully replaced in the next two steps.

- [ ] **Step 2: Replace `showQuestion`**

Replace the entire `showQuestion` function (from `function showQuestion() {` up to but not including `async function submitAnswer`) with:

```javascript
    function showQuestion() {
      var q = _quizQuestions[_quizIndex];
      var total = _quizQuestions.length;
      document.getElementById('quiz-progress').textContent = 'Question ' + (_quizIndex + 1) + ' of ' + total;
      document.getElementById('quiz-progress-fill').style.width = Math.round((_quizIndex / total) * 100) + '%';
      document.getElementById('quiz-question').textContent = q.question;
      document.getElementById('quiz-feedback').style.display = 'none';
      document.getElementById('quiz-next-btn').style.display = 'none';

      var labels = ['A', 'B', 'C', 'D'];
      var optDiv = document.getElementById('quiz-options');
      optDiv.innerHTML = '';
      q.options.forEach(function(opt, i) {
        var lbl = document.createElement('div');
        lbl.id = 'quiz-opt-' + i;
        lbl.style.cssText = 'padding:10px 14px;border-radius:6px;background:var(--surface2);font-size:14px;';
        lbl.textContent = labels[i] + '. ' + opt;
        optDiv.appendChild(lbl);
      });

      var fb = document.getElementById('quiz-feedback');
      fb.style.display = 'block';
      fb.style.background = 'var(--surface2)';
      fb.style.color = 'var(--muted)';
      fb.textContent = 'AI is thinking…';

      setTimeout(function() { aiAnswer(q); }, 600);
    }
```

- [ ] **Step 3: Replace `submitAnswer` with `aiAnswer`**

Replace the entire `async function submitAnswer(answerIndex) { ... }` with:

```javascript
    async function aiAnswer(q) {
      var res = await fetch('/api/quiz/ai_answer', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question_id: q.id})
      });
      var aiData = await res.json();
      var aiIndex = (typeof aiData.answer_index === 'number') ? aiData.answer_index : 0;

      var labels = ['A', 'B', 'C', 'D'];
      for (var i = 0; i < 4; i++) {
        var el = document.getElementById('quiz-opt-' + i);
        if (el) {
          if (i === aiIndex) {
            el.style.background = '#1e3a2e';
            el.style.color = 'var(--accent)';
            el.style.fontWeight = '600';
          } else {
            el.style.background = 'var(--surface2)';
            el.style.color = '';
            el.style.fontWeight = '';
          }
        }
      }

      var submitRes = await fetch('/api/quiz/submit', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question_id: q.id, answer_index: aiIndex})
      });
      var data = await submitRes.json();

      var fb = document.getElementById('quiz-feedback');
      fb.style.display = 'block';
      var reasoningSnippet = (aiData.reasoning || '').substring(0, 200);
      if (data.correct) {
        _quizCorrect++;
        fb.style.background = 'var(--success-dim)';
        fb.style.color = 'var(--accent)';
        fb.textContent = '✓ Correct! ' + data.explanation;
        var note = document.createElement('span');
        note.style.cssText = 'font-size:12px;opacity:0.8;margin-top:6px;display:block';
        note.textContent = 'AI reasoning: ' + reasoningSnippet;
        fb.appendChild(note);
      } else {
        _quizFailedTopics.push(q.topic);
        fb.style.background = 'var(--danger-dim)';
        fb.style.color = 'var(--danger)';
        fb.textContent = '✗ Incorrect. Correct answer: ' + labels[data.correct_index] + '. ' + data.explanation;
        var note = document.createElement('span');
        note.style.cssText = 'font-size:12px;opacity:0.8;margin-top:6px;display:block';
        note.textContent = 'AI reasoning: ' + reasoningSnippet;
        fb.appendChild(note);
      }
      document.getElementById('quiz-next-btn').style.display = 'inline-block';
    }
```

Note: `innerHTML` is intentionally avoided above; `textContent` and `appendChild` are used instead to prevent XSS.

- [ ] **Step 4: Verify no Python syntax errors**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m py_compile hpm_ai_v6/web/web_demo.py && echo "Syntax OK"
```

Expected:
```
Syntax OK
```

- [ ] **Step 5: Smoke-test that the new JS functions are in the rendered HTML**

Create `/tmp/test_quiz_ui_smoke.py`:

```python
import sys
sys.path.insert(0, '/home/mattthomson/workspace/HPM---Learning-Agent')
from hpm_ai_v6.web.web_demo import app

with app.test_client() as client:
    resp = client.get('/')
    html = resp.data.decode()
    assert 'aiAnswer' in html, "aiAnswer function not found in HTML"
    assert 'submitAnswer' not in html, "old submitAnswer still present"
    assert 'AI is thinking' in html, "thinking indicator not found"
    assert '/api/quiz/ai_answer' in html, "ai_answer endpoint reference not found"
    print("PASS")
```

Run:

```bash
python /tmp/test_quiz_ui_smoke.py
```

Expected:
```
PASS
```

- [ ] **Step 6: Commit**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
git add hpm_ai_v6/web/web_demo.py
git commit -m "feat: quiz UI auto-answers via AI — non-clickable labels, aiAnswer replaces submitAnswer"
```

---

## Self-Review

### Spec coverage

| Requirement | Task |
|---|---|
| `POST /api/quiz/ai_answer` endpoint takes `{question_id}` | Task 1 Step 2 |
| Uses `reader.reasoning_agent.reason()` | Task 1 Step 2 |
| Prompt asks for letter + brief explanation | Task 1 Step 2 |
| Parses first A/B/C/D letter to get `answer_index` (A=0 … D=3) | Task 1 Step 2 |
| Returns `{answer_index, reasoning}` | Task 1 Step 2 |
| Options displayed as non-clickable labels | Task 2 Step 2 |
| "AI is thinking…" appears before fetch | Task 2 Step 2 |
| `POST /api/quiz/ai_answer` called automatically after question shown | Task 2 Step 3 |
| AI's chosen option highlighted | Task 2 Step 3 |
| AI reasoning shown in feedback | Task 2 Step 3 |
| Correct/incorrect feedback still uses `/api/quiz/submit` | Task 2 Step 3 |
| "Next" button advances (unchanged) | `nextQuestion()` is untouched |
| Completion screen + "Train on gaps" unchanged | `showCompletion` and `trainGaps` untouched |
| Commit after each task | Task 1 Step 5, Task 2 Step 6 |

No gaps found.

### Placeholder scan

No "TBD", "TODO", "similar to", or "add appropriate" phrases. All code blocks are complete.

### Type consistency

- `answer_index` is `int` in the Python endpoint, checked as `typeof aiData.answer_index === 'number'` in JS — consistent.
- `reasoning` is `str` from `reasoning_agent.reason()`, accessed as `aiData.reasoning` in JS — consistent.
- `_quiz_state[question_id]` is a `QuizQuestion` dataclass; `.question`, `.options[0..3]`, and `.topic` are all valid fields per the spec dataclass definition.
- `aiAnswer(q)` receives the same `q` object shape from `_quizQuestions` that `showQuestion` already constructs — consistent.
- `submitAnswer` is fully removed and has no remaining callers after the replacement (option buttons are now `div` elements with no `onclick`).
