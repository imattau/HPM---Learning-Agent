# Quiz Module Design

**Date:** 2026-05-16  
**Status:** Approved

## Overview

A new Quiz module for the HPM Learning Agent web demo. The module lets the model take multi-choice general knowledge quizzes, provides immediate per-question feedback, and automatically triggers Wikipedia training on topics where it answers incorrectly. Quiz failures become learning signals that direct further training — closing the HPM feedback loop: learn → quiz → identify gaps → train more.

## Architecture

The module has three layers:

1. **`QuizAgent`** (`hpm_ai_v6/agents/quiz_agent.py`) — generates and self-evaluates quiz questions
2. **API endpoints** — four new routes in `web_demo.py`
3. **Quiz UI section** — a new collapsible "Quiz" section in the web demo with two pill sub-tabs

Quiz session state is held in a module-level `_quiz_state` dict in `web_demo.py`, mirroring the existing `_gutenberg_state` pattern. No database or persistence between server restarts.

## QuizAgent

`QuizAgent` composes `ReasoningAgent` and `MultiAgentReader`. It does not answer questions or score responses — those responsibilities stay in the API layer.

### QuizQuestion dataclass

```python
@dataclass
class QuizQuestion:
    id: str
    question: str
    options: list[str]      # always exactly 4
    correct_index: int       # 0-3
    topic: str
    explanation: str         # why the correct answer is right and distractors are wrong
    difficulty: str          # "easy" | "medium" | "hard"
    source: str              # "model" | "bank"
```

### Key methods

**`generate_question(topic: str | None, difficulty: str) -> QuizQuestion`**

1. If no topic provided, selects one from the corpus via `MultiAgentReader`
2. Retrieves relevant passages for the topic
3. Prompts `ReasoningAgent` to construct: question stem, one correct answer, three plausible-but-wrong distractors, and an explanation
4. Self-evaluates: reasons over all four options to confirm the correct answer is actually correct given what the model has learned — discards and regenerates if self-evaluation fails
5. Returns a validated `QuizQuestion`

Distractors must be plausible, not random — they probe understanding rather than pattern-matching on obviously wrong answers.

**`generate_quiz(n: int, difficulty: str, source: str) -> list[QuizQuestion]`**

- `source="model"`: calls `generate_question()` N times with varied topics
- `source="bank"`: loads from the appropriate JSON file in `hpm_ai_v6/data/quiz_banks/`
- `difficulty` affects the reasoning prompt for model-generated questions:
  - **Easy** — factual recall, single-concept questions
  - **Medium** — application of concepts, moderate inference
  - **Hard** — inference across multiple related concepts, relational reasoning

### Difficulty influence on generation

Difficulty is passed into the `ReasoningAgent` prompt as a constraint on abstraction level. Easy questions test whether a fact was learned; Hard questions test whether relationships between facts were learned — which maps directly to HPM's hierarchy of pattern levels.

## Question Banks

Static JSON files in `hpm_ai_v6/data/quiz_banks/`:
- `easy.json`
- `medium.json`  
- `hard.json`

Each file is a JSON array of objects matching the `QuizQuestion` shape (minus `source`, which is set to `"bank"` at load time). These files are authored once and can be extended independently of the model.

## API Endpoints

All new endpoints added to `web_demo.py`:

| Endpoint | Method | Body / Params | Response |
|---|---|---|---|
| `/api/quiz/generate` | POST | `{n, difficulty, source}` | `{questions: [...]}` |
| `/api/quiz/submit` | POST | `{question_id, answer_index}` | `{correct, explanation, correct_index}` |
| `/api/quiz/train_gaps` | POST | `{topics: [str]}` | `{status, message}` |
| `/api/quiz/banks/<difficulty>` | GET | — | JSON file download |

**`/api/quiz/generate`** stores generated questions in `_quiz_state` keyed by `question.id` so that `/api/quiz/submit` can score them without re-generating.

**`/api/quiz/submit`** performs scoring (compare `answer_index` to `correct_index`), returns feedback immediately. Does not trigger training — training is deferred until the user explicitly clicks "Train on gaps" after the full quiz.

**`/api/quiz/train_gaps`** calls `_start_wikipedia_cycle(topics=topics)` with all weak topics in a single call. Uses the existing Wikipedia cycle machinery — no new training infrastructure needed.

## Web UI

New "Quiz" collapsible section in `web_demo.py`, styled identically to the existing Train / Generate / Reason sections.

### Take Quiz sub-tab

Controls:
- Source: Model-generated | Question Bank (radio)
- Difficulty: Easy | Medium | Hard (radio)
- Number of questions: 5 | 10 | 20 (radio)
- "Start Quiz" button

Quiz flow:
- Questions displayed one at a time
- Four option buttons (A/B/C/D)
- On answer selection: immediate feedback shown — correct/incorrect indicator, explanation text, which option was correct
- Progress bar: "Question X of N"
- "Next" button advances to next question

Completion screen:
- Score: X / N correct
- List of topics answered incorrectly
- "Train on gaps" button — triggers `POST /api/quiz/train_gaps` and shows Wikipedia cycle status

### Download Banks sub-tab

Three rows (Easy / Medium / Hard), each with a description and a download button. Served via `GET /api/quiz/banks/<difficulty>` with `Content-Disposition: attachment` header.

## Data Flow

```
User starts quiz
  → POST /api/quiz/generate
    → QuizAgent.generate_quiz()
      → [source=model] ReasoningAgent generates + self-evaluates questions
      → [source=bank]  loads from hpm_ai_v6/data/quiz_banks/<difficulty>.json
    → questions stored in _quiz_state
    → questions returned to UI (without correct_index)

User answers each question
  → POST /api/quiz/submit {question_id, answer_index}
    → scores against _quiz_state[question_id].correct_index
    → returns {correct, explanation, correct_index}

User clicks "Train on gaps"
  → POST /api/quiz/train_gaps {topics: [...failed topics...]}
    → calls _start_wikipedia_cycle(topics=topics)
    → Wikipedia cycle runs in background, same as manual training
```

Note: `correct_index` is withheld from the initial `/api/quiz/generate` response to prevent client-side cheating. It is only returned by `/api/quiz/submit`.

## Files to Create / Modify

| File | Action |
|---|---|
| `hpm_ai_v6/agents/quiz_agent.py` | Create |
| `hpm_ai_v6/agents/__init__.py` | Add `QuizAgent` export |
| `hpm_ai_v6/data/quiz_banks/easy.json` | Create |
| `hpm_ai_v6/data/quiz_banks/medium.json` | Create |
| `hpm_ai_v6/data/quiz_banks/hard.json` | Create |
| `hpm_ai_v6/web/web_demo.py` | Add Quiz section + 4 API endpoints |
| `hpm_ai_v6/tests/test_quiz_agent.py` | Create |
