# Quiz Reasoning Confidence — Implementation Plan

**Date:** 2026-05-16
**Branch:** hpm-ai-v6

## Problem

`POST /api/quiz/ai_answer` called `reasoning_agent.reason(prompt)` and parsed the first A/B/C/D letter. When the model has no learned patterns for a topic it guesses a letter, but the reasoning text shows no actual path. The quiz accepted this guess as a genuine answer and did not flag the topic as a knowledge gap.

## Fix

Replace `reasoning_agent.reason()` with `reasoning_agent.reason_with_trace()`.

### `reason_with_trace` return structure

Returns a `Dict[str, Any]` with keys:
- `answer` — text response (contains the A/B/C/D letter)
- `candidate_paths` — list of reasoning paths found in the learned graph
- `chosen_path` — the best path chosen (or `None` if none found)
- `evidence` — supporting evidence list
- `terms`, `anchors`, `intent`, `mode`, `method` — parsing metadata

**Confidence criterion:** `len(candidate_paths) > 0 OR chosen_path is not None`

When the agent has no learned structure for a topic, both `candidate_paths` is empty and `chosen_path` is `None`, so `confident = False`.

## Backend changes (`web_demo.py`, `api_quiz_ai_answer`)

1. Call `reason_with_trace(prompt)` instead of `reason(prompt)`
2. Extract answer text from `trace["answer"]`
3. Compute `confident = bool(candidate_paths) or chosen_path is not None`
4. Return `{answer_index, reasoning, confident}` in the JSON response

## Frontend changes (`aiAnswer` JS function)

- Read `aiData.confident`
- If `confident === false`:
  - Push topic to `_quizFailedTopics` regardless of correctness (gap always counted)
  - If answer was correct: show success colours but append "(guessing — no learned path)" and note "counted as gap"
  - If answer was wrong: show failure colours with "(guessing — no learned path)" label
- If `confident === true`: normal flow (only push to gaps on wrong answer)
