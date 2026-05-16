# Quiz CLI Design

**Date:** 2026-05-16
**Status:** Approved

## Overview

A standalone CLI quiz tool that runs the HPM model through general knowledge quizzes from the terminal. The AI answers all questions using its learned pattern graph. Results are printed to the terminal; weak topics trigger Wikipedia training to close the HPM feedback loop.

## Invocation

```bash
python -m hpm_ai_v6.cli.quiz_cli [--source bank|model] [--difficulty easy|medium|hard] [--n N] [--auto]
```

- `--source`: `bank` (default) or `model` (AI-generated questions)
- `--difficulty`: `easy` (default), `medium`, `hard`
- `--n`: number of questions, default 5
- `--auto`: skip "Press Enter to continue" pauses; fully autonomous run

## Architecture

Single file: `hpm_ai_v6/cli/quiz_cli.py`

Builds `MultiAgentReader` and `QuizAgent` directly (no Flask, no web server). Uses the same corpus path as the web demo. Calls `QuizAgent.generate_quiz()` and `reasoning_agent.reason_with_trace()` for AI answering. After completion, triggers Wikipedia training on weak topics via `reader.train_sequence()` with fetched Wikipedia sentences.

## Quiz Loop

1. Parse args, build reader + quiz agent
2. Generate questions via `quiz_agent.generate_quiz(n, difficulty, source)`
3. For each question:
   - Print question number, question text, 4 options (A/B/C/D)
   - Call `reasoning_agent.reason_with_trace(prompt)` to get answer + confidence
   - Print AI's chosen option, confidence (confident / guessing), reasoning snippet
   - Print correct/incorrect; reveal correct answer if wrong
   - In interactive mode: print "Press Enter for next question" and wait
4. Print completion summary: score X/N, list of weak topics
5. If weak topics exist: trigger Wikipedia training on those topics, print status

## Output Style

Plain ANSI escape codes — no external colour library.
- Green (`\033[92m`) for correct answers
- Red (`\033[91m`) for incorrect
- Yellow (`\033[93m`) for guessing / no learned path
- Reset: `\033[0m`

## Confidence

Uses the same `reason_with_trace()` logic as the web demo:
- `confident = True` when `candidate_paths` is non-empty or `chosen_path is not None`
- `confident = False` → flagged as guess, topic counted as gap regardless of correctness

## Training After Quiz

Fetch Wikipedia summary sentences for each weak topic using `urllib` (stdlib only), then call `reader.train_sequence(sentences)` to incorporate the new knowledge. `maintenance_cycle()` is available for a fuller hydration + retrain cycle: `reader.maintenance_cycle(sentences, retrain_epochs=1)`.

The exact call used is `reader.train_sequence(wiki_sentences)` for lightweight post-quiz updates, or `reader.maintenance_cycle(wiki_sentences, retrain_epochs=1)` when a deeper rehydration pass is wanted.

## Files

| File | Action |
|---|---|
| `hpm_ai_v6/cli/__init__.py` | Create (empty) |
| `hpm_ai_v6/cli/quiz_cli.py` | Create |
| `hpm_ai_v6/tests/test_quiz_cli.py` | Create |
