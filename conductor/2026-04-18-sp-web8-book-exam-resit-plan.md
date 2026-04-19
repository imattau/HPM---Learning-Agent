# Plan: SP-Web8 - Book Reading & Exam Resit Workflow

The goal is to verify the agent's ability to "Read a Book", "Take an Exam", "Identify Failures", and "Resit after Targeted Research".

## Objective
- Ingest a multi-chapter "book" (e.g., "The History of AI").
- Take a 5-question exam covering multiple chapters.
- Identify questions where the answer was "I don't know" or incorrect.
- For each failed question, use the `LibrarianAgent` to find the relevant chapter/section.
- "Research" by re-ingesting or deeply analyzing the failed section.
- Resit the exam and verify improved results.

## Key Files & Context
- `hpm_ai_v2/agents/reader_agent.py`: Ingestion specialist.
- `hpm_ai_v2/agents/librarian_agent.py`: Topic and document retrieval specialist.
- `hpm_ai_v2/agents/writer_agent.py`: Question answering specialist.
- `hpm_ai_v2/experiments/experiment_sp_web8_book_exam_resit.py`: New experiment script.

## Implementation Steps

### 1. Content Definition
- [ ] Define "The History of AI" book with at least 5 distinct chapters.
- [ ] Define a "Bank" of exam questions, some of which are "hard" or require specific detail.

### 2. Initial Ingestion (The Reading Phase)
- [ ] Initialize the HPM society (Reader, Librarian, Writer).
- [ ] Have the `ReaderAgent` ingest the book chapter by chapter.
- [ ] The `LibrarianAgent` autonomously maps the chapters to topics.

### 3. First Exam (The Assessment Phase)
- [ ] Loop through the exam questions.
- [ ] Use `WriterAgent.answer_natural()` to get answers.
- [ ] Implement a `Grader` that checks for keywords or "I don't know" responses.
- [ ] Collect "Failed Questions" and their associated keywords.

### 4. Remedial Research (The Study Phase)
- [ ] For each failed question:
  - [ ] Use `LibrarianAgent.search_knowledge()` with the question text to find the most relevant document/paragraph node.
  - [ ] "Re-read" the specific section (simulate deeper focus by re-ingesting with `proactive_lookup=True`).
  - [ ] Explicitly link the question topic to the found document.

### 5. Final Resit (The Success Phase)
- [ ] Re-attempt only the failed questions.
- [ ] Verify that the answers are now correct/detailed.

## Verification & Testing
- Run `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp_web8_book_exam_resit.py`.
- Verify the "Initial Score" vs. "Resit Score".
- Verify that the `LibrarianAgent` correctly identified the "remedial" sections.
