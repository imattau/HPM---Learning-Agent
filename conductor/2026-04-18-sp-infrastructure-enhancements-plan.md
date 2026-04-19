# Plan: HPM Society Infrastructure Enhancements

This plan details four specific improvements to the `hpm_ai_v2` agents to enhance robustness, performance, and code quality. These changes include better keyword extraction, unified sentence splitting, web request caching, and an optimized BFS search strategy.

## Objective
- Improve topic discovery in `LibrarianAgent` using YAKE/RAKE.
- Unify sentence splitting in `ReaderAgent` using spaCy (matching `HtmlReaderAgent`).
- Add a caching layer to `WebAgent` using `requests-cache` to reduce redundant network traffic.
- Optimize the `BaseHFNAgent._try_bfs` strategy with a priority queue and early termination.

## Key Files & Context
- `hpm_ai_v2/agents/librarian_agent.py`: High-level concept discovery.
- `hpm_ai_v2/agents/reader_agent.py`: Text ingestion and perception.
- `hpm_ai_v2/agents/web_agent.py`: Web interaction.
- `hpm_ai_v2/agents/base_agent.py`: Base agent and search strategies.

## Implementation Steps

### 1. LibrarianAgent Topic Discovery Upgrade
- [ ] Add `yake` and `sklearn` imports (with try/except fallbacks).
- [ ] Refactor `discover_topics` to use `yake.KeywordExtractor` for more robust keyword identification.
- [ ] Keep the simple token counting as a fallback if `yake` is unavailable.

### 2. ReaderAgent Sentence Splitting Unification
- [ ] Add `_init_spacy` method to `ReaderAgent` (copying logic from `HtmlReaderAgent`).
- [ ] Call `_init_spacy` in `__init__`.
- [ ] Update `ingest_text` to use `self._nlp` for sentence splitting when available, falling back to `SentenceSplitter`.

### 3. WebAgent Caching Layer
- [ ] Import `requests_cache`.
- [ ] Initialize a `requests_cache.CachedSession` in `WebAgent.__init__`.
- [ ] Override `fetch_page` (or ensure `WebFetchMixin` uses the session) to leverage the cache.

### 4. BaseHFNAgent BFS Optimization
- [ ] Import `heapq`.
- [ ] Refactor `_try_bfs` to use a priority queue ordered by utility (negative distance).
- [ ] Implement early termination when a solution is found with distance below a threshold (e.g., 0.01).
- [ ] Maintain the beam width constraint by pruning the heap after expansion.

## Verification & Testing
- Install new dependencies: `yake`, `requests-cache`.
- Run the "Book Reading & Exam Resit" experiment: `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp_web8_book_exam_resit.py`.
- Verify that the agent achieves a passing score (or improved performance) on the exam resit.
- Check logs for "Librarian: Selected best sentence" and "External knowledge acquired" to confirm the new logic is active.
