# Session State: Task 1 Sentence-Level Chunking

## Objective
Implement sentence-level chunking for `fetch_passages()` in `hpm_ai_v2/utils/text_fetcher.py`:
- Add `mode` parameter ("sentence" or "paragraph", default "paragraph")
- Add `split_sentences()` helper function
- Write and pass tests in `tests/test_reader_core.py`
- Preserve backward compatibility (existing tests in `tests/test_webpage_reader.py` must pass)

## Progress
- [x] Located files:
  - `/home/mattthomson/workspace/HPM---Learning-Agent/hpm_ai_v2/utils/text_fetcher.py`
  - `/home/mattthomson/workspace/HPM---Learning-Agent/tests/test_webpage_reader.py`
- [ ] Read current `text_fetcher.py` implementation
- [ ] Create `tests/test_reader_core.py` with failing tests
- [ ] Run failing tests
- [ ] Implement `split_sentences()` and modify `fetch_passages()`
- [ ] Run all tests (new + existing)
- [ ] Commit changes
- [ ] Self-review and report

## Implementation Details
Required changes to `fetch_passages`:
```python
def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def fetch_passages(
    url: str = None,
    text: str = None,
    min_length: int = 40,
    mode: str = "paragraph",  # NEW PARAM
) -> List[str]:
    if url is not None:
        text = fetch_url(url)
    if not text:
        return []
    if mode == "sentence":
        chunks = split_sentences(text)
    else:
        chunks = [p.strip() for p in re.split(r'\n\s*\n', text)]
    return [c for c in chunks if len(c) >= min_length]
```

Test cases needed:
1. `test_sentence_mode_splits_sentences()`
2. `test_sentence_mode_filters_short()`
3. `test_default_mode_unchanged()`

## Next Steps
1. Read current `text_fetcher.py` to understand existing implementation
2. Create `tests/test_reader_core.py` with test cases
3. Run pytest to confirm tests fail
4. Implement changes to `text_fetcher.py`
5. Run pytest to confirm all tests pass
6. Commit with message: "feat: add sentence-level chunking mode to fetch_passages"
7. Report final status

execution_mode: unattended
auto_continue: true
