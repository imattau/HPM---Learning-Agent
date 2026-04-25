# 2026-04-23-online-driven-lm-curriculum.md

# Online-Driven LanguageModelPattern Curriculum Implementation Plan

**Goal:** Implement failure-triggered online learning for the `LanguageModelPattern`. The agent identifies failure in linguistic tasks, fetches relevant domain knowledge from Wikipedia, fine-tunes the neural substrate, and retries.

---

### Task 1: OnlineLearningBuffer Infrastructure

**Files:**
- Create: `hpm_ai_v3/online_buffer.py`
- Modify: `hpm_ai_v3/neural_lm_pattern.py`

- [ ] **Step 1: Implement `OnlineLearningBuffer`**
    - [ ] `__init__`: Max buffer size (10,000 chars), max fetches per phase (5).
    - [ ] `extract_keywords(text)`: Derive 2-3 significant keywords from failing task.
    - [ ] `fetch_wikipedia(keywords)`: Use Wikipedia summary API to get clean text.
    - [ ] `add_to_buffer(text)`: Append text, rotate out oldest when full.
    - [ ] `can_fetch(phase_id)`: Check budget and consecutive failure threshold.

- [ ] **Step 2: Update `LanguageModelPattern` for Fine-Tuning**
    - [ ] `fine_tune(text, epochs)`: Expose lightweight training on small text chunks.

---

### Task 2: Failure-Triggered Learning in Discovery Agent

**Files:**
- Modify: `hpm_ai_v3/agents/base_discovery.py`
- Modify: `hpm_ai_v3/agents/discovery_agent.py`

- [ ] **Step 1: Failure Tracking**
    - [ ] Track consecutive failures per task type/domain in `UnifiedDiscoveryAgent`.
    - [ ] Increment counter on reward <= 0.0, reset on reward > 0.9.

- [ ] **Step 2: Trigger Logic in `run_episode`**
    - [ ] If consecutive failures == 3:
        - [ ] Find `LanguageModelPattern` in population.
        - [ ] Call `buffer.fetch_wikipedia` with task keywords.
        - [ ] Call `lm_pattern.fine_tune` with fetched text.
        - [ ] Retry task execution in the same episode (if max_steps allows) or next.

---

### Task 3: Online Linguistic Curriculum

**Files:**
- Create: `hpm_ai_v3/data/curriculums/phase_online_linguistics.json`

- [ ] **Step 1: Define Tasks**
    - [ ] Mix of familiar and unfamiliar domains (e.g., "Extract numeric values from: ...").
    - [ ] Unfamiliar domains designed to fail initially (e.g., medical or astronomical data).

---

### Task 4: Verification & Testing

**Files:**
- Create: `hpm_ai_v3/tests/test_online_lm.py`

- [ ] **Step 1: Test Fetch Trigger**
    - [ ] Mock Wikipedia API.
    - [ ] Verify 3 failures trigger a fetch attempt.

- [ ] **Step 2: Test Accuracy Improvement**
    - [ ] Compare LM performance on a specific task before and after fine-tuning on relevant text.

- [ ] **Step 3: Curriculum Progression Test**
    - [ ] Run agent through the online linguistic phase.
    - [ ] Verify it advances past failure-prone tasks after fetching knowledge.
