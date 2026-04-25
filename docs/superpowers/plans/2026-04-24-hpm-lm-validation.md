# 2026-04-24-hpm-lm-validation.md

# HPM + LM "Actually Working" Validation Plan

**Goal:** Establish empirical proof that the `LanguageModelPattern` and `ToolSelector` improve learning efficiency and population structure.

---

### Task 1: Diagnostic Logging Infrastructure

**Files:**
- Create: `hpm_ai_v3/diagnostics.py`
- Modify: `hpm_ai_v3/task8/train_cold_start.py`

- [ ] **Step 1: Implement `HPMLogger`**
    - [ ] `log_episode(episode_data)`: Record phase, step-0 tool, success, top 3 weights.
    - [ ] `log_lm_stats(loss)`: Record LM loss.
    - [ ] `save_summary(path)`: Save to JSON/CSV for plotting.

- [ ] **Step 2: Wire into `train_cold_start.py`**
    - [ ] Pass the first selected tool of each episode to the logger.

---

### Task 2: Rich Semantic Corpus

**Files:**
- Create: `hpm_ai_v3/data/lm_corpus/rich_corpus.txt`

- [ ] **Step 1: Author Rich Corpus**
    - [ ] Include 500+ lines of tool documentation, task-to-tool mappings, and usage examples (e.g., "To count words, use the split tool.").
    - [ ] Update `LanguageModelPattern.pretrain` to use this corpus by default.

---

### Task 3: ToolSelector Refinement (Dynamic Alpha & Cache)

**Files:**
- Modify: `hpm_ai_v3/tools/tool_selector.py`
- Modify: `hpm_ai_v3/neural_lm_pattern.py`

- [ ] **Step 1: Dynamic Alpha**
    - [ ] Implement `alpha(episode_count)`: Start high (e.g., 2.0), decay to base (0.5) as replicator signal accumulates.

- [ ] **Step 2: Cache Consistency**
    - [ ] Add `clear_cache()` to `ToolSelector`.
    - [ ] Trigger `clear_cache()` in `LanguageModelPattern.fine_tune()` and `pretrain()`.

---

### Task 4: Internal Validation for Distillation

**Files:**
- Modify: `hpm_ai_v3/neural_lm_pattern.py`

- [ ] **Step 1: Internal Validation Set**
    - [ ] Define a small internal set of number extraction tasks.
    - [ ] `validate_internal()`: Measure accuracy on this set.
    - [ ] Update distillation trigger in training scripts to use this internal metric.

---

### Task 5: Comparative Experiment

**Files:**
- Create: `hpm_ai_v3/experiments/verify_lm_integration.py`

- [ ] **Step 1: Side-by-Side Run**
    - [ ] Run 1000 episodes for Agent A (Baseline) and Agent B (LM + ToolSelector).
    - [ ] Compare: Max phase reached, avg episodes per phase, weight concentration (Gini coefficient).
    - [ ] Print "Success" if Agent B outperforms Agent A significantly.
