# SP23.1: Metacognitive Tabooing & Saliency Diversity

**Goal**: Break the "Saliency Trap" loop where the AI repeatedly attempts the same failed refactor on the same module.

---

### Phase 1: Metacognitive Feedback Loop
- [ ] **Update `hpm_ai_v1/core/librarian.py`**:
    - Implement `taboo_list` (Set of modules currently deemed "unrefactorable").
    - Implement `failure_counts` (Track consecutive L5 rejections per module).
    - Implement `report_failure(filepath)`: Increment count and taboo if count >= 3.
    - Implement `target_history`: Track recently targeted modules to ensure diversity.

### Phase 2: Diversity-Driven Saliency
- [ ] **Refactor `Librarian.get_most_salient_target()`**:
    - Apply a **Recency Penalty** to modules in `target_history`.
    - Filter out any module in the `taboo_list`.
    - If all potential targets are taboo, clear 50% of the oldest taboos to allow re-exploration.

### Phase 3: Main Loop Integration
- [ ] **Modify `hpm_ai_v1/main.py`**:
    - When `l5_monitor.evaluate_changeset` returns `False`, call `librarian.report_failure(target_path)`.
    - Ensure the next generation loop correctly picks the *next* most salient target if the previous one was penalized or tabooed.

---

## Verification
- **Test Case**: Run HPM AI on a target that is guaranteed to fail (e.g., due to a complex dependency break).
- **Expected Outcome**: After 3 generations of failure, the Librarian should print `Module ... entered TABOO state` and switch to a different module for Generation 4.
