# Renderer Refactoring Plan

## Objective
Move the list-specific renderer (`ASTRenderer`) from the generic `utils` package into the `domains` package, renaming it to `ListRenderer`. This aligns the renderer architecture with the domain-agnostic principles established during the Oracle refactoring.

## Scope
- Rename `ASTRenderer` to `ListRenderer`.
- Move `hpm_ai_v2/utils/renderer.py` to `hpm_ai_v2/domains/list_renderer.py`.
- Update `BaseHFNAgent` to import `ListRenderer` from `domains.list_renderer`.
- Update `hpm_ai_v2/utils/__init__.py` to remove `ASTRenderer`.
- Update `README.md` to reflect the new structure.

## Proposed Solution
1. **Move and Rename File:**
   - Execute `mv hpm_ai_v2/utils/renderer.py hpm_ai_v2/domains/list_renderer.py`.
2. **Update Class Name inside `list_renderer.py`:**
   - Replace class definition `ASTRenderer` with `ListRenderer`.
   - Update docstrings referencing `ASTRenderer`.
3. **Update Agent Imports:**
   - In `hpm_ai_v2/agents/base_agent.py`, replace `from hpm_ai_v2.utils.renderer import ASTRenderer` with `from hpm_ai_v2.domains.list_renderer import ListRenderer`.
   - Change the fallback initialization to `ListRenderer(config)`.
4. **Update `utils/__init__.py`:**
   - Remove `from hpm_ai_v2.utils.renderer import ASTRenderer`.
   - Remove `"ASTRenderer"` from `__all__`.
5. **Update Documentation:**
   - Find and replace references to `ASTRenderer` in `README.md` and related documentation files, pointing to `ListRenderer`.

## Verification
- Run a `grep_search` to verify all `ASTRenderer` references are gone or properly addressed.
- Run one of the core benchmarks or the SP71/SP73 experiments to ensure no imports are broken.
