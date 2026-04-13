# Renderer Base Class Refactor Plan

## Objective
Introduce a `Renderer` abstract base class to make the HPM agent layer truly domain-agnostic regarding output generation. By decoupling the rendering logic from `BaseHFNAgent`, we enable the system to support diverse output formats (e.g., Python code via `ASTRenderer`, mathematical formulas via `MathRenderer`, JSON, etc.) without modifying the core learning and planning architecture.

## Motivation
Currently, `BaseHFNAgent` hardcodes the instantiation and use of `ASTRenderer`. This ties the agent layer implicitly to generating Python ASTs and source code, contradicting the goal of complete domain-agnosticism. By formalising rendering behind an interface, we improve testability (mock renderers), extensibility (new formats), and separation of concerns.

## Implementation Steps

### 1. Define `Renderer` Abstract Base Class
**File:** `hpm_ai_v2/utils/base_renderer.py` (New File)
- Import `ABC` and `abstractmethod` from `abc`.
- Define the `Renderer(ABC)` class.
- Add abstract method `render(self, node: HFN) -> str` to generate a generic string representation.
- Add abstract method `render_function(self, node: HFN, func_name: str) -> str` to generate a standalone function or macro representation.

### 2. Refactor `ASTRenderer`
**File:** `hpm_ai_v2/utils/renderer.py`
- Import the new `Renderer` ABC from `base_renderer`.
- Update `ASTRenderer` to inherit from `Renderer`.
- Ensure `ASTRenderer` correctly implements the `render` and `render_function` abstract methods (it already does, but we formalise the inheritance).

### 3. Update `BaseHFNAgent` Initialization
**File:** `hpm_ai_v2/agents/base_agent.py`
- Import the `Renderer` ABC.
- Modify `BaseHFNAgent.__init__` to accept an optional `renderer: Optional[Renderer] = None` parameter.
- Update the initialization logic: `self.renderer = renderer if renderer is not None else ASTRenderer(config)`. This provides full backward compatibility for existing code.

### 4. Update Exports
**File:** `hpm_ai_v2/utils/__init__.py`
- Export the `Renderer` ABC alongside `ASTRenderer` so it is easily accessible for future implementations.

## Verification
- **Backward Compatibility:** Run the existing multi-specialist social learning experiment (`experiment_ms_sl.py` via smoke test) and the collaborative math-physics experiment (`experiment_math_physics.py`). Both should execute successfully without modifications, proving the default fallback to `ASTRenderer` works perfectly.
- **Unit Tests:** Run the full `pytest tests/hfn/` test suite to ensure no core functionality is disrupted.
