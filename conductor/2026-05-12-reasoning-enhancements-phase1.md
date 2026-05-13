# Plan: Phase 1 - Analogical Binding

## Objective
Relax strict node unifications to allow analogical substitutions when matching subgraph antecedents in `ReasoningAgent._edge_matches_template`.

## Changes to `hpm_ai_v6/agents/reasoning_agent.py`
Modify `_edge_matches_template` to attempt analogical matching for `exact_key` and `exact_name` when they don't match exactly. We will use the `_analogy_cache` to determine if a node is an analog of the required node.

### Detailed Logic for `_edge_matches_template`
```python
        for endpoint_name, cell in (("source", edge.source), ("target", edge.target)):
            exact_key = template.get(f"{endpoint_name}_key")
            exact_name = template.get(f"{endpoint_name}_name")
            
            # Check exact match first
            is_exact_match = True
            if exact_key and self._cell_key(cell) != str(exact_key):
                is_exact_match = False
            if exact_name and cell.name != str(exact_name):
                is_exact_match = False

            if not is_exact_match:
                # Attempt analogical binding if exact match fails
                # We need the target cell to compare against. Let's find it.
                target_cell_name = str(exact_name) if exact_name else None
                if not target_cell_name and exact_key:
                    # Very simple inference: just strip "word:" or whatever prefix
                    # Realistically, exact_key is e.g. "word:rabbit", exact_name is "word_rabbit"
                    target_cell_name = str(exact_key).replace(":", "_")
                
                analog_found = False
                if target_cell_name:
                    # Look up in analogy cache
                    analogs = self._analogy_cache.get(target_cell_name, [])
                    for score, analog_cell in analogs:
                        if analog_cell.name == cell.name and score >= self.analogy_threshold:
                            analog_found = True
                            penalty *= max(score, 0.5)
                            break
                            
                    # If not found in cache, calculate directly if target_cell is available in node_index
                    if not analog_found:
                         target_cell = self._node_index.get(str(exact_key)) if exact_key else None
                         if target_cell is None and target_cell_name:
                             # try looking up by name
                             for c in self._node_index.values():
                                 if c.name == target_cell_name:
                                     target_cell = c
                                     break
                         
                         if target_cell:
                             try:
                                 score = target_cell.similarity(cell)
                                 if score >= self.analogy_threshold:
                                     analog_found = True
                                     penalty *= max(score, 0.5)
                             except Exception:
                                 pass
                
                if not analog_found:
                    return None

            var_name = template.get(f"{endpoint_name}_var")
            if isinstance(var_name, str) and var_name:
                bound = next_bindings.get(var_name)
                if bound is not None:
                    if self._cell_key(bound) != self._cell_key(cell):
                        # Attempt analogical binding
                        sim = bound.similarity(cell)
                        if sim < self.analogy_threshold:
                            return None
                        # Apply penalty based on similarity (Phase 1, Step 3)
                        penalty *= max(sim, 0.5)
                else:
                    next_bindings[var_name] = cell
```

## Changes to `hpm_ai_v6/tests/test_reasoning_agent.py`
Add a new test `test_backward_chaining_uses_analogical_binding` to verify that backward chaining succeeds when an intermediate node is replaced by an analog.

## Steps
1. Update `_edge_matches_template` in `reasoning_agent.py`.
2. Add the test in `test_reasoning_agent.py`.
3. Verify tests pass.
4. Exit plan mode.