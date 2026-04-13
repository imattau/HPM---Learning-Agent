"""
SequentialCompositionMixin — robust sequential composition of macros via AST.

Provides:
- compose_sequential(m1, m2): generate a wrapper calling m1 then m2
- _try_sequential_compose(inputs, outputs): solve by composing existing macros
"""
from __future__ import annotations

import ast
import textwrap
from typing import Any, List, Optional, Tuple

import numpy as np

from hfn.hfn import HFN


class SequentialCompositionMixin:
    """
    Mixin adding robust sequential composition of stored patterns.

    Must be used with BaseHFNAgent (provides self.patterns, self.renderer,
    self.executor, self._check_outputs, self.m_dim, self.forest, self.observer).
    """

    def compose_sequential(
        self,
        macro_a_name: str,
        macro_b_name: str,
        new_name: str,
        sample_inputs: List[Any] = None,
    ) -> Optional[HFN]:
        """
        Create a new macro that applies macro_a then macro_b.

        Uses AST to generate a new function body that calls the two macros
        as sub-functions.
        """
        macro_a = self.patterns.get(macro_a_name)
        macro_b = self.patterns.get(macro_b_name)
        if macro_a is None or macro_b is None:
            return None

        # Render each macro as a standalone function with a unique name
        func_a_name = f"_macro_{macro_a_name.replace('-', '_')}"
        func_b_name = f"_macro_{macro_b_name.replace('-', '_')}"
        code_a = self.renderer.render_function(macro_a, func_a_name)
        code_b = self.renderer.render_function(macro_b, func_b_name)

        # Build the composite body (direct calls, not a function def)
        composite_body = textwrap.dedent(f"""
            _tmp = {func_a_name}(inp)
            return {func_b_name}(_tmp)
        """).strip()

        # Combine all code
        full_code = code_a + "\n\n" + code_b + "\n\n" + composite_body

        # Parse and validate the code
        try:
            ast.parse(full_code)
        except SyntaxError as e:
            print(f"  [COMPOSE] Syntax error in generated code: {e}")
            return None

        # Create a new HFN node to represent this composite macro
        node = HFN(
            mu=np.zeros(self.m_dim),
            sigma=np.ones(self.m_dim),
            id=f"macro_{new_name}",
            inputs=[macro_a, macro_b],
            relation_type="macro",
            use_diag=True,
        )
        # Store the pre-generated code in node._code (so renderer uses it)
        node._code = full_code
        
        self.patterns[new_name] = node
        if node.id not in self.forest:
            self.observer.register(node, protected=False, initial_weight=0.5)

        return node

    def _try_sequential_compose(
        self,
        inputs: List[Any],
        outputs: List[Any],
    ) -> Optional[List[HFN]]:
        """
        Strategy: attempt to create a composite macro from all pairs of
        existing macros and test if it solves the task.
        """
        macro_names = list(self.patterns.keys())
        if len(macro_names) < 2:
            return None

        for i in range(len(macro_names)):
            for j in range(len(macro_names)):
                # We allow i == j for double application (e.g. +1 then +1)
                new_name = f"compose_{macro_names[i]}_then_{macro_names[j]}"
                composite = self.compose_sequential(
                    macro_names[i],
                    macro_names[j],
                    new_name,
                    sample_inputs=inputs,
                )
                if composite is None:
                    continue
                # Test the composite macro on the actual task
                code = composite._code
                results, errors = self.executor.run_batch(code, inputs)
                if self._check_outputs(results, outputs):
                    return [composite]
        return None
