"""
L2MacroMixin — HPM Level 2: macro-composition of primitive patterns.

Provides:
- register_macro(name, nodes): store a named macro sequence
- _try_decompose(inputs, outputs): solve by trying registered macros
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from hfn.hfn import HFN
from hpm_ai_v2.utils.state import S_DIM, DIM


class L2MacroMixin:
    """
    Mixin adding HPM L2 macro composition.

    Must be used with BaseHFNAgent (provides self.patterns, self.renderer,
    self.executor, self._check_outputs, self._compose_sequence).
    """

    def register_macro(
        self,
        name: str,
        nodes: List[HFN],
        protect: bool = True,
    ) -> HFN:
        """
        Compose nodes into a named macro and register it in the forest.

        Returns the composed macro HFN node.
        """
        macro = self._compose_sequence(nodes)
        if macro is None:
            raise ValueError(f"Cannot compose empty node list for macro '{name}'")
        macro.id = f"macro_{name}"
        macro.relation_type = "macro"
        self.patterns[name] = macro
        # Register in observer so it participates in retrieval
        if macro.id not in self.forest:
            self.observer.register(macro, protected=protect, initial_weight=0.3)
        return macro

    def _try_decompose(
        self,
        inputs: List[Any],
        outputs: List[Any],
    ) -> Optional[List[HFN]]:
        """
        Strategy: try each registered macro directly against the task.

        Returns [macro_node] if a matching macro is found, else None.
        """
        for name, macro in self.patterns.items():
            if macro.relation_type != "macro":
                continue
            code = self.renderer.render(macro)
            results, errors = self.executor.run_batch(code, inputs)
            if self._check_outputs(results, outputs):
                return [macro]
        return None
