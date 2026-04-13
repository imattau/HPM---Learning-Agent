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
        k_components: int = 1,
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

        # Apply pluggable probabilistic model if k_components > 1
        if k_components > 1:
            from hfn.probabilistic_models import GaussianMixtureModel
            # Initialize components near the mean with small noise
            mus = [macro.mu + np.random.randn(self.m_dim) * 0.1 for _ in range(k_components)]
            sigmas = [np.ones(self.m_dim) * 0.5 for _ in range(k_components)]
            macro.prob_model = GaussianMixtureModel.from_params(
                mus, sigmas, weights=[1.0 / k_components] * k_components, use_diag=True
            )

        self.patterns[name] = macro
        # Register in observer so it participates in retrieval
        if macro.id not in self.forest:
            self.observer.register(macro, protected=protect, initial_weight=0.3)
        return macro

    def register_code_macro(
        self,
        name: str,
        code_str: str,
        sample_inputs: Optional[List[Any]] = None,
        k_components: int = 1,
    ) -> "HFN":
        """
        Register a macro directly from a Python code string.

        Useful when the code is known (e.g. domain-expert seeding) and the
        renderer cannot derive it from structural nodes alone.  The code is
        stored as ``node._code`` so the renderer returns it verbatim.
        """
        from hfn.hfn import HFN

        # Run code to obtain an empirical state vector
        if sample_inputs is None:
            sample_inputs = [1]
        results, errors = self.executor.run_batch(code_str, sample_inputs)
        mu = self.oracle.compute_state(results, errors, code_str)
        # Pad to full node dimensionality
        full_mu = np.zeros(self.m_dim)
        full_mu[:len(mu)] = mu

        node = HFN(
            mu=full_mu,
            sigma=np.ones(self.m_dim) * 0.5,
            id=f"macro_{name}",
            relation_type="macro",
            use_diag=True,
        )

        if k_components > 1:
            from hfn.probabilistic_models import GaussianMixtureModel
            mus = [full_mu + np.random.randn(self.m_dim) * 0.1 for _ in range(k_components)]
            sigmas = [np.ones(self.m_dim) * 0.5 for _ in range(k_components)]
            node.prob_model = GaussianMixtureModel.from_params(
                mus, sigmas, weights=[1.0 / k_components] * k_components, use_diag=True
            )

        node._code = code_str  # stored verbatim for renderer
        self.patterns[name] = node
        if node.id not in self.forest:
            self.observer.register(node, protected=True, initial_weight=0.5)
        return node

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
