"""
L4ForwardModelMixin — HPM Level 4: generative forward model / mental simulation.

Provides:
- _record_transitions(path, inputs): learn per-node state deltas
- _try_imagine(inputs, outputs): BFS using forward model (zero oracle calls during search)
"""
from __future__ import annotations

from collections import deque
from typing import Any, List, Optional

import numpy as np

from hfn.hfn import HFN
from hpm_ai_v2.utils.forward_model import StateTransitionModel
from hpm_ai_v2.utils.hfn_forward_model import HFNStateTransitionModel


class L4ForwardModelMixin:
    """
    Mixin adding HPM L4 generative forward model capabilities.

    Must be used with BaseHFNAgent (provides self.forest, self.retriever,
    self.renderer, self.executor, self.oracle, self._check_outputs,
    self._compose_sequence, self.m_dim).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        
        use_hfn = kwargs.get("use_hfn_forward_model", False)
        cold_dir = kwargs.get("forward_model_cold_dir")
        
        if use_hfn:
            self.forward_model = HFNStateTransitionModel(self.config, cold_dir)
        else:
            self.forward_model = StateTransitionModel()
            
        self._oracle_calls_imagination = 0

    # ------------------------------------------------------------------
    # Transition recording (training phases)
    # ------------------------------------------------------------------

    def _record_transitions(
        self,
        path: List[HFN],
        inputs: List[Any],
    ) -> None:
        """
        Step through path one node at a time, calling oracle at each prefix.
        Records per-step deltas in self.forward_model.
        """
        states: List[np.ndarray] = []

        # Baseline (empty path) state
        empty_results, empty_errors = self.executor.run_batch("", inputs)
        baseline = self.oracle.compute_state(empty_results, empty_errors, "")
        states.append(baseline)

        for i in range(1, len(path) + 1):
            prefix = path[:i]
            composed = self._compose_sequence(prefix)
            if composed is None:
                states.append(states[-1].copy())
                continue
            code = self.renderer.render(composed)
            results, errors = self.executor.run_batch(code, inputs)
            state = self.oracle.compute_state(results, errors, code)
            states.append(state)

        self.forward_model.record_path(path, states)

    # ------------------------------------------------------------------
    # Imaginative BFS strategy
    # ------------------------------------------------------------------

    def _try_imagine(
        self,
        inputs: List[Any],
        outputs: List[Any],
        max_depth: int = 4,
        beam_width: int = 8,
    ) -> Optional[List[HFN]]:
        """
        Strategy: BFS using forward model predictions.

        Zero oracle calls during search; oracle called once at end to verify.
        """
        goal_state = self._outputs_to_goal_state(outputs)
        query = HFN(mu=goal_state, sigma=np.ones(self.m_dim), use_diag=True)
        
        if hasattr(self, "_candidate_ops") and self._candidate_ops:
            primitives = self._candidate_ops
        else:
            primitives = self.retriever.retrieve(query, k=beam_width)
            
        if not primitives:
            return None

        # Compute baseline state (no code executed)
        empty_results, empty_errors = self.executor.run_batch("", inputs)
        start_state = self.oracle.compute_state(empty_results, empty_errors, "")

        # Goal: structural dims of the goal state (which is in the delta slice)
        idx_offset = self.s_dim + self.dim
        goal_struct = goal_state[idx_offset + np.array(self.config.STRUCT_DIMS)]

        queue: deque = deque()
        for p in primitives:
            predicted = self.forward_model.predict(start_state, p)
            queue.append(([p], predicted))

        visited_ids: set = set()
        best_path: Optional[List[HFN]] = None
        best_dist = float('inf')

        while queue:
            path, pred_state = queue.popleft()
            path_key = tuple(n.id for n in path)
            if path_key in visited_ids:
                continue
            visited_ids.add(path_key)

            pred_struct = pred_state[self.config.STRUCT_DIMS]
            dist = float(np.sum((pred_struct - goal_struct) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_path = path

            if len(path) < max_depth:
                next_nodes = self.retriever.retrieve(query, k=beam_width)
                for nxt in next_nodes:
                    new_path = path + [nxt]
                    new_key = tuple(n.id for n in new_path)
                    if new_key not in visited_ids:
                        new_pred = self.forward_model.predict(pred_state, nxt)
                        queue.append((new_path, new_pred))

        if best_path is None:
            return None

        # Verify with one real oracle call
        self._oracle_calls_imagination += 1
        composed = self._compose_sequence(best_path)
        if composed is None:
            return None
        code = self.renderer.render(composed)
        results, errors = self.executor.run_batch(code, inputs)
        if self._check_outputs(results, outputs):
            return best_path
        return None
